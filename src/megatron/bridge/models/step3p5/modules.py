# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom modules for Step-3.5-Flash.

Implements the architectural quirks that cannot be expressed via plain
``GPTModelProvider`` flags:

* :class:`Step3p5RotaryEmbedding` -- a stack of 4 per-cycle-slot rotary
  embeddings (theta/partial-rotary cycle ``[5e6,1e4,1e4,1e4]`` /
  ``[0.5,1,1,1]``); slot 0 (full-attention) gets a manually-constructed
  llama3 ``inv_freq`` while slots 1-3 use vanilla RoPE with
  ``attention_scaling=1.0``. Static (no ``@dynamic_rope_update``).
* :class:`Step3p5SelfAttention` -- adds a per-head sigmoid output gate
  (one scalar per query head, broadcast across ``head_dim``) applied
  post-attention pre-``o_proj``; selects the per-layer rope slot via
  ``(layer_number - 1) % 4``.
* :class:`Step3p5TEDotProductAttention` -- mirrors
  :class:`Gemma3TEDotProductAttention`; injects per-layer
  ``window_size = (511, 0)`` for sliding-attention layers.
* :class:`Step3p5MLP` and :class:`Step3p5GroupedMLP` -- runtime-aware
  asymmetric SwiGLU clamp (``gate.clamp(max=L)``,
  ``up.clamp(-L, L)``) gated on ``layer_number`` membership in
  ``config.routed_swiglu_clamp_layers`` /
  ``config.shared_swiglu_clamp_layers``.

References:
    Reference HF code at
    ``tests/unit_tests/models/step3p5/_reference/modeling_step3p5.py``.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch
from megatron.core import parallel_state
from megatron.core.models.common.embeddings.rope_utils import get_pos_emb_on_this_cp_rank
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.moe.experts import TEGroupedMLP
from torch import Tensor, nn

from megatron.bridge.utils.import_utils import safe_import_from


logger = logging.getLogger(__name__)


TEDotProductAttention, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TEDotProductAttention")


def _is_full_attn_layer(layer_number: int, period: int = 4) -> bool:
    """Return True if the (1-indexed Megatron) ``layer_number`` is a full-attention layer.

    Step-3.5-Flash places full-attention layers at HF 0-indexed positions
    ``{0, 4, 8, ...}`` (full FIRST, then 3 sliding) -- see
    ``reference/config.json`` ``layer_types`` and ``rope_theta`` cycle.

    Args:
        layer_number: Megatron-Core 1-indexed layer number.
        period: Cycle length (default 4 = 1 full + 3 sliding).

    Returns:
        True iff the layer is full-attention.
    """
    return (layer_number - 1) % period == 0


def _llama3_inv_freq(
    *,
    rotary_dim: int,
    rope_theta: float,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_position_embeddings: int,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, float]:
    """Compute the llama3 ``inv_freq`` and ``attention_scaling``.

    Mirrors HuggingFace's ``_compute_llama3_parameters`` so the four
    coefficients (factor / low_freq_factor / high_freq_factor /
    original_max_position_embeddings) match the Step-3.5 ``rope_scaling``
    block exactly. Megatron Core's stock ``RotaryEmbedding(rope_scaling=True)``
    hardcodes ``high_freq_factor=4.0`` and ``original_max_pe=8192`` -- those
    do NOT match Step-3.5's ``32.0`` / ``131072``.

    Args:
        rotary_dim: Rotary head dimension (i.e. ``head_dim *
            partial_rotary_factor``).
        rope_theta: Base RoPE theta.
        factor: llama3 scaling factor.
        low_freq_factor: llama3 low-frequency factor.
        high_freq_factor: llama3 high-frequency factor.
        original_max_position_embeddings: Original (pre-scaling) max
            position embeddings.
        device: Device on which to allocate the buffer.

    Returns:
        Tuple of ``(inv_freq, attention_scaling)``. ``attention_scaling``
        is a scalar to multiply cos/sin by post-init. For pure llama3
        scaling this is 1.0.
    """
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim))

    low_freq_wavelen = original_max_position_embeddings / low_freq_factor
    high_freq_wavelen = original_max_position_embeddings / high_freq_factor
    wavelen = 2 * math.pi / inv_freq

    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth_factor = (original_max_position_embeddings / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama, 1.0


class Step3p5RotaryEmbedding(nn.Module):
    """Stacked 4-slot rotary embedding for Step-3.5-Flash.

    Holds one inv_freq per cycle slot (each slot has independent theta,
    partial-rotary factor, and optional llama3 scaling). The ``forward``
    method returns a stacked cos/sin tensor of shape ``(num_slots, ...)``;
    callers index by ``(layer_number - 1) % num_slots``.

    NOTE: ``@dynamic_rope_update`` is intentionally *not* applied. Long-
    context dynamic NTK extension is out of scope for the from-scratch
    pretraining milestone. A future extension would need to break the
    static cache assumption -- see ``_forward_cached``.
    """

    def __init__(
        self,
        *,
        kv_channels: int,
        rope_theta_cycle: Tuple[float, ...],
        partial_rotary_cycle: Tuple[float, ...],
        yarn_only_full_attention: bool,
        llama3_factor: float,
        llama3_low_freq_factor: float,
        llama3_high_freq_factor: float,
        llama3_original_max_pe: int,
        rotary_interleaved: bool = False,
    ) -> None:
        super().__init__()
        if len(rope_theta_cycle) != len(partial_rotary_cycle):
            raise ValueError(
                "rope_theta_cycle and partial_rotary_cycle must be the same length, "
                f"got {len(rope_theta_cycle)} vs {len(partial_rotary_cycle)}."
            )
        self.num_slots = len(rope_theta_cycle)
        self.kv_channels = kv_channels
        self.rotary_interleaved = rotary_interleaved

        # Per-slot rotary dim and inv_freq.
        rotary_dims: list[int] = []
        inv_freqs: list[torch.Tensor] = []
        for slot_idx, (theta, partial_factor) in enumerate(zip(rope_theta_cycle, partial_rotary_cycle, strict=True)):
            rotary_dim = int(kv_channels * partial_factor)
            if rotary_dim % 2 != 0:
                raise ValueError(
                    f"rotary_dim must be even; got {rotary_dim} for slot {slot_idx} "
                    f"(kv_channels={kv_channels}, partial_factor={partial_factor})."
                )
            rotary_dims.append(rotary_dim)

            apply_llama3 = (slot_idx == 0) and yarn_only_full_attention
            if apply_llama3:
                inv_freq, scale = _llama3_inv_freq(
                    rotary_dim=rotary_dim,
                    rope_theta=theta,
                    factor=llama3_factor,
                    low_freq_factor=llama3_low_freq_factor,
                    high_freq_factor=llama3_high_freq_factor,
                    original_max_position_embeddings=llama3_original_max_pe,
                )
            else:
                inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
                scale = 1.0
            if scale != 1.0:
                # Step3p5RotaryEmbedding emits *freqs* (not cos/sin), so a non-unit
                # attention_scaling cannot be folded into the freqs without changing
                # what apply_rotary_pos_emb computes. HF llama3 returns 1.0 here.
                raise NotImplementedError(
                    f"Slot {slot_idx} requested attention_scaling={scale}; only 1.0 is "
                    "currently supported by the Step3p5 rotary module."
                )
            inv_freqs.append(inv_freq)

        self.rotary_dims = tuple(rotary_dims)
        self.max_rotary_dim = max(rotary_dims)
        for i, inv_freq in enumerate(inv_freqs):
            self.register_buffer(f"inv_freq_{i}", inv_freq, persistent=False)

    def _inv_freq(self, slot: int) -> torch.Tensor:
        return getattr(self, f"inv_freq_{slot}")

    def _slot_emb(self, slot: int, max_seq_len: int, offset: int) -> Tensor:
        """Build the per-slot freqs tensor in MCore's standard layout.

        Returns shape ``[seq, 1, 1, slot_rotary_dim]`` -- the ``freqs`` format
        that :func:`apply_rotary_pos_emb` consumes (cos/sin is computed inside
        the apply_* kernels). Mirrors :meth:`RotaryEmbedding.get_emb`.
        """
        inv_freq = self._inv_freq(slot)
        if inv_freq.device.type == "cpu" and torch.cuda.is_available():
            inv_freq = inv_freq.to(device=torch.cuda.current_device())
            self.register_buffer(f"inv_freq_{slot}", inv_freq, persistent=False)
        positions = torch.arange(offset, offset + max_seq_len, dtype=inv_freq.dtype, device=inv_freq.device)
        freqs = torch.outer(positions, inv_freq)  # [seq, slot_rotary_dim/2]
        if not self.rotary_interleaved:
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            emb = torch.stack((freqs.view(-1, 1), freqs.view(-1, 1)), dim=-1).view(freqs.shape[0], -1)
        return emb[:, None, None, :]  # [seq, 1, 1, slot_rotary_dim]

    def forward(
        self,
        max_seq_len: int,
        offset: int = 0,
        packed_seq: bool = False,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> Tensor:
        """Return per-slot freqs stacked along a leading axis.

        Output shape ``[num_slots, seq, 1, 1, max_rotary_dim]`` -- each slot is
        zero-padded on the trailing dim to ``max_rotary_dim``. Consumers
        (:class:`Step3p5SelfAttention`) pick a slot via
        ``(layer_number - 1) % num_slots`` and slice the trailing zero-pad to
        the slot's true ``rotary_dim`` before passing as ``rotary_pos_emb``.

        ``packed_seq=True`` skips CP slicing along the seq dim; the THD apply
        kernel handles CP later.
        """
        if cp_group is None:
            cp_group = parallel_state.get_context_parallel_group(check_initialized=False)
        per_slot: list[Tensor] = []
        for slot in range(self.num_slots):
            emb = self._slot_emb(slot, max_seq_len, offset)
            slot_rot_dim = self.rotary_dims[slot]
            if slot_rot_dim < self.max_rotary_dim:
                emb = torch.nn.functional.pad(emb, (0, self.max_rotary_dim - slot_rot_dim))
            if cp_group is not None and cp_group.size() > 1 and not packed_seq:
                emb = get_pos_emb_on_this_cp_rank(emb, 0, cp_group)
            per_slot.append(emb)
        return torch.stack(per_slot, dim=0)

    # Delegate to upstream — the body is independent of self state, so the unbound
    # method works correctly when bound to a Step3p5RotaryEmbedding instance.
    get_rotary_seq_len = RotaryEmbedding.get_rotary_seq_len


class Step3p5SelfAttention(SelfAttention):
    """Self-attention with per-head sigmoid output gate and per-layer RoPE slot.

    ``head_gate`` is a column-parallel ``Linear(hidden, num_attention_heads)``
    -- one scalar per query head, applied as
    ``attn_output * sigmoid(gate).unsqueeze(-1)`` AFTER core attention but
    BEFORE ``o_proj`` (mirrors ``modeling_step3p5.py:527-531``).

    The input to ``head_gate`` is the **same** ``hidden_states`` that feeds
    ``q_proj/k_proj/v_proj`` -- not ``q_proj`` output.

    Per-layer RoPE slot is selected by ``(self.layer_number - 1) % 4`` from
    the stacked rope tensor produced by :class:`Step3p5RotaryEmbedding`.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        pg_collection: Optional[ProcessGroupCollection] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            pg_collection=pg_collection,
            **kwargs,
        )
        if not getattr(config, "head_wise_attn_gate", False):
            self.head_gate = None
            return

        # head_gate is sharded the same way as q heads under TP: column-parallel along
        # the head axis. Use ColumnParallelLinear for the same partition convention.
        # Note: importing locally to avoid pulling MCore parallel modules at module-import
        # time (matches Gemma3 conventions).
        from megatron.core.tensor_parallel import ColumnParallelLinear

        self.head_gate = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.num_attention_heads,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="head_gate",
        )

        # The HF reference applies the per-head gate BEFORE ``o_proj``
        # (modeling_step3p5.py:527-531). MCore's ``Attention.forward`` calls
        # ``self.linear_proj`` internally, so we splice the gate in by
        # monkey-patching ``self.linear_proj.forward`` to read a
        # gate-logits tensor stashed on ``self`` at the start of forward.
        # Keeping the patch on the bound method preserves the original module's
        # parameters, hooks, and state_dict paths.
        self._cached_gate_logits: Optional[Tensor] = None
        self._head_dim = config.kv_channels
        original_proj_forward = self.linear_proj.forward

        def _gated_linear_proj_forward(x, *args, **kwargs):
            gate_logits = self._cached_gate_logits
            if gate_logits is not None:
                # x is the pre-o_proj activation. Standard path: (sq, b, n_heads_per_tp * head_dim).
                # THD packed: (t, 1, n_heads_per_tp * head_dim).
                n_heads_per_tp = self.num_attention_heads_per_partition
                hidden_per_tp = n_heads_per_tp * self._head_dim
                assert x.shape[-1] == hidden_per_tp, (
                    f"Step3p5 gated linear_proj expected last-dim {hidden_per_tp}, got {x.shape[-1]}"
                )
                gate = torch.sigmoid(gate_logits.float()).to(x.dtype)
                leading = x.shape[:-1]
                x = x.reshape(*leading, n_heads_per_tp, self._head_dim)
                x = x * gate.reshape(*gate.shape, 1)
                x = x.reshape(*leading, hidden_per_tp).contiguous()
                self._cached_gate_logits = None
            return original_proj_forward(x, *args, **kwargs)

        self.linear_proj.forward = _gated_linear_proj_forward

    def _rope_slot_index(self) -> int:
        period = len(getattr(self.config, "rope_theta_cycle", (1,)))
        return (self.layer_number - 1) % period

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context=None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin=None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params=None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params=None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward with per-layer RoPE slot and per-head output gate."""
        # Step3p5RotaryEmbedding emits a stacked freqs tensor of shape
        # [num_slots, seq, 1, 1, max_rotary_dim]. Slice this layer's slot and
        # trim the trailing zero-pad to the slot's true rotary_dim.
        if rotary_pos_emb is not None and rotary_pos_emb.ndim == 5:
            slot = self._rope_slot_index()
            module_rope = self.config._step3p5_rotary_module
            slot_rot_dim = module_rope.rotary_dims[slot]
            slot_emb = rotary_pos_emb[slot]
            if slot_rot_dim < module_rope.max_rotary_dim:
                slot_emb = slot_emb[..., :slot_rot_dim]
            # apply_rotary_pos_emb's fused/unfused kernels assume contiguous freqs.
            rotary_pos_emb = slot_emb.contiguous()
        # Always clear the inference-only cos/sin path -- upstream SelfAttention
        # asserts they are None outside flash-decode / flashinfer rope.
        rotary_pos_cos = None
        rotary_pos_sin = None

        # Compute head-gate logits from the SAME hidden_states feeding q/k/v_proj
        # and stash on self -- the patched ``self.linear_proj.forward`` reads them
        # and applies the gate to the pre-o_proj activation (mirrors HF semantics).
        if self.head_gate is not None:
            gate_out, _ = self.head_gate(hidden_states)
            self._cached_gate_logits = gate_out  # (seq, batch, n_heads_per_tp)

        try:
            attn_output, attn_bias = super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                key_value_states=key_value_states,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                inference_params=inference_params,
            )
        finally:
            # Make sure a stale gate doesn't leak into a later call if super
            # raised before consuming it.
            self._cached_gate_logits = None

        return attn_output, attn_bias


class Step3p5TEDotProductAttention(TEDotProductAttention):  # type: ignore[misc, valid-type]
    """Per-layer-window TE dot-product attention.

    Sliding-attention layers use ``window_size = (sliding_window - 1, 0)``
    (left-window convention, mirror ``Gemma3TEDotProductAttention``).
    Full-attention layers use ``window_size = None``.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        **kwargs,
    ) -> None:
        import copy

        config = copy.deepcopy(config)
        sliding_window = getattr(config, "sliding_window", None)
        if not _is_full_attn_layer(layer_number, period=len(config.rope_theta_cycle)) and sliding_window is not None:
            config.window_size = (sliding_window - 1, 0)
        else:
            config.window_size = None

        super().__init__(
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            attention_dropout=attention_dropout,
            **kwargs,
        )


def _step3p5_swiglu_clamp(gate: Tensor, up: Tensor, limit: float) -> Tuple[Tensor, Tensor]:
    """Apply the asymmetric Step-3.5 SwiGLU clamp.

    ``gate`` is one-sided upper-clamped to ``limit``; ``up`` is two-sided
    clamped to ``[-limit, limit]``. See ``modeling_step3p5.py:227-229``.
    """
    return gate.clamp(max=limit), up.clamp(min=-limit, max=limit)


class Step3p5MLP(MLP):
    """Dense / shared-expert MLP with optional asymmetric SwiGLU clamp.

    Reads ``self.layer_number`` (set by Megatron-Core's transformer layer
    loop) and consults ``config.shared_swiglu_clamp_layers`` /
    ``config.routed_swiglu_clamp_layers`` at forward time -- the spec is
    static, so per-layer behavior must be runtime-decided here.
    """

    def __init__(self, config: TransformerConfig, submodules, **kwargs) -> None:
        if config.bias_activation_fusion:
            raise ValueError(
                "Step3p5MLP requires bias_activation_fusion=False because the asymmetric "
                "SwiGLU clamp cannot be expressed inside the fused activation kernel."
            )
        # Upstream MLP no longer stores `is_expert` as an attribute; capture it here
        # so :meth:`_resolve_clamp_value` can pick the right clamp list.
        self.is_expert = bool(kwargs.get("is_expert", False))
        super().__init__(config=config, submodules=submodules, **kwargs)

    def _resolve_clamp_value(self) -> Optional[float]:
        # Layer numbers are 1-indexed in Megatron-Core; convert to 0-indexed for the
        # config tuples (which mirror HF's per-layer index lists).
        layer_idx_0 = (getattr(self, "layer_number", 1) or 1) - 1
        shared_layers = getattr(self.config, "shared_swiglu_clamp_layers", ()) or ()
        routed_layers = getattr(self.config, "routed_swiglu_clamp_layers", ()) or ()
        if self.is_expert:
            return float(self.config.routed_swiglu_clamp_value) if layer_idx_0 in routed_layers else None
        # Both the dense MLP (in dense layers) and share_expert (in MoE layers) use the
        # *shared* clamp list -- Step3p5DecoderLayer overloads `swiglu_limits_shared`
        # to both code paths (modeling_step3p5.py:559,572,577).
        return float(self.config.shared_swiglu_clamp_value) if layer_idx_0 in shared_layers else None

    def forward(self, hidden_states: Tensor, *args, **kwargs) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward with optional gate/up clamp before the elementwise multiply."""
        clamp = self._resolve_clamp_value()
        if clamp is None:
            return super().forward(hidden_states, *args, **kwargs)

        # Replicate the Megatron MLP forward but interpose the clamp between the
        # gate/up split and the activation. The fused-bias path is forbidden in
        # __init__, so the local activation_func path below is the canonical one.
        intermediate, bias_parallel = self.linear_fc1(hidden_states)
        if bias_parallel is not None:
            intermediate = intermediate + bias_parallel

        gate, up = torch.chunk(intermediate, 2, dim=-1)
        # HF: `gate = silu(gate_proj(x))` (post-SiLU one-sided), `up` raw two-sided.
        gate = torch.nn.functional.silu(gate)
        gate, up = _step3p5_swiglu_clamp(gate, up, clamp)
        intermediate = gate * up

        output, bias = self.linear_fc2(intermediate)
        return output, bias


class Step3p5GroupedMLP(TEGroupedMLP):
    """Routed-expert grouped MLP with Step-3.5 asymmetric SwiGLU clamp.

    Specializes :class:`megatron.core.transformer.moe.experts.TEGroupedMLP` for
    the same per-layer activation clamp that :class:`Step3p5MLP` applies on the
    dense / shared path. The HF reference clamps ``silu(gate)`` (post-SiLU) and
    ``up`` (two-sided) -- this differs from upstream's
    ``activation_func_clamp_value`` knob which clamps the *pre-activation*
    ``gate``. Therefore the override interposes the clamp inside
    :meth:`bias_act_func` rather than enabling the upstream config field.

    Layer membership is decided at runtime by reading ``self.layer_number``
    against ``config.routed_swiglu_clamp_layers`` (1-indexed Megatron layer
    number; converted to 0-indexed for HF parity). ``layer_number`` is set
    externally after model construction by the Step-3.5 provider since the
    upstream :class:`TEGroupedMLP` constructor does not receive it.

    Parameter naming and shapes are unchanged from upstream, so the bridge
    mappings (``FusedExpertMapping`` / ``FusedGatedExpertMapping``) continue
    to match.
    """

    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        submodules,
        pg_collection=None,
    ) -> None:
        if config.bias_activation_fusion:
            raise ValueError(
                "Step3p5GroupedMLP requires bias_activation_fusion=False because the asymmetric "
                "SwiGLU clamp cannot be expressed inside the fused activation kernel."
            )
        if getattr(config, "use_te_activation_func", False):
            raise ValueError(
                "Step3p5GroupedMLP requires use_te_activation_func=False; the post-SiLU clamp "
                "cannot be applied through the TE-fused activation path."
            )
        super().__init__(
            num_local_experts=num_local_experts,
            config=config,
            submodules=submodules,
            pg_collection=pg_collection,
        )
        # Set externally by Step3p5ModelProvider.provide() after MoELayer build.
        self.layer_number: Optional[int] = None

    def _resolve_routed_clamp_value(self) -> Optional[float]:
        """Return the routed clamp value if this layer is in the clamp list, else None."""
        layer_number = getattr(self, "layer_number", None)
        if layer_number is None:
            return None
        layer_idx_0 = layer_number - 1
        routed_layers = getattr(self.config, "routed_swiglu_clamp_layers", ()) or ()
        if layer_idx_0 not in routed_layers:
            return None
        return float(self.config.routed_swiglu_clamp_value)

    def bias_act_func(
        self,
        intermediate_parallel: Tensor,
        bias_parallel: Optional[Tensor],
        permuted_probs: Optional[Tensor],
    ) -> Tensor:
        """Apply Step-3.5 post-SiLU asymmetric clamp on routed-expert layers.

        Mirrors the canonical fallback path in
        :meth:`TEGroupedMLP.bias_act_func` (gated_linear_unit branch with no
        bias-activation fusion) but interposes the Step-3.5 clamp between
        ``silu(gate)`` and the elementwise multiply. Layers that are not in
        ``routed_swiglu_clamp_layers`` defer to the parent implementation.
        """
        clamp = self._resolve_routed_clamp_value()
        if clamp is None:
            return super().bias_act_func(intermediate_parallel, bias_parallel, permuted_probs)

        # Replicate the upstream non-fused, non-TE GLU branch (experts.py:308-324)
        # but with HF-faithful POST-SiLU clamp ordering.
        if bias_parallel is not None:
            intermediate_parallel = intermediate_parallel + bias_parallel
        gate, up = torch.chunk(intermediate_parallel, 2, dim=-1)
        gate = self.config.activation_func(gate)
        gate, up = _step3p5_swiglu_clamp(gate, up, clamp)
        intermediate_parallel = gate * (up + self.config.glu_linear_offset)
        if permuted_probs is not None:
            original_dtype = intermediate_parallel.dtype
            intermediate_parallel = intermediate_parallel * permuted_probs
            intermediate_parallel = intermediate_parallel.to(original_dtype)
        return intermediate_parallel

    def forward(
        self,
        permuted_local_hidden_states: Tensor,
        tokens_per_expert: Tensor,
        permuted_probs: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward with optional per-expert activation-dispersion logging.

        Step-3.5 §4.1.3 / Appendix B. The pathology Muon induces in MoE is
        invisible to training loss but visible in the dispersion of per-expert
        FFN output norms. We compute the local-rank max-to-median ratio of
        per-expert mean L2 norms on a configurable cadence and stash it on the
        config for downstream metric-emission callbacks. No metric is emitted
        when :attr:`enable_moe_dispersion_logging` is False.
        """
        output, output_bias = super().forward(permuted_local_hidden_states, tokens_per_expert, permuted_probs)
        if (
            self.training
            and getattr(self.config, "enable_moe_dispersion_logging", False)
            and self._should_log_dispersion()
        ):
            self._log_expert_norm_dispersion(output, tokens_per_expert)
        return output, output_bias

    def _should_log_dispersion(self) -> bool:
        """Decide whether to emit a dispersion datapoint this iteration.

        The cadence is read from ``config.moe_dispersion_log_every`` (default 50).
        We don't have access to the global iteration counter here; instead each
        instance keeps its own forward-call counter, which advances once per
        micro-batch step and is sufficient for monitoring trends.
        """
        every = int(getattr(self.config, "moe_dispersion_log_every", 50))
        if every <= 0:
            return False
        self._dispersion_step = getattr(self, "_dispersion_step", 0) + 1
        return (self._dispersion_step % every) == 0

    @torch.no_grad()
    def _log_expert_norm_dispersion(self, output: Tensor, tokens_per_expert: Tensor) -> None:
        """Compute local-rank per-expert norm dispersion and stash on config.

        ``output`` is shape ``(sum_local_tokens, hidden_size)`` grouped along the
        leading axis by ``tokens_per_expert``. We compute each local expert's
        mean-of-row-L2-norms (in float32 to avoid overflow), then the max-to-
        median ratio across local experts. The result is appended to
        ``config._moe_dispersion_log[layer_number]`` for collection by an
        external metric-emitter (training loop callback).
        """
        if output.numel() == 0:
            return
        sizes = tokens_per_expert.tolist() if isinstance(tokens_per_expert, Tensor) else list(tokens_per_expert)
        if len(sizes) == 0 or sum(sizes) != output.shape[0]:
            return
        per_expert_mean_norm: list[Tensor] = []
        cursor = 0
        for n in sizes:
            if n <= 0:
                # Skip empty experts so they don't drag the median to 0.
                cursor += n
                continue
            slab = output[cursor : cursor + n].float()
            per_expert_mean_norm.append(slab.norm(dim=-1).mean())
            cursor += n
        if len(per_expert_mean_norm) < 2:
            return
        norms = torch.stack(per_expert_mean_norm)
        max_n = norms.max()
        median_n = norms.median().clamp(min=torch.finfo(norms.dtype).tiny)
        ratio = (max_n / median_n).item()

        log_dict = getattr(self.config, "_moe_dispersion_log", None)
        if log_dict is None:
            log_dict = {}
            self.config._moe_dispersion_log = log_dict
        layer_key = self.layer_number if self.layer_number is not None else -1
        log_dict[layer_key] = {
            "max": max_n.item(),
            "median": median_n.item(),
            "max_over_median": ratio,
        }
