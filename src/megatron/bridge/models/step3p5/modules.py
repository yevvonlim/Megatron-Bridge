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
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP
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
    ) -> None:
        super().__init__()
        if len(rope_theta_cycle) != len(partial_rotary_cycle):
            raise ValueError(
                "rope_theta_cycle and partial_rotary_cycle must be the same length, "
                f"got {len(rope_theta_cycle)} vs {len(partial_rotary_cycle)}."
            )
        self.num_slots = len(rope_theta_cycle)
        self.kv_channels = kv_channels

        # Per-slot rotary dim and inv_freq.
        rotary_dims: list[int] = []
        inv_freqs: list[torch.Tensor] = []
        attention_scalings: list[float] = []
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
            inv_freqs.append(inv_freq)
            attention_scalings.append(scale)

        self.rotary_dims = tuple(rotary_dims)
        self.max_rotary_dim = max(rotary_dims)
        # Register inv_freq buffers per slot. We pad shorter slots with NaN so the
        # stack has uniform last-dim; callers must slice to the per-slot rotary_dim.
        for i, inv_freq in enumerate(inv_freqs):
            self.register_buffer(f"inv_freq_{i}", inv_freq, persistent=False)
        self.register_buffer(
            "attention_scaling",
            torch.tensor(attention_scalings, dtype=torch.float32),
            persistent=False,
        )

    def _inv_freq(self, slot: int) -> torch.Tensor:
        return getattr(self, f"inv_freq_{slot}")

    def forward(self, max_seq_len: int, offset: int = 0) -> Tensor:
        """Return a stacked cos/sin tensor.

        The returned tensor has shape ``(num_slots, max_seq_len,
        max_rotary_dim)`` for cos and sin, packed as ``(2, num_slots,
        ...)`` along a leading axis. Callers index slot first.

        Slots whose ``rotary_dim`` is smaller than ``max_rotary_dim`` are
        zero-padded on the trailing dim; ``Step3p5SelfAttention`` is
        responsible for slicing back to the slot's true ``rotary_dim``.
        """
        positions = torch.arange(offset, offset + max_seq_len, dtype=torch.float32, device=self._inv_freq(0).device)
        cos_per_slot: list[torch.Tensor] = []
        sin_per_slot: list[torch.Tensor] = []
        for slot in range(self.num_slots):
            inv_freq = self._inv_freq(slot)
            freqs = torch.outer(positions, inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            scaling = float(self.attention_scaling[slot])
            cos = emb.cos() * scaling
            sin = emb.sin() * scaling
            # Zero-pad to max_rotary_dim so we can stack.
            if cos.shape[-1] < self.max_rotary_dim:
                pad = self.max_rotary_dim - cos.shape[-1]
                cos = torch.nn.functional.pad(cos, (0, pad))
                sin = torch.nn.functional.pad(sin, (0, pad))
            cos_per_slot.append(cos)
            sin_per_slot.append(sin)
        cos_stack = torch.stack(cos_per_slot, dim=0)
        sin_stack = torch.stack(sin_per_slot, dim=0)
        # Pack as (2, num_slots, seq, max_rotary_dim).
        return torch.stack([cos_stack, sin_stack], dim=0)


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
    ) -> None:
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
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
        # Pick this layer's rope slot from the stacked rope tensor.
        if rotary_pos_emb is not None and rotary_pos_emb.ndim >= 3 and rotary_pos_emb.size(0) == 2:
            # Shape (2, num_slots, seq, max_rotary_dim) -> select slot.
            slot = self._rope_slot_index()
            rotary_pos_cos = rotary_pos_emb[0, slot]
            rotary_pos_sin = rotary_pos_emb[1, slot]
            # Slice trailing zero-padding for the slot's true rotary_dim.
            module_rope = self.config._step3p5_rotary_module
            rotary_dim = module_rope.rotary_dims[slot]
            if rotary_dim < module_rope.max_rotary_dim:
                rotary_pos_cos = rotary_pos_cos[..., :rotary_dim]
                rotary_pos_sin = rotary_pos_sin[..., :rotary_dim]
            rotary_pos_emb = None

        # Compute head-gate logits from the SAME hidden_states feeding q/k/v_proj.
        gate_logits = None
        if self.head_gate is not None:
            gate_out, _ = self.head_gate(hidden_states)
            gate_logits = gate_out  # shape: (seq, batch, n_heads_per_tp)

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

        if gate_logits is not None:
            # attn_output is (seq, batch, hidden_per_tp = n_heads_per_tp * head_dim)
            seq, batch, hidden_per_tp = attn_output.shape
            n_heads_per_tp = gate_logits.shape[-1]
            head_dim = hidden_per_tp // n_heads_per_tp
            gate = torch.sigmoid(gate_logits).to(attn_output.dtype)
            attn_output = attn_output.view(seq, batch, n_heads_per_tp, head_dim) * gate.unsqueeze(-1)
            attn_output = attn_output.view(seq, batch, hidden_per_tp)

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

    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward with optional gate/up clamp before the elementwise multiply."""
        clamp = self._resolve_clamp_value()
        if clamp is None:
            return super().forward(hidden_states)

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
