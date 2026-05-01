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

"""Megatron-Core provider for Step-3.5-Flash.

Implements every Step-3.5-Flash architectural quirk that can be expressed
through MCore configuration plus the few that need our custom modules in
:mod:`step3p5.modules`. See the design doc at
``/Users/ye/.claude/plans/i-want-to-convert-woolly-flask.md`` for the
flag-by-flag rationale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Tuple, Union

import torch
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.transformer import (
    ModuleSpec,
)
from megatron.core.transformer.enums import AttnBackend, AttnMaskType

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.step3p5.modules import (
    Step3p5GroupedMLP,
    Step3p5MLP,
    Step3p5RotaryEmbedding,
    Step3p5SelfAttention,
    Step3p5TEDotProductAttention,
)
from megatron.bridge.utils.import_utils import safe_import_from


logger = logging.getLogger(__name__)


TENorm, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TENorm")
TELayerNormColumnParallelLinear, _ = safe_import_from(
    "megatron.core.extensions.transformer_engine", "TELayerNormColumnParallelLinear"
)
TEColumnParallelLinear, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TEColumnParallelLinear")
TERowParallelLinear, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TERowParallelLinear")


def step3p5_layer_spec(config: "Step3p5ModelProvider", vp_stage=None):
    """Step-3.5-Flash decoder block spec with per-layer dense vs MoE.

    Builds on Megatron-Core's :func:`get_gpt_decoder_block_spec`, which honors
    ``config.moe_layer_freq`` and produces a list of per-layer specs (dense
    layers get a stock ``MLP``; MoE layers get a ``MoELayer`` with grouped-GEMM
    routed experts and a separate ``SharedExpertMLP``). We then patch in the
    Step-3.5 attention overrides on every layer and swap ``MLP -> Step3p5MLP``
    on dense layers so the SwiGLU clamp on the dense path still fires at
    runtime.

    Per-layer behavior (sliding vs full attention, head-gate, per-slot rope) is
    still runtime-decided via ``self.layer_number`` inside
    :class:`Step3p5SelfAttention` / :class:`Step3p5TEDotProductAttention`.

    Routed-expert SwiGLU clamp is wired by replacing the experts module in MoE
    layer specs with :class:`Step3p5GroupedMLP`. The grouped experts module
    does not receive ``layer_number`` natively, so the provider's ``provide``
    method assigns it after MoE-layer construction.

    Shared-expert SwiGLU clamp (``shared_swiglu_clamp_layers``) is not yet
    wired through ``SharedExpertMLP`` and will need a follow-up to apply
    identically to the HF reference. The dense-path clamp works correctly
    today.
    """
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
    from megatron.core.transformer.mlp import MLP
    from megatron.core.transformer.moe.experts import TEGroupedMLP

    block_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=True, vp_stage=vp_stage)

    for layer_spec in block_spec.layer_specs:
        # Step-3.5 attention overrides apply to every layer (dense and MoE).
        layer_spec.submodules.self_attention.module = Step3p5SelfAttention
        layer_spec.submodules.self_attention.params = {"attn_mask_type": AttnMaskType.causal}
        layer_spec.submodules.self_attention.submodules.core_attention = Step3p5TEDotProductAttention
        # SwiGLU clamp on dense layers -- swap stock MLP -> Step3p5MLP.
        if layer_spec.submodules.mlp.module is MLP:
            layer_spec.submodules.mlp.module = Step3p5MLP
        # SwiGLU clamp on routed experts -- swap TEGroupedMLP -> Step3p5GroupedMLP.
        # Guard on the parent class so future Megatron-Core spec changes (e.g. a
        # different default experts module) don't silently break us.
        mlp_submods = getattr(layer_spec.submodules.mlp, "submodules", None)
        experts_spec = getattr(mlp_submods, "experts", None) if mlp_submods is not None else None
        if experts_spec is not None and getattr(experts_spec, "module", None) is TEGroupedMLP:
            experts_spec.module = Step3p5GroupedMLP

    return block_spec


@dataclass
class Step3p5ModelProvider(GPTModelProvider):
    """Configuration and provider for Megatron Core Step-3.5-Flash models.

    The provider sets the static MCore-Core flags that describe Step-3.5's
    behavior; the per-layer-cycle behavior (4-slot rope, full/sliding
    pattern, SwiGLU clamp on a subset of layers) is enacted by the custom
    modules in :mod:`step3p5.modules`.
    """

    # ---- norm + attention ------------------------------------------------
    normalization: str = "RMSNorm"
    layernorm_epsilon: float = 1e-5
    layernorm_zero_centered_gamma: bool = True  # x * (1 + w)
    qk_layernorm: bool = True
    attention_softmax_in_fp32: bool = False
    attention_backend: AttnBackend = AttnBackend.flash
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    # ---- mlp -------------------------------------------------------------
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    activation_func: Callable = torch.nn.functional.silu

    # ---- step-3.5 quirks (consumed by step3p5.modules) -------------------
    head_wise_attn_gate: bool = True
    sliding_window: int = 512
    rope_theta_cycle: Tuple[float, ...] = (5_000_000.0, 10_000.0, 10_000.0, 10_000.0)
    partial_rotary_cycle: Tuple[float, ...] = (0.5, 1.0, 1.0, 1.0)
    yarn_only_full_attention: bool = True
    llama3_factor: float = 2.0
    llama3_low_freq_factor: float = 1.0
    llama3_high_freq_factor: float = 32.0
    llama3_original_max_pe: int = 131072
    routed_swiglu_clamp_layers: Tuple[int, ...] = ()
    routed_swiglu_clamp_value: float = 7.0
    shared_swiglu_clamp_layers: Tuple[int, ...] = ()
    shared_swiglu_clamp_value: float = 16.0

    # Per-MoE-layer expert-norm dispersion logging (paper §4.1.3 / Appendix B).
    # Off by default; enable in pretraining recipes to surface Muon-induced
    # rogue-expert pathology that training loss alone cannot detect.
    enable_moe_dispersion_logging: bool = False
    moe_dispersion_log_every: int = 50

    # Step-3.5 §3.2: route 3D grouped-expert weights to Muon. Default on so
    # provide() tags them via :func:`mark_3d_experts_for_muon`. Tagging is
    # harmless if the active optimizer is not ``muon_step3p5``.
    route_3d_experts_to_muon: bool = True

    # Mid-training / SFT / RL recipes (Step-3.5 §6) freeze the router. Default
    # off (pretraining keeps the router trainable).
    freeze_router: bool = False

    # Step-3.5 §2.2 eq. (1) per-EP-group balance loss coefficient. 0 = off.
    # Paper recommends 1e-3. Computed locally per rank; differs from
    # ``moe_aux_loss_coeff`` (per-expert) and ``global_aux_loss`` (TP+DP+CP-
    # reduced) by being explicitly per-EP-group.
    ep_group_balance_loss_coeff: float = 0.0

    # ---- rope ------------------------------------------------------------
    position_embedding_type: str = "rope"
    apply_rope_fusion: bool = False  # per-layer rope variation -> no fused kernel
    # rotary_base / rope_scaling are unused by us; we install a custom
    # Step3p5RotaryEmbedding in `provide()`. Keep MCore's default stubs.

    # ---- moe -------------------------------------------------------------
    moe_token_dispatcher_type: str = "alltoall"
    moe_grouped_gemm: bool = True
    moe_router_score_function: str = "sigmoid"
    moe_router_pre_softmax: bool = False  # Step-3.5 router_bias_func does NOT pre-normalize
    moe_router_enable_expert_bias: bool = True
    moe_router_bias_update_rate: float = 0.0  # HF router_bias is requires_grad=False
    moe_router_dtype: str = "fp32"
    moe_router_topk_scaling_factor: float = 3.0
    moe_router_load_balancing_type: str = "seq_aux_loss"
    moe_aux_loss_coeff: float = 1e-3
    moe_shared_expert_overlap: bool = True
    # Step-3.5's share_expert in HF has only down/gate/up (no shared-expert gate weight).
    # TransformerConfig defaults moe_shared_expert_gate=False which is what we want;
    # we don't redeclare so dataclass inheritance stays clean.
    moe_permute_fusion: bool = True

    # explicitly OFF -- not the per-Q-feature gate from Qwen3-Next
    attention_output_gate: bool = False

    # ---- fusion knobs ----------------------------------------------------
    bias_activation_fusion: bool = False  # SwiGLU clamp bypasses fused activation
    bias_dropout_fusion: bool = True
    masked_softmax_fusion: bool = True
    persist_layer_norm: bool = True
    cross_entropy_fusion_impl: str = "te"
    cross_entropy_loss_fusion: bool = True

    # ---- precision -------------------------------------------------------
    bf16: bool = True
    fp16: bool = False
    params_dtype: torch.dtype = torch.bfloat16
    autocast_dtype: torch.dtype = torch.bfloat16
    pipeline_dtype: torch.dtype = torch.bfloat16

    # ---- mtp -------------------------------------------------------------
    mtp_num_layers: int = 0
    mtp_loss_scaling_factor: float = 0.3

    # ---- vocab -----------------------------------------------------------
    make_vocab_size_divisible_by: int = 64  # 128896 is divisible by 64 but not 128
    share_embeddings_and_output_weights: bool = False  # tie_word_embeddings=False in HF

    transformer_layer_spec: Union[ModuleSpec, Callable[["Step3p5ModelProvider"], ModuleSpec]] = field(
        default_factory=lambda: step3p5_layer_spec
    )

    def __post_init__(self):
        super().__post_init__()
        if self.attention_output_gate:
            raise ValueError(
                "Step3p5ModelProvider requires attention_output_gate=False; the per-head "
                "head_gate is implemented inside Step3p5SelfAttention and is NOT the "
                "per-Q-feature gate that attention_output_gate enables."
            )
        if len(self.rope_theta_cycle) != len(self.partial_rotary_cycle):
            raise ValueError(
                "rope_theta_cycle and partial_rotary_cycle must have equal length; got "
                f"{len(self.rope_theta_cycle)} vs {len(self.partial_rotary_cycle)}."
            )

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> "MCoreGPTModel":
        """Configure and instantiate a Megatron-Core Step-3.5-Flash GPT model.

        Replaces the model's ``rotary_pos_emb`` with our custom 4-slot
        :class:`Step3p5RotaryEmbedding` after MCoreGPTModel construction
        (mirrors :meth:`Gemma3ModelProvider.provide`).
        """
        # MCore's GPTModel constructor reads `rotary_percent` to size its
        # internal RotaryEmbedding. We don't actually use that one (we
        # replace it below), but the constructor must not crash. Pin
        # rotary_percent=1.0; our custom rope handles per-slot partial.
        self.rotary_percent = 1.0
        if isinstance(self.rotary_base, (list, tuple)):
            self.rotary_base = float(self.rotary_base[0])
        model = super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)

        rope_module = Step3p5RotaryEmbedding(
            kv_channels=self.kv_channels,
            rope_theta_cycle=self.rope_theta_cycle,
            partial_rotary_cycle=self.partial_rotary_cycle,
            yarn_only_full_attention=self.yarn_only_full_attention,
            llama3_factor=self.llama3_factor,
            llama3_low_freq_factor=self.llama3_low_freq_factor,
            llama3_high_freq_factor=self.llama3_high_freq_factor,
            llama3_original_max_pe=self.llama3_original_max_pe,
            rotary_interleaved=self.rotary_interleaved,
        )
        model.rotary_pos_emb = rope_module
        # Stash the module on the config so Step3p5SelfAttention can read
        # per-slot rotary_dims when slicing the stacked rope tensor.
        self._step3p5_rotary_module = rope_module

        if hasattr(model, "embedding") or hasattr(model, "output_layer"):
            model.setup_embeddings_and_output_layer()

        # Wire layer_number onto each MoE layer's grouped-experts module so the
        # Step3p5GroupedMLP override can decide per-layer whether to apply the
        # routed SwiGLU clamp. TEGroupedMLP doesn't accept layer_number through
        # its constructor; the outer TransformerLayer / MoELayer carry it as an
        # attribute, which we propagate here.
        decoder = getattr(model, "decoder", None)
        if decoder is not None and hasattr(decoder, "layers"):
            for layer in decoder.layers:
                mlp = getattr(layer, "mlp", None)
                experts = getattr(mlp, "experts", None) if mlp is not None else None
                if isinstance(experts, Step3p5GroupedMLP):
                    experts.layer_number = getattr(layer, "layer_number", None)

        # Step-3.5 §3.2: paper applies Muon to all 2D matrices including the per-
        # expert routed weights. Megatron-Core's grouped-experts store those as
        # 3D ``(E, M, H)`` tensors; tag them so the Muon routing predicate
        # registered by ``muon_patches.register_step3p5_muon`` permits them
        # through to the Muon param group. Tagging is harmless when the
        # ``muon_step3p5`` optimizer is not selected (the attribute is just
        # ignored). Optional knob :attr:`route_3d_experts_to_muon` lets users
        # opt out; defaulting on matches the paper.
        if getattr(self, "route_3d_experts_to_muon", True):
            from megatron.bridge.models.step3p5 import muon_patches

            muon_patches.mark_3d_experts_for_muon(model)

        # Optional: freeze the MoE router (Step-3.5 mid-training / SFT recipe).
        # Pretraining keeps router trainable.
        if getattr(self, "freeze_router", False):
            for name, p in model.named_parameters():
                if name.endswith("mlp.router.weight") or name.endswith("router.weight"):
                    p.requires_grad_(False)

        # Step-3.5 §2.2 per-EP-group balance loss: opt-in via coefficient > 0.
        # Skipped silently when expert_model_parallel_size <= 1 since "EP groups"
        # is degenerate without EP sharding.
        ep_size = int(getattr(self, "expert_model_parallel_size", 1) or 1)
        if self.ep_group_balance_loss_coeff > 0 and ep_size > 1:
            from megatron.bridge.models.step3p5 import ep_balance

            ep_balance.install_ep_balance_loss(
                model,
                coeff=self.ep_group_balance_loss_coeff,
                num_ep_groups=ep_size,
            )

        return model


@dataclass
class Step3p5ModelProviderMini3B(Step3p5ModelProvider):
    """Step-3.5-Flash-Mini ~3B-total / ~0.27B-active config.

    Sized for from-scratch pretraining on an 8x A100-40GB cluster. See the
    plan doc for the rationale of every dimension.
    """

    # Architecture sizing
    hidden_size: int = 1024
    num_layers: int = 24  # 3 dense + 21 MoE
    num_attention_heads: int = 24  # uniform across full + sliding (deviation #1)
    num_query_groups: int = 4  # GQA 6:1
    kv_channels: int = 64
    ffn_hidden_size: int = 4096

    # MoE sizing
    num_moe_experts: int = 80
    moe_router_topk: int = 4
    moe_ffn_hidden_size: int = 512
    moe_shared_expert_intermediate_size: int = 512
    moe_layer_freq: list[int] = field(default_factory=lambda: [0] * 3 + [1] * 21)

    # MTP
    mtp_num_layers: int = 3

    # SwiGLU clamp at the last 2 routed-expert layers and last shared/dense layer
    routed_swiglu_clamp_layers: Tuple[int, ...] = (22, 23)
    routed_swiglu_clamp_value: float = 7.0
    shared_swiglu_clamp_layers: Tuple[int, ...] = (23,)
    shared_swiglu_clamp_value: float = 16.0

    # Vocab
    vocab_size: int | None = 128896

    # Sequence length for from-scratch pretraining
    seq_length: int = 4096
    max_position_embeddings: int = 262144

    # Init scale (DSv3-style; small-init for stability)
    init_method_std: float = 0.006
