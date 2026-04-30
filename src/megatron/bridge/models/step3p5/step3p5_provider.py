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
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.mlp import MLPSubmodules

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.step3p5.modules import (
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


def step3p5_layer_spec(config: "Step3p5ModelProvider") -> ModuleSpec:
    """Step-3.5-Flash decoder layer spec.

    Static spec (called once for the whole stack); per-layer behavior --
    sliding vs full attention, swiglu clamp -- is decided at *runtime* by
    :class:`Step3p5SelfAttention` / :class:`Step3p5TEDotProductAttention`
    / :class:`Step3p5MLP` reading ``self.layer_number`` and the per-layer
    config tuples. MCore's spec function is invoked only once.
    """
    from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=Step3p5SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=Step3p5TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=TENorm if config.qk_layernorm else None,
                    k_layernorm=TENorm if config.qk_layernorm else None,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            mlp=ModuleSpec(
                module=Step3p5MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TELayerNormColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )


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
