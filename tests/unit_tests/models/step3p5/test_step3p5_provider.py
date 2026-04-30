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

"""Unit tests for the Step-3.5-Flash provider.

These tests run on CPU and only exercise dataclass / configuration
behavior. Forward-pass parity tests live under
``tests/functional_tests/models/step3p5/`` and require GPU + Megatron-Core.
"""

from __future__ import annotations

import pytest

from megatron.bridge.models.step3p5 import (
    Step3p5ModelProvider,
    Step3p5ModelProviderMini3B,
)


def test_default_provider_has_step3p5_quirks_on():
    p = Step3p5ModelProvider()
    assert p.layernorm_zero_centered_gamma is True
    assert p.qk_layernorm is True
    assert p.head_wise_attn_gate is True
    assert p.attention_output_gate is False  # crucial: not the per-Q-feature gate
    assert p.apply_rope_fusion is False  # per-layer rope variation
    assert p.bias_activation_fusion is False  # SwiGLU clamp incompatible with fused activation
    assert p.moe_router_score_function == "sigmoid"
    assert p.moe_router_pre_softmax is False
    assert p.moe_router_enable_expert_bias is True
    assert p.moe_router_bias_update_rate == 0.0  # frozen bias mirrors HF
    assert p.moe_router_dtype == "fp32"
    assert p.normalization == "RMSNorm"


def test_default_rope_cycle_matches_step3p5_pattern():
    p = Step3p5ModelProvider()
    assert p.rope_theta_cycle == (5_000_000.0, 10_000.0, 10_000.0, 10_000.0)
    assert p.partial_rotary_cycle == (0.5, 1.0, 1.0, 1.0)
    assert p.yarn_only_full_attention is True
    assert p.llama3_factor == 2.0
    assert p.llama3_high_freq_factor == 32.0
    assert p.llama3_original_max_pe == 131072


def test_attention_output_gate_must_be_false():
    with pytest.raises(ValueError, match="attention_output_gate=False"):
        Step3p5ModelProvider(attention_output_gate=True)


def test_rope_cycle_length_mismatch_rejected():
    with pytest.raises(ValueError, match="equal length"):
        Step3p5ModelProvider(
            rope_theta_cycle=(5e6, 1e4),
            partial_rotary_cycle=(0.5, 1.0, 1.0, 1.0),
        )


def test_mini3b_sized_config_matches_plan():
    p = Step3p5ModelProviderMini3B()
    assert p.hidden_size == 1024
    assert p.num_layers == 24
    assert p.num_attention_heads == 24  # uniform across full + sliding (deviation #1)
    assert p.num_query_groups == 4
    assert p.kv_channels == 64
    assert p.ffn_hidden_size == 4096
    assert p.num_moe_experts == 80
    assert p.moe_router_topk == 4
    assert p.moe_ffn_hidden_size == 512
    assert p.moe_shared_expert_intermediate_size == 512
    assert p.mtp_num_layers == 3
    assert p.routed_swiglu_clamp_layers == (22, 23)
    assert p.routed_swiglu_clamp_value == 7.0
    assert p.shared_swiglu_clamp_layers == (23,)
    assert p.shared_swiglu_clamp_value == 16.0
    assert p.vocab_size == 128896
    assert p.make_vocab_size_divisible_by == 64
    # 3 dense + 21 MoE.
    assert p.moe_layer_freq == [0] * 3 + [1] * 21


def test_mini3b_q_expansion_preserved():
    """The 1.5x Q-expansion ratio of the original Step-3.5 (full layers) is
    preserved at the smaller scale: heads * head_dim > hidden_size."""
    p = Step3p5ModelProviderMini3B()
    q_total = p.num_attention_heads * p.kv_channels
    assert q_total == 1536
    assert q_total > p.hidden_size, "Q-expansion ratio (heads*head_dim > hidden) lost"
