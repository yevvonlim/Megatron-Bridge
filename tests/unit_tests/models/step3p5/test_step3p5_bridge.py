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

"""Unit tests for the Step-3.5 HF<->Megatron mapping registry.

CPU-only tests that walk the mapping list and check coverage of the
parameters created by the *vendored* HF reference implementation.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from megatron.bridge.models.step3p5.common import get_step3p5_mapping_list


_REFERENCE_DIR = pathlib.Path(__file__).parent / "_reference"


def test_mapping_list_covers_top_level_params():
    mappings = get_step3p5_mapping_list(num_hidden_layers=24, num_mtp_layers=3)
    hf_keys = {m.hf_param if isinstance(m.hf_param, str) else None for m in mappings}
    assert "model.embed_tokens.weight" in hf_keys
    assert "model.norm.weight" in hf_keys
    assert "lm_head.weight" in hf_keys


def test_mapping_list_covers_step3p5_specific_naming():
    mappings = get_step3p5_mapping_list(num_hidden_layers=24, num_mtp_layers=0)
    hf_keys = [m.hf_param if isinstance(m.hf_param, str) else None for m in mappings]
    # Step-3.5 HF naming: `moe.router_bias`, NOT `moe.gate.e_score_correction_bias`
    assert "model.layers.*.moe.router_bias" in hf_keys
    # `share_expert` (singular), NOT `shared_experts`
    assert "model.layers.*.share_expert.down_proj.weight" in hf_keys
    # Per-head sigmoid output gate (head_gate)
    assert "model.layers.*.self_attn.g_proj.weight" in hf_keys
    # QK norm (per-head)
    assert "model.layers.*.self_attn.q_norm.weight" in hf_keys
    assert "model.layers.*.self_attn.k_norm.weight" in hf_keys


def test_mtp_layer_indices_use_loaded_num_hidden_layers():
    """MTP layer indices in HF state-dict are `num_hidden_layers + i`, dynamic.

    For the 45-layer original config they live at indices 45-47; for our 24-layer
    scaled config they should live at 24-26.
    """
    mappings = get_step3p5_mapping_list(num_hidden_layers=24, num_mtp_layers=3)
    hf_keys = [m.hf_param if isinstance(m.hf_param, str) else None for m in mappings]
    # Some MTP-specific extras hit at the dynamic indices.
    assert "model.layers.24.enorm.weight" in hf_keys
    assert "model.layers.25.eh_proj.weight" in hf_keys
    assert "model.layers.26.shared_head.norm.weight" in hf_keys
    # And nothing past 26 (the MTP block stops at num_hidden_layers + num_mtp_layers - 1)
    assert "model.layers.27.enorm.weight" not in hf_keys


def test_megatron_param_names_are_stable():
    """Sanity-check that Megatron-side params use the canonical conversion paths."""
    mappings = get_step3p5_mapping_list(num_hidden_layers=24, num_mtp_layers=0)
    megatron_keys = [m.megatron_param for m in mappings]
    assert "embedding.word_embeddings.weight" in megatron_keys
    assert "decoder.layers.*.mlp.router.expert_bias" in megatron_keys
    assert "decoder.layers.*.self_attention.head_gate.weight" in megatron_keys
    # Grouped-GEMM expert weights use the wildcard suffix Megatron expects.
    has_grouped = any(k.endswith("linear_fc1.weight*") for k in megatron_keys)
    assert has_grouped, "Expected grouped-GEMM expert mapping (linear_fc1.weight*)"


@pytest.mark.skipif(
    not (_REFERENCE_DIR / "config.json").exists(),
    reason="Vendored HF reference config not available.",
)
def test_reference_config_yields_expected_dimensions():
    """Sanity-check that the vendored HF config can be parsed for key fields used by the bridge."""
    cfg = json.loads((_REFERENCE_DIR / "config.json").read_text())
    assert cfg["model_type"] == "step3p5"
    assert cfg["architectures"] == ["Step3p5ForCausalLM"]
    assert cfg["use_moe"] is True
    assert cfg["moe_router_activation"] == "sigmoid"
    assert cfg["use_moe_router_bias"] is True
    assert cfg["need_fp32_gate"] is True
    assert cfg["use_head_wise_attn_gate"] is True
    # The hybrid attention pattern: full-attention layers are at every 4th index.
    layer_types = cfg["layer_types"]
    full_indices = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    assert all((i % 4) == 0 for i in full_indices)
    # `yarn_only_types` selects which layer types receive the llama3 RoPE scaling.
    assert "full_attention" in cfg["yarn_only_types"]
    # Per-layer rope_theta is a 4-cycle starting at 5e6.
    rope_theta = cfg["rope_theta"]
    assert rope_theta[0] == 5_000_000.0
    assert rope_theta[1] == 10_000.0
    # SwiGLU clamps live at the last 2 routed-expert layers and last shared-expert layer.
    routed_clamps = [(i, v) for i, v in enumerate(cfg["swiglu_limits"]) if v]
    shared_clamps = [(i, v) for i, v in enumerate(cfg["swiglu_limits_shared"]) if v]
    assert routed_clamps == [(43, 7), (44, 7)]
    assert shared_clamps == [(44, 16)]
