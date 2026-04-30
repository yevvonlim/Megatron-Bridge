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

"""Shared HF<->Megatron parameter-mapping list for Step-3.5-Flash.

Step-3.5's HF parameter naming differs from DeepSeek-V3 in three ways:

* the MoE module is named ``moe`` (not ``mlp``);
* the shared expert is named ``share_expert`` (singular, not ``shared_experts``);
* the router bias is a fused parameter on the MoE block (``moe.router_bias``)
  -- not under ``moe.gate``.

Expert weights are stored as **two** separate 3D tensors of shape
``(num_experts, intermediate, hidden)`` (``moe.gate_proj.weight`` and
``moe.up_proj.weight``) -- not a single fused ``gate_up_proj``. We use
:class:`FusedExpertMapping` over each independently rather than
:class:`FusedGatedExpertMapping` since there's nothing to un-fuse.
"""

from __future__ import annotations

from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    FusedExpertMapping,
    GatedMLPMapping,
    QKVMapping,
)


def _base_layer_mappings(layer_prefix_megatron: str, layer_prefix_hf: str) -> list:
    """Build the base mapping list for a single decoder-layer prefix.

    Used both for the main decoder stack and the MTP transformer-layer
    blocks (which share the per-layer parameter shape).
    """
    autos = {
        # input layer norm (always standalone, since linear_qkv is TE-fused-LN-fc1)
        f"{layer_prefix_megatron}.input_layernorm.weight": f"{layer_prefix_hf}.input_layernorm.weight",
        # attention output proj
        f"{layer_prefix_megatron}.self_attention.linear_proj.weight": (f"{layer_prefix_hf}.self_attn.o_proj.weight"),
        # qk norm (per-head; weight is head_dim-sized)
        f"{layer_prefix_megatron}.self_attention.q_layernorm.weight": (f"{layer_prefix_hf}.self_attn.q_norm.weight"),
        f"{layer_prefix_megatron}.self_attention.k_layernorm.weight": (f"{layer_prefix_hf}.self_attn.k_norm.weight"),
        # head-wise sigmoid output gate (column-parallel along head axis)
        f"{layer_prefix_megatron}.self_attention.head_gate.weight": (f"{layer_prefix_hf}.self_attn.g_proj.weight"),
        # router weight + frozen bias
        f"{layer_prefix_megatron}.mlp.router.weight": f"{layer_prefix_hf}.moe.gate.weight",
        f"{layer_prefix_megatron}.mlp.router.expert_bias": f"{layer_prefix_hf}.moe.router_bias",
        # shared expert down proj + post-attention layernorm (MoE layer side)
        f"{layer_prefix_megatron}.mlp.shared_experts.linear_fc2.weight": (
            f"{layer_prefix_hf}.share_expert.down_proj.weight"
        ),
        f"{layer_prefix_megatron}.pre_mlp_layernorm.weight": (f"{layer_prefix_hf}.post_attention_layernorm.weight"),
        # dense MLP path: post-attention layernorm is fused into linear_fc1 (TE)
        f"{layer_prefix_megatron}.mlp.linear_fc1.layer_norm_weight": (
            f"{layer_prefix_hf}.post_attention_layernorm.weight"
        ),
        # dense MLP down proj (only present on dense layers)
        f"{layer_prefix_megatron}.mlp.linear_fc2.weight": f"{layer_prefix_hf}.mlp.down_proj.weight",
    }
    mapping = [AutoMapping(megatron_param=k, hf_param=v) for k, v in autos.items()]

    # Fused QKV from the three HF projections.
    mapping.append(
        QKVMapping(
            megatron_param=f"{layer_prefix_megatron}.self_attention.linear_qkv.weight",
            q=f"{layer_prefix_hf}.self_attn.q_proj.weight",
            k=f"{layer_prefix_hf}.self_attn.k_proj.weight",
            v=f"{layer_prefix_hf}.self_attn.v_proj.weight",
        )
    )

    # Dense MLP gate+up fusion (fc1 = [gate; up]).
    mapping.append(
        GatedMLPMapping(
            megatron_param=f"{layer_prefix_megatron}.mlp.linear_fc1.weight",
            gate=f"{layer_prefix_hf}.mlp.gate_proj.weight",
            up=f"{layer_prefix_hf}.mlp.up_proj.weight",
        )
    )

    # Shared expert gate+up fusion.
    mapping.append(
        GatedMLPMapping(
            megatron_param=f"{layer_prefix_megatron}.mlp.shared_experts.linear_fc1.weight",
            gate=f"{layer_prefix_hf}.share_expert.gate_proj.weight",
            up=f"{layer_prefix_hf}.share_expert.up_proj.weight",
        )
    )

    # Routed-expert grouped-GEMM tensors. HF stores `gate_proj` and `up_proj`
    # as two separate 3D tensors of shape (num_experts, intermediate, hidden);
    # Megatron's grouped-GEMM stores them per-expert under linear_fc1.weight*
    # with gate+up concatenated. Use FusedExpertMapping over each independently.
    # The conversion loop merges the per-expert tensors back into the 3D shape
    # via the `is_grouped_export` protocol.
    mapping.append(
        FusedExpertMapping(
            megatron_param=f"{layer_prefix_megatron}.mlp.experts.linear_fc1.weight*",
            hf_param=f"{layer_prefix_hf}.moe.gate_up_proj_fused",  # synthetic; see Step3p5ModelBridge
        )
    )
    mapping.append(
        FusedExpertMapping(
            megatron_param=f"{layer_prefix_megatron}.mlp.experts.linear_fc2.weight*",
            hf_param=f"{layer_prefix_hf}.moe.down_proj.weight",
            transpose_on_export=True,
        )
    )
    return mapping


def get_step3p5_mapping_list(*, num_hidden_layers: int, num_mtp_layers: int) -> list:
    """Return the full HF<->Megatron mapping list for Step-3.5-Flash.

    Args:
        num_hidden_layers: Number of decoder layers in the loaded HF config
            (i.e. ``config.num_hidden_layers``); MTP layers in the HF state
            dict live at indices ``[N, N+1, ..., N+num_mtp_layers-1]``.
        num_mtp_layers: Number of Multi-Token-Prediction layers (i.e.
            ``num_nextn_predict_layers`` in HF config).
    """
    mapping: list = []

    # Top-level (embedding, final norm, lm head).
    top_level = {
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }
    mapping.extend(AutoMapping(megatron_param=k, hf_param=v) for k, v in top_level.items())

    # Decoder layers.
    mapping.extend(_base_layer_mappings("decoder.layers.*", "model.layers.*"))

    # MTP layers (DSv3-style; one transformer-layer-shaped block per MTP head plus
    # enorm/hnorm/eh_proj/final_layernorm). HF places MTP weights at
    # `model.layers.{num_hidden_layers + i}.*`.
    for i in range(num_mtp_layers):
        hf_layer_idx = num_hidden_layers + i
        mtp_megatron_prefix = f"mtp.layers.{i}.mtp_model_layer"
        mtp_hf_prefix = f"model.layers.{hf_layer_idx}"
        mapping.extend(_base_layer_mappings(mtp_megatron_prefix, mtp_hf_prefix))

        # MTP-block-specific extras (enorm/hnorm/eh_proj/final_layernorm).
        # Step-3.5's published modeling code drops these via
        # `_keys_to_ignore_on_load_unexpected = [r"model\.layers\.45\.*",
        #  r"model\.layers\.46\.*", r"model\.layers\.47\.*"]`. We still emit
        # mappings so Megatron-side training is fully observable; HF round-trip
        # will simply drop the keys when loading into Step3p5ForCausalLM.
        mtp_extras = {
            f"mtp.layers.{i}.enorm.weight": f"model.layers.{hf_layer_idx}.enorm.weight",
            f"mtp.layers.{i}.hnorm.weight": f"model.layers.{hf_layer_idx}.hnorm.weight",
            f"mtp.layers.{i}.eh_proj.weight": f"model.layers.{hf_layer_idx}.eh_proj.weight",
            f"mtp.layers.{i}.final_layernorm.weight": (f"model.layers.{hf_layer_idx}.shared_head.norm.weight"),
        }
        mapping.extend(AutoMapping(megatron_param=k, hf_param=v) for k, v in mtp_extras.items())

    return mapping
