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

"""HuggingFace<->Megatron bridge registration for Step-3.5-Flash.

The bridge handles the three Step-3.5-specific HF-naming differences from
DeepSeek-V3 (``moe`` vs ``mlp``, ``share_expert`` vs ``shared_experts``,
``moe.router_bias`` vs ``mlp.gate.e_score_correction_bias``) plus the
fused-3D-MoE expert weight layout where HF stores ``gate_proj`` /
``up_proj`` as two separate ``(num_experts, intermediate, hidden)``
tensors rather than a single ``gate_up_proj``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Tuple

import torch
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, FusedGatedExpertMapping
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.step3p5.common import get_step3p5_mapping_list
from megatron.bridge.models.step3p5.step3p5_provider import Step3p5ModelProvider


logger = logging.getLogger(__name__)


# Register Step-3.5 custom module types for AutoMapping TP-distribution decisions.
# These are the modules added in step3p5.modules; AutoMapping uses the registry
# to decide whether a given parameter is column-, row-, or replicated-parallel.
AutoMapping.register_module_type("Step3p5SelfAttention", "replicated")
AutoMapping.register_module_type("Step3p5TEDotProductAttention", "replicated")


# Synthetic HF-key prefix used to fold HF's two separate 3D expert tensors
# (`moe.gate_proj.weight`, `moe.up_proj.weight`) into a single fused tensor for
# `FusedGatedExpertMapping`. The actual HF state dict never contains this key;
# `Step3p5ModelBridge.maybe_modify_loaded_hf_weight` synthesizes it on the fly.
_GATE_UP_FUSED_SUFFIX = ".moe.gate_up_proj_fused"


def _is_synthetic_gate_up_key(hf_param: Any) -> bool:
    return isinstance(hf_param, str) and hf_param.endswith(_GATE_UP_FUSED_SUFFIX)


def _real_gate_up_keys(synthetic_key: str) -> Tuple[str, str]:
    layer_prefix = synthetic_key[: -len(_GATE_UP_FUSED_SUFFIX)]
    return f"{layer_prefix}.moe.gate_proj.weight", f"{layer_prefix}.moe.up_proj.weight"


@MegatronModelBridge.register_bridge(
    source="Step3p5ForCausalLM",
    target=GPTModel,
    provider=Step3p5ModelProvider,
    model_type="step3p5",
)
class Step3p5ModelBridge(MegatronModelBridge):
    """Megatron Bridge for StepFun's Step-3.5-Flash."""

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Step3p5ModelProvider:
        """Convert a loaded HuggingFace ``Step3p5Config`` into a Megatron provider."""
        provider: Step3p5ModelProvider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        # Reject configs we cannot honor faithfully (per-layer head-count override).
        attn_other = getattr(hf_config, "attention_other_setting", None)
        if attn_other is not None:
            base_n_heads = getattr(hf_config, "num_attention_heads", None)
            override_n_heads = attn_other.get("num_attention_heads") if isinstance(attn_other, dict) else None
            if override_n_heads is not None and override_n_heads != base_n_heads:
                raise ValueError(
                    "Step3p5ModelBridge does not support per-layer heterogeneous attention "
                    f"head counts (config.json has num_attention_heads={base_n_heads} but "
                    f"attention_other_setting overrides to {override_n_heads}). Megatron-Core "
                    "GPTModel does not natively support this; train a uniform-head variant."
                )

        # Architecture sizing / MoE.
        provider.hidden_size = hf_config.hidden_size
        provider.num_layers = hf_config.num_hidden_layers
        provider.num_attention_heads = hf_config.num_attention_heads
        provider.num_query_groups = hf_config.num_attention_groups
        provider.kv_channels = hf_config.head_dim
        provider.ffn_hidden_size = hf_config.intermediate_size
        provider.vocab_size = hf_config.vocab_size

        provider.num_moe_experts = hf_config.moe_num_experts
        provider.moe_router_topk = hf_config.moe_top_k
        provider.moe_ffn_hidden_size = hf_config.moe_intermediate_size
        provider.moe_shared_expert_intermediate_size = hf_config.share_expert_dim

        # `moe_layers_enum` in HF is e.g. "3,4,...,44"; convert to moe_layer_freq.
        moe_layers_str = getattr(hf_config, "moe_layers_enum", None)
        if isinstance(moe_layers_str, str):
            moe_layer_indices = {int(x) for x in moe_layers_str.split(",") if x}
        elif isinstance(moe_layers_str, (list, tuple)):
            moe_layer_indices = {int(x) for x in moe_layers_str}
        else:
            moe_layer_indices = set(range(hf_config.num_hidden_layers))
        provider.moe_layer_freq = [1 if i in moe_layer_indices else 0 for i in range(hf_config.num_hidden_layers)]

        # Routing / scaling. The HF field name is `moe_router_scaling_factor`, not the
        # DSv3 name -- copy explicitly to avoid auto-mapping mismatch.
        provider.moe_router_topk_scaling_factor = float(getattr(hf_config, "moe_router_scaling_factor", 1.0))

        # Per-layer rope cycle.
        rope_theta = getattr(hf_config, "rope_theta", None)
        if isinstance(rope_theta, (list, tuple)):
            cycle_len = self._infer_cycle_length(rope_theta)
            provider.rope_theta_cycle = tuple(float(x) for x in rope_theta[:cycle_len])

        partial_rotary = getattr(hf_config, "partial_rotary_factors", None)
        if isinstance(partial_rotary, (list, tuple)):
            cycle_len = self._infer_cycle_length(partial_rotary)
            provider.partial_rotary_cycle = tuple(float(x) for x in partial_rotary[:cycle_len])

        yarn_only = getattr(hf_config, "yarn_only_types", None)
        provider.yarn_only_full_attention = isinstance(yarn_only, (list, tuple)) and "full_attention" in yarn_only

        rope_scaling = getattr(hf_config, "rope_scaling", None) or {}
        provider.llama3_factor = float(rope_scaling.get("factor", provider.llama3_factor))
        provider.llama3_low_freq_factor = float(rope_scaling.get("low_freq_factor", provider.llama3_low_freq_factor))
        provider.llama3_high_freq_factor = float(
            rope_scaling.get("high_freq_factor", provider.llama3_high_freq_factor)
        )
        provider.llama3_original_max_pe = int(
            rope_scaling.get("original_max_position_embeddings", provider.llama3_original_max_pe)
        )

        # Sliding window + window pattern.
        provider.sliding_window = int(getattr(hf_config, "sliding_window", 512) or 512)
        provider.head_wise_attn_gate = bool(getattr(hf_config, "use_head_wise_attn_gate", False))

        # SwiGLU clamp lists (HF stores per-layer clamp values; we keep only the
        # non-zero indices).
        provider.routed_swiglu_clamp_layers, provider.routed_swiglu_clamp_value = self._parse_clamp_list(
            getattr(hf_config, "swiglu_limits", None)
        )
        provider.shared_swiglu_clamp_layers, provider.shared_swiglu_clamp_value = self._parse_clamp_list(
            getattr(hf_config, "swiglu_limits_shared", None)
        )

        # MTP.
        provider.mtp_num_layers = int(getattr(hf_config, "num_nextn_predict_layers", 0) or 0)

        # Sequence length / max position embeddings.
        provider.seq_length = int(getattr(hf_config, "max_seq_len", provider.seq_length))
        provider.max_position_embeddings = int(
            getattr(hf_config, "max_position_embeddings", provider.max_position_embeddings)
        )

        return provider

    @classmethod
    def megatron_to_hf_config(cls, provider: Step3p5ModelProvider) -> dict:
        """Reconstruct a HuggingFace ``Step3p5Config`` dict from a provider."""
        hf_config = super().megatron_to_hf_config(provider)
        hf_config["moe_num_experts"] = provider.num_moe_experts
        hf_config["moe_top_k"] = provider.moe_router_topk
        hf_config["moe_intermediate_size"] = provider.moe_ffn_hidden_size
        hf_config["share_expert_dim"] = provider.moe_shared_expert_intermediate_size
        hf_config["moe_router_scaling_factor"] = provider.moe_router_topk_scaling_factor
        hf_config["use_moe_router_bias"] = provider.moe_router_enable_expert_bias
        hf_config["need_fp32_gate"] = provider.moe_router_dtype == "fp32"
        hf_config["use_head_wise_attn_gate"] = provider.head_wise_attn_gate
        hf_config["sliding_window"] = provider.sliding_window
        hf_config["zero_centered"] = provider.layernorm_zero_centered_gamma
        hf_config["num_nextn_predict_layers"] = provider.mtp_num_layers or 0
        hf_config["num_attention_groups"] = provider.num_query_groups
        hf_config["head_dim"] = provider.kv_channels
        return hf_config

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Return the HF<->Megatron parameter mapping registry."""
        hf_config = self.hf_config
        num_hidden_layers = hf_config.num_hidden_layers
        num_mtp_layers = int(getattr(hf_config, "num_nextn_predict_layers", 0) or 0)

        mapping_list = get_step3p5_mapping_list(
            num_hidden_layers=num_hidden_layers,
            num_mtp_layers=num_mtp_layers,
        )

        # Replace the `FusedExpertMapping(... gate_up_proj_fused)` synthetic placeholders
        # written by `common.py` with `FusedGatedExpertMapping` so the gate/up split
        # logic kicks in once `maybe_modify_loaded_hf_weight` synthesizes the fused
        # tensor below. We do the swap here (rather than in common.py) because the
        # `FusedGatedExpertMapping` import circulates through model_bridge.
        for i, mapping in enumerate(mapping_list):
            hf = mapping.hf_param
            if isinstance(hf, str) and hf.endswith(_GATE_UP_FUSED_SUFFIX) and "linear_fc1" in mapping.megatron_param:
                mapping_list[i] = FusedGatedExpertMapping(
                    megatron_param=mapping.megatron_param,
                    hf_param=hf,
                )

        return MegatronMappingRegistry(*mapping_list)

    # ------------------------------------------------------------------ #
    # HF state-dict pre/post-processing for the fused-3D MoE expert layout
    # ------------------------------------------------------------------ #

    def maybe_modify_loaded_hf_weight(self, hf_param, hf_state_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Synthesize the fused gate-up tensor on demand for MoE expert mappings.

        HF stores ``moe.gate_proj.weight`` and ``moe.up_proj.weight`` as two
        independent ``(num_experts, intermediate, hidden)`` tensors;
        ``FusedGatedExpertMapping`` expects a single
        ``(num_experts, 2, intermediate, hidden)`` tensor (or fused along the
        intermediate dim). We stack along a new dim 1 -- the
        ``shape[0] == 2`` codepath inside ``FusedGatedExpertMapping`` then
        matches and splits gate/up correctly.
        """
        if _is_synthetic_gate_up_key(hf_param):
            gate_key, up_key = _real_gate_up_keys(hf_param)
            gate = hf_state_dict[gate_key]
            up = hf_state_dict[up_key]
            if gate.shape != up.shape:
                raise ValueError(
                    f"Step-3.5 expects matching shapes for {gate_key} and {up_key}; "
                    f"got {tuple(gate.shape)} vs {tuple(up.shape)}."
                )
            if gate.ndim != 3:
                raise ValueError(
                    f"Step-3.5 expects 3D MoE expert tensors (num_experts, intermediate, hidden); "
                    f"got ndim={gate.ndim} for {gate_key}."
                )
            # (num_experts, intermediate, hidden) -> (num_experts, 2, intermediate, hidden)
            return torch.stack([gate, up], dim=1)
        return super().maybe_modify_loaded_hf_weight(hf_param, hf_state_dict)

    def maybe_modify_converted_hf_weight(
        self,
        task,
        converted_weights_dict: Dict[str, torch.Tensor],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Rename synthesized fused export keys back to real HF names."""
        synthetic_keys = [k for k in list(converted_weights_dict.keys()) if _GATE_UP_FUSED_SUFFIX in k]
        if not synthetic_keys:
            return converted_weights_dict

        renamed: Dict[str, torch.Tensor] = {}
        for k in synthetic_keys:
            tensor = converted_weights_dict.pop(k)
            # `FusedGatedExpertMapping.megatron_to_hf` returns dict with keys
            # `<hf_param>.gate` / `<hf_param>.up` (where hf_param ends with
            # `_GATE_UP_FUSED_SUFFIX`). Map them back to the real HF names.
            if k.endswith(".gate"):
                base = k[: -len(".gate")]
                layer_prefix = base[: -len(_GATE_UP_FUSED_SUFFIX)]
                renamed[f"{layer_prefix}.moe.gate_proj.weight"] = tensor
            elif k.endswith(".up"):
                base = k[: -len(".up")]
                layer_prefix = base[: -len(_GATE_UP_FUSED_SUFFIX)]
                renamed[f"{layer_prefix}.moe.up_proj.weight"] = tensor
            else:
                logger.warning("Unexpected synthetic key during export: %s", k)
                renamed[k] = tensor
        converted_weights_dict.update(renamed)
        return converted_weights_dict

    # ------------------------------------------------------------------ #

    @staticmethod
    def _infer_cycle_length(per_layer_values) -> int:
        """Infer the canonical cycle length of a per-layer config list.

        Step-3.5's ``rope_theta`` and ``partial_rotary_factors`` are both
        provided as lists of length ``num_hidden_layers + num_nextn_predict_layers``
        but follow a fixed 4-cycle. We probe for the smallest period that
        divides the list cleanly.
        """
        n = len(per_layer_values)
        for period in (1, 2, 3, 4, 5, 6, 8, 12):
            if n % period != 0:
                continue
            if all(per_layer_values[i] == per_layer_values[i % period] for i in range(n)):
                return period
        # Fall back to using the full list -- caller will pass through.
        return n

    @staticmethod
    def _parse_clamp_list(per_layer_clamp) -> tuple[tuple[int, ...], float]:
        """Extract non-zero clamp indices and the (single) clamp value.

        Step-3.5 only uses one clamp value per list (e.g. all 7s at the
        active indices, or all 16s). If multiple distinct non-zero values
        appear, raise -- our scaled provider does not yet model per-layer
        per-value clamps.
        """
        if not per_layer_clamp:
            return (), 0.0
        layers = []
        values = set()
        for i, v in enumerate(per_layer_clamp):
            if v is None or float(v) == 0.0:
                continue
            layers.append(i)
            values.add(float(v))
        if not layers:
            return (), 0.0
        if len(values) > 1:
            raise NotImplementedError(
                f"Step-3.5 swiglu clamp with multiple distinct values per layer is not "
                f"implemented (got values {sorted(values)} at layers {layers})."
            )
        return tuple(layers), values.pop()
