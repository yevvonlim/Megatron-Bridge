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

"""Functional GPU tests for Step-3.5-Flash HF<->Megatron conversion.

Step-3.5 is loaded via ``auto_map`` (custom code on the HF Hub), so this test
constructs a toy model from the vendored reference implementation under
``tests/unit_tests/models/step3p5/_reference/`` rather than relying on the
``transformers`` package's built-in registry.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


_REFERENCE_DIR = Path(__file__).resolve().parents[3] / "unit_tests" / "models" / "step3p5" / "_reference"


HF_STEP3P5_TOY_CONFIG = {
    "architectures": ["Step3p5ForCausalLM"],
    "model_type": "step3p5",
    # Auto-map binding so transformers can locate the modeling file at load time.
    "auto_map": {
        "AutoConfig": "configuration_step3p5.Step3p5Config",
        "AutoModelForCausalLM": "modeling_step3p5.Step3p5ForCausalLM",
    },
    # Architecture (toy sizes).
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_attention_groups": 4,
    "head_dim": 32,
    "vocab_size": 1024,
    "max_seq_len": 512,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-5,
    # Hybrid attention (1 full + 3 sliding cycle, applied to 4 layers).
    "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
    "sliding_window": 64,
    # Per-layer rope cycle (4-cycle).
    "rope_theta": [5_000_000.0, 10_000.0, 10_000.0, 10_000.0],
    "rope_scaling": {
        "rope_type": "llama3",
        "factor": 2.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 32.0,
        "original_max_position_embeddings": 256,
    },
    "yarn_only_types": ["full_attention"],
    "partial_rotary_factors": [0.5, 1.0, 1.0, 1.0],
    # MoE config (small).
    "use_moe": True,
    "moe_num_experts": 4,
    "moe_top_k": 2,
    "moe_intermediate_size": 64,
    "share_expert_dim": 64,
    "moe_layer_offset": 0,
    "moe_every_n_layer": 1,
    "moe_layers_enum": "1,2,3",  # layer 0 dense, 1-3 MoE
    "moe_router_activation": "sigmoid",
    "moe_router_scaling_factor": 3.0,
    "use_moe_router_bias": True,
    "need_fp32_gate": True,
    "norm_expert_weight": True,
    # Attention quirks.
    "use_qk_norm": True,
    "use_head_wise_attn_gate": True,
    "att_impl_type": "GQA",
    "tie_word_embeddings": False,
    # MTP (small).
    "num_nextn_predict_layers": 0,  # disable MTP for this toy test
    "use_rope_layers": [],
    # Clamp lists (none for the toy size).
    "swiglu_limits": [0.0] * 4,
    "swiglu_limits_shared": [0.0] * 4,
    "zero_centered": True,
    "sink": False,
    "torch_dtype": "bfloat16",
}


def _import_reference():
    """Import the vendored Step-3.5 modeling code in-process.

    Returns the (Step3p5Config, Step3p5ForCausalLM) classes from the
    vendored ``_reference/`` dir. Skips the test if vendored files are
    missing (e.g. when running outside this repo).
    """
    cfg_file = _REFERENCE_DIR / "configuration_step3p5.py"
    model_file = _REFERENCE_DIR / "modeling_step3p5.py"
    if not cfg_file.exists() or not model_file.exists():
        pytest.skip(f"Vendored Step-3.5 reference not found under {_REFERENCE_DIR}")

    # Ensure the package can resolve the relative import in modeling.
    pkg_name = "_step3p5_reference_pkg"
    if pkg_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            pkg_name,
            _REFERENCE_DIR / "__init__.py" if (_REFERENCE_DIR / "__init__.py").exists() else None,
            submodule_search_locations=[str(_REFERENCE_DIR)],
        )
        if spec is None:
            pytest.skip("Could not build package spec for vendored reference")
        sys.modules[pkg_name] = importlib.util.module_from_spec(spec)

    # Load configuration_step3p5
    cfg_spec = importlib.util.spec_from_file_location(f"{pkg_name}.configuration_step3p5", cfg_file)
    cfg_mod = importlib.util.module_from_spec(cfg_spec)
    cfg_spec.loader.exec_module(cfg_mod)
    sys.modules[f"{pkg_name}.configuration_step3p5"] = cfg_mod

    # Load modeling_step3p5
    model_spec = importlib.util.spec_from_file_location(f"{pkg_name}.modeling_step3p5", model_file)
    model_mod = importlib.util.module_from_spec(model_spec)
    model_spec.loader.exec_module(model_mod)

    return cfg_mod.Step3p5Config, model_mod.Step3p5ForCausalLM


@pytest.mark.run_only_on("GPU")
class TestStep3p5Conversion:
    """HF<->Megatron roundtrip on a tiny Step-3.5 model.

    Builds a 4-layer Step-3.5 with reduced sizes from the vendored
    reference, saves it as an HF directory (with the modeling files
    bundled alongside config.json + safetensors), then runs the standard
    Megatron-Bridge conversion script.
    """

    @pytest.fixture(scope="class")
    def step3p5_toy_model_path(self, tmp_path_factory):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        Step3p5Config, Step3p5ForCausalLM = _import_reference()

        model_dir = tmp_path_factory.mktemp("step3p5_toy") / "model"
        model_dir.mkdir(parents=True)

        config = Step3p5Config(**HF_STEP3P5_TOY_CONFIG)
        config.torch_dtype = torch.bfloat16
        model = Step3p5ForCausalLM(config).bfloat16()
        model.save_pretrained(str(model_dir), safe_serialization=True)

        # Bundle the modeling code alongside the checkpoint so `auto_map`
        # works at load time without external trust_remote_code paths.
        for fname in ("configuration_step3p5.py", "modeling_step3p5.py"):
            shutil.copy(_REFERENCE_DIR / fname, model_dir / fname)

        return str(model_dir)

    @pytest.mark.parametrize("tp,pp,ep", [(1, 1, 1), (2, 1, 1), (1, 1, 2)])
    def test_roundtrip(self, step3p5_toy_model_path, tp, pp, ep, tmp_path):
        """HF -> Megatron -> HF roundtrip with various parallelism configs."""
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={max(tp * pp * ep, 1)}",
                "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
                f"--hf-model-id={step3p5_toy_model_path}",
                f"--output-dir={tmp_path}",
                f"--tp={tp}",
                f"--pp={pp}",
                f"--ep={ep}",
                "--trust-remote-code",
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[4]),
        )
        assert result.returncode == 0, f"Conversion failed:\nstderr={result.stderr}\nstdout={result.stdout}"
