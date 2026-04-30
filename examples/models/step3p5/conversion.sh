#!/usr/bin/env bash
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

# HF<->Megatron conversion for Step-3.5-Flash (321B).
#
# WARNING: the published checkpoint is ~640GB in bf16 -- the Megatron import
# below requires a node with enough aggregate GPU+CPU memory to hold both
# representations during conversion. For most users the more practical path
# is to train the Mini3B variant from scratch via
# `src/megatron/bridge/recipes/step3p5/step3p5_mini.py`.

set -euo pipefail

WORKSPACE=${WORKSPACE:-/workspace}
HF_MODEL=${HF_MODEL:-stepfun-ai/Step-3.5-Flash}
MODEL_NAME=${MODEL_NAME:-Step-3.5-Flash}

# Recommended for the 321B variant on a 16x H100/H200 node.
TP=${TP:-2}
PP=${PP:-8}
EP=${EP:-16}

echo "==> HF -> Megatron import"
uv run python examples/conversion/convert_checkpoints.py import \
    --hf-model "${HF_MODEL}" \
    --megatron-path "${WORKSPACE}/${MODEL_NAME}" \
    --torch-dtype bfloat16 \
    --trust-remote-code

echo "==> Logits parity check"
uv run python -m torch.distributed.run --nproc_per_node=$((TP * PP * EP)) \
    examples/conversion/compare_hf_and_megatron/compare.py \
    --hf_model_path "${HF_MODEL}" \
    --megatron_model_path "${WORKSPACE}/${MODEL_NAME}" \
    --prompt "Hello, how are you?" \
    --tp "${TP}" --pp "${PP}" --ep "${EP}" \
    --trust-remote-code

echo "==> Megatron -> HF export"
uv run python examples/conversion/convert_checkpoints.py export \
    --hf-model "${HF_MODEL}" \
    --megatron-path "${WORKSPACE}/${MODEL_NAME}/iter_0000000" \
    --hf-path "${WORKSPACE}/${MODEL_NAME}-hf-export" \
    --trust-remote-code

echo "==> Roundtrip validation"
uv run python -m torch.distributed.run --nproc_per_node=$((TP * PP * EP)) \
    examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id "${HF_MODEL}" \
    --tp "${TP}" --pp "${PP}" --ep "${EP}" \
    --trust-remote-code
