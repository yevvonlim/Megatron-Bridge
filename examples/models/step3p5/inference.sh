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

# Generation test for Step-3.5-Flash.
# Requires: GPU with enough VRAM to hold the 321B model in bf16 (~640GB
# aggregate) -- typically 8x H200 141GB or 16x H100 80GB.

set -euo pipefail

HF_MODEL=${HF_MODEL:-stepfun-ai/Step-3.5-Flash}
PROMPT=${PROMPT:-"Explain the difference between supervised fine-tuning and reinforcement learning from human feedback."}

uv run python examples/conversion/hf_to_megatron_generate_text.py \
    --hf_model_path "${HF_MODEL}" \
    --prompt "${PROMPT}" \
    --trust-remote-code
