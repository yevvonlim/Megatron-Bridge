#!/usr/bin/env bash
#SBATCH --job-name=step3p5-mini-pretrain
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Pretraining the Step-3.5-Flash-Mini-3B-A0.27B variant from scratch on
# a single 8x A100-40GB node.

set -euo pipefail

WORKSPACE=${WORKSPACE:-/workspace}
DATA_PATH=${DATA_PATH:-/path/to/dataset.bin}
TRAIN_ITERS=${TRAIN_ITERS:-100000}

srun --container-image="${CONTAINER_IMAGE:-megatron-bridge:latest}" \
     --container-mounts="${WORKSPACE}:/workspace" \
     --container-workdir=/workspace/Megatron-Bridge \
     bash -lc "
uv run python -m torch.distributed.run --nproc_per_node=8 \
    scripts/training/run_recipe.py \
    --recipe step3p5_mini_3b_pretrain_config \
    train.train_iters=${TRAIN_ITERS} \
    train.global_batch_size=256 \
    train.micro_batch_size=1 \
    dataset.blend='([${DATA_PATH}],[1.0])' \
    model.tensor_model_parallel_size=2 \
    model.expert_model_parallel_size=4 \
    model.pipeline_model_parallel_size=1
"
