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

"""Pretraining recipe for Step-3.5-Flash-Mini (~3B-total / ~0.27B-active).

Targets an 8x A100-40GB single node:

* TP=2 (sliced query heads), PP=1, EP=4 (80 experts / 4 = 20 per rank), DP=1
* sequence_parallel=True, seq_length=4096, micro_batch=1
* precision-aware optimizer (bf16 grads/moments)
* moe_router_bias_update_rate=0 (frozen bias mirrors HF semantics);
  ``seq_aux_loss`` carries the load balancing instead.
"""

from __future__ import annotations

from typing import Optional

import torch

from megatron.bridge.models.step3p5 import Step3p5ModelProviderMini3B
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.config import ConfigContainer


def step3p5_mini_3b_pretrain_config(
    *,
    train_iters: int = 100_000,
    global_batch_size: int = 256,
    seq_length: int = 4096,
    tensor_model_parallel_size: int = 2,
    expert_model_parallel_size: int = 4,
    pipeline_model_parallel_size: int = 1,
    micro_batch_size: int = 1,
    mock_data: bool = True,
    data_paths: Optional[list[str]] = None,
    tokenizer_model: Optional[str] = None,
) -> ConfigContainer:
    """Pretraining config for the Step-3.5-Flash-Mini ~3B-A0.27B variant.

    Args:
        train_iters: Total training iterations.
        global_batch_size: Global batch size (achieved via grad accum on 8 GPUs).
        seq_length: Sequence length used for pretraining (256K is dropped from
            the original config).
        tensor_model_parallel_size: TP -- 2 is the recommended single-node value.
        expert_model_parallel_size: EP -- ``num_moe_experts`` must be divisible
            by this. Default 4 maps 80 experts to 20 per rank.
        pipeline_model_parallel_size: PP. Default 1 -- there are too few layers
            for asymmetric PP layouts at this scale.
        micro_batch_size: Tokens-per-step per GPU. Stay at 1 unless you have
            spare memory.
        mock_data: If True, train on Megatron's synthetic mock dataset. Useful
            for the smoke test in the verification plan.
        data_paths: List of Megatron mmap dataset paths (``.bin`` / ``.idx``)
            when ``mock_data=False``.
        tokenizer_model: HF tokenizer id to use. If None, falls back to the
            null tokenizer with vocab=128896 -- correct for parameter sizing,
            but text generation requires a real BPE.

    Returns:
        ConfigContainer: A fully-populated pretraining config.
    """
    cfg = _pretrain_common()

    # ---- Model -----------------------------------------------------------
    cfg.model = Step3p5ModelProviderMini3B()

    # Parallelism -- 8x A100-40GB single-node default.
    cfg.model.tensor_model_parallel_size = tensor_model_parallel_size
    cfg.model.pipeline_model_parallel_size = pipeline_model_parallel_size
    cfg.model.expert_model_parallel_size = expert_model_parallel_size
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True

    # Length / dtype.
    cfg.model.seq_length = seq_length
    cfg.model.bf16 = True
    cfg.model.params_dtype = torch.bfloat16
    cfg.model.autocast_dtype = torch.bfloat16
    cfg.model.pipeline_dtype = torch.bfloat16

    # Memory.
    cfg.model.recompute_granularity = "selective"

    # ---- Tokenizer -------------------------------------------------------
    if tokenizer_model:
        cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
        cfg.tokenizer.tokenizer_model = tokenizer_model
    else:
        cfg.tokenizer.tokenizer_type = "NullTokenizer"
        cfg.tokenizer.tokenizer_model = None
        # Pretraining size matches HF Step-3.5 vocab = 128896 (config.json:22).
        # NullTokenizer's default 32K is wrong for this architecture.
        cfg.tokenizer.vocab_size = max(DEFAULT_NULL_TOKENIZER_VOCAB_SIZE, 128896)

    # ---- Dataset ---------------------------------------------------------
    if mock_data or not data_paths:
        cfg.dataset.blend = None
    else:
        # Equal-weight blend across the provided paths.
        weight_each = 1.0 / len(data_paths)
        cfg.dataset.blend = (data_paths, [weight_each] * len(data_paths))
    cfg.dataset.num_workers = 4

    # ---- DDP / sharding (with EP > 1, do NOT shard optim/grads/params) ---
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True

    # ---- Optimizer (precision-aware to fit on 40GB) ----------------------
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_grads_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16
    # Scale-down a bit from the _pretrain_common default (3e-4) for the smaller model.
    cfg.optimizer.lr = 1e-4
    cfg.optimizer.min_lr = 1e-5

    # ---- Training --------------------------------------------------------
    cfg.train.train_iters = train_iters
    cfg.train.global_batch_size = global_batch_size
    cfg.train.micro_batch_size = micro_batch_size
    cfg.scheduler.lr_warmup_iters = max(500, train_iters // 100)

    return cfg
