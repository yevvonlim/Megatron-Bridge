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


def step3p5_mini_3b_muon_pretrain_config(
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
    enable_fp16_ns: bool = True,
    enable_dispersion_logging: bool = True,
    routed_clamp_value: float = 7.0,
    routed_clamp_layers: tuple[int, ...] = (22, 23),
) -> ConfigContainer:
    """Paper-aligned Muon pretraining config for Step-3.5-Flash-Mini.

    Diverges from :func:`step3p5_mini_3b_pretrain_config` in three ways
    (Step-3.5 §3.2 / §4.1):

    * **Optimizer**: ``muon_step3p5`` registered by
      :mod:`megatron.bridge.models.step3p5.muon_patches`. 2D matrices and
      tagged 3D grouped-expert weights are orthogonalized with Polar Express
      Newton-Schulz (T=6); embeddings, output projection, biases, and 1D
      params route to AdamW.
    * **Stability defenses**: routed-expert SwiGLU clamp on the last two
      MoE layers (paper Appendix B), per-layer max-to-median activation-norm
      logging, and (opt-in) FP16 NS intermediates.
    * **Sharding**: ``use_distributed_optimizer`` left True;
      ``data_parallel_sharding_strategy="no_shard"`` mirrors the Kimi/Muon
      ZeRO-1 recipe required by the layer-wise distributed Muon path.

    Args mirror :func:`step3p5_mini_3b_pretrain_config` plus:
        enable_fp16_ns: Monkey-patch ``newton_schulz`` to use FP16 instead
            of BF16 intermediates (paper §4.1.1). Default True.
        enable_dispersion_logging: Emit per-layer expert-norm max-to-median
            ratio every ``moe_dispersion_log_every`` iterations. Default True.
        routed_clamp_value: SwiGLU clamp magnitude for layers in
            ``routed_clamp_layers``. Setting to 0 disables the clamp.
        routed_clamp_layers: 0-indexed HF layer indices to apply the clamp.

    Returns:
        ConfigContainer: A pretraining config wired for paper-faithful Muon.
    """
    cfg = step3p5_mini_3b_pretrain_config(
        train_iters=train_iters,
        global_batch_size=global_batch_size,
        seq_length=seq_length,
        tensor_model_parallel_size=tensor_model_parallel_size,
        expert_model_parallel_size=expert_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        micro_batch_size=micro_batch_size,
        mock_data=mock_data,
        data_paths=data_paths,
        tokenizer_model=tokenizer_model,
    )

    # Register and (optionally) FP16-patch the optimizer registry. Both calls
    # are idempotent; safe to invoke from a recipe entry point.
    from megatron.bridge.models.step3p5 import muon_patches

    muon_patches.register_step3p5_muon()
    if enable_fp16_ns:
        muon_patches.enable_fp16_ns()

    # Step-3.5 §4.1 stability defenses on the model provider side.
    cfg.model.routed_swiglu_clamp_value = routed_clamp_value
    cfg.model.routed_swiglu_clamp_layers = tuple(routed_clamp_layers)
    cfg.model.enable_moe_dispersion_logging = enable_dispersion_logging
    # Paper §2.2 eq. (1): per-EP-group balance loss with coef 1e-3. Active only
    # when EP > 1.
    cfg.model.ep_group_balance_loss_coeff = 1e-3

    # Optimizer routing: Muon-managed matrices use Polar Express NS; everything
    # else falls back to AdamW per ``muon_step3p5``'s param-override factory.
    # use_precision_aware_optimizer is incompatible with Muon's state machinery
    # -- the precision-aware path is keyed to Adam exp_avg / exp_avg_sq tensors
    # which Muon does not maintain in the same form. Disable it.
    cfg.optimizer.optimizer = muon_patches.get_muon_optimizer_name()
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.muon_coefficient_type = "polar_express"
    cfg.optimizer.muon_num_ns_steps = 6
    cfg.optimizer.muon_momentum = 0.95
    cfg.optimizer.muon_scale_mode = "spectral"
    cfg.optimizer.muon_split_qkv = True
    cfg.optimizer.muon_tp_mode = "blockwise"

    # Paper §4.2.3: lr=2.5e-4 at 196B; scaled down for Mini3B. Sweep this if
    # max-to-median dispersion grows during early iterations.
    cfg.optimizer.lr = 1.5e-4
    cfg.optimizer.min_lr = 1.5e-5
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.clip_grad = 1.0

    # Paper §4.2.3 stage 1.
    cfg.scheduler.lr_warmup_iters = max(2000, train_iters // 50)

    # MTP loss scaling per paper stage 1 (only takes effect if the underlying
    # model has mtp_num_layers > 0).
    if hasattr(cfg.model, "mtp_loss_scaling_factor"):
        cfg.model.mtp_loss_scaling_factor = 0.3

    return cfg
