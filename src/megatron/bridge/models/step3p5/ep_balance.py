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

"""Step-3.5-Flash per-EP-group balance loss.

Paper §2.2 eq. (1):

    L_EP = G * sum_g f_g * p_g

where ``G`` is the number of expert-parallel groups, ``f_g`` is the
fraction of tokens that get routed to *any* expert in EP group ``g``,
and ``p_g`` is the sum of router probabilities over experts in group
``g``. Coefficient defaults to 1e-3.

Megatron-Core ships ``aux_loss`` / ``seq_aux_loss`` / ``global_aux_loss``
but none of them split the experts by EP group. ``global_aux_loss``
reduces across TP+DP+CP rather than EP. The Step-3.5 EP loss is per-EP-
group and computed *locally* on each rank's routing decisions -- it
penalizes the case where one EP rank's expert pool gets disproportionate
traffic, which causes EP straggler effects independent of overall
expert load balance.

This module installs the loss at runtime by wrapping each router's
``forward`` to compute the term and fold it into the gradient path via
the same :class:`MoEAuxLossAutoScaler` mechanism the upstream losses
use, so no fork of upstream router code is required. All state lives
in the bridge.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler


logger = logging.getLogger(__name__)


def _step3p5_ep_balance_loss(
    probs: torch.Tensor,
    routing_map: torch.Tensor,
    *,
    num_ep_groups: int,
    coeff: float,
) -> torch.Tensor:
    """Compute the Step-3.5 §2.2 per-EP-group balance loss.

    Args:
        probs: Router probabilities post-topk, shape ``(num_tokens, num_experts)``.
            For sigmoid routers this is non-normalized; for softmax it sums to 1
            per token. The loss is invariant to the constant scale.
        routing_map: One-hot mask of which experts each token activates,
            shape ``(num_tokens, num_experts)``. Effectively boolean even when
            stored as float.
        num_ep_groups: Number of EP groups -- equal to
            ``expert_model_parallel_size``. Each group owns
            ``num_experts // num_ep_groups`` experts.
        coeff: Loss coefficient. Paper recommends 1e-3.

    Returns:
        Scalar loss tensor (already multiplied by ``coeff``).
    """
    if coeff == 0:
        return probs.new_zeros(())
    num_tokens, num_experts = probs.shape
    if num_experts % num_ep_groups != 0:
        raise ValueError(f"num_experts ({num_experts}) must be divisible by num_ep_groups ({num_ep_groups})")
    experts_per_group = num_experts // num_ep_groups

    # Reshape (num_tokens, num_experts) -> (num_tokens, G, experts_per_group)
    # then sum over the experts_per_group axis to get per-group quantities.
    rmap = routing_map.float().view(num_tokens, num_ep_groups, experts_per_group)
    probs_g = probs.view(num_tokens, num_ep_groups, experts_per_group)
    # f_g: fraction of tokens that activate ANY expert in group g.
    # Use clamp(max=1) to avoid double-counting when a token's topk lands on
    # >1 expert in the same group.
    f_per_token = rmap.sum(dim=-1).clamp(max=1.0)
    f_g = f_per_token.mean(dim=0)  # (G,)
    # p_g: average router probability sum per group, averaged over tokens.
    p_g = probs_g.sum(dim=-1).mean(dim=0)  # (G,)

    loss = num_ep_groups * (f_g * p_g).sum() * coeff
    return loss


def install_ep_balance_loss(
    model: torch.nn.Module,
    *,
    coeff: float,
    num_ep_groups: int,
) -> int:
    """Wrap each MoE router's ``forward`` to add the EP-group balance loss.

    Walks ``model.decoder.layers`` for any ``MoELayer`` and replaces the
    bound method ``layer.mlp.router.forward`` with a thin wrapper that
    calls the original, computes the EP-group loss on the returned
    ``(probs, routing_map)``, and folds it into ``probs`` via
    :class:`MoEAuxLossAutoScaler` so the loss back-propagates through
    the router weights. Idempotent: calling twice on the same model is
    a no-op.

    Args:
        model: A constructed Megatron GPTModel.
        coeff: Loss coefficient. Pass 0 to disable (the wrapper is still
            installed but the loss term is skipped).
        num_ep_groups: ``expert_model_parallel_size``.

    Returns:
        The number of routers wrapped.
    """
    decoder = getattr(model, "decoder", None)
    if decoder is None or not hasattr(decoder, "layers"):
        return 0

    wrapped = 0
    for layer in decoder.layers:
        mlp = getattr(layer, "mlp", None)
        router = getattr(mlp, "router", None) if mlp is not None else None
        if router is None:
            continue
        if getattr(router, "_step3p5_ep_loss_installed", False):
            continue

        original_forward = router.forward

        def _make_wrapped(orig, router_ref):
            def _wrapped(input: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
                probs, routing_map = orig(input, padding_mask=padding_mask)
                if coeff > 0 and router_ref.training and torch.is_grad_enabled():
                    ep_loss = _step3p5_ep_balance_loss(
                        probs=probs,
                        routing_map=routing_map,
                        num_ep_groups=num_ep_groups,
                        coeff=coeff,
                    )
                    probs = MoEAuxLossAutoScaler.apply(probs, ep_loss)
                return probs, routing_map

            return _wrapped

        router.forward = _make_wrapped(original_forward, router)
        router._step3p5_ep_loss_installed = True
        wrapped += 1

    if wrapped > 0:
        logger.info(
            "Installed Step-3.5 EP-group balance loss (coeff=%.4g, num_ep_groups=%d) on %d routers.",
            coeff,
            num_ep_groups,
            wrapped,
        )
    return wrapped
