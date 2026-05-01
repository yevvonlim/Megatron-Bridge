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

"""Bridge-side Muon patches for Step-3.5-Flash pretraining.

Step-3.5 §3.2 / §4.1.1 / §4.1.2 require three deviations from the stock
``emerging_optimizers.TensorParallelMuon`` integration shipped with
Megatron-Core:

1. **Polar Express coefficients** -- already a native option in
   ``NSCoeffT`` (no patch needed; set
   ``OptimizerConfig.muon_coefficient_type = "polar_express"``).
2. **FP16 Newton-Schulz intermediates** -- the upstream package switches
   to BF16 only when ``muon_fp32_matmul_prec="medium"``. The paper §4.1.1
   reports BF16 cumulative addition error causes rare unrecoverable loss
   spikes that FP16 mantissa precision eliminates. We expose this via
   :func:`enable_fp16_ns`, an opt-in monkey-patch that re-routes the
   iteration through an FP16 cast.
3. **3D grouped-expert weights** -- ``TEGroupedMLP`` stores routed expert
   weights as ``(E, M, H)`` tensors. The default Muon routing predicate
   ``_is_nonlinear_or_embedding`` rejects ``ndim != 2``, so those weights
   silently fall through to AdamW. This file registers
   :class:`Step3p5TensorParallelMuon` under the optimizer name
   ``"muon_step3p5"`` with a 3D-aware override that loops over the leading
   expert axis and applies NS to each ``(M, H)`` slice independently.
   This matches the paper's per-expert orthogonalization semantics.

All patches are opt-in -- importing this module merely makes the helpers
available; you must call :func:`register_step3p5_muon` and (optionally)
:func:`enable_fp16_ns` from the recipe to activate them. None of these
patches modify ``3rdparty/Megatron-LM/`` or the upstream
``emerging_optimizers`` package on disk.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
from megatron.core.optimizer.emerging_optimizers import (
    _EMERGING_OPTIMIZERS,
    HAVE_EMERGING_OPTIMIZERS,
    EmergingOptimizerEntry,
    _eopt_init_state_fn,
)
from megatron.core.optimizer.optimizer_config import ParamKey, ParamPredicate


logger = logging.getLogger(__name__)


_STEP3P5_MUON_NAME = "muon_step3p5"
_FP16_NS_INSTALLED = False
_REGISTERED = False


# ---------------------------------------------------------------------------
# 1. 3D-aware Muon subclass + routing predicate
# ---------------------------------------------------------------------------


def _step3p5_is_nonlinear_or_embedding(param: torch.nn.Parameter) -> bool:
    """Routing predicate for ``muon_step3p5``.

    Returns ``True`` when *param* should be routed to AdamW (not Muon).
    Differs from upstream ``_is_nonlinear_or_embedding`` (which returns
    ``True`` for any ``ndim != 2``) by carving out 3D grouped-expert
    weights tagged with the ``_step3p5_route_to_muon`` attribute -- those
    are sent to Muon and orthogonalized per-expert by
    :class:`Step3p5TensorParallelMuon`.
    """
    if getattr(param, "is_embedding_or_output_parameter", False):
        return True
    if param.ndim == 2:
        return False
    if param.ndim == 3 and getattr(param, "_step3p5_route_to_muon", False):
        return False
    return True


def _step3p5_default_param_overrides_factory() -> Dict[ParamKey, Dict[str, Any]]:
    """Default param overrides for ``muon_step3p5``.

    Routes parameters that are not Muon-eligible (per
    :func:`_step3p5_is_nonlinear_or_embedding`) to AdamW. The Muon-eligible
    set is the complement of this predicate -- 2D matrices and the
    explicitly-tagged 3D grouped-expert weights.
    """
    return {
        ParamKey(
            predicate=ParamPredicate(
                name="step3p5_nonlinear_or_embedding",
                fn=_step3p5_is_nonlinear_or_embedding,
            )
        ): {"optimizer": "adam"}
    }


if HAVE_EMERGING_OPTIMIZERS:
    from megatron.core.optimizer.emerging_optimizers import TensorParallelMuon

    class Step3p5TensorParallelMuon(TensorParallelMuon):
        """Muon optimizer that handles 3D ``(E, M, H)`` grouped-expert weights.

        Upstream :meth:`TensorParallelMuon.orthogonalize` calls
        ``newton_schulz_tp`` directly. ``newton_schulz_step`` inside that
        kernel uses :func:`torch.addmm` which is strictly 2D -- so a 3D
        grouped-expert grad would error out on ``addmm`` even if it passed
        the routing predicate. We therefore detect 3D inputs in
        :meth:`orthogonalize` and loop over the leading expert dimension,
        delegating to the parent for each ``(M, H)`` slice. Each expert's
        weight matrix is functionally independent, so per-expert NS is
        the semantically correct mapping (cross-expert orthogonalization
        would not be meaningful).
        """

        def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            """Apply NS per leading-axis slice when *grad* is 3D, else delegate."""
            if grad.ndim != 3:
                return super().orthogonalize(p, grad, **kwargs)

            # Per-expert NS: copy attributes onto each slice so the parent's
            # tp_group / partition_dim logic still works. The leading axis is
            # the expert axis, fully replicated across TP/EP (no NS comm needed
            # along it). We keep partition_dim and expert_tp on the slices.
            partition_dim = getattr(p, "partition_dim", None)
            expert_tp = getattr(p, "expert_tp", False)
            out = torch.empty_like(grad)
            for e in range(grad.shape[0]):
                slice_p = p[e]
                slice_g = grad[e]
                # The parent reads `partition_dim` / `expert_tp` from the
                # *parameter* (slice_p), not the grad. Mirror them.
                if partition_dim is not None:
                    slice_p.partition_dim = partition_dim  # type: ignore[attr-defined]
                if expert_tp:
                    slice_p.expert_tp = True  # type: ignore[attr-defined]
                out[e] = super().orthogonalize(slice_p, slice_g, **kwargs)
            return out

else:
    Step3p5TensorParallelMuon = None  # type: ignore[assignment, misc]


def _step3p5_muon_config_to_kwargs(config: Any, model_chunks: Any, pg_collection: Any) -> Dict[str, Any]:
    """Translate ``OptimizerConfig`` muon_* fields into ``Step3p5TensorParallelMuon`` kwargs.

    Mirrors the kwarg shape used by the stock ``muon`` registration but
    routes through our 3D-aware subclass.
    """
    return {
        "lr": config.lr,
        "momentum": config.muon_momentum,
        "nesterov": config.muon_nesterov,
        "weight_decay": config.weight_decay,
        "use_decoupled_weight_decay": True,
        "split_qkv": config.muon_split_qkv,
        "fp32_matmul_prec": config.muon_fp32_matmul_prec,
        "coefficient_type": config.muon_coefficient_type,
        "num_ns_steps": config.muon_num_ns_steps,
        "scale_mode": config.muon_scale_mode,
        "extra_scale_factor": config.muon_extra_scale_factor,
        "pg_collection": pg_collection,
        "tp_mode": config.muon_tp_mode,
    }


def register_step3p5_muon() -> None:
    """Populate :data:`_EMERGING_OPTIMIZERS` with the ``muon_step3p5`` entry.

    Idempotent; safe to call multiple times. Raises if
    ``emerging_optimizers >= 0.2`` is not installed.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    if not HAVE_EMERGING_OPTIMIZERS:
        raise RuntimeError(
            "emerging_optimizers >= 0.2 is required to use 'muon_step3p5'. "
            "Install via `pip install git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@v0.2.0`."
        )
    _EMERGING_OPTIMIZERS[_STEP3P5_MUON_NAME] = EmergingOptimizerEntry(
        optimizer_cls=Step3p5TensorParallelMuon,
        init_state_fn=_eopt_init_state_fn,
        config_to_kwargs=_step3p5_muon_config_to_kwargs,
        default_param_overrides=_step3p5_default_param_overrides_factory(),
    )
    _REGISTERED = True
    logger.info("Registered emerging optimizer entry: %s", _STEP3P5_MUON_NAME)


# ---------------------------------------------------------------------------
# 2. FP16 NS intermediates
# ---------------------------------------------------------------------------


def enable_fp16_ns() -> None:
    """Monkey-patch :func:`newton_schulz` to use FP16 (not BF16) intermediates.

    Step-3.5 §4.1.1 reports rare unrecoverable training-loss spikes when
    NS runs in BF16 (cumulative addition error from BF16's 7-bit mantissa).
    FP16's 10-bit mantissa eliminates the failure mode at the cost of a
    smaller exponent range. The wrapped iteration normalizes to spectral
    norm <= 1 before iterating, so the dynamic range fits comfortably in
    FP16.

    The patch is idempotent. Removing it would require restarting the
    process; the original function reference is captured but not exposed
    (we don't expect a need to revert mid-run).
    """
    global _FP16_NS_INSTALLED
    if _FP16_NS_INSTALLED:
        return
    if not HAVE_EMERGING_OPTIMIZERS:
        raise RuntimeError("emerging_optimizers >= 0.2 is required to enable FP16 NS.")
    from emerging_optimizers.orthogonalized_optimizers import muon_utils

    # Capture the symbols the wrapper needs so we don't re-resolve per call.
    get_coefficient_iterator = muon_utils.get_coefficient_iterator
    distributed_normalize_p2 = muon_utils.distributed_normalize_p2
    coefficient_sets_map = muon_utils._COEFFICIENT_SETS
    newton_schulz_step = muon_utils.newton_schulz_step

    def _fp16_newton_schulz(
        x: torch.Tensor,
        steps: int,
        coefficient_type: str = "quintic",
        custom_coefficient_sets: Any = None,
        eps: float = 1e-7,
        transpose: Optional[bool] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        use_syrk: bool = False,
    ) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError("Input tensor x must have at least 2 dimensions since Muon is not for 1d parameters.")
        if x.dtype != torch.float32:
            raise ValueError(f"Input tensor x must be in float32, got {x.dtype}")
        if transpose is None:
            transpose = x.size(-2) > x.size(-1)
        if transpose:
            x = x.mT
        if tp_group is not None:
            X = distributed_normalize_p2(x, eps, tp_group)
        else:
            X = torch.nn.functional.normalize(x, p=2, dim=(-2, -1), eps=eps)  # type: ignore[arg-type]

        if coefficient_type in coefficient_sets_map:
            coefficient_sets = coefficient_sets_map[coefficient_type]
        elif coefficient_type == "custom":
            if custom_coefficient_sets is None:
                raise ValueError("custom_coefficient_sets must be provided when coefficient_type is 'custom'.")
            coefficient_sets = custom_coefficient_sets
        else:
            raise ValueError(f"Invalid coefficient type: {coefficient_type}")

        iter_mode = "cycle" if coefficient_type != "polar_express" else "repeat_last"
        coeff_iter = get_coefficient_iterator(steps, coefficient_sets, mode=iter_mode)

        # Step-3.5 §4.1.1: run NS iterations in FP16, not BF16.
        X = X.to(torch.float16)
        if use_syrk:
            # use_syrk path requires bfloat16 per upstream contract; we don't override.
            X = X.to(torch.bfloat16)
            ns_step_fn = muon_utils.newton_schulz_step_tsyrk
        else:
            ns_step_fn = newton_schulz_step

        for a, b, c in coeff_iter:
            X = ns_step_fn(X, a, b, c, tp_group=tp_group)

        X = X.to(torch.float32)
        if transpose:
            X = X.mT
        return X

    muon_utils.newton_schulz = _fp16_newton_schulz
    _FP16_NS_INSTALLED = True
    logger.info("Patched newton_schulz to use FP16 NS intermediates.")


# ---------------------------------------------------------------------------
# 3. Tag grouped-expert weights for Muon routing
# ---------------------------------------------------------------------------


def mark_3d_experts_for_muon(model: torch.nn.Module) -> int:
    """Tag 3D grouped-expert weights with ``_step3p5_route_to_muon = True``.

    Walks the model's parameters; any parameter whose name matches the
    grouped-expert pattern *and* is 3D gets tagged so that
    :func:`_step3p5_is_nonlinear_or_embedding` allows it through the Muon
    side of the routing split.

    Returns the number of parameters tagged (for logging).
    """
    tagged = 0
    for name, param in model.named_parameters():
        if param.ndim != 3:
            continue
        # Match TEGroupedMLP weight names (linear_fc1.weight*, linear_fc2.weight*
        # under .experts.). The "weight" prefix covers both the bare `.weight`
        # and TE's bucketed suffixes (e.g. `weight0`, `weight_quantizer`).
        if "experts.linear_fc" in name and "weight" in name.rsplit(".", 1)[-1]:
            param._step3p5_route_to_muon = True  # type: ignore[attr-defined]
            tagged += 1
    if tagged > 0:
        logger.info("Tagged %d 3D grouped-expert weights for Muon routing.", tagged)
    return tagged


def get_muon_optimizer_name() -> str:
    """Return the registry name to set on ``OptimizerConfig.optimizer``."""
    return _STEP3P5_MUON_NAME
