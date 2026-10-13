"""Framework-free defaults and validation for operational layer options."""
from __future__ import annotations
from collections.abc import Mapping
import math
from numbers import Integral, Real

SUPERONN_DEFAULTS = {
    "q": 3, "max_shift": 2.0, "shift_mode": "learned", "shift_sharing": "full",
    "initialization": "linear", "input_bound": 1.0, "output_chunk_size": 4,
    "recompute": True, "order_regularization": 0.0, "order_decay": 1.5,
    "conv_precision": "stable",
}
TF_EXECUTION_DEFAULTS = {"jit_compile": False}
TF_OPERATOR_OPTIONS = {"seed", "shift_initializer", "boundary", "include_constant"}

def superonn_options(options=None, *, spatial_rank=1):
    if spatial_rank not in (1, 2) or isinstance(spatial_rank, bool):
        raise ValueError("spatial_rank must be 1 or 2")
    if options is not None and not isinstance(options, Mapping):
        raise ValueError("superonn options must be a mapping")
    options = dict(options or {})
    unknown = set(options) - (set(SUPERONN_DEFAULTS) | set(TF_EXECUTION_DEFAULTS) | TF_OPERATOR_OPTIONS)
    if unknown:
        raise ValueError(f"Unknown SuperONN hyperparameters: {sorted(unknown, key=str)}")
    result = {**SUPERONN_DEFAULTS, **TF_EXECUTION_DEFAULTS, **options}
    for key in ("q", "output_chunk_size"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"SuperONN {key} must be a positive integer")
        result[key] = int(value)
    value = result["max_shift"]
    try:
        limits = (value,) if isinstance(value, Real) else tuple(value)
    except TypeError as error:
        raise ValueError("Invalid SuperONN max_shift") from error
    if len(limits) not in (1, spatial_rank) or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or v < 0 for v in limits):
        raise ValueError("Invalid SuperONN max_shift")
    if not isinstance(value, Real) and len(limits) != spatial_rank:
        raise ValueError("max_shift tuple must match spatial_rank")
    result["max_shift"] = float(limits[0]) if spatial_rank == 1 or isinstance(value, Real) else tuple(float(v) for v in limits)
    for key in ("order_regularization", "order_decay"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < (1 if key == "order_decay" else 0):
            raise ValueError(f"Invalid SuperONN {key}")
        result[key] = float(value)
    if result["shift_mode"] not in ("learned", "random_fixed") or result["shift_sharing"] not in ("full", "shared"):
        raise ValueError("Invalid SuperONN shift model or sharing")
    if result["initialization"] not in ("linear", "all_orders"):
        raise ValueError("Invalid SuperONN initialization")
    bound = result["input_bound"]
    if bound is not None and (isinstance(bound, bool) or not isinstance(bound, Real) or not math.isfinite(bound) or bound <= 0):
        raise ValueError("SuperONN input_bound must be positive or None")
    if not isinstance(result["recompute"], bool):
        raise ValueError("SuperONN recompute must be boolean")
    if result["conv_precision"] not in ("stable", "autocast"):
        raise ValueError("SuperONN conv_precision must be stable or autocast")
    if not isinstance(result["jit_compile"], bool):
        raise ValueError("SuperONN jit_compile must be boolean")
    if "seed" in result:
        seed = result["seed"]
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral)):
            raise ValueError("SuperONN seed must be an integer or None")
        result["seed"] = None if seed is None else int(seed)
    if result.get("shift_initializer") not in (None, "zeros", "random"):
        raise ValueError("Invalid SuperONN shift_initializer")
    if result.get("boundary", "zeros") not in ("zeros", "border", "reflection", "circular"):
        raise ValueError("Invalid SuperONN boundary")
    if "include_constant" in result and not isinstance(result["include_constant"], bool):
        raise ValueError("SuperONN include_constant must be boolean")
    return result
