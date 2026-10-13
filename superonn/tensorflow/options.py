"""Hyperparameter defaults and validation for TensorFlow spatial operators."""
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

def superonn_options(options=None):
    if options is not None and not isinstance(options, Mapping):
        raise ValueError("superonn options must be a mapping")
    options = dict(options or {})
    unknown = set(options) - (set(SUPERONN_DEFAULTS) | set(TF_EXECUTION_DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown SuperONN hyperparameters: {sorted(unknown)}")
    result = {**SUPERONN_DEFAULTS, **TF_EXECUTION_DEFAULTS, **options}
    for key in ("q", "output_chunk_size"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"SuperONN {key} must be a positive integer")
        result[key] = int(value)
    for key in ("max_shift", "order_regularization", "order_decay"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < (1 if key == "order_decay" else 0):
            raise ValueError(f"Invalid SuperONN {key}")
        result[key] = float(value)
    if result["shift_mode"] not in ("learned", "random_fixed") or result["shift_sharing"] not in ("full", "shared"):
        raise ValueError("Invalid SuperONN shift model or sharing")
    if result["shift_mode"] == "random_fixed" and not result["max_shift"].is_integer():
        raise ValueError("random_fixed max_shift must be an integer")
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
    return result
