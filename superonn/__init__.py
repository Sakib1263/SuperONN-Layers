"""Reusable operators with lazy loading of the optional framework backends."""
from .presets import SuperONNConfig, superonn_regularization_loss

__version__ = "0.3.1"
__all__ = ["SuperONN1d", "SuperONN2d", "SuperONNTranspose1d", "SuperONNTranspose2d", "SuperONNConfig", "superonn_regularization_loss"]


def __getattr__(name):
    if name in ("SuperONN1d", "SuperONN2d", "SuperONNTranspose1d", "SuperONNTranspose2d"):
        from . import layers
        value = getattr(layers, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
