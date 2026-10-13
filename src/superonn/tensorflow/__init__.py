"""Native TensorFlow/Keras layers and interchangeable convolution factories."""
from .layers import SuperONN1D, SuperONNBlock1D
from .spatial import (
    DepthwiseSuperONN1D, SeparableSuperONN1D,
    spectral_conv1d, spectral_depthwise_conv1d, spectral_separable_conv1d,
    model_regularization_loss,
)

# General-purpose names; the existing spectral names remain compatible aliases.
conv1d = spectral_conv1d
depthwise_conv1d = spectral_depthwise_conv1d
separable_conv1d = spectral_separable_conv1d

__all__ = ["SuperONN1D", "SuperONNBlock1D", "DepthwiseSuperONN1D", "SeparableSuperONN1D",
           "conv1d", "depthwise_conv1d", "separable_conv1d",
           "spectral_conv1d", "spectral_depthwise_conv1d", "spectral_separable_conv1d", "model_regularization_loss"]
