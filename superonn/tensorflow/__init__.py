"""Native TensorFlow/Keras layers and interchangeable convolution factories."""
import tensorflow as tf

if tf.keras.backend.backend() != "tensorflow":
    raise RuntimeError(
        "superonn.tensorflow requires KERAS_BACKEND=tensorflow; select it before importing Keras."
    )

from .layers import SuperONN1D, SuperONNBlock1D
from .nd_layers import SuperONN2D, SuperONNTranspose1D, SuperONNTranspose2D
from .spatial import (
    DepthwiseSuperONN1D, SeparableSuperONN1D,
    spectral_conv1d, spectral_depthwise_conv1d, spectral_separable_conv1d,
    model_regularization_loss, conv2d, conv_transpose1d, conv_transpose2d,
)

# General-purpose names; the existing spectral names remain compatible aliases.
conv1d = spectral_conv1d
depthwise_conv1d = spectral_depthwise_conv1d
separable_conv1d = spectral_separable_conv1d

__all__ = ["SuperONN2D", "SuperONNTranspose1D", "SuperONNTranspose2D", "SuperONN1D", "SuperONNBlock1D", "DepthwiseSuperONN1D", "SeparableSuperONN1D",
           "conv1d", "conv2d", "conv_transpose1d", "conv_transpose2d", "depthwise_conv1d", "separable_conv1d",
           "spectral_conv1d", "spectral_depthwise_conv1d", "spectral_separable_conv1d", "model_regularization_loss"]
