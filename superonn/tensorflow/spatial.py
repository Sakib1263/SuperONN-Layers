"""Native Keras CNN/SuperONN factories; no PyTorch/NumPy graph bridge."""
from __future__ import annotations
import tensorflow as tf
from .options import superonn_options, SUPERONN_DEFAULTS, TF_EXECUTION_DEFAULTS
from .layers import SuperONN1D, _spatial_integer


@tf.keras.utils.register_keras_serializable(package="SuperONN")
class DepthwiseSuperONN1D(tf.keras.layers.Layer):
    def __init__(self, kernel_size, depth_multiplier=1, strides=1, padding="same",
                 dilation_rate=1, use_bias=True, kernel_initializer="glorot_uniform",
                 superonn=None, activation=None, bias_initializer="zeros",
                 depthwise_initializer=None, depthwise_regularizer=None,
                 bias_regularizer=None, depthwise_constraint=None, bias_constraint=None,
                 jit_compile=None, **kwargs):
        super().__init__(autocast=False, **kwargs)
        kernel_size = _spatial_integer(kernel_size, "kernel_size")
        strides = _spatial_integer(strides, "strides")
        dilation_rate = _spatial_integer(dilation_rate, "dilation_rate")
        if isinstance(depth_multiplier, bool) or int(depth_multiplier) != depth_multiplier or depth_multiplier < 1:
            raise ValueError("depth_multiplier must be a positive integer")
        self.settings = dict(kernel_size=kernel_size, depth_multiplier=int(depth_multiplier),
                             strides=strides, padding=padding, dilation_rate=dilation_rate,
                             use_bias=use_bias, kernel_initializer=tf.keras.initializers.serialize(tf.keras.initializers.get(kernel_initializer if depthwise_initializer is None else depthwise_initializer)),
                             activation=tf.keras.activations.serialize(tf.keras.activations.get(activation)),
                             bias_initializer=tf.keras.initializers.serialize(tf.keras.initializers.get(bias_initializer)),
                             bias_regularizer=tf.keras.regularizers.serialize(tf.keras.regularizers.get(bias_regularizer)),
                             bias_constraint=tf.keras.constraints.serialize(tf.keras.constraints.get(bias_constraint)))
        self.depthwise_regularizer = tf.keras.regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = tf.keras.constraints.get(depthwise_constraint)
        self.options = superonn_options(superonn)
        if jit_compile is not None:
            self.options = superonn_options({**self.options, "jit_compile": jit_compile})
        self.operator = None

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None or channels < 1:
            raise ValueError("DepthwiseSuperONN1D requires known positive channels")
        args = {k: v for k, v in self.settings.items() if k != "depth_multiplier"}
        self.operator = SuperONN1D(int(channels) * self.settings["depth_multiplier"],
                                  groups=int(channels), **args, **self.options,
                                  kernel_regularizer=self.depthwise_regularizer,
                                  kernel_constraint=self.depthwise_constraint,
                                  dtype=self.dtype_policy, name="operator")
        self.operator.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        return self.operator(inputs)

    def compute_output_shape(self, input_shape):
        if self.operator is None:
            length = input_shape[1]
            effective = (self.settings["kernel_size"] - 1) * self.settings["dilation_rate"] + 1
            if length is not None:
                length = ((length + self.settings["strides"] - 1) // self.settings["strides"] if self.settings["padding"] == "same" else
                          (length - effective) // self.settings["strides"] + 1)
            return tf.TensorShape((input_shape[0], length, input_shape[-1] * self.settings["depth_multiplier"]))
        return self.operator.compute_output_shape(input_shape)

    def get_config(self):
        return {**super().get_config(), **self.settings, "superonn": self.options,
                "depthwise_regularizer": tf.keras.regularizers.serialize(self.depthwise_regularizer),
                "depthwise_constraint": tf.keras.constraints.serialize(self.depthwise_constraint)}


@tf.keras.utils.register_keras_serializable(package="SuperONN")
class SeparableSuperONN1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding="same", dilation_rate=1,
                 use_bias=True, depth_multiplier=1, depthwise_initializer="glorot_uniform",
                 pointwise_initializer="glorot_uniform", superonn=None, activation=None,
                 bias_initializer="zeros", depthwise_regularizer=None, pointwise_regularizer=None,
                 bias_regularizer=None, depthwise_constraint=None, pointwise_constraint=None,
                 bias_constraint=None, jit_compile=None, **kwargs):
        super().__init__(autocast=False, **kwargs)
        self.settings = dict(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                             dilation_rate=dilation_rate, use_bias=use_bias, depth_multiplier=depth_multiplier,
                             depthwise_initializer=tf.keras.initializers.serialize(tf.keras.initializers.get(depthwise_initializer)),
                             pointwise_initializer=tf.keras.initializers.serialize(tf.keras.initializers.get(pointwise_initializer)))
        self.options = superonn_options(superonn)
        if jit_compile is not None:
            self.options = superonn_options({**self.options, "jit_compile": jit_compile})
        self.settings.update({"activation": tf.keras.activations.serialize(tf.keras.activations.get(activation)),
            "bias_initializer": tf.keras.initializers.serialize(tf.keras.initializers.get(bias_initializer)),
            **{name: tf.keras.regularizers.serialize(tf.keras.regularizers.get(value)) for name,value in
               (("depthwise_regularizer",depthwise_regularizer),("pointwise_regularizer",pointwise_regularizer),("bias_regularizer",bias_regularizer))},
            **{name: tf.keras.constraints.serialize(tf.keras.constraints.get(value)) for name,value in
               (("depthwise_constraint",depthwise_constraint),("pointwise_constraint",pointwise_constraint),("bias_constraint",bias_constraint))}})
        self.depthwise = DepthwiseSuperONN1D(kernel_size, depth_multiplier, strides, padding, dilation_rate,
                                            use_bias=False, kernel_initializer=depthwise_initializer,
                                            depthwise_regularizer=depthwise_regularizer, depthwise_constraint=depthwise_constraint,
                                            superonn=self.options, dtype=self.dtype_policy)
        self.pointwise = tf.keras.layers.Conv1D(filters, 1, use_bias=use_bias,
                                              activation=activation, bias_initializer=bias_initializer,
                                              kernel_regularizer=pointwise_regularizer, bias_regularizer=bias_regularizer,
                                              kernel_constraint=pointwise_constraint, bias_constraint=bias_constraint,
                                              kernel_initializer=pointwise_initializer, dtype=self.dtype_policy)

    def build(self, input_shape):
        self.depthwise.build(input_shape)
        self.pointwise.build(self.depthwise.compute_output_shape(input_shape))
        super().build(input_shape)

    def call(self, inputs):
        return self.pointwise(self.depthwise(inputs))

    def compute_output_shape(self, input_shape):
        return self.pointwise.compute_output_shape(self.depthwise.compute_output_shape(input_shape))

    def get_config(self):
        return {**super().get_config(), **self.settings, "superonn": self.options}


def spectral_conv1d(filters, kernel_size, *, operator="cnn", superonn=None, **kwargs):
    if operator == "cnn":
        return tf.keras.layers.Conv1D(filters, kernel_size, **kwargs)
    if operator == "superonn":
        options = {**dict(superonn or {}), **{key: kwargs.pop(key) for key in tuple(kwargs) if key in SUPERONN_DEFAULTS or key in TF_EXECUTION_DEFAULTS}}
        return SuperONN1D(filters, kernel_size, **superonn_options(options), **kwargs)
    raise ValueError("operator must be cnn or superonn")


def spectral_depthwise_conv1d(kernel_size, *, operator="cnn", superonn=None, **kwargs):
    if operator == "cnn":
        return tf.keras.layers.DepthwiseConv1D(kernel_size, **kwargs)
    if operator == "superonn":
        options = {**dict(superonn or {}), **{key: kwargs.pop(key) for key in tuple(kwargs) if key in SUPERONN_DEFAULTS or key in TF_EXECUTION_DEFAULTS}}
        return DepthwiseSuperONN1D(kernel_size, superonn=options, **kwargs)
    raise ValueError("operator must be cnn or superonn")


def spectral_separable_conv1d(filters, kernel_size, *, operator="cnn", superonn=None, **kwargs):
    if operator == "cnn":
        return tf.keras.layers.SeparableConv1D(filters, kernel_size, **kwargs)
    if operator == "superonn":
        options = {**dict(superonn or {}), **{key: kwargs.pop(key) for key in tuple(kwargs) if key in SUPERONN_DEFAULTS or key in TF_EXECUTION_DEFAULTS}}
        return SeparableSuperONN1D(filters, kernel_size, superonn=options, **kwargs)
    raise ValueError("operator must be cnn or superonn")


def model_regularization_loss(model):
    """Weight regularizers are included once, even when a shared layer is called repeatedly."""
    losses = tuple(model.losses)
    dtype = tf.float64 if any(tf.as_dtype(value.dtype) == tf.float64 for value in losses) else tf.float32
    return tf.add_n([tf.cast(value, dtype) for value in losses]) if losses else tf.constant(0., dtype)
