"""Native Keras CNN/SuperONN factories; no PyTorch/NumPy graph bridge."""
from __future__ import annotations
import tensorflow as tf
from ..options import superonn_options, SUPERONN_DEFAULTS, TF_EXECUTION_DEFAULTS, TF_OPERATOR_OPTIONS
from .layers import SuperONN1D, _spatial_integer, _positive_integer
from .nd_layers import SuperONN2D, SuperONNTranspose1D, SuperONNTranspose2D
from .functional import normalize_padding, output_shape_1d


@tf.keras.utils.register_keras_serializable(package="SuperONN")
class DepthwiseSuperONN1D(tf.keras.layers.Layer):
    def __init__(self, kernel_size, depth_multiplier=1, strides=1, padding="same",
                 dilation_rate=1, use_bias=True, kernel_initializer="glorot_uniform",
                 superonn=None, activation=None, bias_initializer="zeros",
                 depthwise_initializer=None, depthwise_regularizer=None,
                 bias_regularizer=None, depthwise_constraint=None, bias_constraint=None,
                 jit_compile=None, data_format="channels_last", **kwargs):
        super().__init__(autocast=False, **kwargs)
        kernel_size = _spatial_integer(kernel_size, "kernel_size")
        strides = _spatial_integer(strides, "strides")
        dilation_rate = _spatial_integer(dilation_rate, "dilation_rate")
        depth_multiplier = _positive_integer(depth_multiplier, "depth_multiplier")
        padding = normalize_padding(padding, 1)
        if data_format not in ("channels_last", "channels_first"):
            raise ValueError("Invalid data_format")
        self.data_format = data_format
        self.input_spec = tf.keras.layers.InputSpec(ndim=3)
        self.settings = dict(kernel_size=kernel_size, depth_multiplier=depth_multiplier, data_format=data_format,
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
        shape = tf.TensorShape(input_shape)
        if shape.rank != 3:
            raise ValueError("DepthwiseSuperONN1D requires rank-3 input")
        channel_axis = -1 if self.data_format == "channels_last" else 1
        channels = shape[channel_axis]
        if channels is None or channels < 1:
            raise ValueError("DepthwiseSuperONN1D requires known positive channels")
        args = {k: v for k, v in self.settings.items() if k != "depth_multiplier"}
        self.operator = SuperONN1D(int(channels) * self.settings["depth_multiplier"],
                                  groups=int(channels), **args, **self.options,
                                  kernel_regularizer=self.depthwise_regularizer,
                                  kernel_constraint=self.depthwise_constraint,
                                  dtype=self.dtype_policy, name="operator")
        self.operator.build(input_shape)
        self.input_spec = tf.keras.layers.InputSpec(ndim=3, axes={channel_axis: int(channels)})
        super().build(input_shape)

    def call(self, inputs):
        return self.operator(inputs)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape)
        if shape.rank != 3:
            raise ValueError("DepthwiseSuperONN1D requires rank-3 input")
        channels = shape[-1 if self.data_format == "channels_last" else 1]
        filters = None if channels is None else channels * self.settings["depth_multiplier"]
        return output_shape_1d(shape, filters, self.settings["kernel_size"], self.settings["strides"],
                               self.settings["padding"], self.settings["dilation_rate"], self.data_format)

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
                 bias_constraint=None, jit_compile=None, data_format="channels_last", **kwargs):
        super().__init__(autocast=False, **kwargs)
        if data_format not in ("channels_last", "channels_first"):
            raise ValueError("Invalid data_format")
        self.data_format = data_format
        self.input_spec = tf.keras.layers.InputSpec(ndim=3)
        self.settings = dict(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                             dilation_rate=dilation_rate, use_bias=use_bias, depth_multiplier=depth_multiplier, data_format=data_format,
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
                                            superonn=self.options, data_format=data_format, dtype=self.dtype_policy)
        self.pointwise = tf.keras.layers.Conv1D(filters, 1, use_bias=use_bias,
                                              activation=activation, bias_initializer=bias_initializer,
                                              kernel_regularizer=pointwise_regularizer, bias_regularizer=bias_regularizer,
                                              kernel_constraint=pointwise_constraint, bias_constraint=bias_constraint,
                                              kernel_initializer=pointwise_initializer, data_format="channels_last", dtype=self.dtype_policy)

    def build(self, input_shape):
        self.depthwise.build(input_shape)
        shape = self.depthwise.compute_output_shape(input_shape)
        if self.data_format == "channels_first":
            shape = (shape[0], shape[2], shape[1])
        self.pointwise.build(shape)
        super().build(input_shape)

    def call(self, inputs):
        result = self.depthwise(inputs)
        if self.data_format == "channels_first":
            result = tf.transpose(result, (0, 2, 1))
        result = self.pointwise(result)
        return tf.transpose(result, (0, 2, 1)) if self.data_format == "channels_first" else result

    def compute_output_shape(self, input_shape):
        shape = self.depthwise.compute_output_shape(input_shape)
        if self.data_format == "channels_first":
            shape = (shape[0], shape[2], shape[1])
        result = self.pointwise.compute_output_shape(shape)
        return tf.TensorShape((result[0], result[2], result[1])) if self.data_format == "channels_first" else result

    def get_config(self):
        return {**super().get_config(), **self.settings, "superonn": self.options}


def _factory_options(superonn, kwargs, spatial_rank=1):
    if superonn is not None and not hasattr(superonn, "items"):
        raise ValueError("superonn options must be a mapping")
    known = set(SUPERONN_DEFAULTS) | set(TF_EXECUTION_DEFAULTS) | TF_OPERATOR_OPTIONS
    options = {**dict(superonn or {}), **{key: kwargs.pop(key) for key in tuple(kwargs) if key in known}}
    return superonn_options(options, spatial_rank=spatial_rank)


def spectral_conv1d(filters, kernel_size, *, operator="cnn", superonn=None, **kwargs):
    if operator == "cnn":
        return tf.keras.layers.Conv1D(filters, kernel_size, **kwargs)
    if operator == "superonn":
        return SuperONN1D(filters, kernel_size, **_factory_options(superonn, kwargs), **kwargs)
    raise ValueError("operator must be cnn or superonn")


def spectral_depthwise_conv1d(kernel_size, *, operator="cnn", superonn=None, **kwargs):
    if operator == "cnn":
        return tf.keras.layers.DepthwiseConv1D(kernel_size, **kwargs)
    if operator == "superonn":
        return DepthwiseSuperONN1D(kernel_size, superonn=_factory_options(superonn, kwargs), **kwargs)
    raise ValueError("operator must be cnn or superonn")


def spectral_separable_conv1d(filters, kernel_size, *, operator="cnn", superonn=None, **kwargs):
    if operator == "cnn":
        return tf.keras.layers.SeparableConv1D(filters, kernel_size, **kwargs)
    if operator == "superonn":
        return SeparableSuperONN1D(filters, kernel_size, superonn=_factory_options(superonn, kwargs), **kwargs)
    raise ValueError("operator must be cnn or superonn")


def _spatial_factory(cnn_class, operational_class, filters, kernel_size, operator, superonn, kwargs):
    if operator == "cnn":
        return cnn_class(filters, kernel_size, **kwargs)
    if operator == "superonn":
        options = _factory_options(superonn, kwargs, operational_class.spatial_rank)
        return operational_class(filters, kernel_size, **options, **kwargs)
    raise ValueError("operator must be cnn or superonn")


def conv2d(filters, kernel_size, *, operator="cnn", superonn=None, **kwargs):
    """Select native Keras Conv2D or SuperONN2D with explicit operator settings."""
    return _spatial_factory(tf.keras.layers.Conv2D, SuperONN2D, filters, kernel_size, operator, superonn, kwargs)


def conv_transpose1d(filters, kernel_size, *, operator="cnn", superonn=None, **kwargs):
    """Select native Keras Conv1DTranspose or SuperONNTranspose1D."""
    return _spatial_factory(tf.keras.layers.Conv1DTranspose, SuperONNTranspose1D, filters, kernel_size, operator, superonn, kwargs)


def conv_transpose2d(filters, kernel_size, *, operator="cnn", superonn=None, **kwargs):
    """Select native Keras Conv2DTranspose or SuperONNTranspose2D."""
    return _spatial_factory(tf.keras.layers.Conv2DTranspose, SuperONNTranspose2D, filters, kernel_size, operator, superonn, kwargs)


def model_regularization_loss(model):
    """Weight regularizers are included once, even when a shared layer is called repeatedly."""
    losses = tuple(model.losses)
    dtype = tf.float64 if any(tf.as_dtype(value.dtype) == tf.float64 for value in losses) else tf.float32
    return tf.add_n([tf.cast(value, dtype) for value in losses]) if losses else tf.constant(0., dtype)
