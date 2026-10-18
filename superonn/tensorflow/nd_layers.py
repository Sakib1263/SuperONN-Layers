"""Native Keras spatial and transpose SuperONN layers."""
from __future__ import annotations

import itertools
import math
from numbers import Integral, Real

import tensorflow as tf

from .functional import position_grid, sample_nd, sparse_upsample, spatial_tuple, normalize_padding
from .layers import _positive_integer, _CombinedRegularizer, _compiled_recompute, _compiled_chunk_vjp, _bounded_recomputed_chunks


@tf.keras.utils.register_keras_serializable(package="SuperONN")
class _NDPolynomialInitializer(tf.keras.initializers.Initializer):
    def __init__(self, initializer="glorot_uniform", initialization="linear", include_constant=False):
        self.initializer = tf.keras.initializers.get(initializer)
        self.initialization = initialization
        self.include_constant = bool(include_constant)

    def __call__(self, shape, dtype=None):
        spatial, channels, orders, filters = tuple(shape[:-3]), *shape[-3:]
        if self.initialization == "all_orders":
            return tf.reshape(self.initializer((*spatial, channels * orders, filters), dtype=dtype), shape)
        linear = self.initializer((*spatial, channels, filters), dtype=dtype)
        return tf.stack([linear if i == int(self.include_constant) else tf.zeros_like(linear)
                         for i in range(orders)], axis=-2)

    def get_config(self):
        return dict(initializer=tf.keras.initializers.serialize(self.initializer),
                    initialization=self.initialization, include_constant=self.include_constant)


@tf.keras.utils.register_keras_serializable(package="SuperONN")
class _NDOrderRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, strength=0., decay=1.5, include_constant=False):
        self.strength, self.decay = float(strength), float(decay)
        self.include_constant = bool(include_constant)

    def __call__(self, kernel):
        dtype = tf.float64 if tf.as_dtype(kernel.dtype) == tf.float64 else tf.float32
        values = tf.cast(kernel, dtype)
        values = values[..., int(self.include_constant):, :]
        degree = tf.range(tf.shape(values)[-2])
        factors = tf.pow(tf.constant(self.decay, dtype), tf.cast(degree, dtype))
        return tf.constant(self.strength, dtype) * tf.reduce_sum(values ** 2 * factors[:, None])

    def get_config(self):
        return dict(strength=self.strength, decay=self.decay, include_constant=self.include_constant)


@tf.keras.utils.register_keras_serializable(package="SuperONN")
class _NDShiftInitializer(tf.keras.initializers.Initializer):
    def __init__(self, shift_mode, shift_initializer, max_shift, seed=None):
        self.shift_mode, self.shift_initializer = shift_mode, shift_initializer
        self.max_shift, self.seed = tuple(max_shift), seed

    def __call__(self, shape, dtype=None):
        dtype = tf.as_dtype(dtype or tf.float32)
        components = []
        for axis, limit in enumerate(self.max_shift):
            if self.shift_initializer == "zeros" or limit == 0:
                value = tf.zeros(shape[:-1], dtype)
            else:
                integer = self.shift_mode == "random_fixed"
                draw_dtype = tf.int32 if integer else dtype
                lo, hi = (-math.floor(limit), math.floor(limit) + 1) if integer else (-.5, .5)
                if self.seed is None:
                    value = tf.random.uniform(shape[:-1], lo, hi, dtype=draw_dtype)
                else:
                    value = tf.random.stateless_uniform(shape[:-1], [self.seed, axis], lo, hi, dtype=draw_dtype)
                value = tf.cast(value, dtype) if integer else tf.math.atanh(value)
            components.append(value)
        return tf.stack(components, axis=-1)

    def get_config(self):
        return dict(shift_mode=self.shift_mode, shift_initializer=self.shift_initializer,
                    max_shift=self.max_shift, seed=self.seed)


class _SuperONNND(tf.keras.layers.Layer):
    spatial_rank = 2
    transposed = False

    def __init__(self, filters, kernel_size, q=3, strides=1, padding=None,
                 dilation_rate=1, groups=1, shift_mode="learned", shift_sharing="full",
                 max_shift=2., shift_initializer=None, use_bias=True,
                 kernel_initializer="glorot_uniform", initialization="linear",
                 boundary="zeros", output_chunk_size=4, recompute=True, seed=None,
                 input_bound=None, include_constant=False, output_padding=None,
                 order_regularization=0., order_decay=1.5, conv_precision="stable",
                 jit_compile=False, data_format="channels_last", activation=None,
                 bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, **kwargs):
        super().__init__(autocast=False, **kwargs)
        d = self.spatial_rank
        self.filters = _positive_integer(filters, "filters")
        self.kernel_size = spatial_tuple(kernel_size, d, "kernel_size")
        self.strides = spatial_tuple(strides, d, "strides")
        self.dilation_rate = spatial_tuple(dilation_rate, d, "dilation_rate")
        self.q = _positive_integer(q, "q")
        self.include_constant = bool(include_constant)
        self.orders = self.q + int(self.include_constant)
        self.groups = _positive_integer(groups, "groups")
        if self.filters % self.groups:
            raise ValueError("filters must be divisible by groups")
        padding = ("valid" if self.transposed else "same") if padding is None else padding
        self.padding = normalize_padding(padding, d)
        if not isinstance(self.padding, str) and d == 1:
            self.padding = (self.padding,)
        self.output_padding = None if output_padding is None else spatial_tuple(output_padding, d, "output_padding", 0)
        if not self.transposed and self.output_padding is not None:
            raise ValueError("output_padding belongs to transpose layers")
        if self.transposed and self.include_constant:
            raise ValueError("A per-tap constant is undefined on a sparse transpose lattice; use bias")
        if self.output_padding is not None:
            limits = self.strides if isinstance(self.padding, str) else tuple(max(s, dil) for s, dil in zip(self.strides, self.dilation_rate))
            if any(op >= limit for op, limit in zip(self.output_padding, limits)):
                raise ValueError("output_padding must be smaller than stride (or dilation for numeric padding)")
        if shift_mode not in ("learned", "random_fixed") or shift_sharing not in ("full", "shared"):
            raise ValueError("Invalid shift_mode or shift_sharing")
        limits = (max_shift,) * d if isinstance(max_shift, Real) else tuple(max_shift)
        if len(limits) != d or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or v < 0 for v in limits):
            raise ValueError("max_shift must contain finite nonnegative numbers")
        self.max_shift = tuple(float(v) for v in limits)
        self.shift_mode, self.shift_sharing = shift_mode, shift_sharing
        self.shift_initializer = shift_initializer or ("random" if shift_mode == "random_fixed" else "zeros")
        if self.shift_initializer not in ("zeros", "random"):
            raise ValueError("shift_initializer must be zeros or random")
        if initialization not in ("linear", "all_orders"):
            raise ValueError("initialization must be linear or all_orders")
        self.initialization = initialization
        if boundary not in ("zeros", "border", "reflection", "circular"):
            raise ValueError("Invalid boundary")
        if self.transposed and boundary != "zeros":
            raise ValueError("Transpose layers support boundary=zeros")
        self.boundary = boundary
        if data_format not in ("channels_last", "channels_first"):
            raise ValueError("Invalid data_format")
        self.data_format = data_format
        if not tf.as_dtype(self.compute_dtype).is_floating:
            raise ValueError("A floating-point dtype policy is required")
        if input_bound is not None and (isinstance(input_bound, bool) or not isinstance(input_bound, Real) or not math.isfinite(input_bound) or input_bound <= 0):
            raise ValueError("input_bound must be positive and finite")
        self.input_bound = None if input_bound is None else float(input_bound)
        for value, name, lower in ((order_regularization, "order_regularization", 0), (order_decay, "order_decay", 1)):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < lower:
                raise ValueError(f"{name} must be finite and >= {lower}")
        self.order_regularization, self.order_decay = float(order_regularization), float(order_decay)
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral)):
            raise ValueError("seed must be an integer or None")
        self.seed = None if seed is None else int(seed)
        self.output_chunk_size = _positive_integer(output_chunk_size, "output_chunk_size")
        self.recompute = bool(recompute)
        if conv_precision not in ("stable", "autocast") or not isinstance(jit_compile, bool):
            raise ValueError("Invalid conv_precision or jit_compile")
        self.conv_precision, self.jit_compile = conv_precision, jit_compile
        self.use_bias = bool(use_bias)
        self.activation = tf.keras.activations.get(activation)
        for name, getter, value in (
            ("kernel_initializer", tf.keras.initializers.get, kernel_initializer),
            ("bias_initializer", tf.keras.initializers.get, bias_initializer),
            ("kernel_regularizer", tf.keras.regularizers.get, kernel_regularizer),
            ("bias_regularizer", tf.keras.regularizers.get, bias_regularizer),
            ("kernel_constraint", tf.keras.constraints.get, kernel_constraint),
            ("bias_constraint", tf.keras.constraints.get, bias_constraint)):
            setattr(self, name, getter(value))
        self.input_spec = tf.keras.layers.InputSpec(ndim=d + 2)
        self._compiled_operation = tf.function(self._operation, jit_compile=True, autograph=False, reduce_retracing=True)

    @property
    def stable_dtype(self):
        return tf.float64 if self.compute_dtype == "float64" else tf.float32

    def build(self, input_shape):
        shape = tf.TensorShape(input_shape)
        channel_axis = -1 if self.data_format == "channels_last" else 1
        if shape.rank != self.spatial_rank + 2 or shape[channel_axis] is None:
            raise ValueError("Input rank must match the layer and channels must be known")
        self.input_channels = int(shape[channel_axis])
        if self.input_channels < 1 or self.input_channels % self.groups:
            raise ValueError("input channels must be positive and divisible by groups")
        self.channels_per_group = self.input_channels // self.groups
        self.filters_per_group = self.filters // self.groups
        self.input_spec = tf.keras.layers.InputSpec(ndim=self.spatial_rank + 2, axes={channel_axis: self.input_channels})
        regularizers = [self.kernel_regularizer] if self.kernel_regularizer is not None else []
        if self.order_regularization:
            regularizers.append(_NDOrderRegularizer(self.order_regularization, self.order_decay, self.include_constant))
        regularizer = _CombinedRegularizer(regularizers) if len(regularizers) > 1 else (regularizers[0] if regularizers else None)
        self.kernel = self.add_weight(name="kernel", shape=(*self.kernel_size, self.channels_per_group, self.orders, self.filters),
                                      initializer=_NDPolynomialInitializer(self.kernel_initializer, self.initialization, self.include_constant),
                                      regularizer=regularizer, constraint=self.kernel_constraint, autocast=False)
        rows = self.filters if self.shift_sharing == "full" else self.groups
        self.shift_parameter = self.add_weight(name="shift_parameter", shape=(rows, self.channels_per_group, self.spatial_rank),
                                               initializer=_NDShiftInitializer(self.shift_mode, self.shift_initializer, self.max_shift, self.seed),
                                               trainable=self.shift_mode == "learned", autocast=False)
        self.bias = self.add_weight(name="bias", shape=(self.filters,), initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer, constraint=self.bias_constraint, autocast=False) if self.use_bias else None
        dtype = self.stable_dtype
        values = tf.TensorSpec((None,) * (self.spatial_rank + 1) + (self.input_channels,), dtype)
        kernel_shape = (*self.kernel_size, self.channels_per_group, self.orders)
        shifts_shape = (None, self.channels_per_group, self.spatial_rank)
        self._recomputed_full_chunk = _compiled_recompute(
            self._full_chunk,
            (values, tf.TensorSpec((*kernel_shape, None), dtype),
             tf.TensorSpec(shifts_shape, dtype),
             tf.TensorSpec((None, self.spatial_rank), dtype),
             tf.TensorSpec((self.channels_per_group,), tf.int32)), 4,
        )
        self._full_chunk_vjp = _compiled_chunk_vjp(
            self._full_chunk,
            (values, tf.TensorSpec((*kernel_shape, None), dtype),
             tf.TensorSpec(shifts_shape, dtype),
             tf.TensorSpec((None, self.spatial_rank), dtype),
             tf.TensorSpec((self.channels_per_group,), tf.int32)),
        )
        self._recomputed_shared = _compiled_recompute(
            self._shared_recompute_operation,
            (values, tf.TensorSpec((*kernel_shape, self.filters), dtype),
             tf.TensorSpec(shifts_shape, dtype),
             tf.TensorSpec((self.spatial_rank,), tf.int32),
             tf.TensorSpec((self.spatial_rank,), tf.int32)), 3,
        )
        self._graph_operation = tf.function(
            self._operation,
            input_signature=(values,
                             tf.TensorSpec((*kernel_shape, self.filters), dtype),
                             tf.TensorSpec(self.shift_parameter.shape, dtype),
                             tf.TensorSpec((self.filters,) if self.bias is not None else (), dtype)),
            autograph=False, jit_compile=False,
            experimental_attributes={"_noinline": True},
        )
        super().build(input_shape)

    def effective_shifts(self):
        parameter = tf.cast(self.shift_parameter, self.stable_dtype)
        limits = tf.constant(self.max_shift, self.stable_dtype)
        if self.shift_mode == "random_fixed":
            return tf.round(tf.clip_by_value(parameter, -tf.floor(limits), tf.floor(limits)))
        return limits * tf.tanh(parameter)

    def _geometry(self, sizes):
        effective = tuple((k - 1) * dil + 1 for k, dil in zip(self.kernel_size, self.dilation_rate))
        counts, origins = [], []
        for axis, (stride, kernel) in enumerate(zip(self.strides, effective)):
            n = sizes[axis]
            op = None if self.output_padding is None else self.output_padding[axis]
            if self.transposed:
                if not isinstance(self.padding, str):
                    count = (n - 1) * stride - 2 * self.padding[axis] + kernel + (op or 0)
                    left = self.padding[axis]
                elif self.padding == "valid":
                    count = (n - 1) * stride + kernel + (max(kernel, stride) - kernel if op is None else op)
                    left = 0
                else:
                    count = n * stride if op is None else (n - 1) * stride + kernel % 2 + op
                    left = tf.maximum((n - 1) * stride + kernel - count, 0) // 2
                # Equivalent convolution on the zero-inserted input lattice.
                origin = left - (kernel - 1)
            else:
                if self.padding == "same":
                    count = (n + stride - 1) // stride
                    left = tf.maximum((count - 1) * stride + kernel - n, 0) // 2
                else:
                    left = 0 if self.padding == "valid" else self.padding[axis]
                    count = (n + 2 * left - kernel) // stride + 1
                origin = -left
            counts.append(count)
            origins.append(origin)
        return tf.stack(counts), tf.stack(origins), effective

    def _full_chunk(self, inputs, kernel, shifts, positions, channels):
        result = tf.zeros((tf.shape(inputs)[0], tf.shape(positions)[0], tf.shape(shifts)[0]), self.stable_dtype)
        for tap in itertools.product(*(range(k) for k in self.kernel_size)):
            offset = tf.constant(tuple(t * dil for t, dil in zip(tap, self.dilation_rate)), self.stable_dtype)
            coords = positions[:, None, None, :] + offset + shifts[None]
            sampled = sample_nd(inputs, coords, channels, self.boundary, self.spatial_rank)
            weights = kernel[tap]
            polynomial = tf.transpose(weights[:, -1, :])[None, None]
            for degree in range(self.orders - 2, -1, -1):
                polynomial = polynomial * sampled + tf.transpose(weights[:, degree, :])[None, None]
            if not self.include_constant:
                polynomial = polynomial * sampled
            result = result + tf.reduce_sum(polynomial, axis=-1)
        return result

    def _shared(self, inputs, kernel, shifts, counts, origins, effective):
        steps = (1,) * self.spatial_rank if self.transposed else self.strides
        extents = (counts - 1) * tf.constant(steps, tf.int32) + tf.constant(effective, tf.int32)
        positions = position_grid(extents, (1,) * self.spatial_rank, origins, self.stable_dtype)
        conv_dtype = tf.as_dtype(self.compute_dtype) if self.conv_precision == "autocast" else self.stable_dtype
        if self.spatial_rank == 2 and self.groups == self.input_channels:
            channels = tf.range(self.input_channels)
            coords = positions[:, None, None, :] + tf.reshape(shifts, (1, 1, self.input_channels, 2))
            sampled = sample_nd(inputs, coords, channels, self.boundary, 2)[:, :, 0]
            powers = [tf.ones_like(sampled)] if self.include_constant else []
            power = sampled
            for degree in range(self.q):
                if degree:
                    power = power * sampled
                powers.append(power)
            basis = tf.reshape(tf.stack(powers, -1), tf.concat(([tf.shape(inputs)[0]], extents, [self.input_channels * self.orders]), 0))
            multiplier = self.filters_per_group
            weights = tf.reshape(kernel[..., 0, :, :], (*self.kernel_size, self.orders, self.input_channels, multiplier))
            weights = tf.reshape(tf.transpose(weights, (0, 1, 3, 2, 4)), (*self.kernel_size, self.input_channels * self.orders, multiplier))
            result = tf.nn.depthwise_conv2d(tf.cast(basis, conv_dtype), tf.cast(weights, conv_dtype),
                                           strides=(1, 1, 1, 1), padding="VALID", dilations=self.dilation_rate, data_format="NHWC")
            result = tf.cast(result[:, ::steps[0], ::steps[1]], self.stable_dtype)
            result = tf.reshape(result, tf.concat(([tf.shape(inputs)[0]], counts, [self.input_channels, self.orders, multiplier]), 0))
            return tf.reshape(tf.reduce_sum(result, axis=-2), tf.concat(([tf.shape(inputs)[0]], counts, [self.filters]), 0))
        outputs = []
        for group in range(self.groups):
            channels = tf.range(group * self.channels_per_group, (group + 1) * self.channels_per_group)
            coords = positions[:, None, None, :] + shifts[group][None, None]
            sampled = sample_nd(inputs, coords, channels, self.boundary, self.spatial_rank)[:, :, 0]
            powers = [tf.ones_like(sampled)] if self.include_constant else []
            power = sampled
            for degree in range(self.q):
                if degree:
                    power = power * sampled
                powers.append(power)
            basis = tf.reshape(tf.stack(powers, -1), tf.concat(([tf.shape(inputs)[0]], extents, [self.channels_per_group * self.orders]), 0))
            start, end = group * self.filters_per_group, (group + 1) * self.filters_per_group
            weights = tf.reshape(kernel[..., start:end], (*self.kernel_size, self.channels_per_group * self.orders, self.filters_per_group))
            # The rank-agnostic API implements portable dilated gradients using
            # space-to-batch where direct CPU Conv2D backprop lacks support.
            result = tf.nn.convolution(tf.cast(basis, conv_dtype), tf.cast(weights, conv_dtype),
                                       padding="VALID", dilations=self.dilation_rate,
                                       data_format="NWC" if self.spatial_rank == 1 else "NHWC")
            if self.spatial_rank == 1:
                result = result[:, ::steps[0], :]
            else:
                result = result[:, ::steps[0], ::steps[1], :]
            outputs.append(tf.cast(result, self.stable_dtype))
        return tf.concat(outputs, axis=-1)

    def _shared_recompute_operation(self, inputs, kernel, shifts, counts, origins):
        effective = tuple((k - 1) * dil + 1 for k, dil in zip(self.kernel_size, self.dilation_rate))
        return self._shared(inputs, kernel, shifts, counts, origins, effective)

    def _operation(self, inputs, kernel, shifts, bias):
        values = tf.cast(inputs, self.stable_dtype)
        if self.input_bound is not None:
            values = tf.tanh(values / tf.constant(self.input_bound, self.stable_dtype))
        sizes = tf.shape(values)[1:1 + self.spatial_rank]
        counts, origins, effective = self._geometry(sizes)
        tf.debugging.assert_positive(sizes, message="Input spatial sizes must be positive")
        tf.debugging.assert_positive(counts, message="Output spatial sizes must be positive")
        kernel, shifts = tf.cast(kernel, self.stable_dtype), tf.cast(shifts, self.stable_dtype)
        if self.transposed:
            values = sparse_upsample(values, self.strides)
            kernel = tf.reverse(kernel, list(range(self.spatial_rank)))
            shifts = shifts * tf.constant(self.strides, self.stable_dtype)
        steps = (1,) * self.spatial_rank if self.transposed else self.strides
        if self.shift_sharing == "shared" or self.filters_per_group == 1 or not any(self.max_shift):
            if not any(self.max_shift):
                shifts = tf.zeros((self.groups, self.channels_per_group, self.spatial_rank), self.stable_dtype) + tf.reduce_sum(shifts) * 0
            result = (self._recomputed_shared(values, kernel, shifts, counts, origins)
                      if self.recompute else self._shared(values, kernel, shifts, counts, origins, effective))
        else:
            positions = position_grid(counts, steps, origins, self.stable_dtype)
            if self.recompute and not self.jit_compile:
                flat = _bounded_recomputed_chunks(self, values, kernel, shifts, positions)
            else:
                chunks = []
                for group in range(self.groups):
                    channels = tf.range(group * self.channels_per_group, (group + 1) * self.channels_per_group)
                    group_end = (group + 1) * self.filters_per_group
                    for start in range(group * self.filters_per_group, group_end, self.output_chunk_size):
                        end = min(start + self.output_chunk_size, group_end)
                        w, delta = kernel[..., start:end], shifts[start:end]
                        chunks.append(self._recomputed_full_chunk(values, w, delta, positions, channels)
                                      if self.recompute else self._full_chunk(values, w, delta, positions, channels))
                flat = tf.concat(chunks, -1)
            result = tf.reshape(flat, tf.concat(([tf.shape(inputs)[0]], counts, [self.filters]), 0))
        if self.bias is not None:
            result = result + tf.cast(bias, self.stable_dtype)
        return tf.cast(self.activation(result), self.compute_dtype)

    def call(self, inputs):
        original_shape = inputs.shape
        if self.data_format == "channels_first":
            inputs = tf.transpose(inputs, (0, *range(2, self.spatial_rank + 2), 1))
        # Read variables outside tf.function, including during symbolic build.
        args = (inputs, tf.convert_to_tensor(self.kernel), self.effective_shifts(),
                tf.convert_to_tensor(self.bias) if self.bias is not None else tf.constant(0., self.stable_dtype))
        sizes = tf.shape(inputs)[1:1 + self.spatial_rank]
        tf.debugging.assert_positive(sizes, message="Input spatial sizes must be positive")
        tf.debugging.assert_positive(self._geometry(sizes)[0], message="Output spatial sizes must be positive")
        if self.jit_compile:
            result = self._compiled_operation(*args)
        elif self.recompute:
            result = self._graph_operation(*[tf.cast(value, self.stable_dtype) for value in args])
        else:
            result = self._operation(*args)
        if self.data_format == "channels_first":
            result = tf.transpose(result, (0, self.spatial_rank + 1, *range(1, self.spatial_rank + 1)))
        result.set_shape(self.compute_output_shape(original_shape))
        return result

    def compute_output_shape(self, input_shape):
        spatial = input_shape[1:-1] if self.data_format == "channels_last" else input_shape[2:]
        output = []
        for axis, n in enumerate(spatial):
            if n is None:
                output.append(None)
                continue
            kernel = (self.kernel_size[axis] - 1) * self.dilation_rate[axis] + 1
            stride = self.strides[axis]
            op = None if self.output_padding is None else self.output_padding[axis]
            if self.transposed:
                if self.padding == "same":
                    count = n * stride if op is None else (n - 1) * stride + kernel % 2 + op
                elif self.padding == "valid":
                    count = (n - 1) * stride + kernel + (max(kernel, stride) - kernel if op is None else op)
                else:
                    count = (n - 1) * stride - 2 * self.padding[axis] + kernel + (op or 0)
            else:
                count = (n + stride - 1) // stride if self.padding == "same" else (n + 2 * (0 if self.padding == "valid" else self.padding[axis]) - kernel) // stride + 1
            if count < 1:
                raise ValueError("Output spatial sizes must be positive")
            output.append(count)
        return tf.TensorShape((input_shape[0], *output, self.filters) if self.data_format == "channels_last" else (input_shape[0], self.filters, *output))

    def get_config(self):
        names = ("filters", "kernel_size", "q", "strides", "padding", "dilation_rate", "groups",
                 "shift_mode", "shift_sharing", "max_shift", "shift_initializer", "use_bias", "initialization",
                 "boundary", "output_chunk_size", "recompute", "seed", "input_bound", "include_constant",
                 "output_padding", "order_regularization", "order_decay", "conv_precision", "jit_compile", "data_format")
        config = {**super().get_config(), **{name: getattr(self, name) for name in names}}
        for name, serializer in (("activation", tf.keras.activations.serialize),
                                 ("kernel_initializer", tf.keras.initializers.serialize), ("bias_initializer", tf.keras.initializers.serialize),
                                 ("kernel_regularizer", tf.keras.regularizers.serialize), ("bias_regularizer", tf.keras.regularizers.serialize),
                                 ("kernel_constraint", tf.keras.constraints.serialize), ("bias_constraint", tf.keras.constraints.serialize)):
            config[name] = serializer(getattr(self, name))
        return config


@tf.keras.utils.register_keras_serializable(package="SuperONN")
class SuperONN2D(_SuperONNND):
    """2D polynomial convolution with bilinear connection-specific shifts.

    Uses Keras channels-last tensors by default, or channels_first. ``padding``
    accepts same/valid or per-axis symmetric integers. Kernel layout is
    [height,width,input_channels/groups,orders,filters]. ``max_shift`` accepts
    a scalar or (height,width) limits in input-feature pixels.
    """
    spatial_rank = 2


@tf.keras.utils.register_keras_serializable(package="SuperONN")
class SuperONNTranspose1D(_SuperONNND):
    """1D polynomial transpose convolution on a sparse upsampled lattice.

    Offsets are measured in original input bins and scaled by strides on the
    lattice. Numeric padding matches PyTorch; same/valid strings follow Keras
    output sizing. output_padding=None enables Keras automatic sizing.
    """
    spatial_rank = 1
    transposed = True


@tf.keras.utils.register_keras_serializable(package="SuperONN")
class SuperONNTranspose2D(_SuperONNND):
    """2D polynomial transpose convolution with bilinear lattice sampling.

    Offsets use original input-feature pixels. Numeric padding matches PyTorch;
    Keras same/valid output sizing is supported. Constant per-tap bases are
    rejected on sparse lattices; use the ordinary output bias instead.
    """
    spatial_rank = 2
    transposed = True
