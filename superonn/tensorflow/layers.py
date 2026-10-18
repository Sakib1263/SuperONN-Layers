"""Native Keras 1D SuperONNs with explicit shift and connectivity contracts.

The two paper models are ``shift_mode='random_fixed'`` (integer, frozen shifts)
and ``shift_mode='learned'`` (continuous, BP-optimized shifts). Independently,
``shift_sharing='full'`` assigns a shift to each connected input/output pair;
``'shared'`` ties shifts across outputs in an ordinary convolution group.

For output bin t, kernel tap r, and connected input c, sample at
    t * strides - pad_left + r * dilation_rate + delta[output, c],
then raise the *interpolated* value to powers 1..q and sum learned coefficients.
Zero padding extends the source domain before sampling, avoiding the cropped
shifted-map edge loss in the reference archives. Positive delta reads higher
input indices (an impulse consequently moves toward lower output indices).

This is a new implementation of the published mechanism, not a claim of a new
neuron model or an exact reproduction of a published training experiment.
"""

from __future__ import annotations

import math
import weakref
from numbers import Integral
from typing import Any

import tensorflow as tf
from .functional import normalize_padding, output_shape_1d


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _spatial_integer(value, name):
    """Match Keras Conv1D's integer or one-element spatial tuple syntax."""
    if isinstance(value, (tuple, list)):
        if len(value) != 1:
            raise ValueError(f"{name} must contain one spatial entry")
        value = value[0]
    return _positive_integer(value, name)


def _weak_bound_operation(function):
    """Do not let TensorFlow's gradient registry retain a discarded layer.

    Graph-mode custom gradients are registered beyond a layer's lifetime.
    Their Python closures must not strongly capture the owning bound method.
    The normal layer/model owns its executable functions while in use.
    """
    reference = weakref.WeakMethod(function)

    def invoke(*args):
        operation = reference()
        if operation is None:
            raise RuntimeError("The owning SuperONN layer no longer exists")
        return operation(*args)

    return invoke


def _compiled_recompute(function, signature, differentiable_count):
    """Keep recomputation's custom-gradient walk inside a reusable FuncGraph.

    Inlining one recompute_grad per output chunk into a large caller graph
    makes TensorFlow repeatedly walk upstream forward/gradient subgraphs.
    Explicit tensor arguments and a dynamic signature bound that walk and
    reuse the same forward/backward function for batches and final chunks.
    Integer geometry is captured only from this function's own placeholders,
    not watched by the localized gradient tape or captured from a caller.
    """
    function = _weak_bound_operation(function)

    def recomputed(*args):
        def operation(*values):
            return function(*values, *args[differentiable_count:])
        return tf.recompute_grad(operation)(*args[:differentiable_count])

    # Preserve this boundary through Grappler as well as Python tracing.
    # Otherwise the function optimizer can inline every chunk and its backward
    # body again, consuming excessive CPU memory on a full-width model. TF
    # carries this attribute into generated forward/backward functions.
    return tf.function(recomputed, input_signature=signature, autograph=False,
                       jit_compile=False, experimental_attributes={"_noinline": True})


def _compiled_chunk_vjp(function, signature):
    """Recompute and differentiate only one chunk, with no enclosing loop tape."""
    function = _weak_bound_operation(function)

    def vjp(values, kernel, shifts, positions, channels, upstream):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch((values, kernel, shifts))
            result = function(values, kernel, shifts, positions, channels)
        gradients = tape.gradient(
            result, (values, kernel, shifts), output_gradients=upstream,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )
        return tuple(tf.convert_to_tensor(value) for value in gradients)

    return tf.function(
        vjp, input_signature=(*signature, tf.TensorSpec((None, None, None), signature[0].dtype)),
        autograph=False, jit_compile=False, experimental_attributes={"_noinline": True},
    )


def _bounded_recomputed_chunks(layer, inputs, kernel, shifts, positions):
    """Stream chunk forward/VJPs without saving a separate loop gradient tape.

    Autodifferentiating a While containing per-chunk custom gradients retains
    their passthrough inputs and allocates input-sized zero gradients per chunk.
    A sequential forward loop alone therefore does NOT bound backward memory.
    One custom VJP around the complete loop retains its common inputs once;
    the backward loop recomputes/differentiates one chunk at a time and sums
    its input gradient. Coefficient/offset gradients are concatenated in their
    original order. Chunk size, arithmetic and per-chunk recomputation remain
    unchanged. XLA retains its separate static-loop path.
    """
    # The registered custom-gradient closure can outlive its owning model.
    layer = weakref.proxy(layer)
    count = (layer.filters_per_group + layer.output_chunk_size - 1) // layer.output_chunk_size

    @tf.custom_gradient
    def streamed(values, coefficients, offsets):
        outputs = []
        for group in range(layer.groups):
            first = group * layer.filters_per_group
            last = first + layer.filters_per_group
            channels = tf.range(group * layer.channels_per_group, (group + 1) * layer.channels_per_group)
            chunks = tf.TensorArray(layer.stable_dtype, size=count, infer_shape=False,
                                    element_shape=(None, None, None))

            def body(index, array, first=first, last=last, channels=channels):
                start = first + index * layer.output_chunk_size
                end = tf.minimum(start + layer.output_chunk_size, last)
                result = layer._recomputed_full_chunk(
                    values, coefficients[..., start:end], offsets[start:end], positions, channels)
                return index + 1, array.write(index, tf.transpose(result, (2, 0, 1)))

            _, chunks = tf.while_loop(lambda index, _: index < count, body,
                                      (tf.constant(0), chunks), parallel_iterations=1)
            outputs.append(tf.transpose(chunks.concat(), (1, 2, 0)))

        def gradient(upstream):
            input_gradient = tf.zeros_like(values)
            kernel_gradients, shift_gradients = [], []
            rank = coefficients.shape.rank
            output_first = (rank - 1, *range(rank - 1))
            output_last = (*range(1, rank), 0)
            for group in range(layer.groups):
                first = group * layer.filters_per_group
                last = first + layer.filters_per_group
                channels = tf.range(group * layer.channels_per_group, (group + 1) * layer.channels_per_group)
                kernel_chunks = tf.TensorArray(layer.stable_dtype, size=count, infer_shape=False)
                shift_chunks = tf.TensorArray(layer.stable_dtype, size=count, infer_shape=False)

                def backward(index, accumulated, kernel_array, shift_array,
                             first=first, last=last, channels=channels):
                    start = first + index * layer.output_chunk_size
                    end = tf.minimum(start + layer.output_chunk_size, last)
                    dx, dw, ds = layer._full_chunk_vjp(
                        values, coefficients[..., start:end], offsets[start:end], positions,
                        channels, upstream[..., start:end],
                    )
                    return (index + 1, accumulated + dx,
                            kernel_array.write(index, tf.transpose(dw, output_first)),
                            shift_array.write(index, ds))

                _, input_gradient, kernel_chunks, shift_chunks = tf.while_loop(
                    lambda index, *_: index < count, backward,
                    (tf.constant(0), input_gradient, kernel_chunks, shift_chunks),
                    parallel_iterations=1,
                )
                kernel_gradients.append(tf.transpose(kernel_chunks.concat(), output_last))
                shift_gradients.append(shift_chunks.concat())
            return input_gradient, tf.concat(kernel_gradients, axis=-1), tf.concat(shift_gradients, axis=0)

        return tf.concat(outputs, axis=-1), gradient

    return streamed(inputs, kernel, shifts)


@tf.keras.utils.register_keras_serializable(package="SuperONN")
class _OrderAwareRegularizer(tf.keras.regularizers.Regularizer):
    """Published Frobenius sum over orders; strength is task dependent."""
    def __init__(self, strength=0.0, decay=1.5, include_constant=False):
        self.strength, self.decay = float(strength), float(decay)
        self.include_constant = bool(include_constant)

    def __call__(self, kernel):
        dtype = tf.float64 if tf.as_dtype(kernel.dtype) == tf.float64 else tf.float32
        values = tf.cast(kernel, dtype)
        values = values[:, :, int(self.include_constant):, :]
        degree = tf.range(tf.shape(values)[2])
        factors = tf.pow(tf.constant(self.decay, dtype), tf.cast(degree, dtype))
        return tf.constant(self.strength, dtype) * tf.reduce_sum(tf.square(values) * factors[None, None, :, None])

    def get_config(self):
        return {"strength": self.strength, "decay": self.decay, "include_constant": self.include_constant}


@tf.keras.utils.register_keras_serializable(package="SuperONN")
class _CombinedRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, regularizers):
        self.regularizers = [tf.keras.regularizers.get(value) for value in regularizers]

    def __call__(self, kernel):
        dtype = tf.float64 if tf.as_dtype(kernel.dtype) == tf.float64 else tf.float32
        return tf.add_n([tf.cast(value(kernel), dtype) for value in self.regularizers])

    def get_config(self):
        return {"regularizers": [tf.keras.regularizers.serialize(value) for value in self.regularizers]}


@tf.keras.utils.register_keras_serializable(package="SuperONN")
class _PolynomialInitializer(tf.keras.initializers.Initializer):
    """Initialize a linear convolution, with optional independent higher orders."""

    def __init__(self, initializer="glorot_uniform", initialization="linear", include_constant=False):
        self.initializer = tf.keras.initializers.get(initializer)
        self.initialization = initialization
        self.include_constant = bool(include_constant)

    def __call__(self, shape, dtype=None):
        k, channels, q, filters = shape
        if self.initialization == "all_orders":
            flat = self.initializer((k, channels * q, filters), dtype=dtype)
            return tf.reshape(flat, shape)
        linear = self.initializer((k, channels, filters), dtype=dtype)
        return tf.stack([linear if i == int(self.include_constant) else tf.zeros_like(linear) for i in range(q)], axis=2)

    def get_config(self):
        return {
            "initializer": tf.keras.initializers.serialize(self.initializer),
            "initialization": self.initialization,
            "include_constant": self.include_constant,
        }


@tf.keras.utils.register_keras_serializable(package="SuperONN")
class _ShiftInitializer(tf.keras.initializers.Initializer):
    def __init__(self, shift_mode, shift_initializer, max_shift, seed=None):
        self.shift_mode = shift_mode
        self.shift_initializer = shift_initializer
        self.max_shift = max_shift
        self.seed = seed

    def __call__(self, shape, dtype=None):
        dtype = tf.as_dtype(dtype or tf.float32)
        if self.shift_initializer == "zeros" or self.max_shift == 0:
            return tf.zeros(shape, dtype=dtype)
        if self.shift_mode == "random_fixed":
            # An integer-uniform draw is made once in build, and saved as state.
            limit = int(self.max_shift)
            draw = (tf.random.stateless_uniform(shape, [self.seed, 0], -limit, limit + 1, dtype=tf.int32)
                    if self.seed is not None else tf.random.uniform(shape, -limit, limit + 1, dtype=tf.int32))
            return tf.cast(draw, dtype)
        # Uniform initial effective shifts in +/- half the bound. Raw parameters
        # are unconstrained; the forward mapping max_shift*tanh(raw) is bounded.
        normalized = (tf.random.stateless_uniform(shape, [self.seed, 0], -0.5, 0.5, dtype=dtype)
                      if self.seed is not None else tf.random.uniform(shape, -0.5, 0.5, dtype=dtype))
        return tf.math.atanh(normalized)

    def get_config(self):
        return {
            "shift_mode": self.shift_mode,
            "shift_initializer": self.shift_initializer,
            "max_shift": self.max_shift,
            "seed": self.seed,
        }


def _sample_connections_1d(inputs, coordinates, channels, boundary):
    """Sample [N,L,C] at [T,O,I] coordinates for original channel indices [I].

    Returns [N,T,O,I]. Integer neighbors are extended independently for zero
    boundaries, including the fractional halo outside either endpoint.
    """
    length = tf.shape(inputs)[1]
    last = tf.cast(length - 1, inputs.dtype)
    if boundary == "border":
        coordinates = tf.clip_by_value(coordinates, 0, last)
    elif boundary == "reflection":
        period = tf.maximum(2 * last, tf.cast(1, inputs.dtype))
        folded = tf.math.floormod(coordinates, period)
        coordinates = tf.where(folded <= last, folded, period - folded)
        coordinates = tf.where(length > 1, coordinates, tf.zeros_like(coordinates))
    elif boundary == "circular":
        coordinates = tf.math.floormod(coordinates, tf.cast(length, inputs.dtype))

    lower = tf.floor(coordinates)
    fraction = coordinates - lower
    lower = tf.cast(lower, tf.int32)
    upper = lower + 1
    # [I,L,N] permits one gather per input channel without replicating batch
    # indices or an entire source signal for every output connection.
    values = tf.transpose(tf.gather(inputs, channels, axis=2), (2, 1, 0))

    def neighbor(index):
        clipped = tf.math.floormod(index, length) if boundary == "circular" else tf.clip_by_value(index, 0, length - 1)
        gathered = tf.gather(values, tf.transpose(clipped, (2, 0, 1)), axis=1, batch_dims=1)
        gathered = tf.transpose(gathered, (3, 1, 2, 0))
        if boundary == "zeros":
            valid = tf.cast((index >= 0) & (index < length), inputs.dtype)
            gathered = gathered * valid[tf.newaxis]
        return gathered

    lo, hi = neighbor(lower), neighbor(upper)
    return lo + fraction[tf.newaxis] * (hi - lo)


@tf.keras.utils.register_keras_serializable(package="SuperONN")
class SuperONN1D(tf.keras.layers.Layer):
    """Polynomial spectral convolution with fixed or learnable kernel locations.

    Default layout is [batch, bins, channels]; channels_first is also supported.
    ``groups`` partitions original
    input channels, with both channels and filters divisible by groups. Kernel
    state has shape [kernel_size, input_channels/groups, orders, filters]. An
    optional constant basis increases orders from q to q+1. Bias is one scalar
    per output channel. Numeric padding or Keras same/valid sizing is supported.

    ``random_fixed`` uses nontrainable integer shifts initialized uniformly once.
    ``learned`` uses continuous shifts ``max_shift*tanh(shift_parameter)`` with a
    zero initialization by default. Both modes can use full connection-specific
    shifts or the cheaper shared-input-channel restriction. Full is the default.

    Half/bfloat16 policies compute interpolation and polynomial arithmetic in
    float32, then cast the result to the layer compute dtype. Float64 is retained
    for gradient checks. The core imposes no input bound; use bounded features or
    SuperONNBlock1D when polynomial orders require a range policy.

    ``output_chunk_size`` limits simultaneous output connections in full mode.
    ``recompute=True`` also recomputes pure full-chunk intermediates in backward
    to reduce saved activations, at extra compute cost. Neither changes shifts
    or coefficients. Shared mode uses channel-major polynomial expansion and
    native valid convolutions on an explicitly sampled extended source grid.
    """

    spatial_rank = 1
    transposed = False

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        q: int = 3,
        strides: int = 1,
        padding: str = "same",
        dilation_rate: int = 1,
        groups: int = 1,
        shift_mode: str = "learned",
        shift_sharing: str = "full",
        max_shift: float = 2.0,
        shift_initializer: str | None = None,
        use_bias: bool = True,
        kernel_initializer: Any = "glorot_uniform",
        initialization: str = "linear",
        boundary: str = "zeros",
        output_chunk_size: int = 4,
        recompute: bool = True,
        seed: int | None = None,
        input_bound: float | None = None,
        order_regularization: float = 0.0,
        order_decay: float = 1.5,
        conv_precision: str = "stable",
        jit_compile: bool = False,
        activation=None,
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        include_constant=False,
        data_format="channels_last",
        **kwargs,
    ):
        super().__init__(autocast=False, **kwargs)
        self.filters = _positive_integer(filters, "filters")
        self.kernel_size = _spatial_integer(kernel_size, "kernel_size")
        self.q = _positive_integer(q, "q")
        self.include_constant = bool(include_constant)
        self.orders = self.q + int(self.include_constant)
        if data_format not in ("channels_last", "channels_first"):
            raise ValueError("Invalid data_format")
        self.data_format = data_format
        self.strides = _spatial_integer(strides, "strides")
        self.dilation_rate = _spatial_integer(dilation_rate, "dilation_rate")
        self.groups = _positive_integer(groups, "groups")
        self.output_chunk_size = _positive_integer(output_chunk_size, "output_chunk_size")
        padding = normalize_padding(padding, 1)
        if shift_mode not in {"random_fixed", "learned"}:
            raise ValueError("shift_mode must be 'random_fixed' or 'learned'")
        if shift_sharing not in {"shared", "full"}:
            raise ValueError("shift_sharing must be 'shared' or 'full'")
        if boundary not in {"zeros", "border", "reflection", "circular"}:
            raise ValueError("boundary must be zeros, border, reflection, or circular")
        if initialization not in {"linear", "all_orders"}:
            raise ValueError("initialization must be 'linear' or 'all_orders'")
        if isinstance(max_shift, bool) or not math.isfinite(float(max_shift)) or max_shift < 0:
            raise ValueError("max_shift must be finite and nonnegative")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral)):
            raise ValueError("seed must be an integer or None")
        shift_initializer = shift_initializer or ("random" if shift_mode == "random_fixed" else "zeros")
        if shift_initializer not in {"zeros", "random"}:
            raise ValueError("shift_initializer must be 'zeros' or 'random'")
        if self.filters % self.groups:
            raise ValueError("filters must be divisible by groups")
        if not tf.as_dtype(self.compute_dtype).is_floating:
            raise ValueError("SuperONN1D requires a floating-point dtype policy")
        self.padding = padding
        self.shift_mode = shift_mode
        self.shift_sharing = shift_sharing
        self.max_shift = float(max_shift)
        self.shift_initializer = shift_initializer
        self.use_bias = bool(use_bias)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.initialization = initialization
        self.boundary = boundary
        self.recompute = bool(recompute)
        if conv_precision not in ("stable", "autocast"):
            raise ValueError("conv_precision must be stable or autocast")
        self.conv_precision = conv_precision
        if not isinstance(jit_compile, bool):
            raise ValueError("jit_compile must be boolean")
        self.jit_compile = jit_compile
        self._compiled_operation = tf.function(self._operation, jit_compile=True,
                                                autograph=False, reduce_retracing=True)
        self.seed = int(seed) if seed is not None else None
        if input_bound is not None and (isinstance(input_bound, bool) or not math.isfinite(input_bound) or input_bound <= 0):
            raise ValueError("input_bound must be positive and finite")
        if isinstance(order_regularization, bool) or not math.isfinite(order_regularization) or order_regularization < 0:
            raise ValueError("order_regularization must be finite and nonnegative")
        if isinstance(order_decay, bool) or not math.isfinite(order_decay) or order_decay < 1:
            raise ValueError("order_decay must be finite and >= 1")
        self.input_bound = None if input_bound is None else float(input_bound)
        self.order_regularization, self.order_decay = float(order_regularization), float(order_decay)
        self.activation = tf.keras.activations.get(activation)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.input_spec = tf.keras.layers.InputSpec(ndim=3)

    def _coefficient_regularizer(self):
        items = [self.kernel_regularizer] if self.kernel_regularizer is not None else []
        if self.order_regularization:
            items.append(_OrderAwareRegularizer(self.order_regularization, self.order_decay, self.include_constant))
        return _CombinedRegularizer(items) if len(items) > 1 else (items[0] if items else None)

    @property
    def stable_dtype(self):
        return tf.float64 if self.compute_dtype == "float64" else tf.float32

    def build(self, input_shape):
        shape = tf.TensorShape(input_shape)
        channel_axis = -1 if self.data_format == "channels_last" else 1
        if shape.rank != 3 or shape[channel_axis] is None:
            raise ValueError("SuperONN1D requires rank 3 with known channels")
        self.input_channels = int(shape[channel_axis])
        if self.input_channels < 1 or self.input_channels % self.groups:
            raise ValueError("input channels must be positive and divisible by groups")
        self.channels_per_group = self.input_channels // self.groups
        self.filters_per_group = self.filters // self.groups
        self.input_spec = tf.keras.layers.InputSpec(ndim=3, axes={channel_axis: self.input_channels})
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.kernel_size, self.channels_per_group, self.orders, self.filters),
            initializer=_PolynomialInitializer(self.kernel_initializer, self.initialization, self.include_constant),
            regularizer=self._coefficient_regularizer(),
            constraint=self.kernel_constraint,
            dtype=self.variable_dtype,
            autocast=False,
        )
        shift_rows = self.filters if self.shift_sharing == "full" else self.groups
        self.shift_parameter = self.add_weight(
            name="shift_parameter",
            shape=(shift_rows, self.channels_per_group),
            initializer=_ShiftInitializer(self.shift_mode, self.shift_initializer, self.max_shift, self.seed),
            trainable=self.shift_mode == "learned",
            dtype=self.variable_dtype,
            autocast=False,
        )
        self.bias = (
            self.add_weight(name="bias", shape=(self.filters,), initializer=self.bias_initializer,
                            regularizer=self.bias_regularizer, constraint=self.bias_constraint,
                            dtype=self.variable_dtype, autocast=False)
            if self.use_bias else None
        )
        dtype = self.stable_dtype
        values = tf.TensorSpec((None, None, self.input_channels), dtype)
        kernel_shape = (self.kernel_size, self.channels_per_group, self.orders)
        self._recomputed_full_chunk = _compiled_recompute(
            self._full_chunk,
            (values, tf.TensorSpec((*kernel_shape, None), dtype),
             tf.TensorSpec((None, self.channels_per_group), dtype),
             tf.TensorSpec((None,), dtype),
             tf.TensorSpec((self.channels_per_group,), tf.int32)), 4,
        )
        self._full_chunk_vjp = _compiled_chunk_vjp(
            self._full_chunk,
            (values, tf.TensorSpec((*kernel_shape, None), dtype),
             tf.TensorSpec((None, self.channels_per_group), dtype),
             tf.TensorSpec((None,), dtype),
             tf.TensorSpec((self.channels_per_group,), tf.int32)),
        )
        self._recomputed_shared = _compiled_recompute(
            self._shared_recompute_operation,
            (values, tf.TensorSpec((*kernel_shape, self.filters), dtype),
             tf.TensorSpec((None, self.channels_per_group), dtype),
             tf.TensorSpec((), tf.int32), tf.TensorSpec((), tf.int32)), 3,
        )
        # Reuse the layer's chunk-call graph as well. Multiple model branches
        # can call the same layer many times in a single training
        # step; inlining those loops repeatedly needlessly duplicates graphs.
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
        """Current offsets in input-feature bins, with no dummy axis or rounding."""
        parameter = tf.cast(self.shift_parameter, self.stable_dtype)
        if self.shift_mode == "random_fixed":
            limit = math.floor(self.max_shift)
            return tf.round(tf.clip_by_value(parameter, -limit, limit))
        return tf.constant(self.max_shift, self.stable_dtype) * tf.tanh(parameter)

    def _output_grid(self, length):
        effective_kernel = (self.kernel_size - 1) * self.dilation_rate + 1
        if self.padding == "same":
            count = (length + self.strides - 1) // self.strides
            total_padding = tf.maximum((count - 1) * self.strides + effective_kernel - length, 0)
            left = total_padding // 2
        else:
            left = tf.constant(0 if self.padding == "valid" else self.padding, tf.int32)
            count = (length + 2 * left - effective_kernel) // self.strides + 1
        tf.debugging.assert_positive(count, message="valid kernel must fit the spectral input")
        return count, left, effective_kernel

    def _shared(self, inputs, kernel, shifts, count, left, effective_kernel):
        extent = (count - 1) * self.strides + effective_kernel
        position = tf.cast(tf.range(extent) - left, self.stable_dtype)
        conv_dtype = tf.as_dtype(self.compute_dtype) if self.conv_precision == "autocast" else self.stable_dtype
        if self.groups == self.input_channels:
            # Depthwise polynomial filtering can process every original channel
            # in one native kernel. Q separate depthwise responses are summed
            # within each original channel, rather than mixed across channels.
            channels = tf.range(self.input_channels)
            coords = position[:, None, None] + tf.reshape(shifts, (1, 1, self.input_channels))
            sampled = _sample_connections_1d(inputs, coords, channels, self.boundary)[:, :, 0, :]
            powers = [tf.ones_like(sampled)] if self.include_constant else []
            power = sampled
            for degree in range(self.q):
                if degree:
                    power = power * sampled
                powers.append(power)
            basis = tf.reshape(tf.stack(powers, axis=-1), (tf.shape(inputs)[0], 1, extent, self.input_channels * self.orders))
            multiplier = self.filters_per_group
            weights = tf.reshape(kernel[:, 0, :, :], (self.kernel_size, self.orders, self.input_channels, multiplier))
            weights = tf.reshape(tf.transpose(weights, (0, 2, 1, 3)), (1, self.kernel_size, self.input_channels * self.orders, multiplier))
            result = tf.nn.depthwise_conv2d(tf.cast(basis, conv_dtype), tf.cast(weights, conv_dtype),
                                          strides=(1, 1, 1, 1), padding="VALID", dilations=(1, self.dilation_rate), data_format="NHWC")
            result = tf.cast(result, self.stable_dtype)
            result = tf.squeeze(result, axis=1)[:, ::self.strides, :]
            result = tf.reshape(result, (tf.shape(inputs)[0], count, self.input_channels, self.orders, multiplier))
            return tf.reshape(tf.reduce_sum(result, axis=3), (tf.shape(inputs)[0], count, self.filters))
        outputs = []
        for group in range(self.groups):
            channels = tf.range(group * self.channels_per_group, (group + 1) * self.channels_per_group)
            coords = position[:, None, None] + shifts[group][None, None, :]
            sampled = _sample_connections_1d(inputs, coords, channels, self.boundary)[:, :, 0, :]
            powers = [tf.ones_like(sampled)] if self.include_constant else []
            power = sampled
            for degree in range(self.q):
                if degree:
                    power = power * sampled
                powers.append(power)
            # [N,E,I,Q] makes each original channel's powers contiguous, which
            # preserves group/depthwise semantics instead of grouping by power.
            basis = tf.reshape(tf.stack(powers, axis=-1), (tf.shape(inputs)[0], extent, self.channels_per_group * self.orders))
            start, end = group * self.filters_per_group, (group + 1) * self.filters_per_group
            weights = tf.reshape(kernel[:, :, :, start:end], (self.kernel_size, self.channels_per_group * self.orders, self.filters_per_group))
            # Compute unit stride then slice so simultaneous stride+dilation is
            # defined without relying on a backend-specific convolution limit.
            result = tf.nn.convolution(tf.cast(basis, conv_dtype), tf.cast(weights, conv_dtype),
                                      padding="VALID", dilations=(self.dilation_rate,), data_format="NWC")
            result = tf.cast(result, self.stable_dtype)
            outputs.append(result[:, ::self.strides, :])
        return tf.concat(outputs, axis=-1)

    def _full_chunk(self, inputs, kernel, shifts, positions, channels):
        result = tf.zeros((tf.shape(inputs)[0], tf.shape(positions)[0], tf.shape(shifts)[0]), dtype=self.stable_dtype)
        for tap in range(self.kernel_size):
            coords = positions[:, None, None] + tf.cast(tap * self.dilation_rate, self.stable_dtype) + shifts[None, :, :]
            sampled = _sample_connections_1d(inputs, coords, channels, self.boundary)
            polynomial = tf.transpose(kernel[tap, :, self.orders - 1, :])[None, None, :, :]
            for degree in range(self.orders - 2, -1, -1):
                coefficient = tf.transpose(kernel[tap, :, degree, :])[None, None, :, :]
                polynomial = polynomial * sampled + coefficient
            result = result + tf.reduce_sum(polynomial if self.include_constant else polynomial * sampled, axis=-1)
        return result

    def _shared_recompute_operation(self, inputs, kernel, shifts, count, left):
        effective_kernel = (self.kernel_size - 1) * self.dilation_rate + 1
        return self._shared(inputs, kernel, shifts, count, left, effective_kernel)

    def _operation(self, inputs, kernel, shifts, bias):
        values = tf.cast(inputs, self.stable_dtype)
        if self.input_bound is not None:
            values = tf.tanh(values / tf.constant(self.input_bound, self.stable_dtype))
        tf.debugging.assert_positive(tf.shape(values)[1], message="spectral length must be positive")
        kernel = tf.cast(kernel, self.stable_dtype)
        shifts = tf.cast(shifts, self.stable_dtype)
        count, left, effective_kernel = self._output_grid(tf.shape(values)[1])
        if self.shift_sharing == "shared" or self.filters_per_group == 1 or self.max_shift == 0:
            # With one output per group, full and shared have identical offset
            # shapes and no output-sharing restriction remains.
            if self.max_shift == 0:
                # Keep the old trainable-variable/checkpoint order and a zero
                # derivative to all offsets in the no-shift ablation.
                shifts = tf.zeros((self.groups, self.channels_per_group), dtype=self.stable_dtype) + tf.reduce_sum(shifts) * tf.constant(0., self.stable_dtype)
            if self.recompute:
                result = self._recomputed_shared(values, kernel, shifts, count, left)
            else:
                result = self._shared(values, kernel, shifts, count, left, effective_kernel)
        else:
            positions = tf.cast(tf.range(count) * self.strides - left, self.stable_dtype)
            if self.recompute and not self.jit_compile:
                result = _bounded_recomputed_chunks(self, values, kernel, shifts, positions)
            else:
                chunks = []
                for group in range(self.groups):
                    channels = tf.range(group * self.channels_per_group, (group + 1) * self.channels_per_group)
                    group_end = (group + 1) * self.filters_per_group
                    for start in range(group * self.filters_per_group, group_end, self.output_chunk_size):
                        end = min(start + self.output_chunk_size, group_end)
                        chunk_kernel, chunk_shifts = kernel[:, :, :, start:end], shifts[start:end]
                        if self.recompute:
                            chunks.append(self._recomputed_full_chunk(
                                values, chunk_kernel, chunk_shifts, positions, channels))
                        else:
                            chunks.append(self._full_chunk(values, chunk_kernel, chunk_shifts, positions, channels))
                result = tf.concat(chunks, axis=-1)
        if self.bias is not None:
            result = result + tf.cast(bias, self.stable_dtype)
        return tf.cast(self.activation(result), self.compute_dtype)

    def call(self, inputs):
        original_shape = inputs.shape
        if self.data_format == "channels_first":
            inputs = tf.transpose(inputs, (0, 2, 1))
        # Explicit tensor arguments prevent Keras' stateless shape-inference
        # scope from being captured as frozen weights/offsets by tf.function.
        # Real calls always read the current variables outside the XLA cluster.
        args = (inputs, tf.convert_to_tensor(self.kernel), self.effective_shifts(),
                tf.convert_to_tensor(self.bias) if self.bias is not None else tf.constant(0., self.stable_dtype))
        if self.jit_compile:
            # Validate length outside the optional XLA cluster as well. XLA
            # may discard Assert nodes inside a compiled cluster.
            tf.debugging.assert_positive(tf.shape(inputs)[1], message="spectral length must be positive")
            self._output_grid(tf.shape(inputs)[1])
            result = self._compiled_operation(*args)
        elif self.recompute:
            result = self._graph_operation(*[tf.cast(value, self.stable_dtype) for value in args])
        else:
            result = self._operation(*args)
        result = tf.transpose(result, (0, 2, 1)) if self.data_format == "channels_first" else result
        result.set_shape(self.compute_output_shape(original_shape))
        return result

    def compute_output_shape(self, input_shape):
        return output_shape_1d(input_shape, self.filters, self.kernel_size, self.strides,
                               self.padding, self.dilation_rate, self.data_format)

    def get_config(self):
        return {
            **super().get_config(),
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "q": self.q,
            "include_constant": self.include_constant,
            "data_format": self.data_format,
            "strides": self.strides,
            "padding": self.padding,
            "dilation_rate": self.dilation_rate,
            "groups": self.groups,
            "shift_mode": self.shift_mode,
            "shift_sharing": self.shift_sharing,
            "max_shift": self.max_shift,
            "shift_initializer": self.shift_initializer,
            "use_bias": self.use_bias,
            "kernel_initializer": tf.keras.initializers.serialize(self.kernel_initializer),
            "initialization": self.initialization,
            "boundary": self.boundary,
            "output_chunk_size": self.output_chunk_size,
            "recompute": self.recompute,
            "seed": self.seed,
            "input_bound": self.input_bound,
            "order_regularization": self.order_regularization,
            "order_decay": self.order_decay,
            "conv_precision": self.conv_precision,
            "jit_compile": self.jit_compile,
            "activation": tf.keras.activations.serialize(self.activation),
            "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": tf.keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": tf.keras.regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": tf.keras.constraints.serialize(self.kernel_constraint),
            "bias_constraint": tf.keras.constraints.serialize(self.bias_constraint),
        }


@tf.keras.utils.register_keras_serializable(package="SuperONN")
class SuperONNBlock1D(tf.keras.layers.Layer):
    """Length-preserving residual block with an explicit bounded-feature branch.

    The branch receives ``tanh(inputs/input_bound)``; the residual
    retains the original features. No channel normalization is applied to a raw
    one-channel spectrum. This wrapper is an engineering composition, distinct
    from the core paper neuron and not a claim of architectural novelty.
    """

    def __init__(self, filters, kernel_size, q=3, shift_mode="learned", shift_sharing="full",
                 max_shift=2.0, groups=1, dilation_rate=1, input_bound=1.0, dropout=0.0,
                 layer_scale_init=1e-3, output_chunk_size=4, recompute=True, seed=None,
                 order_regularization=0.0, order_decay=1.5, normalization="unit", conv_precision="stable",
                 jit_compile=False, **kwargs):
        super().__init__(autocast=False, **kwargs)
        if not math.isfinite(float(input_bound)) or input_bound <= 0:
            raise ValueError("input_bound must be finite and positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if not math.isfinite(float(layer_scale_init)):
            raise ValueError("layer_scale_init must be finite")
        self.settings = dict(filters=filters, kernel_size=kernel_size, q=q, shift_mode=shift_mode,
                             shift_sharing=shift_sharing, max_shift=max_shift, groups=groups,
                             dilation_rate=dilation_rate, output_chunk_size=output_chunk_size,
                             recompute=recompute, seed=seed)
        self.settings.update({"order_regularization": order_regularization, "order_decay": order_decay,
                              "conv_precision": conv_precision, "jit_compile": jit_compile})
        self.input_bound = float(input_bound)
        if normalization not in ("unit", "scaled"):
            raise ValueError("normalization must be unit or scaled")
        self.normalization = normalization
        self.dropout_rate = float(dropout)
        self.layer_scale_init = float(layer_scale_init)
        self.operator = SuperONN1D(**self.settings, padding="same", dtype=self.dtype_policy, name="superonn")
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate, dtype=self.dtype_policy)
        self.projection = None

    def build(self, input_shape):
        self.operator.build(input_shape)
        if int(input_shape[-1]) != self.operator.filters:
            self.projection = tf.keras.layers.Conv1D(self.operator.filters, 1, use_bias=False, dtype=self.dtype_policy, name="residual_projection")
            self.projection.build(input_shape)
        self.layer_scale = self.add_weight(name="layer_scale", shape=(self.operator.filters,),
                                           initializer=tf.keras.initializers.Constant(self.layer_scale_init),
                                           autocast=False)
        super().build(input_shape)

    def call(self, inputs, training=False):
        values = tf.cast(inputs, self.operator.stable_dtype)
        bound = tf.constant(self.input_bound, values.dtype)
        bounded = tf.tanh(values / bound)
        response = self.operator(bounded * bound if self.normalization == "scaled" else bounded)
        response = self.dropout(tf.nn.gelu(response), training=training)
        residual = inputs if self.projection is None else self.projection(inputs)
        result = tf.cast(residual, values.dtype) + tf.cast(self.layer_scale, values.dtype) * tf.cast(response, values.dtype)
        return tf.cast(result, self.compute_dtype)

    def get_config(self):
        return {**super().get_config(), **self.settings, "input_bound": self.input_bound,
                "dropout": self.dropout_rate, "layer_scale_init": self.layer_scale_init,
                "normalization": self.normalization}

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        # Preserve the scaled-tanh map of older serialized wrappers.
        config.setdefault("normalization", "scaled")
        return cls(**config)
