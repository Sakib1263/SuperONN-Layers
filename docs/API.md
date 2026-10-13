# API reference

This page describes the public layer interfaces and numerical contracts. See the [README](../README.md) for installation and complete model examples.

## PyTorch layers

```python
from superonn import (
    SuperONN1d, SuperONN2d,
    SuperONNTranspose1d, SuperONNTranspose2d,
    SuperONNConfig, superonn_regularization_loss,
)
```

| Layer | Batched input | Batched output |
|---|---|---|
| `SuperONN1d` | `[N, Cin, L]` | `[N, Cout, Lout]` |
| `SuperONN2d` | `[N, Cin, H, W]` | `[N, Cout, Hout, Wout]` |
| `SuperONNTranspose1d` | `[N, Cin, L]` | `[N, Cout, Lout]` |
| `SuperONNTranspose2d` | `[N, Cin, H, W]` | `[N, Cout, Hout, Wout]` |

Unbatched channel-first tensors are accepted. Eager execution accepts noncontiguous tensors and empty batches. Spatial dimensions must be positive and inputs must use real floating-point dtypes.

### Constructor arguments

Forward layers use:

```python
SuperONN1d(
    in_channels, out_channels, kernel_size,
    stride=1, padding=0, dilation=1, groups=1, bias=True,
    padding_mode="zeros", device=None, dtype=None,
    *, q=3, shift_mode="learned", shift_sharing="full", max_shift=2.0,
    shift_init=None, output_chunk_size=4, recompute=True,
    initialization="linear", input_bound=None, include_constant=False,
    order_regularization=0.0, order_decay=1.5, conv_precision="stable",
)
```

`SuperONN2d` accepts the same options with scalar or two-element spatial arguments. Transposed layers use the corresponding ConvTranspose leading argument order:

```python
SuperONNTranspose1d(
    in_channels, out_channels, kernel_size,
    stride=1, padding=0, output_padding=0, groups=1, bias=True,
    dilation=1, padding_mode="zeros", device=None, dtype=None,
    **superonn_options,
)
```

Restrictions:

- Input and output channel counts must be divisible by `groups`.
- Forward `padding` accepts nonnegative spatial values, `"same"`, or `"valid"`. As in PyTorch Conv, `"same"` requires stride 1.
- Forward boundary modes are `"zeros"`, `"replicate"`, `"reflect"`, and `"circular"`. Transposed layers accept numerical padding and `padding_mode="zeros"`.
- `output_padding` is smaller than stride or dilation along each transposed axis.
- `q` and `output_chunk_size` are positive integers. Shift bounds are finite and nonnegative. In 2D, `max_shift=(height_bound, width_bound)` controls each axis separately.
- `input_bound` is `None` or a finite positive number. `order_regularization` is nonnegative; `order_decay` is at least 1.
- `include_constant=True` adds a degree-zero basis in forward layers. Transposed layers reject it because a constant basis is undefined on their sparse lattice; use bias instead.

### Initialization and offset inspection

`shift_init=None` means `"zeros"` for learned mode and `"random"` for fixed mode. Other choices are `"random_int"`, `"half"`, and `"half_int"`; half variants use half the bound. Fixed mode always samples integer offsets within the allowed interval.

`initialization="linear"` initializes the degree-one coefficients and zeros the other orders. `"all_orders"` initializes every order; `"tanh"` and `"selu"` select the corresponding Xavier gains in the PyTorch implementation. Input bounding is a separate choice.

```python
layer = SuperONN1d(2, 4, 3, padding=1, shift_sharing="full")
offsets = layer.shifts  # effective offsets; do not mutate this computed tensor
layer.set_shifts_(offsets.detach() * 0.5)
```

Coefficients have shape `[Cout, Cin/groups, Q, *kernel]`, where `Q=q+1` with a constant basis and `Q=q` otherwise. Full offsets have shape `[Cout, Cin/groups, spatial_rank]`; shared offsets have shape `[Cin, spatial_rank]`. Learned raw offsets are parameters; fixed offsets are persistent buffers. `set_shifts_()` validates shape, range, finiteness, and integer values in fixed mode.

`LayerClass.from_conv(conv, q=3, **options)` accepts the corresponding PyTorch convolution class, preserves its device/dtype/grouping and linear coefficients, and starts offsets at zero. Leave `input_bound=None` to preserve the original convolution's initial mapping.

## TensorFlow/Keras layers

```python
from superonn.tensorflow import (
    SuperONN1D, SuperONNBlock1D,
    DepthwiseSuperONN1D, SeparableSuperONN1D,
    conv1d, depthwise_conv1d, separable_conv1d,
    model_regularization_loss,
)
```

All Keras operators require `[N, L, Cin]` inputs with known positive channel counts when built. `SuperONN1D` produces `[N, Lout, filters]`.

```python
SuperONN1D(
    filters, kernel_size, q=3, strides=1, padding="same",
    dilation_rate=1, groups=1,
    shift_mode="learned", shift_sharing="full", max_shift=2.0,
    shift_initializer=None, use_bias=True, kernel_initializer="glorot_uniform",
    initialization="linear", boundary="zeros", output_chunk_size=4,
    recompute=True, seed=None, input_bound=None,
    order_regularization=0.0, order_decay=1.5,
    conv_precision="stable", jit_compile=False, activation=None,
    bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None,
    kernel_constraint=None, bias_constraint=None, **kwargs,
)
```

Keras-specific details:

- `kernel_size`, `strides`, and `dilation_rate` accept integers or one-element tuples.
- Padding is `"same"` or `"valid"`; boundary is `"zeros"`, `"border"`, or `"reflection"`.
- `initialization` accepts `"linear"` and `"all_orders"`. `shift_initializer` accepts `"zeros"` and `"random"`, with the same mode-dependent default as PyTorch.
- Fixed mode requires an integer `max_shift`. `seed` optionally seeds offset initialization; use ordinary Keras random-seed controls for the whole model.
- Coefficients use the Keras layout `[kernel_size, Cin/groups, q, filters]`. Raw learned offsets or fixed offsets are stored in `shift_parameter`; call `effective_shifts()` to inspect effective offsets.
- The API uses TensorFlow operations and TensorFlow compilation. Other Keras backends are outside its supported scope.

### Compositions and interchangeable factories

`DepthwiseSuperONN1D(kernel_size, depth_multiplier=1, ..., superonn=None)` applies an operational filter group to each original input channel. `SeparableSuperONN1D(filters, kernel_size, ..., superonn=None)` adds a conventional pointwise Conv1D after a depthwise SuperONN.

`SuperONNBlock1D(filters, kernel_size, ...)` is a length-preserving residual wrapper. It bounds the operational branch, optionally drops branch activations, and scales the branch by a trainable per-channel scale initialized to `1e-3`. A pointwise residual projection is used when channel counts differ. The default `normalization="unit"` supplies `tanh(x/input_bound)` to the branch; `"scaled"` multiplies that bounded value by the bound.

Factories default to `operator="cnn"`; explicitly select `"superonn"` for operational layers:

```python
layer = conv1d(
    16, 7, operator="superonn", padding="same", activation="gelu",
    superonn={"q": 3, "max_shift": 2, "shift_mode": "learned"},
)
```

Factory options can be supplied in the `superonn` mapping or as recognized SuperONN keyword arguments. Direct keyword arguments take precedence. The factories apply the shared bounded defaults in `SUPERONN_DEFAULTS`, including `input_bound=1`, and reject unknown mapping keys. Depthwise/separable constructors also use that preset. `jit_compile` defaults to `False`.

`spectral_conv1d`, `spectral_depthwise_conv1d`, and `spectral_separable_conv1d` remain compatible aliases for the three general-purpose factories.

## Numerical and boundary contract

For output position `t`, tap `r`, and connected input channel `c`, the forward operator samples at `t * stride - pad_left + r * dilation + offset[output, c]`. In 2D, the same rule applies per axis. Positive offsets read larger input indices. Offsets do not depend on the input example or output position.

Optional input bounding is applied to source features first. Boundary extension defines the source before sampling; interpolation then precedes polynomial evaluation. Thus interpolation of `x**q` is not used as a substitute for `(interpolate(x))**q`.

Transposed PyTorch operators use a sparse upsampled lattice with input-feature offset units retained through stride. They are decoder extensions, rather than an exact reproduction of a paper's transposed neuron. Their ordinary zero-offset, degree-one limit follows the matching ConvTranspose mapping.

Half/bfloat16 policies retain FP32 interpolation and polynomial arithmetic; float64 remains float64. `conv_precision="autocast"` affects only eligible native convolution paths. Neither backend applies compilation or changes global precision settings automatically.

## Regularization and serialization

The optional coefficient penalty is `strength * sum(order_decay**(order-1) * ||W_order||²)` for orders 1 through `q`. It excludes offsets, bias, and any degree-zero basis. Its strength is zero by default.

PyTorch `superonn_regularization_loss(model)` aggregates each registered layer once and must be added to the task loss. Keras includes registered penalties in `model.losses`; the optional `model_regularization_loss(model)` sums them for a custom training loop. It includes ordinary Keras weight regularizers as well, so add it once.

PyTorch state dictionaries store semantic configuration metadata and support `weights_only=True` loading. Recreate a compatible model before loading. Chunk size, recomputation, convolution precision, and compilation are execution choices selected on reconstruction. Older compatible library metadata is handled where supported; arbitrary external checkpoint formats require explicit conversion.

Keras layers implement `get_config()` and register in the `SuperONN` serialization namespace. Import the TensorFlow API before loading a saved `.keras` model in a fresh process. Custom objects supplied by the user need their own serialization support.
