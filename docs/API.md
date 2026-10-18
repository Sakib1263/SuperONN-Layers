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
    SuperONN1D, SuperONN2D, SuperONNTranspose1D, SuperONNTranspose2D,
    SuperONNBlock1D,
    DepthwiseSuperONN1D, SeparableSuperONN1D,
    conv1d, conv2d, conv_transpose1d, conv_transpose2d,
    depthwise_conv1d, separable_conv1d,
    model_regularization_loss,
)
```

Core Keras layers require batched inputs with known positive channel counts
when built. `data_format="channels_last"` is the default; `"channels_first"`
transposes into native channels-last computation internally for CPU portability.
1D depthwise/separable compositions also support both layouts; the residual
block retains its channels-last interface.

| Layer | Default input | Default output | Default padding |
|---|---|---|---|
| `SuperONN1D` | `[N,L,Cin]` | `[N,Lout,filters]` | `"same"` |
| `SuperONN2D` | `[N,H,W,Cin]` | `[N,Hout,Wout,filters]` | `"same"` |
| `SuperONNTranspose1D` | `[N,L,Cin]` | `[N,Lout,filters]` | `"valid"` |
| `SuperONNTranspose2D` | `[N,H,W,Cin]` | `[N,Hout,Wout,filters]` | `"valid"` |

```python
SuperONN1D(
    filters, kernel_size, q=3, strides=1, padding="same",
    dilation_rate=1, groups=1,
    shift_mode="learned", shift_sharing="full", max_shift=2.0,
    shift_initializer=None, use_bias=True, kernel_initializer="glorot_uniform",
    initialization="linear", boundary="zeros", output_chunk_size=4,
    recompute=True, seed=None, input_bound=None, include_constant=False,
    data_format="channels_last",
    order_regularization=0.0, order_decay=1.5,
    conv_precision="stable", jit_compile=False, activation=None,
    bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None,
    kernel_constraint=None, bias_constraint=None, **kwargs,
)
```

Keras-specific details:

- `kernel_size`, `strides`, and `dilation_rate` accept scalar integers or tuples matching spatial rank. Both stride and dilation can exceed one; the shared path uses a unit-stride native convolution then slices the output.
- Forward padding is `"same"`, `"valid"`, or nonnegative symmetric per-axis integers. Boundaries are `"zeros"`, `"border"`, `"reflection"`, or `"circular"`. Transpose requires `boundary="zeros"`.
- `initialization` accepts `"linear"` and `"all_orders"`. `shift_initializer` accepts `"zeros"` and `"random"`, with the same mode-dependent default as PyTorch.
- `max_shift` is a finite nonnegative scalar or a per-axis 2D tuple. Fixed offsets are sampled from integer bins within `floor(max_shift)`. `seed` optionally seeds offset initialization; use ordinary Keras random-seed controls for the whole model.
- Coefficients use `[*kernel, Cin/groups, orders, filters]`, where `orders=q+int(include_constant)`. Constant bases are allowed in forward layers. Raw learned offsets or fixed offsets are stored in `shift_parameter`; call `effective_shifts()` to inspect them. Full offsets use `[filters,Cin/groups,rank]` and shared offsets use `[groups,Cin/groups,rank]`. The existing `SuperONN1D` retains its compatible offset layout without the trailing size-one rank axis.
- The API uses TensorFlow operations and TensorFlow compilation. Other Keras backends are outside its supported scope.

### 2D and transposed constructors

`SuperONN2D(filters, kernel_size, ...)` accepts the same common arguments as
`SuperONN1D`, with two spatial axes. The transpose classes add
`output_padding=None`; their default `padding="valid"` follows Keras.
All four classes support ordinary `build`, `get_config`, `.keras` saving,
mixed precision, `tf.function`, and optional XLA.

Let `n` be an input size, `s` its stride, and `k=(kernel_size-1)*dilation+1`:

| Transpose padding | Output size per axis |
|---|---|
| Numeric `p` | `(n-1)*s - 2*p + k + op`, with omitted `op=0` |
| `"valid"`, `output_padding=None` | `(n-1)*s + max(k,s)` |
| `"valid"`, explicit `op` | `(n-1)*s + k + op` |
| `"same"`, `output_padding=None` | `n*s` |
| `"same"`, explicit `op` | `(n-1)*s + k%2 + op` |

Explicit `output_padding` must be nonnegative and smaller than stride for
Keras string padding. Numeric padding follows PyTorch and permits values
smaller than stride **or** dilation. Transpose constant bases are rejected;
use bias. Kernel layout remains `[*kernel,Cin/groups,q,filters]` in both
directions; it differs from a conventional Keras ConvTranspose kernel layout.

Use [conversion helpers](INTEROPERABILITY.md) for native cross-framework state
transfer, including layout handling and exact-padding restrictions.

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

`conv2d`, `conv_transpose1d`, and `conv_transpose2d` select the corresponding
standard Keras layer or native operational class. Defaults are those of the
selected class; supply explicit padding to preserve a desired architecture shape.
The conventional Keras transpose classes do not support SuperONN's grouped
transpose extension.

Factory options can be supplied in the `superonn` mapping or as recognized SuperONN keyword arguments. Direct keyword arguments take precedence. The factories apply the shared bounded defaults in `SUPERONN_DEFAULTS`, including `input_bound=1`, and reject unknown mapping keys. Depthwise/separable constructors also use that preset. Optional `seed`,
`shift_initializer`, `boundary`, and `include_constant` settings are supported
in mappings or factory keyword arguments. Two-dimensional factories validate
per-axis shift bounds. Direct arguments override mapping entries. `jit_compile` defaults to `False`.

`spectral_conv1d`, `spectral_depthwise_conv1d`, and `spectral_separable_conv1d` remain compatible aliases for the three general-purpose factories.

## Numerical and boundary contract

For output position `t`, tap `r`, and connected input channel `c`, the forward operator samples at `t * stride - pad_left + r * dilation + offset[output, c]`. In 2D, the same rule applies per axis. Positive offsets read larger input indices. Offsets do not depend on the input example or output position.

Optional input bounding is applied to source features first. Boundary extension defines the source before sampling; interpolation then precedes polynomial evaluation. Thus interpolation of `x**q` is not used as a substitute for `(interpolate(x))**q`.

Transposed operators in both backends use a sparse upsampled lattice with input-feature offset units retained through stride. They are decoder extensions, rather than an exact reproduction of a paper's transposed neuron. Their ordinary zero-offset, degree-one limit follows the matching ConvTranspose mapping.

Half/bfloat16 policies retain FP32 interpolation and polynomial arithmetic; float64 remains float64. `conv_precision="autocast"` affects only eligible native convolution paths. Neither backend applies compilation or changes global precision settings automatically.

## Regularization and serialization

The optional coefficient penalty is `strength * sum(order_decay**(order-1) * ||W_order||²)` for orders 1 through `q`. It excludes offsets, bias, and any degree-zero basis. Its strength is zero by default.

PyTorch `superonn_regularization_loss(model)` aggregates each registered layer once and must be added to the task loss. Keras includes registered penalties in `model.losses`; the optional `model_regularization_loss(model)` sums them for a custom training loop. It includes ordinary Keras weight regularizers as well, so add it once.

PyTorch state dictionaries store semantic configuration metadata and support `weights_only=True` loading. Recreate a compatible model before loading. Chunk size, recomputation, convolution precision, and compilation are execution choices selected on reconstruction. Older compatible library metadata is handled where supported; arbitrary external checkpoint formats require explicit conversion.

Keras layers implement `get_config()` and register in the `SuperONN` serialization namespace. Import the TensorFlow API before loading a saved `.keras` model in a fresh process. Custom objects supplied by the user need their own serialization support.


## Training lifecycle and deployment

All four native cores support optimizer checkpoint resume and supplied-weight
execution: Keras `stateless_call()` or PyTorch `torch.func.functional_call()`.
The latter preserves the ordinary module buffers unless replacements are
provided. Offset buffers/non-trainable weights remain part of the model state.

PyTorch constructors accept `device="meta"` for allocation-free structure
creation. Materialize with a concrete state dictionary and `assign=True`, or
call `to_empty(device=...)` followed by `reset_parameters()`. Construct the
optimizer afterward. `reset_parameters()` also restores the shift-limit buffer.

TensorFlow `Model.export(format="tf_saved_model")` supports serving graphs
with explicit batch/spatial input signatures. A saved serving endpoint can be
loaded without importing the layer package. Use `.keras` for optimizer resume.
PyTorch `torch.export` save/load supports evaluated cores with a declared
dynamic batch dimension on PyTorch 2.14.1. Spatial/compiler/runtime constraints depend on the
exported graph; additional deployment formats need separate validation.

Synchronous two-CPU DDP and MirroredStrategy updates, including coefficient
regularization, match equivalent single-process updates for the four cores.
Custom distributed training loops must use the framework's loss reduction and
regularizer scaling rules. Multi-host/GPU distributed paths remain outside
this checked scope.

## Nested input-gradient penalties

TensorFlow input-gradient penalties, such as R1/gradient penalties, require
`recompute=False` in every participating core or composition. Recomputed paths
do not provide a supported higher-order derivative contract; the TensorFlow
rematerialization helper can reject nested derivatives. With
recomputation disabled, output/input/coefficient/offset derivatives of these
penalties are checked against PyTorch for all four classes, both sharing
layouts, and grouped/depthwise connectivity. Saved activation memory increases.

The implementation supplies explicit native convolution data formats so 2D
shared/depthwise second derivatives avoid a missing-attribute error. This is
independent of the public channels-first/channels-last tensor layout.

The TensorFlow namespace checks the selected Keras backend and requires
`KERAS_BACKEND=tensorflow` before Keras is imported. Other Keras backends use
different tensor/variable systems; use the native PyTorch namespace for
PyTorch models.

## Execution and trace lifetime (0.5.2)

TensorFlow recomputation uses explicit current tensor weights and layer-owned
functions with dynamic batch/spatial/final-chunk signatures. Non-XLA full-layout
chunks execute in sequential runtime loops, retaining per-chunk activation
control. Their custom backward streams individual chunk VJPs, accumulates one
input-gradient buffer and avoids automatic per-chunk loop tape accumulators.
Registered gradient closures use weak owner references, allowing discarded
full/shared layers to be collected. Frozen weights and `training=False` still
permit input gradients.
Function-local experimental no-inline boundaries are verified on TensorFlow
2.21.0. See [execution details](PERFORMANCE.md).

For whole-function/model XLA, construct participating cores/compositions with
`jit_compile=True` to select static chunk loops. The outer compiler flag alone
does not select that path. PyTorch retains non-reentrant checkpointing in
training mode with gradients enabled, and ordinary autograd in eval mode.

Cached executable functions are not serialized state. Mathematical settings,
weight axes, public names and 0.5.x checkpoint compatibility are unchanged.
Reconstruct layers to change traced configuration and restart existing workers
after package/source updates. Stateless/functional calls use supplied weights;
they do not mutate persistent variables or transfer optimizer state.
