# Framework interoperability

SuperONN provides native PyTorch and native TensorFlow implementations of the
same four operators. Choose the backend that matches your containing model.

| Operation | PyTorch | TensorFlow/Keras |
|---|---|---|
| Forward 1D | `SuperONN1d` | `SuperONN1D` |
| Forward 2D | `SuperONN2d` | `SuperONN2D` |
| Transpose 1D | `SuperONNTranspose1d` | `SuperONNTranspose1D` |
| Transpose 2D | `SuperONNTranspose2d` | `SuperONNTranspose2D` |

PyTorch imports work from `superonn` or `superonn.pytorch`. TensorFlow imports
come from `superonn.tensorflow`. Both run with their ordinary optimizers,
automatic differentiation, model composition, precision policies, and native
checkpoint formats. Importing a single backend does not initialize the other.

## Transfer a trained layer

Install both backend extras (`python -m pip install ".[torch,tensorflow]"`)
to use the optional conversion helpers:

```python
import numpy as np
import importlib.util
import torch
if importlib.util.find_spec("triton") is not None:
    import triton
import tensorflow as tf
from superonn import SuperONN2d
from superonn.interop import to_tensorflow, to_pytorch

original = SuperONN2d(3, 8, 3, padding=1, q=3, input_bound=1)
native_keras = to_tensorflow(original)

x = torch.rand(2, 3, 12, 10)
x_keras = tf.convert_to_tensor(x.permute(0, 2, 3, 1).numpy())
y_torch = original(x).detach().permute(0, 2, 3, 1).numpy()
y_keras = native_keras(x_keras).numpy()
np.testing.assert_allclose(y_torch, y_keras, atol=3e-6, rtol=3e-6)

# Train with Keras as usual, then export the updated layer to PyTorch.
restored = to_pytorch(native_keras, device="cpu")
```

`to_tensorflow(layer, data_format="channels_last", dtype=None, jit_compile=False)`
constructs and builds a Keras counterpart with the original input channel count.
The default preserves the source coefficient dtype. Pass `data_format="channels_first"`
to retain PyTorch input/output layouts. `dtype` can also select a Keras mixed
precision policy; that choice can change numerical precision.

`to_pytorch(layer, device=None, dtype=None)` requires a built Keras layer and uses
its variable dtype by default. For mixed policies, this preserves coefficient
precision; PyTorch AMP can be selected separately during training. The returned
module starts on CPU unless a device is supplied.

Use `copy_to_tensorflow(pytorch_layer, tensorflow_layer)` or
`copy_to_pytorch(tensorflow_layer, pytorch_layer)` to update an existing compatible
layer. Semantic settings, declared tensor shapes, finite cast values, and fixed
offset bounds are checked before any state mutation. These functions
retain the destination device, dtype, execution controls, and trainability.
Assignment casts to the destination variable dtype; use matching dtypes when
precision preservation matters.

## What is preserved

Conversions transfer polynomial coefficients, raw learned offsets or fixed
integer offsets, and bias. They preserve grouping, degree, constant bases,
per-axis shift bounds, input bounding, boundary extension, stride, dilation,
output geometry, and the order-aware coefficient penalty. Kernel axes and
shared-shift layouts are converted explicitly. Zero-inserted transpose offsets
retain their units in the original input-feature grid.

Conversion produces an independent layer. It does not transfer an optimizer,
random state, data preprocessing, or a live cross-framework gradient graph.
Whole-model conversion requires rebuilding the destination architecture and
converting its supported layers. Arbitrary surrounding model operations are
outside these helpers.

## Using both CUDA frameworks in one process

Framework CUDA and compiler libraries must be compatible independently of the
operator. In the local Linux combination of PyTorch 2.14.1/Triton 3.8.0 and
TensorFlow 2.21.0, loading Triton's native compiler after TensorFlow caused a
runtime crash. Loading the installed compiler first allowed the joint checks:

```python
import importlib.util
import torch
if importlib.util.find_spec("triton") is not None:
    import triton
import tensorflow as tf
```

This setup is only relevant when running both frameworks in the same process
with that compiler stack. Native TensorFlow SuperONN does not import PyTorch or
Triton. Do not install a CUDA compiler solely for CPU layer transfer. For GPU
models, also allocate framework GPU memory explicitly when both runtimes run
together; ordinary separate framework training avoids competing GPU allocators.

## Exact geometry and intentional restrictions

- Numeric symmetric `padding` gives the most direct exact mapping across
  frameworks. Transpose `output_padding` specifies additional right/bottom
  output bins. It does not add input samples.
- TensorFlow forward `same` accepts strides greater than one, with padding
  depending on input size. Automatic conversion rejects that setting because
  a PyTorch layer cannot express it with one static padding configuration.
  Choose numeric padding in a portable architecture.
- TensorFlow transpose `same` follows Keras sizing and can need asymmetric
  cropping. Conversion rejects asymmetric cases that PyTorch symmetric padding
  cannot represent. Representable `same` and `valid` cases are supported.
- An inline Keras activation is rejected during conversion to a bare PyTorch
  layer. Put activations in separate architecture layers for straightforward
  transfer. User-supplied Keras regularizers and constraints are backend-specific;
  only the common order-aware penalty transfers automatically. Automatic
  conversion rejects other training settings; explicitly construct the
  destination and use a copy helper after selecting its training policy.
- Whole-layer freezing is preserved. Partial parameter freezing in either
  framework requires constructing the destination explicitly and using a copy
  helper. Automatic conversion rejects mixed flags rather than enabling
  frozen parameters.
- Half/bfloat16 computations can differ in rounding or kernel accumulation.
  Float64 output/input/coefficient/offset gradient tests establish the common
  operator contract; they do not imply bitwise equality in every precision.

## Platform support

The operator code uses native tensor operations and no custom compiled extension.
CPU CI covers Linux, Windows, and macOS with TensorFlow, plus Linux PyTorch
version checks. Framework wheels, GPU support, and compiler availability depend
on the operating system and hardware. Local numerical checks are performed on
Linux under WSL; additional operating-system verification comes from CI.

## Execution settings after transfer

Transfer preserves supported mathematical configuration and copies current
state. It does not transfer compiled graphs or replace one backend's
checkpointing machinery with the other's. TensorFlow 0.5.2 functions read
explicit current weights, including after transfer or `stateless_call()`.
There is no cached trainable result to invalidate.

Choose TensorFlow `jit_compile=True` during construction/conversion when using
a containing XLA function. Reconstruct layers to change traced options.
PyTorch eval-mode autograd and TensorFlow requested recomputation have different
policies; both support frozen-model input gradients. TensorFlow nested
input-gradient losses require `recompute=False` in participating layers.
See [current compiler/device scope](PERFORMANCE.md); CPU conversion parity does
not validate joint GPU startup.
