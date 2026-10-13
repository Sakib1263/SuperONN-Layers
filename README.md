# SuperONN

**Polynomial convolution layers with fixed or learned receptive fields, using familiar PyTorch and TensorFlow/Keras APIs.**

SuperONN combines learnable polynomial nodal operations with receptive-field offsets. Use it to build signal and image models, encoders, decoders, generators, regressors, or discriminators alongside ordinary neural-network layers.

The package provides PyTorch 1D/2D forward and transposed layers, plus native TensorFlow/Keras 1D layers with grouped, depthwise, and separable compositions. The two offset modes and both connectivity layouts are implemented. Release **0.3.1** is licensed under **MIT**.

[Installation](#installation) · [Quick start](#quick-start) · [Offset modes](#offset-modes-and-sharing) · [Hyperparameters](#hyperparameters) · [Examples](#examples) · [API reference](docs/API.md) · [Performance](docs/PERFORMANCE.md)

## What the layer does

A conventional convolution multiplies sampled features by weights. SuperONN evaluates a learned polynomial of each sampled feature:

```text
sample = interpolate(input, output_position * stride - padding
                            + kernel_position * dilation + offset)
output = sum_over_connections_and_taps(sum(q=1..Q, coefficient[q] * sample**q)) + bias
```

Interpolation happens before polynomial evaluation. Offsets can be frozen integers or learned continuous values; they are shared across spatial positions and examples. Higher polynomial orders and the offset range provide extra controls over expressiveness and cost.

With `q=1`, `max_shift=0`, and `input_bound=None`, a forward layer reduces to an ordinary convolution when its linear weights and bias match. `from_conv()` also supports a convolution-equivalent initialization with additional polynomial orders.

## Installation

Requires **Python 3.11 or newer**. Clone or download this repository, then run these commands from the folder containing `pyproject.toml`:

```bash
# PyTorch API
python -m pip install .

# PyTorch and optional native TensorFlow/Keras API
python -m pip install ".[tensorflow]"

# Editable installation for development
python -m pip install -e ".[test]"
```

These commands install from the local repository; they do not assume a published package index release. The base dependency is `torch>=2.6,<3`. The TensorFlow extra uses TensorFlow 2.21.0 and Keras 3.15 or newer within the 3.x series. Both backends are loaded lazily: importing `superonn.tensorflow` does not initialize PyTorch.

For GPU use, install the framework build appropriate to your platform first. TensorFlow GPU setups can use its CUDA dependency extra where supported:

```bash
python -m pip install "tensorflow[and-cuda]==2.21.0"
```

The locally verified runtimes are Python 3.13.14, PyTorch 2.6.0/2.14.1, and TensorFlow 2.21.0 with Keras 3.15.0/3.15.1. CPU CI is configured for Python 3.11/3.13 with PyTorch and Python 3.13 with TensorFlow. See [framework support](docs/PERFORMANCE.md#framework-support) for the tested scope.

## Quick start

### PyTorch

```python
import torch
from torch import nn
from superonn import SuperONN1d, SuperONNConfig

options = SuperONNConfig().layer_kwargs()
model = nn.Sequential(
    SuperONN1d(1, 16, 7, padding=3, **options),
    nn.GELU(),
    nn.AdaptiveAvgPool1d(1),
    nn.Flatten(),
    nn.Linear(16, 1),
)

x = torch.randn(4, 1, 256)  # batch, channels, length
prediction = model(x)       # (4, 1)
prediction.square().mean().backward()
```

Forward constructors follow the leading arguments of `nn.Conv1d` and `nn.Conv2d`. Transposed constructors follow `nn.ConvTranspose1d` and `nn.ConvTranspose2d`, including `output_padding`. Grouping partitions the original channels and keeps their polynomial orders together.

```python
from superonn import SuperONN2d, SuperONNTranspose1d

image_layer = SuperONN2d(3, 8, 3, padding=1, input_bound=1)
decoder = SuperONNTranspose1d(16, 1, 4, stride=2, padding=1, input_bound=1)

# Preserve an existing convolution's initial behavior.
conv = nn.Conv1d(1, 16, 7, padding=3)
replacement = SuperONN1d.from_conv(conv, q=3, input_bound=None)
```

`from_conv()` copies linear weights and bias, zeros higher-order coefficients, and initializes offsets to zero. A nonzero input bound changes the initial mapping. Use an ordinary activation after PyTorch layers when your architecture requires one.

### TensorFlow/Keras

```python
import tensorflow as tf
from superonn.tensorflow import SuperONN1D

model = tf.keras.Sequential([
    tf.keras.Input((256, 1)),  # length, channels
    SuperONN1D(16, 7, padding="same", input_bound=1, activation="gelu"),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")
loss = model.train_on_batch(tf.random.normal((4, 256, 1)), tf.zeros((4, 1)))
```

Keras uses batched, channels-last tensors and supports ordinary activation, initializer, regularizer, constraint, and serialization APIs. The implementation uses native TensorFlow operations and requires the TensorFlow backend.

Optional factories let an architecture switch between conventional and operational layers:

```python
from superonn.tensorflow import conv1d, depthwise_conv1d, separable_conv1d

dense = conv1d(16, 7, operator="superonn", padding="same")
depthwise = depthwise_conv1d(7, operator="superonn", padding="same")
separable = separable_conv1d(16, 7, operator="superonn", padding="same")
# operator="cnn" selects the corresponding standard Keras layer.
```

Factories apply the bounded preset (`input_bound=1`). The direct core layer defaults to `input_bound=None`. The separable variant combines a depthwise SuperONN with a conventional pointwise Conv1D.

## Offset modes and sharing

The offset mode and sharing layout are independent choices:

| Setting | Behavior | Typical tradeoff |
|---|---|---|
| `shift_mode="random_fixed"` | Integer offsets sampled at initialization and stored without gradients | A fixed nonlocalized receptive field |
| `shift_mode="learned"` | Continuous offsets optimized through interpolation, bounded by `max_shift * tanh(raw)` | Adaptive receptive fields; extra trainable parameters |
| `shift_sharing="full"` | An offset for each connected input/output pair | Greater flexibility and higher activation cost |
| `shift_sharing="shared"` | One offset per input channel, tied across outputs in its group | Lower cost and fewer independent offsets |

Learned offsets start at zero by default. Fixed offsets start randomly by default. These descriptive settings identify the mechanisms directly; there are no ambiguous V1/V2 aliases. Both modes work with either sharing layout.

## Hyperparameters

`SuperONNConfig()` supplies a reusable PyTorch preset. Keras factories use the same shared defaults and additionally support `jit_compile`.

| Parameter | Preset default | Purpose |
|---|---|---|
| `q` | `3` | Highest polynomial degree; increasing it adds coefficients and computation |
| `max_shift` | `2.0` | Maximum offset in current input-feature positions; PyTorch 2D accepts one value per axis |
| `shift_mode` | `"learned"` | Fixed integer or trainable continuous offsets |
| `shift_sharing` | `"full"` | Connection-specific or shared input-channel offsets |
| `initialization` | `"linear"` | Start from linear coefficients with higher orders zero; `"all_orders"` initializes every order |
| `input_bound` | `1.0` | Optional `tanh(input / bound)` before sampling and polynomial evaluation |
| `output_chunk_size` | `4` | Output connections processed together in the full layout |
| `recompute` | `True` | Recompute intermediate activations during backward to reduce saved memory |
| `order_regularization` | `0.0` | Strength of the optional order-weighted coefficient penalty |
| `order_decay` | `1.5` | Multiplier between consecutive polynomial orders in that penalty |
| `conv_precision` | `"stable"` | Keep arithmetic stable; `"autocast"` permits lower precision in eligible native convolution paths |

These are starting settings, not a universal optimum. The direct PyTorch and TensorFlow cores use `input_bound=None`; the preset makes bounding explicit. Bounding changes the function and does not replace dataset preprocessing. Offset units follow the current feature grid, including after downsampling.

Start with `q=3` and tune degree, offset range, width, kernel size, grouping, and regularization using validation data. Full and shared layouts have different expressiveness. Treat them as architecture choices; chunking and recomputation are execution controls. Detailed defaults, restrictions, shapes, and checkpoint contracts are in the [API reference](docs/API.md).

## Training and checkpoints

PyTorch layers participate in ordinary optimizers, device transfers, autograd, and state dictionaries. Add the optional coefficient penalty explicitly:

```python
from superonn import superonn_regularization_loss

loss = task_loss + superonn_regularization_loss(model)
```

Keras registers kernel penalties in `model.losses`; standard `fit()` and `train_on_batch()` include them. In a custom training loop, add `model_regularization_loss(model)` from `superonn.tensorflow` to the task loss once.

Save and restore PyTorch weights after reconstructing the same architecture and semantic options:

```python
torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt", map_location="cpu", weights_only=True))
```

For Keras, use `model.save("model.keras")` and `tf.keras.models.load_model("model.keras")`. Import `superonn.tensorflow` before loading in a fresh process to register the custom layers. Store architecture and preprocessing settings alongside any research checkpoint.

## Performance

The implementation uses Horner polynomial evaluation, native grouped/depthwise convolutions where equivalent, and optional activation recomputation. Compilation is an explicit opt-in:

```python
# PyTorch: compile a layer or a supported containing model.
model.compile(fullgraph=True, dynamic=False)

# Keras: compile a supported whole training function.
model.compile(optimizer="adam", loss="mse", jit_compile=True)

# TensorFlow: alternatively compile an individual operator.
layer = SuperONN1D(16, 7, input_bound=1, jit_compile=True)
```

The snippets show separate framework-specific alternatives. Compilation has a first-call cost, and new shapes may trigger recompilation. Eager execution is the default. Under half/bfloat16 policies, interpolation and polynomial evaluation remain FP32; float64 remains float64. Reduced native convolution precision is optional and may not improve a particular workload.

See [performance and troubleshooting](docs/PERFORMANCE.md) for measurements, memory controls, compiler requirements, and tested limitations.

## Examples

All examples use synthetic tensors and run after installation:

```bash
python examples/basic_models.py
python examples/pytorch_training.py
python examples/pytorch_training.py --compile

# Requires the TensorFlow extra:
python examples/tensorflow_training.py
python examples/tensorflow_training.py --compile
```

- [Basic models](examples/basic_models.py): a generator, regressor, and discriminator using both offset modes.
- [PyTorch training](examples/pytorch_training.py): a complete training loop with AMP, optional compilation, and regularization.
- [TensorFlow training](examples/tensorflow_training.py): native Keras composition, training, and optional XLA.

## Repository layout

```text
src/superonn/             PyTorch layers, sampling, presets, and optional TensorFlow API
examples/                Runnable model and training examples
docs/                    API reference and performance guidance
tests/                   Public API regression tests
.github/workflows/       CPU test automation
pyproject.toml           Package metadata and dependencies
LICENSE / NOTICE.md      MIT license and underlying research credits
```

## Development and support

```bash
python -m pip install -e ".[test]"
python -m pytest tests/test_public_api.py -q

# Optional TensorFlow tests, including CPU XLA and serialization:
python -m pip install -e ".[test,tensorflow]"
python -m pytest tests/test_tensorflow_api.py -q

# Build distributable wheel and source archive:
python -m pip install build
python -m build
```

When reporting a problem, include a minimal reproducer, package/framework versions, input shape and dtype, device, relevant hyperparameters, and whether compilation is enabled. Avoid attaching private data; a small synthetic tensor is usually enough to reproduce an operator issue.

This release covers the listed eager, compiled, gradient, precision, and serialization paths. ONNX export, distributed execution, additional accelerators, and task-specific model quality require separate validation.

## Research and license

Polynomial neurons and nonlocalized receptive fields build on published Self-ONN and Super Neuron research. See [NOTICE.md](NOTICE.md) for references and implementation scope. Cite the underlying papers relevant to your method and record the exact library version or commit used in experiments. This implementation does not by itself establish scientific novelty or task performance.

The library is available under the [MIT license](LICENSE). Framework dependencies retain their own licenses. See [CHANGELOG.md](CHANGELOG.md) for release history.
