# SuperONN Layers

**Native polynomial convolution layers with fixed or learned receptive-field
offsets for PyTorch and TensorFlow/Keras.**

Build signal and image models, encoders, decoders, generators, regressors, and
discriminators using familiar layer interfaces. The library implements the
[published Self-ONN/Super Neuron mechanisms](#research-and-citation) with explicit
grouping, interpolation, boundary, precision, and checkpoint contracts. This checkout targets **0.5.2**
and is licensed under **MIT**.

| Operation | PyTorch | TensorFlow/Keras |
|---|---|---|
| Forward 1D | `SuperONN1d` | `SuperONN1D` |
| Forward 2D | `SuperONN2d` | `SuperONN2D` |
| Transpose 1D | `SuperONNTranspose1d` | `SuperONNTranspose1D` |
| Transpose 2D | `SuperONNTranspose2d` | `SuperONNTranspose2D` |

All four operators support grouped connections, fixed/learned shifts,
full/shared connectivity, configurable polynomial degree, and optional input
bounding. TensorFlow also provides 1D depthwise/separable compositions and
factories for switching between standard Keras convolutions and SuperONNs.

[Installation](#installation) · [Quick starts](#quick-starts) ·
[Hyperparameters](#hyperparameters) · [Interoperability](#framework-interoperability) ·
[API](docs/API.md) · [Performance](docs/PERFORMANCE.md) · [Citation](#research-and-citation)

## Installation

Requires Python **3.11+**. Clone/download this repository and run the appropriate
command in the directory containing `pyproject.toml`:

```bash
# Choose a native backend:
python -m pip install ".[torch]"
python -m pip install ".[tensorflow]"

# Both backends, including layer conversion:
python -m pip install ".[torch,tensorflow]"

# Editable installation for local development:
python -m pip install -e ".[torch,tensorflow]"
```

These commands install this local checkout; they do not assume a package-index
release. The base package contains framework-independent configuration and
loads backends on demand. The `torch` extra requires PyTorch `>=2.6,<3`; the
`tensorflow` extra installs TensorFlow `2.21.0` and Keras `>=3.15,<4`.
Importing the TensorFlow API does not import PyTorch, and importing the PyTorch
API does not import TensorFlow.

For an existing research environment, register the checkout without changing
its installed framework versions:

```bash
python -m pip install --no-deps -e .
```

Install a framework build suitable for your hardware before using this option.
On supported CUDA platforms, TensorFlow's CUDA dependency extra is available:

```bash
python -m pip install "tensorflow[and-cuda]==2.21.0"
```

Framework GPU support and compiler availability depend on the operating system.
See [TensorFlow installation](https://www.tensorflow.org/install/pip),
[PyTorch installation](https://pytorch.org/get-started/locally/), and the
[performance guide](docs/PERFORMANCE.md). When using both CUDA frameworks in
one process, follow the [tested startup setup](docs/INTEROPERABILITY.md#using-both-cuda-frameworks-in-one-process).

### Upgrading from 0.4.x

PyTorch is an explicit installation extra in 0.5.0. Choose `[torch]`,
`[tensorflow]`, or both for a fresh environment. Existing environments retain
their installed frameworks. Existing layer names, weight layouts, and the
original 1D factory aliases remain compatible.

This version fixes replicate-boundary offset gradients, composition shape
inference, and state-transfer validation. Record the library version when
resuming an experiment, since corrected edge gradients can change subsequent
training updates. See [CHANGELOG.md](CHANGELOG.md) for the full changes.

### Updating an existing checkout

0.5.2 preserves 0.5.x coefficient/offset layouts, mathematical defaults and
serialization names. Reconstruct compatible layers and load existing state;
no weight conversion is required. After installation or source updates, restart
Python/notebook workers and rebuild/reload models so old traced functions are
not reused. Execution flags are chosen when reconstructing a layer.

Development snapshots can share the `0.5.2` version string while containing
different execution fixes. A version comparison alone does not certify an
unreleased snapshot. Install the updated source/package, restart workers and
reconstruct models; record the eventual release commit for reproducibility.


## Quick starts

### PyTorch signal regressor

```python
import torch
from torch import nn
from superonn import SuperONN1d, SuperONNConfig

model = nn.Sequential(
    SuperONN1d(1, 16, 7, padding=3, **SuperONNConfig().layer_kwargs()),
    nn.GELU(),
    nn.AdaptiveAvgPool1d(1),
    nn.Flatten(),
    nn.Linear(16, 1),
)

x = torch.randn(4, 1, 256)  # batch, channels, length
prediction = model(x)
prediction.square().mean().backward()
```

PyTorch uses channels-first tensors. Constructors follow the leading arguments
of `nn.Conv1d`/`nn.Conv2d` or their transposed counterparts, including grouping
and transpose `output_padding`. The explicit backend namespace
`superonn.pytorch` exports the same classes.

To start from an existing convolution's function:

```python
conv = nn.Conv1d(1, 16, 7, padding=3)
replacement = SuperONN1d.from_conv(conv, q=3, input_bound=None)
```

PyTorch `from_conv()` copies linear coefficients/bias, zeros higher orders,
and starts shifts at zero. A nonzero input bound changes that initial function.

### TensorFlow signal regressor

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

TensorFlow layers are native Keras layers using TensorFlow operations. The
TensorFlow Keras backend is required. They support ordinary activations,
initializers, regularizers, constraints, optimizers, and `.keras` serialization.

### TensorFlow image encoder/decoder

```python
import tensorflow as tf
from superonn.tensorflow import SuperONN2D, SuperONNTranspose2D

model = tf.keras.Sequential([
    tf.keras.Input((32, 32, 3)),
    SuperONN2D(16, 4, strides=2, padding=1, input_bound=1, activation="gelu"),
    SuperONNTranspose2D(3, 4, strides=2, padding=1, input_bound=1),
])
assert model(tf.zeros((2, 32, 32, 3))).shape == (2, 32, 32, 3)
```

`SuperONNTranspose1D` supplies the corresponding signal decoder. Transposed
operators use a sparse upsampled lattice; shifts retain units in the original
input-feature grid. Their degree-one, zero-shift limit follows the matching
standard transposed convolution.

### Interchangeable Keras factories

```python
from superonn.tensorflow import (
    conv1d, conv2d, conv_transpose1d, conv_transpose2d,
    depthwise_conv1d, separable_conv1d,
)

encoder = conv2d(16, 4, operator="superonn", strides=2, padding=1,
                 superonn={"q": 3, "max_shift": (2., 1.), "seed": 37})
decoder = conv_transpose2d(3, 4, operator="superonn", strides=2, padding=1)
depthwise = depthwise_conv1d(3, operator="superonn", padding="same",
                           data_format="channels_first")
```

Factories default to `operator="cnn"`. Select `"superonn"` explicitly to use
operational layers. Supply explicit padding when switching operators to retain
the intended shape: standard Keras layers accept `"same"`/`"valid"`, while
SuperONN also accepts symmetric numeric padding. Keras transposed CNN layers
do not provide the additional grouped connectivity exposed by SuperONN.

Operational options can be passed directly or in a `superonn` mapping; direct
arguments override mapping entries. Seed, shift initializer, boundary policy,
and forward constant bases are supported alongside the shared presets.
The separable composition combines a depthwise SuperONN with a conventional
pointwise Conv1D. Both compositions support channels-last and channels-first.

## Shapes and numerical contract

| Backend | 1D batched input | 2D batched input |
|---|---|---|
| PyTorch | `[N,C,L]` | `[N,C,H,W]` |
| TensorFlow default | `[N,L,C]` | `[N,H,W,C]` |
| TensorFlow `channels_first` | `[N,C,L]` | `[N,C,H,W]` |

TensorFlow requires known input channels at build time; spatial dimensions can
be dynamic in ordinary graph execution. Groups partition original input
channels, with both input and output channel counts divisible by groups.
PyTorch also accepts its unbatched channel-first equivalents.

For each connected input/output pair and kernel tap:

```text
sample = interpolate(input, output_position * stride - padding
                            + kernel_position * dilation + offset)
output = sum_over_connections_and_taps(sum(p=1..q, weight[p] * sample**p)) + bias
```

Interpolation precedes polynomial evaluation. Positive offsets read larger
input coordinates. Optional bounding applies `tanh(input/input_bound)` to
source features before sampling. Forward boundaries are zero, replicate/border,
reflection, and circular; transposed operators use zero boundaries.

TensorFlow core forward layers default to `padding="same"`; PyTorch defaults
to numeric zero padding. TensorFlow transpose defaults to `padding="valid"`
and `output_padding=None`. Numeric TensorFlow transpose padding uses the
PyTorch-compatible output formula, with omitted output padding interpreted as
zero. Keras string padding uses standard Keras automatic sizing. The
[API reference](docs/API.md) gives exact per-axis formulas.

With `q=1`, zero shifts, no input bound, and no constant basis, matching
coefficients/bias yield the standard convolution or transpose function.
Forward `include_constant=True` adds a degree-zero term; transpose rejects it
on a sparse lattice and supports ordinary bias instead.

## Hyperparameters

The direct core classes default to `input_bound=None`. `SuperONNConfig()` and
Keras factories use the bounded starting preset below:

| Parameter | Preset default | Effect |
|---|---|---|
| `q` | `3` | Highest polynomial degree; larger values increase coefficients/cost |
| `shift_mode` | `"learned"` | Continuous trained offsets or `"random_fixed"` frozen integers |
| `shift_sharing` | `"full"` | Per-connection offsets or `"shared"` offsets tied across group outputs |
| `max_shift` | `2.0` | Bound in current input-feature positions; 2D accepts per-axis limits |
| `initialization` | `"linear"` | Initialize degree one and zero higher orders; `"all_orders"` initializes all |
| `input_bound` | `1.0` | Apply `tanh(input/bound)` before sampling |
| `output_chunk_size` | `4` | Simultaneous output connections in the full path |
| `recompute` | `True` | Recompute intermediates in backward to reduce saved activations |
| `order_regularization` | `0.0` | Optional order-weighted coefficient penalty strength |
| `order_decay` | `1.5` | Penalty multiplier between consecutive degrees |
| `conv_precision` | `"stable"` | Stable arithmetic; `"autocast"` permits reduced native-convolution precision |

Learned offsets start at zero and use `max_shift*tanh(raw)` without rounding.
Fixed offsets start randomly and are stored as integers within `floor(max_shift)`.
Both modes support both sharing layouts. There are no ambiguous V1/V2 aliases.

These settings are validation starting points, not universal optima. Tune
degree, receptive-field range, width, grouping, input range, and regularization
for the task. Full/shared connectivity changes the model's function; chunking,
recomputation, and compilation are execution controls. Bounding changes the
function and does not replace data preprocessing.

## Training, precision, and compilation

Use ordinary PyTorch or Keras training loops. For the optional PyTorch penalty:

```python
from superonn import superonn_regularization_loss
loss = task_loss + superonn_regularization_loss(model)
```

Keras includes registered penalties in `model.losses`, and its standard
training APIs include them. Custom loops can add
`superonn.tensorflow.model_regularization_loss(model)` once. The penalty
excludes offsets, bias, and any constant basis; additional user Keras
regularizers retain their ordinary behavior.

Half/bfloat16 policies retain float32 interpolation and polynomial arithmetic;
float64 stays float64. `conv_precision="autocast"` affects eligible native
convolution paths. Use PyTorch AMP or a Keras mixed-precision policy as usual.
The library does not change global precision or device settings.

Compilation is opt-in:

```python
# PyTorch model or layer:
model.compile(fullgraph=True, dynamic=False)

# TensorFlow: select static XLA chunks on each participating operator:
layer = SuperONN2D(8, 3, padding="same", jit_compile=True)

# A containing model built with those operators can also opt into XLA:
model.compile(optimizer="adam", loss="mse", jit_compile=True)
```

These are separate framework examples. Compilation has a first-call cost;
new shapes can trigger new compilations. Start with fixed shapes, verify outputs
and gradients, and measure the actual workload. See
[performance and troubleshooting](docs/PERFORMANCE.md) for setup and limitations.

TensorFlow 0.5.2 reuses dynamically shaped recomputation functions and bounds
non-XLA full-layout forward and backward chunk loops. The streaming backward
recomputes one chunk at a time and avoids retained per-chunk tape buffers.
Registered gradient closures use weak owner references so discarded layers
can be collected.
Bounded operator graphs do not guarantee low startup cost for every containing
training graph. Layer XLA selects static chunks. PyTorch retains its native
non-reentrant training checkpoint policy. For additive independent branches,
see the [gradient-aggregation example](examples/tensorflow_branches.py).
The library does not change global execution or optimizer settings.

For TensorFlow losses that differentiate an input gradient, such as R1 or
gradient penalties, use `recompute=False` in every participating operational
layer or factory. Recomputed paths do not provide a supported higher-order
gradient contract; TensorFlow's rematerialization helper can reject nested
derivatives. The native
path with recomputation disabled supports the tested input-gradient penalties;
this choice uses more saved activation memory.

### Distributed training

The cores use ordinary native framework variables and optimizers. Two-process
CPU PyTorch DDP (Gloo) and two-CPU TensorFlow MirroredStrategy updates have been
compared with equivalent single-process updates, including the order penalty,
for all four classes in full/learned and shared/fixed configurations. These
checks use PyTorch 2.14.1 and TensorFlow 2.21.0/Keras 3.15.1; they do not establish
multi-host, NCCL, or TPU behavior.

Create Keras models and optimizers inside `strategy.scope()`. In custom
distributed TensorFlow loops, reduce per-example task losses using the global
batch size and scale weight regularization once per replica; standard Keras
training APIs handle this scaling. See the
[TensorFlow custom distributed training guide](https://www.tensorflow.org/tutorials/distribute/custom_training).

## Checkpoints and serialization

```python
# Reconstruct a matching PyTorch architecture before loading:
torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt", map_location="cpu", weights_only=True))

# Keras:
model.save("model.keras")
restored = tf.keras.models.load_model("model.keras")
```

The snippets describe separate framework models. PyTorch state dictionaries
validate semantic configuration, including degree, connectivity, boundaries,
and input bounds. Keras classes are registered under `SuperONN`; import
`superonn.tensorflow` before loading in a fresh process. User-supplied custom
objects need their own serialization registrations. For resumable training,
save both model and optimizer state in PyTorch; a compiled Keras `.keras` model
includes its optimizer. Restore these together before the next update. AdamW
resume and Keras Adam resume with EMA and gradient accumulation have been
checked for both offset modes and sharing layouts in all four core classes.
Preserve architecture,
preprocessing, and software versions alongside research checkpoints.

## Deployment and functional execution

For TensorFlow inference, export a built Keras model with `model.export()` and
load the serving artifact independently of the Python layer classes. For the
TensorFlow signal regressor above:

```python
model.export("saved_model", format="tf_saved_model")
serving = tf.saved_model.load("saved_model")
prediction = serving.serve(tf.zeros((1, 256, 1)))
```

Use an explicit `input_signature` for the dimensions and dtype accepted by your
endpoint. Dynamic batch and spatial sizes were checked for all four TensorFlow
cores in both channel layouts. SavedModel serves inference; use `.keras` to
restore the model and its optimizer for training. See the
[Keras export API](https://keras.io/api/models/model_saving_apis/export/).

PyTorch inference graphs support `torch.export.export()` followed by
`torch.export.save()`/`load()`. Export an evaluation model with representative
inputs and declare supported dynamic dimensions; dynamic batch was checked for
all four cores and both sharing layouts on PyTorch 2.14.1. Exported programs specialize Python
configuration and enforce their input constraints. This does not establish ONNX
or mobile-runtime compatibility.

Keras `stateless_call()` and PyTorch `torch.func.functional_call()` accept supplied
weights without changing the original layer. PyTorch also supports construction
on the `meta` device, then `load_state_dict(..., assign=True)` from concrete
weights, or `to_empty(device=...)` followed by `reset_parameters()`. Create the
optimizer after materialization or assignment so it references the active
parameters.

## Framework interoperability

`superonn.interop` transfers compatible configurations and weights in either
direction. Install both backend extras, then use `to_tensorflow(layer)` or
`to_pytorch(built_keras_layer)`. The result is an independent native layer.

Copy helpers update an existing destination while retaining its device, dtype,
and trainability. They validate semantic settings, shapes, finite values, and
fixed offsets before mutation. Automatic conversion rejects partial parameter
freezing and unsupported inline activations/training settings; explicitly
construct the target and use a copy helper when selecting its training policy.

Tensor layouts and some `same` padding cases differ between frameworks.
Numeric symmetric padding is the clearest exact-parity path. Conversion does
not transfer optimizer state or a live cross-framework gradient graph.
Whole-model transfer requires rebuilding the destination architecture. See
[the complete interoperability guide](docs/INTEROPERABILITY.md).

## Examples and release checks

All shipped examples use synthetic tensors:

```bash
python examples/basic_models.py
python examples/pytorch_training.py
python examples/pytorch_training.py --compile
python examples/tensorflow_training.py
python examples/tensorflow_training.py --compile
python examples/tensorflow_branches.py
python examples/backend_interop.py
```

- [Basic models](examples/basic_models.py): PyTorch generator, regressor, and discriminator.
- [PyTorch training](examples/pytorch_training.py): AMP, regularization, optional compilation.
- [TensorFlow training](examples/tensorflow_training.py): native Keras training and optional XLA.
- [Independent branches](examples/tensorflow_branches.py): reusable branch graphs, one update and one regularizer.
- [Layer transfer](examples/backend_interop.py): native image encoder/decoders and bidirectional transfer.

The repository and source distribution retain a compact [regression suite](tests/).
It includes an independent scalar reference and separate optional backend suites;
the installed runtime wheel contains only the library. GitHub Actions builds and
checks distributions, checks installed APIs outside the checkout, and runs CPU
regressions in the existing backend matrix. Remote CI results become available
only after those jobs run.

The focused 0.5.2 checks cover all eight CPU cores with independent outputs and
input/coefficient/shift gradients, recomputation on/off, state round trips,
stateless/functional calls and mixed precision. TensorFlow CPU operator and
whole-function XLA checks pass with layer XLA selected explicitly. PyTorch CUDA
outputs/gradients/AMP pass on 2.6.0. Real PyTorch 2.14.1 CPU Inductor compilation
passes for all four cores with full/shared offsets and recomputation on/off,
including input/parameter gradients and three optimizer updates per configuration.
TensorFlow CUDA is unavailable in the unchanged local environment.
See the [verification scope and measurements](docs/PERFORMANCE.md#focused-052-verification)
for versions, tolerances and limits. Earlier lifecycle/export/distributed checks
remain historical evidence in the changelog.

Task accuracy, multi-host/GPU distributed execution, ONNX export, and additional accelerators
require their own validation. When reporting an issue, include a small
reproducer, versions, input shape/dtype, device, relevant options, and compiler
settings.

## Repository and development

```text
superonn/                 Lazy root API, PyTorch operators, shared configuration
superonn/pytorch/         Explicit PyTorch namespace
superonn/tensorflow/      Native Keras cores, compositions, and factories
superonn/interop.py       Validated configuration/weight transfer
examples/                 Runnable models and training loops
docs/                     API, performance, and interoperability guides
tests/                    Independent oracle, optional backend suites, bounded runner
.github/workflows/        Distribution, installed-API and CPU regression checks
pyproject.toml            Packaging and backend extras
LICENSE / NOTICE.md       MIT license and research credits
```

Run maintained development checks with the backend(s) you use:

```bash
python -m pip install -e ".[test,torch]"   # or .[test,tensorflow]
python -B -m tests.run_checks base
python -B -m tests.run_checks torch
python -B -m tests.run_checks tensorflow
```

The `test` extra adds NumPy for the oracle; it is not a base runtime dependency.
Optional compiler/device probes and resource limits are explained in the
[performance guide](docs/PERFORMANCE.md#reproduce-the-focused-checks).

Build distributions with isolated build dependencies:

```bash
python -m pip install build
python -m build
```

## Research and citation

SuperONN-Layers is an independent implementation based on research by
**Serkan Kiranyaz and coauthors**. Credit for the original ONN, Self-ONN, and
Super Neuron methods belongs to the authors below. These references use the
published journal versions and their DOIs.

1. **Super Neurons**.
   S. Kiranyaz, J. Malik, M. Yamac, M. Duman, I. Adalioglu, E. Guldogan,
   T. Ince, and M. Gabbouj.
   *IEEE Transactions on Emerging Topics in Computational Intelligence*,
   **8**(1), 206–228 (2024).
   DOI: [10.1109/TETCI.2023.3314658](https://doi.org/10.1109/TETCI.2023.3314658).

   Introduces nonlocalized receptive fields with random or learnable kernel
   shifts—the basis for the library's fixed and learned offset modes.

2. **Self-organized Operational Neural Networks with Generative Neurons**.
   S. Kiranyaz, J. Malik, H. Ben Abdallah, T. Ince, A. Iosifidis, and M. Gabbouj.
   *Neural Networks*, **140**, 294–308 (2021).
   DOI: [10.1016/j.neunet.2021.02.028](https://doi.org/10.1016/j.neunet.2021.02.028).

   Introduces generative neurons that learn connection-specific nodal functions,
   providing the polynomial Self-ONN foundation used here.

3. **Operational neural networks**.
   S. Kiranyaz, T. Ince, A. Iosifidis, and M. Gabbouj.
   *Neural Computing and Applications*, **32**(11), 6645–6668 (2020).
   DOI: [10.1007/s00521-020-04780-3](https://doi.org/10.1007/s00521-020-04780-3).

   Introduces the ONN framework and its generalization of conventional
   convolutional neurons.

When using this library in research, cite *Super Neurons* for receptive-field
offsets and the generative-neuron Self-ONN paper for polynomial nodal operations.
The original ONN paper provides the broader methodological background. Also
report the library version or commit, offset mode/sharing, shift bounds,
polynomial degree, input range, and regularization settings.

Decoder extensions, execution controls, API integration, and additional
implementation references are described in [NOTICE.md](NOTICE.md).
Implementation correctness alone does not establish scientific novelty or
task performance.

## License

The library uses the [MIT license](LICENSE). Dependencies retain their own
licenses. See [CHANGELOG.md](CHANGELOG.md) for release history.
