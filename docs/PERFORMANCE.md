# Performance and troubleshooting

This guide covers execution controls, observed timings, and common setup issues. For constructor details, see the [API reference](API.md); for installation and examples, see the [README](../README.md).

## Framework support

| API | Locally verified versions | Coverage |
|---|---|---|
| PyTorch | 2.6.0+cu126 and 2.14.1+cu126 | 1D/2D forward and transpose, eager/compiled execution, gradients, AMP, and checkpoints |
| TensorFlow/Keras | TensorFlow 2.21.0; Keras 3.15.0 and 3.15.1 | 1D/2D forward/transpose, grouped layers and 1D depthwise/separable compositions, eager/graph/XLA training, gradients, mixed precision, and serialization |
| Python | 3.13.14 | Local CPU and CUDA checks |

The GitHub Actions release workflow builds/checks distributions, verifies framework-free metadata, and checks installed backend APIs. Backend jobs use Python 3.11/3.13 with PyTorch 2.6.0/2.14.1, plus Linux/Windows/macOS CPU jobs with Python 3.13 and TensorFlow 2.21.0/Keras 3.15.1. Remote CI results become available after the repository is published and the workflow runs.

The package supports the tested paths above. ONNX export, multi-host/GPU distributed execution, MPS/XPU, other GPUs, and combinations outside the tested matrix require their own checks.

Additional lifecycle checks use PyTorch 2.14.1 and TensorFlow 2.21.0/Keras
3.15.1 on Linux CPU: optimizer resume, functional/stateless calls, independent
SavedModel serving, dynamic-batch PyTorch export, and synchronous two-CPU
training. PyTorch meta construction/materialization passes on 2.6.0/2.14.1 CPU
and 2.14.1 CUDA. Compiled TensorFlow operators also accept supplied stateless
weights after tracing. These checks are scoped workflows, not deployment or
distributed-runtime benchmarks.

## Choosing execution controls

- **Compilation:** PyTorch `Module.compile()` and Keras `Model.compile(jit_compile=True)` can fuse compatible operations. Compilation is disabled by default and has a first-call cost. Fixed input shapes generally make timing and compiler reuse easier to assess.
- **Operator-level XLA:** TensorFlow layers accept `jit_compile=True`. This is useful inside a custom training loop. Compiling the whole training function can fuse more operations; the two approaches have different overhead.
- **Chunk size:** `output_chunk_size` limits simultaneous connections in the full sharing layout. Smaller chunks may reduce activation memory while increasing overhead.
- **Recomputation:** `recompute=True` reduces saved activations by recomputing pure intermediates during backward. Measure the memory/time tradeoff on the actual model.
- **Precision:** `conv_precision="stable"` is the default. Sampling and polynomial evaluation remain FP32 under half/bfloat16 policies. `"autocast"` permits reduced precision only in eligible native convolution paths; it may not be faster. Float64 remains float64.

Polynomial degree, sharing layout, offset range, and grouping affect the model's function. Changing them for speed also changes the architecture. Compare execution controls using identical model settings and weights.

## Measuring your workload

Record the library/framework versions, hardware, batch size, input shape,
channel width, polynomial degree, groups, offset mode/sharing, and precision.
Use identical weights and model settings when comparing execution controls.

Warm up the runtime and synchronize the device around GPU measurements. Report
first-call compilation latency separately from steady-state forward/backward
time. Include the measurement count, median and spread, and activation-memory
measurement method. New shapes may trigger new compilation.

Shared and full offsets define different models, so their timing difference
is an architecture tradeoff. Chunking, recomputation, and compilation can be
compared while preserving the mathematical settings. Validate the actual
workload before relying on reduced precision or a compiler speedup.

The training examples provide starting models for such measurements. Historical
timings from older releases have been removed from this guide; they do not
establish the performance of the current 2D/transpose implementations.

## Common issues

### Input shape or channel mismatch

PyTorch expects `[batch, channels, length]` or `[batch, channels, height, width]`; its unbatched equivalents are also accepted. Keras cores and 1D depthwise/separable compositions default to `[batch, length, channels]` or `[batch, height, width, channels]`; `data_format="channels_first"` is also supported. Keras needs the channel count to be known when the layer is built. Both input and output channel counts must be divisible by `groups`.

### Large values or unstable higher-order terms

Check input and intermediate feature ranges before increasing `q`. Consider explicit feature scaling, `input_bound`, and order-aware coefficient regularization. Bounding computes `tanh(input / bound)`; it changes the mapping and must be recorded with the model settings. The coefficient penalty is disabled unless its strength is positive.

### Compilation fails or recompiles repeatedly

Run the same model eagerly to isolate API and data problems. Try fixed shapes first, then enable compilation. New shapes or unsupported surrounding model operations can cause new graphs or compiler errors. PyTorch's default compiler requires a working supported toolchain; TensorFlow XLA requires the relevant compiler libraries for the selected device.

CUDA XLA needs compatible CUDA libraries, `libdevice`, and `ptxas`. Install the appropriate framework GPU dependencies in your environment. The library does not modify global runtime settings or search paths. XLA may omit in-graph assertions, so validate input contracts before compiled execution.

### Checkpoint loading fails

Reconstruct the same architecture, polynomial degree, offset mode, sharing layout, boundary policy, and input bound. A PyTorch state dictionary validates semantic metadata. Execution controls such as chunk size, recomputation, precision policy, and compilation can be selected when rebuilding the layer.

For Keras, import `superonn.tensorflow` before loading a `.keras` file in a new process. Custom third-party activations, regularizers, and constraints need their own registrations. Checkpoints from unrelated implementations require explicit conversion.

### Missing backend or runtime resources

Install the selected backend extra (`[torch]` or `[tensorflow]`) in the same environment that runs your model. Install both for layer conversion. XLA needs a working compiler runtime. Limit TensorFlow thread counts on constrained systems if the runtime cannot create its worker threads.


## Nested input-gradient penalties

TensorFlow input-gradient penalties, such as R1/gradient penalties, require
`recompute=False` in every participating core or composition. The installed
TensorFlow rematerialization API rejects the tested nested derivatives. With
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
