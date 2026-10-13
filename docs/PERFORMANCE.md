# Performance and troubleshooting

This guide covers execution controls, observed timings, and common setup issues. For constructor details, see the [API reference](API.md); for installation and examples, see the [README](../README.md).

## Framework support

| API | Locally verified versions | Coverage |
|---|---|---|
| PyTorch | 2.6.0+cu126 and 2.14.1+cu126 | 1D/2D forward and transpose, eager/compiled execution, gradients, AMP, and checkpoints |
| TensorFlow/Keras | TensorFlow 2.21.0; Keras 3.15.0 and 3.15.1 | 1D grouped/depthwise/separable layers, eager/graph execution, XLA, gradients, and serialization |
| Python | 3.13.14 | Local CPU and CUDA checks |

The GitHub Actions workflow is configured for Python 3.11/3.13 with PyTorch 2.6.0/2.14.1, plus Python 3.13 with TensorFlow 2.21.0/Keras 3.15.1. Remote CI results become available after the repository is published and the workflow runs.

The package supports the tested paths above. ONNX export, distributed execution, MPS/XPU, other GPUs, and combinations outside the tested matrix require their own checks.

## Choosing execution controls

- **Compilation:** PyTorch `Module.compile()` and Keras `Model.compile(jit_compile=True)` can fuse compatible operations. Compilation is disabled by default and has a first-call cost. Fixed input shapes generally make timing and compiler reuse easier to assess.
- **Operator-level XLA:** TensorFlow layers accept `jit_compile=True`. This is useful inside a custom training loop. Compiling the whole training function can fuse more operations; the two approaches have different overhead.
- **Chunk size:** `output_chunk_size` limits simultaneous connections in the full sharing layout. Smaller chunks may reduce activation memory while increasing overhead.
- **Recomputation:** `recompute=True` reduces saved activations by recomputing pure intermediates during backward. Measure the memory/time tradeoff on the actual model.
- **Precision:** `conv_precision="stable"` is the default. Sampling and polynomial evaluation remain FP32 under half/bfloat16 policies. `"autocast"` permits reduced precision only in eligible native convolution paths; it may not be faster. Float64 remains float64.

Polynomial degree, sharing layout, offset range, and grouping affect the model's function. Changing them for speed also changes the architecture. Compare execution controls using identical model settings and weights.

## Measured operator timings

Measurements were made on **13 October 2026**, using an **NVIDIA RTX 3080 Ti Laptop GPU** and Python 3.13.14. The synthetic workload had batch size 2, 8 input/output channels, length 1400, kernel size 7, polynomial degree 3, full learned offsets, input bounding, and recomputation enabled. The depthwise workload used 8 groups.

Three warmup passes preceded seven timed forward/backward passes. Timings below exclude compilation from steady-state execution. Reference and compiled paths used the same weights within each framework; outputs and input/coefficient/offset gradients were checked for numerical agreement.

| Framework/workload | Reference | Compiled | Observed speedup | First compiled call |
|---|---:|---:|---:|---:|
| PyTorch 2.14.1, dense | 46.57 ms eager | 3.02 ms Inductor | 15.4x | 41.0 s |
| TensorFlow 2.21 / Keras 3.15.1, dense | 57.46 ms graph | 1.68 ms XLA training function | 34.2x | 10.2 s |
| PyTorch 2.14.1, depthwise | 4.71 ms eager | 1.11 ms Inductor | 4.2x | 0.8 s |
| TensorFlow 2.21 / Keras 3.15.1, depthwise | 5.68 ms graph | 2.18 ms XLA training function | 2.6x | 1.4 s |

TensorFlow timings compare operator training functions, rather than complete application models. Operator-level XLA inside an ordinary graph measured approximately 2.49 ms dense and 2.32 ms depthwise. An eager outer gradient tape has a different overhead profile.

PyTorch's compiled incremental activation peak was approximately 10.4 MB dense and 0.72 MB depthwise, versus 40.4 MB and 1.97 MB eager. These are incremental allocator measurements, not total process or model memory.

These measurements describe one short workload. They are not universal speed guarantees, accuracy comparisons, or long-run thermal measurements. First calls may benefit from persistent compiler caches and are not guaranteed cold-cache measurements. Device, batch size, width, shape changes, determinism settings, and compiler versions all affect results.

To assess your model, time synchronized forward/backward steps after warmup, report first-call latency separately, use identical weights and settings, and compare outputs and gradients before interpreting speedups. Runnable training examples provide starting points for a representative workload.

## Common issues

### Input shape or channel mismatch

PyTorch expects `[batch, channels, length]` or `[batch, channels, height, width]`; its unbatched equivalents are also accepted. Keras expects `[batch, length, channels]`. Keras needs the channel count to be known when the layer is built. Both input and output channel counts must be divisible by `groups`.

### Large values or unstable higher-order terms

Check input and intermediate feature ranges before increasing `q`. Consider explicit feature scaling, `input_bound`, and order-aware coefficient regularization. Bounding computes `tanh(input / bound)`; it changes the mapping and must be recorded with the model settings. The coefficient penalty is disabled unless its strength is positive.

### Compilation fails or recompiles repeatedly

Run the same model eagerly to isolate API and data problems. Try fixed shapes first, then enable compilation. New shapes or unsupported surrounding model operations can cause new graphs or compiler errors. PyTorch's default compiler requires a working supported toolchain; TensorFlow XLA requires the relevant compiler libraries for the selected device.

CUDA XLA needs compatible CUDA libraries, `libdevice`, and `ptxas`. Install the appropriate framework GPU dependencies in your environment. The library does not modify global runtime settings or search paths. XLA may omit in-graph assertions, so validate input contracts before compiled execution.

### Checkpoint loading fails

Reconstruct the same architecture, polynomial degree, offset mode, sharing layout, boundary policy, and input bound. A PyTorch state dictionary validates semantic metadata. Execution controls such as chunk size, recomputation, precision policy, and compilation can be selected when rebuilding the layer.

For Keras, import `superonn.tensorflow` before loading a `.keras` file in a new process. Custom third-party activations, regularizers, and constraints need their own registrations. Checkpoints from unrelated implementations require explicit conversion.

### TensorFlow tests cannot run

Install the TensorFlow and test extras. The optional test file skips when TensorFlow is absent; its XLA cases still require a working CPU XLA runtime when it is installed. Limit TensorFlow thread counts on constrained systems if the runtime cannot create its worker threads.
