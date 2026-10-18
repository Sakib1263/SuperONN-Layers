# Performance and troubleshooting

Execution controls preserve the operator's mathematical settings. Their benefit
depends on shape, device and containing model. See the [API](API.md) for contracts
and the [README](../README.md) for installation.

## Execution controls

| Control | Purpose | Cost or limit |
|---|---|---|
| `output_chunk_size` | Bound simultaneous full-layout output connections | Smaller chunks add sampling/dispatch work; chunking alone does not bound all tensors retained by autodiff |
| `recompute` | Recompute pure intermediates in backward instead of saving them | Additional compute; backend-specific execution policy below |
| Compilation | Optional XLA/Inductor optimization | Cold compilation, toolchain requirements and shape specialization |
| `conv_precision` | `stable` retains stable native-convolution accumulation; `autocast` permits eligible reduced-precision convolutions | Sampling and powers remain FP32 under half/bfloat16 policies; float64 stays float64 |

Degree, full/shared offsets, groups, offset bounds and input bounding change the
model. Hold them and the weights/inputs fixed when comparing execution controls.
No defaults were changed for the 0.5.2 execution update.

### TensorFlow/Keras

With `recompute=True`, explicit coefficient and effective-shift tensors enter
reusable functions; they are read anew on every call, including stateless calls.
Compatible batch/spatial sizes and smaller final chunks share signatures.
Non-XLA full-layout execution uses one sequential runtime chunk loop per group.
Shared/depthwise paths retain their native convolution implementations.

A custom gradient surrounds the complete full-layout loop, but its backward
recomputes and differentiates **one chunk at a time**. It accumulates a common
input-gradient buffer and assembles coefficient/offset gradients in their
original order. A serial forward loop alone was insufficient: TensorFlow's
automatic loop gradient retained per-chunk passthrough tensors in accumulator
lists. The maintained regression rejects those lists for all four cores.
This is chunk-wise recomputation, not simultaneous whole-layer activation replay.

The containing layer and localized recomputation functions use function-local
no-inline boundaries, including generated backward functions. The experimental
`_noinline` attribute is verified on the declared TensorFlow **2.21.0** runtime;
this is not a portability guarantee for other TensorFlow versions. There are
no process-global graph optimizer or eager-execution changes.

Recomputation remains active when requested, including inference and frozen
models: these calls can still need input gradients. Integer geometry is not
watched by the localized recomputation tape. Functions are owned by their layer;
registered gradient closures hold weak layer references, allowing discarded
full/shared layers to be collected. There is no library-global input/model cache
or trainable-result cache. Reconstruct a
layer when changing execution settings after tracing.

**XLA is separate and opt-in.** Set `jit_compile=True` on participating cores
or pass it through each composition/factory when compiling a containing training
function with XLA. This selects static chunks instead of the non-XLA runtime
TensorArray path, whose variable-width final chunks are not generally XLA
compatible. Merely enabling `Model.compile(jit_compile=True)` does not select
the layer's static chunk path. See the [training example](../examples/tensorflow_training.py).

Bounded operator graphs do not guarantee inexpensive native graph optimization
or backward preparation for every large, coupled training graph. Python tracing,
native graph instantiation, and backward preparation are different cold costs.
For additive losses on independent branch inputs, the
[branch example](../examples/tensorflow_branches.py) reuses a compiled branch
gradient function and sums its gradients before **one** optimizer update.
It adds regularization once. Choose branch weights to match the exact objective.
Activations in other layers, branches, coupled losses and optimizer state can
still dominate model memory. Per-sample accumulation is not generally equivalent
for losses that couple examples.
This recipe does not split losses with cross-branch dependencies, and it does
not silently detach those dependencies or alter an application's trainer.

### PyTorch

The existing native implementation retains bounded full-connection Horner
evaluation and non-reentrant checkpointing. Checkpointing runs when
`recompute=True`, the module is in training mode, and gradients are enabled.
Eval mode uses ordinary autograd and still supports input gradients, including
frozen parameters. This policy is intentionally different from TensorFlow.

The checkpointed operations do not draw random numbers; RNG preservation is
disabled locally for those pure operations. Autocast behavior is handled by the
non-reentrant checkpoint API and the operator's precision contexts. Connection
indices and grids remain call-local, with no new persistent buffers/caches or
device-move invalidation requirements. No PyTorch numerical rewrite was needed
for this update. First use can include lazy framework/checkpoint imports.

`torch.compile(..., backend="eager", fullgraph=True)` checks Dynamo capture;
it does not establish that Inductor compiled native kernels successfully.

## Focused 0.5.2 verification

Local checks on Linux/WSL use Python 3.13.14, NumPy 2.5.1, PyTorch
2.6.0+cu126 / 2.14.1+cu126, and TensorFlow 2.21.0 with Keras 3.15.0.
The independent scalar NumPy oracle samples before taking powers. Float64
forward tolerance is `rtol=2e-11, atol=2e-12`; gradients use central differences
with step `1e-6` away from sampling kinks. Full cases check every input,
coefficient and raw-offset coordinate; grouped/shared/depthwise cases use two
deterministic directions per tensor. Recomputation on/off is checked in addition
to the independent oracle.

All eight CPU cores cover learned shifts, partial chunks, grouping, depthwise/
shared connectivity, stride/dilation, numeric padding, transpose output padding,
weight mutation, supplied weights, serialization, mixed BF16 and frozen input
gradients. TensorFlow also covers fixed-offset save/load, nonzero boundary
extensions, factories, compositions, nested derivatives without recomputation,
dynamic shapes, function boundaries, absence of backward accumulator lists,
and garbage collection of full/shared layer owners. Saved serving is checked
after original model objects are released; use `.keras` for training/state
resume rather than assuming exported serving custom gradients are trainable.
Explicit conversion checks both
directions and gradients for all four operator pairs.

| Additional path | Current scoped result |
|---|---|
| TensorFlow operator XLA, CPU | All four cores: outputs and first-order gradients |
| TensorFlow whole-function XLA, CPU, layer `jit_compile=True` | All four cores: finite loss and input/parameter gradients |
| PyTorch Dynamo full-graph capture, CPU | All four cores on 2.6.0 and 2.14.1 |
| PyTorch Inductor, 2.14.1 CPU | 1D full-layout output/gradient parity passed; process reached 150 s while compiling 2D; transpose cases not reached |
| PyTorch 2.6.0 CUDA | All four cores: oracle outputs, directional gradients and FP16 AMP |
| TensorFlow CUDA / CUDA XLA | Unavailable in the unchanged environment: no GPU exposed to TensorFlow |
| Windows/macOS | CI gates configured; not executed locally for this update |

Older lifecycle/export/distributed evidence is recorded in the changelog.
It does not extend the new execution checks to unexecuted devices or compilers.

### Small resource probes

These are diagnostics, not throughput benchmarks or a before/after comparison
against a retained legacy implementation.

A TensorFlow 1D layer (`q=2`, kernel 2, chunk 2, full learned shifts,
`input_bound=.7`) was reused in three additive branches, input `[1,8,1]`.
Five warmed forward/backward calls followed each cold call:

| Output width | Forward Python trace | First forward execution | Backward Python trace | First backward execution | Warm backward median (min–max) |
|---|---|---|---|---|---|
| 3 | 0.103 s | 0.164 s | 0.359 s | 0.525 s | 1.58 ms (1.53–2.22) |
| 9 | 0.096 s | 0.124 s | 0.379 s | 0.538 s | 4.56 ms (3.84–5.05) |

Both widths had **357** layer-forward graph nodes and **1,780** containing
training graph nodes, each including its function definitions. Containing
forward/backward call graphs had 33/36 nodes. Layer, chunk and VJP functions each
retained one trace after changed batch/spatial sizes. The bounded CPU probe
peaked at 835.2 MiB sampled process-tree RSS.
RSS includes framework overhead and is not activation memory. These numbers do
not establish behavior for arbitrary monolithic model graphs.

On an RTX 3080 Ti Laptop GPU, PyTorch 2.6.0 full-layout toy cases used identical
weights, `q=2`, chunk 2, one input/three outputs, kernel 2, batch 1, and input
length 4 or image 3×3. Forward cases included a constant term, dilation 2 and
stride 2; transpose used stride 2 and output padding 1. Both used bound .7.

| Core | Warm backward step median, recompute off → on | Incremental peak allocated tensor memory, off → on |
|---|---|---|
| 1D | 7.38 → 11.98 ms | 35,328 → 24,576 bytes |
| 2D | 24.43 → 46.88 ms | 130,560 → 72,192 bytes |
| Transpose 1D | 6.57 → 11.67 ms | 36,864 → 25,600 bytes |
| Transpose 2D | 24.21 → 42.28 ms | 132,096 → 73,216 bytes |

GPU timings synchronize around each call; five samples report min/max in probe
output. Peak memory subtracts persistent allocations and is captured before
oracle/AMP validation. It counts PyTorch allocated tensors, not allocator
reservation, process memory or external GPU consumers. Small tensors are
dominated by dispatch overhead. Recomputation saved tensor memory at additional
compute cost; no general throughput improvement is claimed.

## Reproduce the focused checks

Install development dependencies in your chosen development environment:

```bash
python -m pip install -e ".[test,torch]"       # or .[test,tensorflow]
python -B -m tests.run_checks base
python -B -m tests.run_checks torch
python -B -m tests.run_checks tensorflow
python -B -m tests.run_checks interop         # requires both
```

The runner bounds child wall time (180 s by default) and samples process-tree
RSS on Linux (3,072 MiB by default). Summed RSS conservatively counts shared
pages in each descendant; GPU allocations are separate. RSS accounting is
unavailable on other operating systems. Use separate backend processes for
device/compiler checks. Set process-local environment variables, for example:

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 \
SUPERONN_PROBE_BACKEND=tensorflow SUPERONN_RUN_COMPILER=1 \
python -B -m tests.run_checks execution --seconds 150

SUPERONN_PROBE_BACKEND=torch SUPERONN_PROBE_DEVICE=cuda \
python -B -m tests.run_checks execution --test tests.test_execution.TorchExecutionTests.test_full_chunk_profile
```

Compiler probes are conditional and can exceed the budget. An unavailable
requested device is reported explicitly; it is not a CPU pass. GitHub Actions
retains distribution/installed-API checks and adds compact CPU regression gates
without adding matrix dimensions. Remote results are only known after jobs run.

## Nested derivatives and troubleshooting

TensorFlow losses that differentiate input gradients, including R1/gradient
penalties, require `recompute=False` in every participating core/composition.
Recomputed paths do not provide a supported higher-order derivative contract;
the TensorFlow rematerialization helper can reject nested derivatives. The
library does not silently disable recomputation. Native nested derivatives retain
more activations. This restriction applies separately from compilation.

Use positive spatial sizes, known channels at Keras build time and channel counts
divisible by groups. PyTorch expects channels first; Keras supports both layouts
through native channels-last calculation. Bound or scale large inputs before
increasing degree; input bounding changes the function.

For checkpoints, reconstruct identical mathematical settings. Execution controls
can be selected on reconstruction. Import `superonn.tensorflow` before loading
a `.keras` file; user-defined custom objects need their own registration.
After updating the installed package or checkout, restart existing Python/
notebook workers and rebuild/reload models to discard old traced functions.
An existing model instance does not automatically replace cached executable code.

Install backend/compiler dependencies explicitly for your platform. The library
does not change global eager settings, optimizer configuration, runtime search
paths, or silently switch execution semantics. Missing CUDA libraries/`libdevice`/
`ptxas` are environment issues; additional platforms, ONNX, multi-host/GPU
distributed training and other accelerators need their own verification.
