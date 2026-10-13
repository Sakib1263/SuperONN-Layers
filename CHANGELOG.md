# Changelog

## 0.5.1 — 2026-10-13

- Enabled PyTorch meta-device construction and deferred materialization without
  value-dependent initialization checks; reset restores the shift-limit buffer.
- Checked optimizer resume in all four cores, including Keras EMA/gradient
  accumulation state and fixed-offset non-trainable weights.
- Checked supplied-weight functional/stateless calls, SavedModel serving with
  dynamic shapes/channel layouts, and PyTorch ExportedProgram save/load.
- Compared two-process PyTorch CPU DDP and two-CPU TensorFlow MirroredStrategy
  updates with equivalent single-process updates, including coefficient penalties.
- Checked supplied stateless weights after operator-level TensorFlow XLA tracing.
- Removed the standalone regression scripts and their packaging/dependency
  references; usage examples remain available.
- Removed older-release benchmark tables and converted the transfer example
  into a native training/usage walkthrough.
- Replaced regression CI with distribution and installed-backend release checks.
- Added repository metadata links, consistent text-file rules, and updated
  installation, training-resume, deployment, and troubleshooting guidance.

## 0.5.0 — 2026-10-13

- Fixed PyTorch replicate-boundary offset gradients at exact edge samples,
  including one-pixel inputs, without changing forward values.
- Made replicate endpoint subgradients explicit across supported PyTorch versions.
- Normalized NumPy-produced tuning scalars for safe checkpoint/config serialization.
- Fixed numeric-padding shape inference for Keras depthwise/separable layers
  and added CPU-portable channels-first composition support.
- Aligned factory validation with fractional fixed-shift bounds and exposed
  seeded initialization, boundary, and forward constant-basis options.
- Added `conv2d`, `conv_transpose1d`, and `conv_transpose2d` factories for the
  new TensorFlow cores.
- Made transfer failures atomic for malformed shapes and cast-invalid offsets;
  recognized equivalent odd-kernel same/numeric padding and rejected silent
  changes to partially frozen training policies.
- Used explicit convolution layouts for shared/depthwise second derivatives
  and verified nested input-gradient penalties with `recompute=False`.
- Added an early error for an incompatible Keras backend.
- Validated short 1D valid inputs before individual XLA compilation.
- Separated PyTorch/TensorFlow dependency extras. Fresh installations now choose
  `[torch]`, `[tensorflow]`, or both; the base package has no framework dependency.
- Extended regression coverage and updated installation/migration, model,
  precision, serialization, interoperability, and API documentation.


## 0.4.0 — 2026-10-13

- Added native Keras `SuperONN2D`, `SuperONNTranspose1D`, and
  `SuperONNTranspose2D`, completing both framework layer families.
- Implemented grouped/full/shared offsets, per-axis bounds, Keras and numeric
  padding, sparse transpose geometry, channel layouts, recomputation, and XLA.
- Added numeric padding, circular boundaries, optional constant bases, and
  channels-first support to the compatible Keras 1D core.
- Used portable TensorFlow convolution for dilated CPU gradients and native
  depthwise execution for eligible 2D connectivity.
- Added validated bidirectional native layer conversion/copy helpers and an
  explicit `superonn.pytorch` namespace.
- Fixed disconnected PyTorch offset gradients on one-pixel reflected axes.
- Allowed fractional fixed-mode bounds while retaining integer offset bins.
- Documented the tested joint CUDA framework/compiler startup order.
- Added output/input/coefficient/offset-gradient parity, finite differences,
  Keras convolution limits, state/serialization, mixed precision, and XLA tests.
- Added interoperability documentation, a native encoder/decoder example,
  and multi-platform CPU CI. Existing Keras 1D weight layouts remain compatible.

## 0.3.2 — 2026-10-13

- Fixed package discovery for the flat `superonn/` source layout.
- Restored the test workflow to the required `.github/workflows/` directory.
- Exposed framework-free defaults and option validation in `superonn.options`;
  existing TensorFlow option imports remain compatible.
- Preserved numerical operators and Keras serialization behavior.

## 0.3.1 — 2026-10-13

- Published a general-purpose package layout and neutral distribution metadata.
- Reworked installation, quick starts, hyperparameter guidance, API documentation,
  performance guidance, and runnable training examples.
- Added `conv1d`, `depthwise_conv1d`, and `separable_conv1d` as convenient Keras
  factory names; existing `spectral_*` names remain compatible aliases.
- Removed development benchmark scripts, raw result records, and cached bytecode
  from the source release. Retained public API regression tests and CPU CI.
- Retained the numerical operator implementation and checkpoint semantics.

## 0.3.0 — 2026-10-13

- Fixed full-graph compilation of sampling by replacing runtime corner-product
  iteration with static integer traversal and precomputing kernel taps.
- Verified PyTorch 2.6/2.14.1 compiler capture, CUDA Inductor, standard checkpoint
  keys after in-place Module.compile, and stable output/gradient parity.
- Added optional native TensorFlow/Keras operators to the standalone package.
- Added serializable per-operator TensorFlow XLA control and validated Keras
  whole-model/graph compilation on current TensorFlow 2.21/Keras 3.15.1.
- Added optional native-convolution precision control; interpolation and
  polynomial evaluation retain stable precision.
- Accepted Keras Conv1D one-element spatial tuples; completed separable output
  shape inference.
- Avoided unnecessary RNG-state preservation in pure PyTorch checkpoint regions.
- Added independent release tests, CPU CI matrix, examples, provenance notices
  and an MIT license. Stable arithmetic and explicit opt-in compilation remain
  defaults; no universal performance/accuracy optimum is claimed.

## 0.2.0 — 2026-10-13

- Added configurable model integration and order-aware coefficient penalties.
- Improved Horner/native execution, precision and checkpoint validation.
- Repaired native Keras serialization/normalization and legacy optimizer behavior.

## 0.1.0 — 2026-10-13

- Introduced standalone 1D/2D PyTorch forward/transpose SuperONNs.
- Implemented both fixed and learned offsets with full/shared connectivity,
  independent scalar/gradient oracles and CNN-limit checks.
