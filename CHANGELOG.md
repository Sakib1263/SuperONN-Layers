# Changelog

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
