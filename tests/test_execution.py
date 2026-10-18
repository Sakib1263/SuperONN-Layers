"""Opt-in, bounded execution probes, kept separate from default CPU checks.

SUPERONN_PROBE_BACKEND=tensorflow|torch
SUPERONN_PROBE_DEVICE=cpu|cuda
SUPERONN_RUN_COMPILER=1 enables real XLA/Inductor compilation.
Run through tests.run_checks to bound process time and RSS.
"""
import json
import os
import statistics
import time
import unittest
import numpy as np

from .reference import Case, forward

BACKEND = os.environ.get("SUPERONN_PROBE_BACKEND")
DEVICE = os.environ.get("SUPERONN_PROBE_DEVICE", "cpu")
COMPILER = os.environ.get("SUPERONN_RUN_COMPILER") == "1"


def report(**values):
    print(json.dumps(values, sort_keys=True), flush=True)


@unittest.skipUnless(BACKEND == "tensorflow", "opt-in TensorFlow process")
class TensorFlowExecutionTests(unittest.TestCase):

    def setUp(self):
        import tensorflow as tf
        if DEVICE == "cuda" and not tf.config.list_physical_devices("GPU"):
            self.skipTest("requested TensorFlow CUDA runtime unavailable")
        placement = tf.device("/GPU:0" if DEVICE == "cuda" else "/CPU:0")
        placement.__enter__()
        self.addCleanup(placement.__exit__, None, None, None)


    def test_composed_startup_and_graph_bounds(self):
        import tensorflow as tf
        from superonn.tensorflow import SuperONN1D
        if DEVICE == "cuda":
            if not tf.config.list_physical_devices("GPU"):
                self.skipTest("requested TensorFlow CUDA runtime unavailable")
        for width in (3, 9):
            with tf.device("/GPU:0" if DEVICE == "cuda" else "/CPU:0"):
                layer = SuperONN1D(width, 2, q=2, padding="same", output_chunk_size=2,
                                   input_bound=.7, initialization="all_orders")
                layer.build((None, None, 1))
                @tf.function(input_signature=[tf.TensorSpec((None, None, 1), tf.float32)],
                             autograph=False, jit_compile=False)
                def forward(x):
                    return layer(x) + layer(x * .9) + layer(x * .8)
                @tf.function(input_signature=[tf.TensorSpec((None, None, 1), tf.float32)],
                             autograph=False, jit_compile=False)
                def backward(x):
                    with tf.GradientTape() as tape:
                        tape.watch(x)
                        y = forward(x)
                        loss = tf.reduce_mean(y ** 2)
                    return loss, tape.gradient(loss, (x, *layer.trainable_variables))
                x = tf.ones((1, 8, 1)) * .2
                start = time.perf_counter()
                f = forward.get_concrete_function()
                trace_forward = time.perf_counter() - start
                start = time.perf_counter()
                forward(x).numpy()
                first_forward = time.perf_counter() - start
                start = time.perf_counter()
                b = backward.get_concrete_function()
                trace_backward = time.perf_counter() - start
                start = time.perf_counter()
                loss, grads = backward(x)
                loss.numpy()
                first_backward = time.perf_counter() - start
                self.assertTrue(all(g is not None and bool(tf.reduce_all(tf.math.is_finite(g))) for g in grads))
                samples = []
                for _ in range(5):
                    start = time.perf_counter()
                    backward(x)[0].numpy()
                    samples.append(time.perf_counter() - start)
                # Reuse the same helper with different batch/spatial shapes.
                backward(tf.ones((2, 11, 1)) * .2)[0].numpy()
                self.assertEqual(layer._graph_operation.experimental_get_tracing_count(), 1)
                self.assertEqual(layer._recomputed_full_chunk.experimental_get_tracing_count(), 1)
                self.assertEqual(layer._full_chunk_vjp.experimental_get_tracing_count(), 1)
                body = layer._graph_operation.get_concrete_function().graph.as_graph_def()
                definitions = b.graph.as_graph_def().library.function
                forward_boundaries = [v.signature.name for v in definitions
                    if "_noinline" in v.attr and v.attr["_noinline"].b and "forward" in v.signature.name]
                backward_boundaries = [v.signature.name for v in definitions
                    if "_noinline" in v.attr and v.attr["_noinline"].b and "backward" in v.signature.name]
                self.assertTrue(forward_boundaries)
                self.assertTrue(backward_boundaries)
                node_count = len(body.node) + sum(len(fn.node_def) for fn in body.library.function)
                backward_graph = b.graph.as_graph_def()
                training_nodes = len(backward_graph.node) + sum(
                    len(fn.node_def) for fn in backward_graph.library.function)
                if width == 3:
                    baseline_nodes = node_count
                    baseline_training_nodes = training_nodes
                else:
                    self.assertLessEqual(node_count, baseline_nodes + 5)
                    self.assertLessEqual(training_nodes, baseline_training_nodes + 5)
                report(backend="tensorflow", device=DEVICE, width=width, branches=3,
                       forward_trace_s=trace_forward, first_forward_s=first_forward,
                       backward_trace_s=trace_backward, first_backward_s=first_backward,
                       warm_backward_median_s=statistics.median(samples), warm_min_s=min(samples),
                       warm_max_s=max(samples), layer_nodes_including_functions=node_count,
                       containing_forward_nodes=len(f.graph.as_graph_def().node),
                       containing_backward_nodes=len(b.graph.as_graph_def().node),
                       training_nodes_including_functions=training_nodes,
                       vjp_traces=layer._full_chunk_vjp.experimental_get_tracing_count(),
                       layer_traces=layer._graph_operation.experimental_get_tracing_count(),
                       chunk_traces=layer._recomputed_full_chunk.experimental_get_tracing_count())


    def test_device_core_oracle(self):
        import tensorflow as tf
        from .test_tensorflow import make, assign, run
        from .reference import check_gradients
        if DEVICE == "cuda":
            if not tf.config.list_physical_devices("GPU"):
                self.skipTest("requested TensorFlow CUDA runtime unavailable")
        with tf.device("/GPU:0" if DEVICE == "cuda" else "/CPU:0"):
            for rank, transposed in ((1, False), (2, False), (1, True), (2, True)):
                case = Case(rank, transposed)
                arrays = case.data()
                layer = make(case)
                assign(layer, arrays)
                y, grads = run(layer, case, arrays)
                np.testing.assert_allclose(y, forward(case, *arrays), rtol=2e-11, atol=2e-12)
                check_gradients(self, case, arrays, grads)
                report(backend="tensorflow", device=DEVICE, rank=rank, transpose=transposed,
                       reference_and_gradients="passed")

    @unittest.skipUnless(COMPILER, "XLA explicitly opt-in")
    def test_xla_all_cores(self):
        import tensorflow as tf
        from .test_tensorflow import make, assign, tensor
        for rank, transposed in ((1, False), (2, False), (1, True), (2, True)):
            case = Case(rank, transposed)
            arrays = case.data()
            layer = make(case, dtype="float32", jit=True)
            ordinary = make(case, dtype="float32", jit=False)
            assign(layer, arrays)
            assign(ordinary, arrays)
            x = tf.constant(tensor(arrays[0]), tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(x)
                y = layer(x)
                loss = tf.reduce_sum(y ** 2)
            grads = tape.gradient(loss, (x, *layer.trainable_variables))
            with tf.GradientTape() as tape:
                tape.watch(x)
                expected = ordinary(x)
                expected_loss = tf.reduce_sum(expected ** 2)
            expected_grads = tape.gradient(expected_loss, (x, *ordinary.trainable_variables))
            np.testing.assert_allclose(y, expected, rtol=3e-5, atol=3e-6)
            for a, b in zip(grads, expected_grads):
                np.testing.assert_allclose(a, b, rtol=2e-4, atol=3e-6)
            report(backend="tensorflow", device=DEVICE, rank=rank, transpose=transposed,
                   compiler="XLA", result="passed")



    @unittest.skipUnless(COMPILER, "whole-function XLA explicitly opt-in")
    def test_whole_function_xla(self):
        import tensorflow as tf
        from .test_tensorflow import make, assign, tensor
        for rank, transposed in ((1, False), (2, False), (1, True), (2, True)):
            case = Case(rank, transposed)
            arrays = case.data()
            layer = make(case, dtype="float32", jit=True)
            assign(layer, arrays)
            @tf.function(jit_compile=True, autograph=False)
            def update_values(x):
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    loss = tf.reduce_sum(layer(x) ** 2)
                return loss, tape.gradient(loss, (x, *layer.trainable_variables))
            loss, gradients = update_values(tf.constant(tensor(arrays[0]), tf.float32))
            self.assertTrue(bool(tf.math.is_finite(loss)))
            self.assertTrue(all(g is not None and bool(tf.reduce_all(tf.math.is_finite(g))) for g in gradients))
            report(backend="tensorflow", rank=rank, transpose=transposed, device=DEVICE,
                   compiler="whole-function XLA with layer jit_compile=True", result="passed")



    @unittest.skipUnless(COMPILER, "outer-only XLA diagnostic explicitly opt-in")
    def test_outer_only_xla_limit(self):
        import tensorflow as tf
        from superonn.tensorflow import SuperONN1D
        layer = SuperONN1D(3, 2, output_chunk_size=2, jit_compile=False)
        layer.build((None, None, 1))
        @tf.function(jit_compile=True, autograph=False)
        def outer(x):
            with tf.GradientTape() as tape:
                tape.watch(x)
                loss = tf.reduce_sum(layer(x) ** 2)
            return tape.gradient(loss, x)
        try:
            result = outer(tf.ones((1, 8, 1)))
        except tf.errors.InvalidArgumentError as error:
            self.assertIn("tf2xla conversion failed", str(error))
            report(backend="tensorflow", device=DEVICE, outer_only_xla="unsupported",
                   error=str(error).splitlines()[0][:240])
        else:
            self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(result))))
            report(backend="tensorflow", device=DEVICE, outer_only_xla="supported on this runtime/shape")



    @unittest.skipUnless(COMPILER, "training example XLA explicitly opt-in")
    def test_compiled_training_example(self):
        import tensorflow as tf
        from examples.tensorflow_training import regressor
        model = regressor(length=8, jit_compile=True)
        model.compile(optimizer=tf.keras.optimizers.SGD(.01), loss="mse", jit_compile=True)
        before = [v.numpy().copy() for v in model.trainable_variables]
        for _ in range(2):
            loss = model.train_on_batch(tf.ones((1, 8, 1)) * .2, tf.zeros((1, 1)))
            self.assertTrue(np.isfinite(loss))
        self.assertTrue(any(np.any(a != v.numpy()) for a, v in zip(before, model.trainable_variables)))


@unittest.skipUnless(BACKEND == "torch", "opt-in PyTorch process")
class TorchExecutionTests(unittest.TestCase):
    def test_full_chunk_profile(self):
        import torch
        from .test_torch import make, assign
        if DEVICE == "cuda":
            self.assertTrue(torch.cuda.is_available(), "requested CUDA unavailable")
        sync = torch.cuda.synchronize if DEVICE == "cuda" else lambda: None
        for rank, transposed in ((1, False), (2, False), (1, True), (2, True)):
            case = Case(rank, transposed)
            arrays = case.data()
            for remat in (False, True):
                layer = make(case, recompute=remat, dtype=torch.float32, device=DEVICE)
                assign(layer, arrays)
                x = torch.tensor(arrays[0], dtype=torch.float32, device=DEVICE, requires_grad=True)
                def step():
                    return torch.autograd.grad(layer(x).square().mean(), (x, *layer.parameters()))
                if DEVICE == "cuda":
                    baseline_memory = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                sync()
                start = time.perf_counter()
                step()
                sync()
                cold = time.perf_counter() - start
                timings = []
                for _ in range(5):
                    sync()
                    start = time.perf_counter()
                    step()
                    sync()
                    timings.append(time.perf_counter() - start)

                peak = torch.cuda.max_memory_allocated() - baseline_memory if DEVICE == "cuda" else None
                np.testing.assert_allclose(layer(x).detach().cpu().numpy(), forward(case, *arrays),
                                           rtol=3e-5, atol=3e-6)
                if DEVICE == "cuda" and remat:
                    from .reference import check_gradients, target_for
                    loss = (layer(x) * torch.tensor(target_for(case, arrays), device=DEVICE, dtype=x.dtype)).sum()
                    gradients = torch.autograd.grad(loss, (x, layer.weight, layer.raw_shifts))
                    check_gradients(self, case, arrays, [g.detach().cpu().numpy() for g in gradients])
                    with torch.autocast("cuda", dtype=torch.float16):
                        y = layer(x)
                        loss = y.float().square().sum()
                    self.assertEqual(y.dtype, torch.float16)
                    torch.testing.assert_close(y.float(), layer(x), rtol=.02, atol=.002)
                    self.assertTrue(all(torch.isfinite(g).all() for g in
                        torch.autograd.grad(loss, (x, *layer.parameters()))))
                    del y, loss, gradients

                report(backend="torch", device=DEVICE, rank=rank, transpose=transposed,
                       recompute=remat, cold_s=cold, warm_median_s=statistics.median(timings),
                       warm_min_s=min(timings), warm_max_s=max(timings),
                       peak_incremental_device_bytes=peak)

    def _inductor_core(self, rank, transposed):
        import torch
        from .test_torch import make, assign
        # Each public test is independently selectable so a cold compilation
        # cannot consume the budget of a later core. Both implementations
        # (full Horner and shared/native convolution) need real code generation.
        for kind in ("full", "shared"):
            for remat in (False, True):
                case = Case(rank, transposed, kind)
                arrays = case.data()
                eager = make(case, recompute=remat, dtype=torch.float32, device=DEVICE)
                candidate = make(case, recompute=remat, dtype=torch.float32, device=DEVICE)
                assign(eager, arrays)
                assign(candidate, arrays)
                compiled = torch.compile(candidate, backend="inductor",
                                         fullgraph=True, dynamic=False)
                optimizers = [torch.optim.SGD(model.parameters(), lr=.01)
                              for model in (eager, candidate)]
                synchronize = torch.cuda.synchronize if DEVICE == "cuda" else lambda: None
                report(backend="torch", version=torch.__version__, device=DEVICE,
                       rank=rank, transpose=transposed, layout=kind, recompute=remat,
                       compiler="Inductor", stage="starting")
                start = time.perf_counter()
                for step in range(3):
                    inputs = [torch.tensor(arrays[0] * (1 + .05 * step),
                                           dtype=torch.float32, device=DEVICE,
                                           requires_grad=True) for _ in range(2)]
                    for optimizer in optimizers:
                        optimizer.zero_grad(set_to_none=True)
                    expected, actual = eager(inputs[0]), compiled(inputs[1])
                    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
                    if step == 0:
                        np.testing.assert_allclose(actual.detach().cpu().numpy(),
                                                   forward(case, *arrays),
                                                   rtol=3e-5, atol=3e-6)
                    expected.square().sum().backward()
                    actual.square().sum().backward()
                    for a, b in zip((inputs[1], *candidate.parameters()),
                                    (inputs[0], *eager.parameters())):
                        self.assertIsNotNone(a.grad)
                        self.assertIsNotNone(b.grad)
                        self.assertTrue(bool(torch.isfinite(a.grad).all()))
                        torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=3e-6)
                    for optimizer in optimizers:
                        optimizer.step()
                    for a, b in zip(candidate.parameters(), eager.parameters()):
                        torch.testing.assert_close(a, b, rtol=2e-4, atol=3e-6)
                    synchronize()
                    if step == 0:
                        cold_seconds = time.perf_counter() - start
                report(backend="torch", version=torch.__version__, device=DEVICE,
                       rank=rank, transpose=transposed, layout=kind, recompute=remat,
                       compiler="Inductor", result="passed", optimizer_steps=3,
                       cold_forward_backward_s=cold_seconds,
                       total_s=time.perf_counter() - start)

    @unittest.skipUnless(COMPILER, "Inductor explicitly opt-in")
    def test_inductor_1d(self):
        self._inductor_core(1, False)

    @unittest.skipUnless(COMPILER, "Inductor explicitly opt-in")
    def test_inductor_2d(self):
        self._inductor_core(2, False)

    @unittest.skipUnless(COMPILER, "Inductor explicitly opt-in")
    def test_inductor_transpose1d(self):
        self._inductor_core(1, True)

    @unittest.skipUnless(COMPILER, "Inductor explicitly opt-in")
    def test_inductor_transpose2d(self):
        self._inductor_core(2, True)
