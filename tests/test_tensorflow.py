"""CPU oracle, state, dynamic graph and mixed-policy checks for native Keras."""
import importlib.util
import os
import tempfile
import unittest
import numpy as np

if importlib.util.find_spec("tensorflow"):
    import tensorflow as tf
    import superonn.tensorflow as sonn
else:
    tf = None

from .reference import Case, forward, check_gradients, target_for

CLASSES = ("SuperONN1D", "SuperONN2D", "SuperONNTranspose1D", "SuperONNTranspose2D")


def tensor(values):
    return np.moveaxis(values, 1, -1)


def canonical(values):
    return np.moveaxis(np.asarray(values), -1, 1)


def make(case, recompute=True, dtype="float64", jit=False):
    index = case.rank - 1 + 2 * case.transpose
    args = dict(filters=case.outputs, kernel_size=case.kernel, strides=case.stride,
                dilation_rate=case.dilation, padding=case.padding, groups=case.groups,
                q=case.q, shift_sharing=case.sharing, max_shift=case.limit,
                input_bound=case.bound, include_constant=case.constant,
                output_chunk_size=2, recompute=recompute, dtype=dtype, jit_compile=jit,
                boundary=case.boundary)
    if case.transpose:
        args["output_padding"] = case.output_padding
    layer = getattr(sonn, CLASSES[index])(**args)
    layer.build((None, *((None,) * case.rank), case.channels))
    return layer


def assign(layer, arrays):
    _, w, raw, bias = arrays
    d = layer.spatial_rank
    layer.kernel.assign(w.transpose(*range(3, 3 + d), 1, 2, 0))
    layer.shift_parameter.assign(raw.reshape(layer.shift_parameter.shape))
    layer.bias.assign(bias)


def run(layer, case, arrays):
    x = tf.constant(tensor(arrays[0]), tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = layer(x)
        loss = tf.reduce_sum(y * tensor(target_for(case, arrays)))
    grads = tape.gradient(loss, (x, layer.kernel, layer.shift_parameter))
    d = layer.spatial_rank
    return canonical(y.numpy()), (canonical(grads[0].numpy()),
        grads[1].numpy().transpose(d + 2, d, d + 1, *range(d)),
        grads[2].numpy().reshape(arrays[2].shape))



@unittest.skipIf(tf is None, "TensorFlow is not installed")
class TensorFlowCoreTests(unittest.TestCase):
    def test_reference_and_gradients(self):
        for rank, transposed in ((1, False), (2, False), (1, True), (2, True)):
            for kind in ("full", "grouped", "shared", "depthwise"):
                case = Case(rank, transposed, kind)
                arrays = case.data()
                results = []
                for recompute in (False, True):
                    with self.subTest(rank=rank, transpose=transposed, kind=kind, recompute=recompute):
                        layer = make(case, recompute)
                        assign(layer, arrays)
                        y, grads = run(layer, case, arrays)
                        np.testing.assert_allclose(y, forward(case, *arrays), rtol=2e-11, atol=2e-12)
                        check_gradients(self, case, arrays, grads, exhaustive=kind == "full")
                        results.append((y, grads))
                for a, b in zip(results[0][1], results[1][1]):
                    np.testing.assert_allclose(a, b, rtol=2e-10, atol=2e-11)

    def test_dynamic_stateless_serialization_and_frozen_gradients(self):
        for rank, transposed in ((1, False), (2, False), (1, True), (2, True)):
            with self.subTest(rank=rank, transpose=transposed):
                case = Case(rank, transposed)
                layer = make(case)
                arrays = case.data()
                assign(layer, arrays)
                original = layer(tf.constant(tensor(arrays[0])))
                # Dynamic batch/spatial sizes use the same layer and helper graphs.
                for size in (3, 5):
                    shape = (2, *((size,) * rank), case.channels)
                    with tf.GradientTape() as tape:
                        x = tf.ones(shape, tf.float64) * .2
                        tape.watch(x)
                        loss = tf.reduce_sum(layer(x))
                    self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(tape.gradient(loss, x)))))
                self.assertEqual(layer._graph_operation.experimental_get_tracing_count(), 1)
                self.assertEqual(layer._recomputed_full_chunk.experimental_get_tracing_count(), 1)
                # Both forward and generated gradient functions must retain the local boundary.
                graph = layer._graph_operation.get_concrete_function().graph.as_graph_def()
                marked = [f for f in graph.library.function if f.attr.get("_noinline", None) and f.attr["_noinline"].b]
                self.assertTrue(marked)
                changed = [tf.identity(v) for v in layer.trainable_variables]
                kernel_index = next(i for i, v in enumerate(layer.trainable_variables) if v is layer.kernel)
                changed[kernel_index] = changed[kernel_index] + .05
                supplied, _ = layer.stateless_call(changed, layer.non_trainable_variables,
                                                  tf.constant(tensor(arrays[0])))
                self.assertGreater(float(tf.reduce_max(tf.abs(supplied - original))), 1e-7)
                np.testing.assert_array_equal(layer(tf.constant(tensor(arrays[0]))), original)
                layer.kernel.assign_add(tf.ones_like(layer.kernel) * .05)
                np.testing.assert_allclose(layer(tf.constant(tensor(arrays[0]))), supplied, rtol=1e-12, atol=1e-12)
                inputs = tf.keras.Input(shape=(*((None,) * rank), case.channels), dtype="float64")
                model = tf.keras.Model(inputs, layer(inputs))
                with tempfile.TemporaryDirectory() as directory:
                    path = os.path.join(directory, "model.keras")
                    model.save(path)
                    restored = tf.keras.models.load_model(path)
                    np.testing.assert_allclose(restored(tensor(arrays[0])), supplied, rtol=1e-12, atol=1e-12)
                layer.trainable = False
                x = tf.constant(tensor(arrays[0]))
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    y = tf.reduce_sum(model(x, training=False))
                self.assertIsNotNone(tape.gradient(y, x))

    def test_mixed_precision_and_nested_derivatives(self):
        for rank, transposed in ((1, False), (2, False), (1, True), (2, True)):
            with self.subTest(rank=rank, transpose=transposed):
                case = Case(rank, transposed)
                arrays = case.data()
                full = make(case, dtype="float32")
                half = make(case, dtype="mixed_bfloat16")
                assign(full, arrays)
                assign(half, arrays)
                x = tf.constant(tensor(arrays[0]), tf.float32)
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    y = half(x)
                    loss = tf.reduce_sum(tf.cast(y, tf.float32) ** 2)
                self.assertEqual(y.dtype, tf.bfloat16)
                np.testing.assert_allclose(tf.cast(y, tf.float32), full(x), rtol=.02, atol=.002)
                for grad in tape.gradient(loss, [x, *half.trainable_variables]):
                    self.assertIsNotNone(grad)
                    self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(grad))))
                ordinary = make(case, recompute=False)
                assign(ordinary, arrays)
                x = tf.constant(tensor(arrays[0]))
                with tf.GradientTape() as outer:
                    outer.watch(x)
                    with tf.GradientTape() as inner:
                        inner.watch(x)
                        loss = tf.reduce_sum(ordinary(x) ** 2)
                    dx = inner.gradient(loss, x)
                    penalty = tf.reduce_sum(dx ** 2)
                for grad in outer.gradient(penalty, [x, *ordinary.trainable_variables]):
                    self.assertIsNotNone(grad)
                    self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(grad))))

    def test_factories_and_compositions(self):
        settings = dict(q=2, recompute=True, output_chunk_size=2)
        x = tf.ones((1, 6, 2)) * .1
        self.assertIs(sonn.conv1d, sonn.spectral_conv1d)
        for factory, args in ((sonn.conv1d, (3, 2)), (sonn.depthwise_conv1d, (2,)),
                              (sonn.separable_conv1d, (3, 2))):
            layer = factory(*args, operator="superonn", superonn=settings)
            with tf.GradientTape() as tape:
                tape.watch(x)
                y = tf.reduce_sum(layer(x))
            self.assertTrue(all(g is not None for g in tape.gradient(y, [x, *layer.trainable_variables])))
            clone = type(layer).from_config(layer.get_config())
            clone(x)
            clone.set_weights(layer.get_weights())
            np.testing.assert_allclose(clone(x), layer(x), rtol=1e-6, atol=1e-7)
        for factory, shape in ((sonn.conv2d, (1, 4, 4, 1)),
                               (sonn.conv_transpose1d, (1, 4, 1)),
                               (sonn.conv_transpose2d, (1, 4, 4, 1))):
            layer = factory(3, 2, operator="superonn", superonn=settings)
            self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(layer(tf.ones(shape))))))

    def test_q1_native_and_fixed_offsets(self):
        for rank, transposed in ((1, False), (2, False), (1, True), (2, True)):
            with self.subTest(rank=rank, transpose=transposed):
                case = Case(rank, transposed)
                case.q, case.constant, case.bound, case.limit = 1, False, None, 0.
                case.dilation = (1,) * rank
                arrays = case.data()
                layer = make(case)
                assign(layer, arrays)
                x = tf.constant(tensor(arrays[0]))
                actual = layer(x)
                weights = tf.squeeze(layer.kernel, axis=-2)
                if transposed:
                    weights = tf.transpose(weights, (*range(rank), rank + 1, rank))
                    raw_shape = [tf.shape(x)[0], *[n + 2 for n in actual.shape[1:-1]], case.outputs]
                    expected = tf.nn.conv_transpose(x, weights, raw_shape,
                        strides=(1, *case.stride, 1), padding="VALID",
                        data_format="NWC" if rank == 1 else "NHWC")
                    expected = expected[(slice(None), *((slice(1, -1),) * rank), slice(None))]
                else:
                    expected = tf.nn.convolution(tf.pad(x, [[0, 0], *([[1, 1]] * rank), [0, 0]]),
                        weights, strides=case.stride, padding="VALID",
                        data_format="NWC" if rank == 1 else "NHWC")
                np.testing.assert_allclose(actual, expected + layer.bias, rtol=2e-11, atol=2e-12)
                fixed = type(layer).from_config({**layer.get_config(), "shift_mode": "random_fixed", "max_shift": 2.})
                fixed.build(x.shape)
                fixed.kernel.assign(layer.kernel)
                fixed.bias.assign(layer.bias)
                fixed.shift_parameter.assign(tf.ones_like(fixed.shift_parameter))
                self.assertFalse(fixed.shift_parameter.trainable)
                self.assertTrue(bool(tf.reduce_all(tf.equal(fixed.effective_shifts(), 1.))))
                model = tf.keras.Sequential([tf.keras.Input(x.shape[1:], dtype="float64"), fixed])
                with tempfile.TemporaryDirectory() as directory:
                    path = os.path.join(directory, "fixed.keras")
                    model.save(path)
                    restored = tf.keras.models.load_model(path)
                    np.testing.assert_allclose(restored(x), model(x), rtol=2e-11, atol=2e-12)

    def test_boundary_and_block_paths(self):
        for rank in (1, 2):
            for boundary in ("border", "reflection", "circular"):
                case = Case(rank, False)
                case.boundary = boundary
                arrays = case.data()
                layer = make(case)
                assign(layer, arrays)
                y, grads = run(layer, case, arrays)
                np.testing.assert_allclose(y, forward(case, *arrays), rtol=2e-11, atol=2e-12)
                check_gradients(self, case, arrays, grads)
        block = sonn.SuperONNBlock1D(3, 3, q=2, output_chunk_size=2, recompute=True)
        x = tf.ones((1, 5, 2)) * .1
        with tf.GradientTape() as tape:
            tape.watch(x)
            value = tf.reduce_sum(block(x, training=False))
        self.assertTrue(all(g is not None for g in tape.gradient(value, [x, *block.trainable_variables])))
        clone = type(block).from_config(block.get_config())
        clone(x)
        clone.set_weights(block.get_weights())
        np.testing.assert_allclose(clone(x), block(x), rtol=1e-6, atol=1e-7)

    def test_independent_branch_recipe(self):
        from examples.tensorflow_branches import branch_functions
        model = tf.keras.Sequential([tf.keras.Input((None, 1)),
            sonn.conv1d(3, 2, operator="superonn", order_regularization=1e-5),
            tf.keras.layers.GlobalAveragePooling1D(), tf.keras.layers.Dense(1)])
        branch, regularizer = branch_functions(model)
        batches = [(tf.ones((1, length, 1)) * .2, tf.zeros((1, 1))) for length in (4, 5)]
        parts = [branch(x, y) for x, y in batches]
        penalty, extras = regularizer()
        combined = [tf.add_n([g[i] for _, g in parts]) / 2 + extra for i, extra in enumerate(extras)]
        with tf.GradientTape() as tape:
            loss = tf.add_n([tf.reduce_mean((model(x, training=True) - y) ** 2) for x, y in batches]) / 2
            loss = loss + sonn.model_regularization_loss(model)
        expected = tape.gradient(loss, model.trainable_variables)
        for a, b in zip(combined, expected):
            np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-7)
        self.assertEqual(branch.experimental_get_tracing_count(), 1)

    def test_streamed_backward_has_no_chunk_tape_accumulators(self):
        for rank, transposed in ((1, False), (2, False), (1, True), (2, True)):
            with self.subTest(rank=rank, transpose=transposed):
                case = Case(rank, transposed, "grouped")
                layer = make(case)
                assign(layer, case.data())
                @tf.function(input_signature=[tf.TensorSpec(
                    (None, *((None,) * rank), case.channels), tf.float64)],
                    autograph=False, jit_compile=False)
                def gradients(x):
                    with tf.GradientTape() as tape:
                        tape.watch(x)
                        loss = tf.reduce_sum(layer(x) ** 2)
                    return tape.gradient(loss, (x, *layer.trainable_variables))
                concrete = gradients.get_concrete_function()
                graph = concrete.graph.as_graph_def()
                definitions = {f.signature.name: f for f in graph.library.function}
                nodes = [node for f in definitions.values() for node in f.node_def]
                loops = [node for node in nodes if node.op in ("While", "StatelessWhile")]
                self.assertGreaterEqual(len(loops), 2 * case.groups)
                # The previous autodifferentiated primal loop stored one list
                # per passthrough value for every chunk. Sequential primal
                # execution alone did not bound those backward buffers.
                accumulated = [f"{f.signature.name}/{node.name}" for f in definitions.values()
                               for node in f.node_def if node.op == "TensorListPushBack"]
                self.assertFalse(accumulated, accumulated)
                values = gradients(tf.constant(tensor(case.data()[0])))
                self.assertTrue(all(g is not None and bool(tf.reduce_all(tf.math.is_finite(g)))
                                    for g in values))
                self.assertEqual(layer._full_chunk_vjp.experimental_get_tracing_count(), 1)

    def test_shared_dynamic_and_instance_lifetime(self):
        import gc
        import weakref
        def exercise(rank, transposed, kind):
            case = Case(rank, transposed, kind)
            layer = make(case)
            assign(layer, case.data())
            for size in (3, 5):
                x = tf.ones((2, *((size,) * rank), case.channels), tf.float64) * .2
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    loss = tf.reduce_sum(layer(x))
                self.assertIsNotNone(tape.gradient(loss, x))
            self.assertEqual(layer._graph_operation.experimental_get_tracing_count(), 1)
            helper = layer._recomputed_shared if kind == "shared" else layer._full_chunk_vjp
            self.assertEqual(helper.experimental_get_tracing_count(), 1)
            return weakref.ref(layer)
        references = [exercise(rank, transposed, kind) for rank, transposed in
                      ((1, False), (2, False), (1, True), (2, True))
                      for kind in ("full", "shared")]
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))

    def test_saved_serving_after_owner_release(self):
        import gc
        for rank, transposed in ((1, False), (2, False), (1, True), (2, True)):
            with self.subTest(rank=rank, transpose=transposed):
                case = Case(rank, transposed)
                arrays = case.data()
                layer = make(case)
                assign(layer, arrays)
                inputs = tf.keras.Input(shape=(*((None,) * rank), case.channels), dtype="float64")
                model = tf.keras.Model(inputs, layer(inputs))
                expected = tensor(forward(case, *arrays))
                with tempfile.TemporaryDirectory() as directory:
                    model.export(directory, format="tf_saved_model", verbose=False)
                    del model, layer
                    gc.collect()
                    serving = tf.saved_model.load(directory)
                    result = serving.serve(tf.constant(tensor(arrays[0])))
                    np.testing.assert_allclose(result, expected, rtol=2e-11, atol=2e-12)
