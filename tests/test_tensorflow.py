"""CPU oracle, state, dynamic graph and mixed-policy checks for native Keras."""
import importlib.util
from contextlib import contextmanager
import os
import tempfile
import unittest
from unittest.mock import patch
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


@unittest.skipIf(tf is None, "TensorFlow is not installed")
class TensorFlowSharedStrideTests(unittest.TestCase):
    """Direct-stride contracts only; no training workload or timing benchmark."""

    @staticmethod
    def case(channels=2, groups=1, outputs=3, stride=2, dilation=1):
        case = Case(1, False, "shared")
        case.channels, case.groups, case.outputs = channels, groups, outputs
        case.sizes, case.kernel = (9,), (3,)
        case.stride, case.dilation, case.padding = (stride,), (dilation,), (1,)
        case.q, case.bound = 3, .7
        return case

    @staticmethod
    @contextmanager
    def unit_stride_reference():
        """Reproduce the old dense-convolution-then-slice computation."""
        convolution, depthwise = tf.nn.convolution, tf.nn.depthwise_conv2d

        def dense_convolution(inputs, filters, strides=None, padding="VALID", **kwargs):
            rank = filters.shape.rank - 2
            strides = strides or (1,) * rank
            result = convolution(inputs, filters, strides=(1,) * rank, padding=padding, **kwargs)
            return result[(slice(None), *(slice(None, None, s) for s in strides), slice(None))]

        def dense_depthwise(inputs, filter, **kwargs):
            strides = kwargs.pop("strides")[1:3]
            result = depthwise(inputs, filter, strides=(1, 1, 1, 1), **kwargs)
            return result[:, ::strides[0], ::strides[1], :]

        with patch.object(tf.nn, "convolution", side_effect=dense_convolution), \
                patch.object(tf.nn, "depthwise_conv2d", side_effect=dense_depthwise):
            yield

    def tearDown(self):
        tf.keras.backend.clear_session()

    def test_values_and_gradients_match_legacy_and_scalar_oracle(self):
        specifications = (
            (2, 1, 3, 2, 1),  # ordinary shared convolution
            (4, 2, 6, 3, 1),  # grouped shared convolution
            (2, 2, 2, 2, 1),  # depthwise, multiplier one (also full-equivalent)
            (2, 2, 4, 3, 1),  # depthwise, multiplier two
            (2, 1, 3, 2, 2),  # stride+dilation fallback, ordinary
            (2, 2, 4, 2, 2),  # stride+dilation fallback, depthwise
            (2, 1, 3, 1, 2),  # unit stride remains unchanged
        )
        for index, specification in enumerate(specifications):
            case = self.case(*specification)
            case.constant = index % 2 == 0
            arrays = case.data()
            arrays[2].flat[::2] *= -1  # positive and negative fractional shifts
            reference = make(case, recompute=False)
            assign(reference, arrays)
            with self.unit_stride_reference():
                expected, expected_grads = run(reference, case, arrays)
            np.testing.assert_allclose(expected, forward(case, *arrays), rtol=2e-11, atol=2e-12)
            for recompute in (False, True):
                with self.subTest(specification=specification, recompute=recompute):
                    layer = make(case, recompute=recompute)
                    assign(layer, arrays)
                    actual, gradients = run(layer, case, arrays)
                    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)
                    for actual_grad, expected_grad in zip(gradients, expected_grads):
                        np.testing.assert_allclose(actual_grad, expected_grad, rtol=2e-10, atol=2e-11)
                    if not recompute:
                        check_gradients(self, case, arrays, gradients)

    def test_native_convolutions_do_not_compute_discarded_outputs(self):
        for groups in (1, 2, 4):
            for dilation in (1, 2):
                with self.subTest(groups=groups, dilation=dilation):
                    case = self.case(4, groups, 8, stride=3, dilation=dilation)
                    layer = make(case, recompute=False)
                    convolution, depthwise = tf.nn.convolution, tf.nn.depthwise_conv2d
                    calls = []

                    def record_convolution(inputs, filters, **kwargs):
                        result = convolution(inputs, filters, **kwargs)
                        calls.append((kwargs["strides"][0], int(result.shape[1])))
                        return result

                    def record_depthwise(inputs, filter, **kwargs):
                        self.assertEqual(kwargs["strides"][1], kwargs["strides"][2])
                        result = depthwise(inputs, filter, **kwargs)
                        calls.append((kwargs["strides"][2], int(result.shape[2])))
                        return result

                    with patch.object(tf.nn, "convolution", side_effect=record_convolution), \
                            patch.object(tf.nn, "depthwise_conv2d", side_effect=record_depthwise):
                        result = layer(tf.constant(tensor(case.data()[0])))
                    count = int(result.shape[1])
                    native_stride = 3 if dilation == 1 else 1
                    native_count = count if dilation == 1 else (count - 1) * 3 + 1
                    self.assertEqual(calls, [(native_stride, native_count)] * (1 if groups == 4 else groups))
                    self.assertGreater(count, 1)
                    if dilation == 1:
                        self.assertLess(native_count, (count - 1) * 3 + 1)

    def test_dynamic_padding_boundary_and_layout_parity(self):
        for groups, channels_first in ((1, False), (2, True)):
            for padding in ("same", "valid", 2):
                for boundary in ("zeros", "border", "reflection", "circular"):
                    with self.subTest(groups=groups, padding=padding, boundary=boundary):
                        options = dict(filters=4, kernel_size=3, strides=2, groups=groups,
                                       q=3, padding=padding, boundary=boundary, shift_sharing="shared",
                                       data_format="channels_first" if channels_first else "channels_last",
                                       dtype="float64", recompute=True)
                        layer = sonn.SuperONN1D(**options)
                        legacy = sonn.SuperONN1D(**{**options, "recompute": False})
                        shape = (None, 2, None) if channels_first else (None, None, 2)
                        layer.build(shape)
                        legacy.build(shape)
                        layer.shift_parameter.assign(tf.ones_like(layer.shift_parameter) * -.23)
                        legacy.set_weights(layer.get_weights())
                        for length in (8, 9):
                            x = tf.reshape(tf.linspace(tf.constant(-.6, tf.float64), .8, length * 2), (1, length, 2))
                            if channels_first:
                                x = tf.transpose(x, (0, 2, 1))
                            with self.unit_stride_reference():
                                expected = legacy(x)
                            np.testing.assert_allclose(layer(x), expected, rtol=2e-11, atol=2e-12)
                        self.assertEqual(layer._graph_operation.experimental_get_tracing_count(), 1)

    def test_float32_and_mixed_policy_gradients_and_weight_round_trip(self):
        for groups in (1, 2):
            for dtype in ("float32", "mixed_float16"):
                with self.subTest(groups=groups, dtype=dtype):
                    options = dict(filters=4, kernel_size=3, strides=2, groups=groups,
                                   padding="same", q=3, shift_sharing="shared", dtype=dtype)
                    layer = sonn.SuperONN1D(**options, recompute=True)
                    legacy = sonn.SuperONN1D(**options, recompute=False)
                    layer.build((None, None, 2))
                    legacy.build((None, None, 2))
                    layer.shift_parameter.assign(tf.ones_like(layer.shift_parameter) * .27)
                    legacy.set_weights(layer.get_weights())
                    x = tf.reshape(tf.linspace(-.5, .8, 36), (2, 9, 2))

                    def evaluate(core):
                        with tf.GradientTape() as tape:
                            tape.watch(x)
                            y = core(x)
                            loss = tf.reduce_sum(tf.cast(y, tf.float32) ** 2)
                        return y, tape.gradient(loss, (x, *core.trainable_variables))

                    with self.unit_stride_reference():
                        expected, expected_grads = evaluate(legacy)
                    actual, gradients = evaluate(layer)
                    tolerance = 2e-3 if dtype == "mixed_float16" else 2e-6
                    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
                    for actual_grad, expected_grad in zip(gradients, expected_grads):
                        self.assertIsNotNone(actual_grad)
                        self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(actual_grad))))
                        np.testing.assert_allclose(actual_grad, expected_grad, rtol=tolerance, atol=tolerance)
                    restored = sonn.SuperONN1D.from_config(layer.get_config())
                    restored.build((None, None, 2))
                    restored.set_weights(layer.get_weights())
                    np.testing.assert_array_equal(restored(x), actual)


@unittest.skipIf(tf is None, "TensorFlow is not installed")
class TensorFlowNDStrideTests(unittest.TestCase):
    """Compact-path striding, with independent values and derivative contracts."""

    @staticmethod
    def case(channels=2, groups=1, outputs=3, stride=(2, 3), dilation=(1, 1),
             kind="shared", rank=2, transpose=False):
        case = Case(rank, transpose, kind)
        case.channels, case.groups, case.outputs = channels, groups, outputs
        case.sizes, case.kernel = ((5, 7), (2, 3)) if rank == 2 else ((5,), (3,))
        case.stride, case.dilation = stride, dilation
        case.padding = (1,) * rank
        case.output_padding = (1,) * rank if transpose else (0,) * rank
        case.q, case.bound = 3, .7
        return case

    def tearDown(self):
        tf.keras.backend.clear_session()

    @staticmethod
    def evaluate(layer, case, arrays):
        x = tf.constant(tensor(arrays[0]), tf.float64)
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = layer(x)
            loss = tf.reduce_sum(y * tensor(target_for(case, arrays)))
        gradients = tape.gradient(loss, (x, layer.kernel, layer.shift_parameter, layer.bias))
        d = case.rank
        return canonical(y), (
            canonical(gradients[0]),
            gradients[1].numpy().transpose(d + 2, d, d + 1, *range(d)),
            gradients[2].numpy().reshape(arrays[2].shape), gradients[3].numpy())

    def test_native_counts_and_structural_dispatch(self):
        cases = [
            self.case(), self.case(4, 2, 6),
            self.case(2, 2, 4, stride=(2, 2)),
            self.case(2, 2, 4),  # unequal CPU depthwise strides retain fallback
            self.case(dilation=(2, 1)),
            self.case(2, 2, 4, stride=(2, 2), dilation=(1, 2)),
            self.case(rank=1, transpose=True, stride=(2,), dilation=(1,)),
            self.case(transpose=True, stride=(2, 2)),
        ]
        zero = self.case(kind="full")
        zero.limit = 0.
        cases.append(zero)
        for case in cases:
            with self.subTest(rank=case.rank, transpose=case.transpose,
                              groups=case.groups, stride=case.stride,
                              dilation=case.dilation, bound=case.limit):
                layer = make(case, recompute=False)
                convolution, depthwise = tf.nn.convolution, tf.nn.depthwise_conv2d
                calls = []
                def record_convolution(inputs, filters, **kwargs):
                    result = convolution(inputs, filters, **kwargs)
                    calls.append((tuple(kwargs.get("strides", (1,) * case.rank)),
                                  tuple(result.shape[1:-1])))
                    return result
                def record_depthwise(inputs, filter, **kwargs):
                    result = depthwise(inputs, filter, **kwargs)
                    native = tuple(kwargs["strides"][1:3])
                    self.assertEqual(native[0], native[1])
                    calls.append((native, tuple(result.shape[1:-1])))
                    return result
                with patch.object(tf.nn, "convolution", side_effect=record_convolution), \
                        patch.object(tf.nn, "depthwise_conv2d", side_effect=record_depthwise):
                    actual = layer(tf.constant(tensor(case.data()[0])))
                steps = (1,) * case.rank if case.transpose else case.stride
                native = steps if all(d == 1 for d in case.dilation) else (1,) * case.rank
                is_depthwise = case.rank == 2 and case.groups == case.channels
                if is_depthwise and len(set(native)) != 1:
                    native = (1, 1)
                counts = tuple(actual.shape[1:-1])
                expected_counts = tuple((n - 1) * (s // ns) + 1
                                        for n, s, ns in zip(counts, steps, native))
                self.assertEqual(calls, [(native, expected_counts)] *
                                 (1 if is_depthwise else case.groups))
        # A learned full-layout zero initialization is not structural sharing.
        case = self.case(kind="full")
        arrays = case.data()
        arrays[2].fill(0)
        layer = make(case, recompute=False)
        assign(layer, arrays)
        with patch.object(layer, "_shared", wraps=layer._shared) as compact:
            actual = layer(tf.constant(tensor(arrays[0])))
            compact.assert_not_called()
        np.testing.assert_allclose(canonical(actual), forward(case, *arrays),
                                   rtol=2e-11, atol=2e-12)

    def test_oracle_legacy_and_all_gradients(self):
        cases = [
            self.case(), self.case(4, 2, 6),
            self.case(2, 2, 2, stride=(2, 2), kind="depthwise"),
            self.case(2, 2, 4, stride=(2, 2)), self.case(2, 2, 4),
            self.case(dilation=(2, 1)),
            self.case(2, 2, 4, stride=(2, 2), dilation=(1, 2)),
            self.case(kind="full"),
            self.case(rank=1, transpose=True, stride=(2,), dilation=(1,)),
            self.case(transpose=True, stride=(2, 2)),
            self.case(2, 2, 4, transpose=True, stride=(2, 2)),
        ]
        cases[7].limit = 0.
        short = self.case()
        short.sizes, short.kernel, short.stride = (1, 3), (2, 2), (2, 2)
        cases.append(short)
        for index, case in enumerate(cases):
            case.constant = not case.transpose and index % 2 == 0
            arrays = case.data()
            arrays[2].flat[::2] *= -1
            legacy = make(case, recompute=False)
            assign(legacy, arrays)
            with TensorFlowSharedStrideTests.unit_stride_reference():
                expected, expected_grads = self.evaluate(legacy, case, arrays)
            np.testing.assert_allclose(expected, forward(case, *arrays),
                                       rtol=2e-11, atol=2e-12)
            for recompute in (False, True):
                with self.subTest(index=index, recompute=recompute):
                    layer = make(case, recompute=recompute)
                    assign(layer, arrays)
                    actual, gradients = self.evaluate(layer, case, arrays)
                    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)
                    for got, wanted in zip(gradients, expected_grads):
                        np.testing.assert_allclose(got, wanted, rtol=2e-10, atol=2e-11)
                    if not recompute:
                        check_gradients(self, case, arrays, gradients)

    def test_dynamic_boundaries_stateless_serialization_and_frozen_inputs(self):
        for boundary in ("zeros", "border", "reflection", "circular"):
            for padding in ("same", "valid", (1, 2)):
                with self.subTest(boundary=boundary, padding=padding):
                    options = dict(filters=4, kernel_size=(2, 3), strides=(2, 3),
                                   groups=2, q=3, padding=padding, boundary=boundary,
                                   shift_sharing="shared", data_format="channels_first",
                                   dtype="float64", recompute=True,
                                   initialization="all_orders", input_bound=.7, order_regularization=1e-4)
                    layer = sonn.conv2d(operator="superonn", **options)
                    legacy = sonn.SuperONN2D(**{**options, "recompute": False})
                    layer.build((None, 4, None, None))
                    legacy.build((None, 4, None, None))
                    # Exact edge coordinates exercise boundary contracts separately
                    # from the fractional-coordinate finite differences above.
                    for sizes in ((5, 7), (6, 8)):
                        x = tf.reshape(tf.linspace(tf.constant(-.6, tf.float64), .8,
                                                   4 * np.prod(sizes)), (1, 4, *sizes))
                        legacy.set_weights(layer.get_weights())
                        with TensorFlowSharedStrideTests.unit_stride_reference():
                            expected = legacy(x)
                        np.testing.assert_allclose(layer(x), expected, rtol=2e-11, atol=2e-12)
                    self.assertEqual(layer._graph_operation.experimental_get_tracing_count(), 1)
                    before = layer(x)
                    supplied = [tf.identity(v) + .03 if v is layer.kernel else tf.identity(v)
                                for v in layer.trainable_variables]
                    wanted, _ = layer.stateless_call(supplied, layer.non_trainable_variables, x)
                    self.assertGreater(float(tf.reduce_max(tf.abs(wanted - before))), 1e-7)
                    np.testing.assert_array_equal(layer(x), before)
                    layer.kernel.assign_add(tf.ones_like(layer.kernel) * .03)
                    layer.shift_parameter.assign_add(tf.ones_like(layer.shift_parameter) * .19)
                    legacy.set_weights(layer.get_weights())
                    with TensorFlowSharedStrideTests.unit_stride_reference():
                        expected = legacy(x)
                    np.testing.assert_allclose(layer(x), expected, rtol=2e-11, atol=2e-12)
                    if boundary == "zeros" and padding == "same":
                        inputs = tf.keras.Input((4, None, None), dtype="float64")
                        model = tf.keras.Model(inputs, layer(inputs) + layer(inputs))
                        self.assertEqual(len(model.losses), 1)
                        np.testing.assert_allclose(sonn.model_regularization_loss(model),
                                                   layer.losses[0], rtol=1e-12)
                        with tempfile.TemporaryDirectory() as directory:
                            path = os.path.join(directory, "strided.keras")
                            model.save(path)
                            restored = tf.keras.models.load_model(path)
                            np.testing.assert_allclose(restored(x), model(x), rtol=2e-11, atol=2e-12)
                    layer.trainable = False
                    with tf.GradientTape() as tape:
                        tape.watch(x)
                        loss = tf.reduce_sum(layer(x, training=False) ** 2)
                    self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(tape.gradient(loss, x)))))

    def test_precision_and_nested_derivatives(self):
        for groups in (1, 2):
            for dtype in ("float32", "mixed_float16", "mixed_bfloat16"):
                for precision in ("stable", "autocast"):
                    with self.subTest(groups=groups, dtype=dtype, precision=precision):
                        options = dict(filters=4, kernel_size=2, strides=2, groups=groups,
                                       padding="same", q=3, shift_sharing="shared",
                                       dtype=dtype, conv_precision=precision, initialization="all_orders")
                        layer = sonn.SuperONN2D(**options, recompute=True)
                        legacy = sonn.SuperONN2D(**options, recompute=False)
                        layer.build((None, None, None, 2))
                        legacy.build((None, None, None, 2))
                        layer.shift_parameter.assign(tf.ones_like(layer.shift_parameter) * .23)
                        legacy.set_weights(layer.get_weights())
                        x = tf.reshape(tf.linspace(-.5, .8, 70), (1, 5, 7, 2))
                        def evaluate(core):
                            with tf.GradientTape() as tape:
                                tape.watch(x)
                                y = core(x)
                                loss = tf.reduce_sum(tf.cast(y, tf.float32) ** 2)
                            return y, tape.gradient(loss, (x, *core.trainable_variables))
                        with TensorFlowSharedStrideTests.unit_stride_reference():
                            expected, expected_grads = evaluate(legacy)
                        actual, gradients = evaluate(layer)
                        tolerance = .02 if dtype == "mixed_bfloat16" else .003 if dtype == "mixed_float16" else 2e-6
                        np.testing.assert_allclose(tf.cast(actual, tf.float32),
                                                   tf.cast(expected, tf.float32),
                                                   rtol=tolerance, atol=tolerance)
                        for got, wanted in zip(gradients, expected_grads):
                            self.assertIsNotNone(got)
                            self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(got))))
                            np.testing.assert_allclose(got, wanted, rtol=tolerance, atol=tolerance)
        for groups in (1, 2):
            case = self.case(2, groups, 4, stride=(2, 2))
            arrays = case.data()
            layer = make(case, recompute=False)
            assign(layer, arrays)
            legacy = make(case, recompute=False)
            assign(legacy, arrays)
            x = tf.constant(tensor(arrays[0]))
            def nested(core):
                with tf.GradientTape() as outer:
                    outer.watch(x)
                    with tf.GradientTape() as inner:
                        inner.watch(x)
                        loss = tf.reduce_sum(core(x) ** 2)
                    penalty = tf.reduce_sum(inner.gradient(loss, x) ** 2)
                return outer.gradient(penalty, (x, *core.trainable_variables))
            with TensorFlowSharedStrideTests.unit_stride_reference():
                wanted = nested(legacy)
            for got, expected in zip(nested(layer), wanted):
                self.assertIsNotNone(got)
                np.testing.assert_allclose(got, expected, rtol=2e-10, atol=2e-11)

    def test_strided_factories_wrappers_and_shared_updates(self):
        for factory, arguments in ((sonn.depthwise_conv1d, (3,)),
                                   (sonn.separable_conv1d, (3, 3))):
            with self.subTest(factory=factory.__name__):
                settings = dict(q=3, shift_sharing="shared", input_bound=.7, max_shift=.8,
                                recompute=True, initialization="all_orders",
                                order_regularization=1e-4, include_constant=True)
                layer = factory(*arguments, operator="superonn", superonn=settings,
                                depth_multiplier=2, strides=2, padding=1,
                                data_format="channels_first", dtype="float64")
                x = tf.reshape(tf.linspace(tf.constant(-.4, tf.float64), .6, 36), (2, 2, 9))
                layer.build(x.shape)
                core = layer.operator if hasattr(layer, "operator") else layer.depthwise.operator
                self.assertTrue(core.recompute)
                self.assertEqual(core.strides, 2)
                case = TensorFlowSharedStrideTests.case(2, 2, 4, 2, 1)
                case.constant = True
                arrays = case.data()
                arrays = (np.asarray(x), *arrays[1:])
                if core.bias is None:
                    arrays[3].fill(0)
                core.kernel.assign(arrays[1].transpose(3, 1, 2, 0))
                core.shift_parameter.assign(arrays[2].reshape(core.shift_parameter.shape))
                if core.bias is not None:
                    core.bias.assign(arrays[3])
                expected = forward(case, *arrays)
                if hasattr(layer, "pointwise"):
                    expected = np.einsum("ncl,co->nol", expected, layer.pointwise.kernel.numpy()[0])
                    expected += layer.pointwise.bias.numpy()[None, :, None]
                np.testing.assert_allclose(layer(x), expected, rtol=2e-11, atol=2e-12)
                config = layer.get_config()
                config["superonn"] = {**config["superonn"], "recompute": False}
                legacy = type(layer).from_config(config)
                legacy.build(x.shape)
                legacy.set_weights(layer.get_weights())
                with TensorFlowSharedStrideTests.unit_stride_reference():
                    with tf.GradientTape() as tape:
                        tape.watch(x)
                        loss = tf.reduce_sum(legacy(x) ** 2)
                    wanted = tape.gradient(loss, (x, *legacy.trainable_variables))
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    loss = tf.reduce_sum(layer(x) ** 2)
                for got, reference in zip(tape.gradient(loss, (x, *layer.trainable_variables)), wanted):
                    self.assertIsNotNone(got)
                    np.testing.assert_allclose(got, reference, rtol=2e-10, atol=2e-11)
                inputs = tf.keras.Input((2, None), dtype="float64")
                model = tf.keras.Model(inputs, layer(inputs) + layer(inputs))
                self.assertEqual(len(model.losses), 1)
                optimizer = tf.keras.optimizers.SGD(.001)
                previous = layer(x).numpy()
                for _ in range(2):
                    with tf.GradientTape() as tape:
                        loss = tf.reduce_mean(model(x) ** 2) + sonn.model_regularization_loss(model)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    self.assertTrue(all(g is not None and bool(tf.reduce_all(tf.math.is_finite(g)))
                                        for g in gradients))
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                self.assertGreater(float(np.max(np.abs(layer(x).numpy() - previous))), 1e-7)
                with tempfile.TemporaryDirectory() as directory:
                    path = os.path.join(directory, "wrapper.keras")
                    model.save(path)
                    restored = tf.keras.models.load_model(path)
                    np.testing.assert_allclose(restored(x), model(x), rtol=2e-11, atol=2e-12)

    def test_boundary_gradients_at_exact_edge_samples(self):
        for boundary in ("zeros", "border", "reflection", "circular"):
            for groups in (1, 2):
                with self.subTest(boundary=boundary, groups=groups):
                    case = self.case(2, groups, 4, stride=(2, 2))
                    case.sizes, case.boundary = (1, 3), boundary
                    arrays = case.data()
                    arrays[2].fill(0.)
                    layer, legacy = make(case), make(case, recompute=False)
                    assign(layer, arrays)
                    assign(legacy, arrays)
                    with TensorFlowSharedStrideTests.unit_stride_reference():
                        expected, wanted = self.evaluate(legacy, case, arrays)
                    actual, gradients = self.evaluate(layer, case, arrays)
                    np.testing.assert_allclose(actual, forward(case, *arrays),
                                               rtol=2e-11, atol=2e-12)
                    for got, reference in zip(gradients, wanted):
                        np.testing.assert_allclose(got, reference, rtol=2e-10, atol=2e-11)

    def test_mixed_precision_stride_dilation_fallbacks(self):
        specifications = ((1, 1, (2,), (2,)), (1, 2, (2,), (2,)),
                          (2, 1, (2, 3), (2, 1)), (2, 2, (2, 2), (1, 2)),
                          (2, 2, (2, 3), (1, 1)))
        for rank, groups, strides, dilation in specifications:
            case = self.case(2, groups, 4, rank=rank, stride=strides, dilation=dilation)
            arrays = case.data()
            for policy in ("mixed_float16", "mixed_bfloat16"):
                with self.subTest(rank=rank, groups=groups, policy=policy, dilation=dilation):
                    ordinary = make(case, dtype=policy, recompute=False)
                    ordinary.conv_precision = "autocast"
                    assign(ordinary, arrays)
                    x = tf.constant(tensor(arrays[0]), tf.float32)
                    def evaluate(core):
                        with tf.GradientTape() as tape:
                            tape.watch(x)
                            y = core(x)
                            loss = tf.reduce_sum(tf.cast(y, tf.float32) ** 2)
                        return y, tape.gradient(loss, (x, *core.trainable_variables))
                    with TensorFlowSharedStrideTests.unit_stride_reference():
                        expected, wanted = evaluate(ordinary)
                    for recompute in (False, True):
                        layer = type(ordinary).from_config(
                            {**ordinary.get_config(), "recompute": recompute})
                        layer.build((None, *((None,) * rank), case.channels))
                        layer.set_weights(ordinary.get_weights())
                        actual, gradients = evaluate(layer)
                        tolerance = .02 if policy == "mixed_bfloat16" else .003
                        np.testing.assert_allclose(
                            canonical(tf.cast(actual, tf.float32)), forward(case, *arrays),
                            rtol=tolerance, atol=tolerance)
                        np.testing.assert_allclose(tf.cast(actual, tf.float32),
                                                   tf.cast(expected, tf.float32),
                                                   rtol=tolerance, atol=tolerance)
                        for got, reference in zip(gradients, wanted):
                            self.assertIsNotNone(got)
                            self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(got))))
                            np.testing.assert_allclose(got, reference,
                                                       rtol=tolerance, atol=tolerance)
