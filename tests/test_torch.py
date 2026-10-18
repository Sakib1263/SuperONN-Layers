"""Native PyTorch checks; no TensorFlow import required."""
import importlib.util
import io
import unittest
import numpy as np

if importlib.util.find_spec("torch"):
    import torch
    import superonn.pytorch as sonn
else:
    torch = None

from .reference import Case, forward, check_gradients, target_for

CLASSES = ("SuperONN1d", "SuperONN2d", "SuperONNTranspose1d", "SuperONNTranspose2d")


def make(case, recompute=True, dtype=None, device=None):
    cls = getattr(sonn, CLASSES[case.rank - 1 + 2 * case.transpose])
    return cls(case.channels, case.outputs, case.kernel, stride=case.stride,
               padding=case.padding, dilation=case.dilation, groups=case.groups,
               q=case.q, shift_sharing=case.sharing, max_shift=case.limit,
               input_bound=case.bound, include_constant=case.constant,
               output_padding=case.output_padding, output_chunk_size=4 if case.kind == "grouped" else 2,
               recompute=recompute, dtype=dtype or torch.float64, device=device,
               padding_mode={"border": "replicate", "reflection": "reflect"}.get(case.boundary, case.boundary))


def assign(layer, arrays):
    with torch.no_grad():
        for destination, value in ((layer.weight, arrays[1]), (layer.raw_shifts, arrays[2]), (layer.bias, arrays[3])):
            destination.copy_(torch.as_tensor(value))


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TorchCoreTests(unittest.TestCase):
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
                        x = torch.tensor(arrays[0], requires_grad=True)
                        y = layer(x)
                        loss = (y * torch.tensor(target_for(case, arrays))).sum()
                        grads = [g.detach().numpy() for g in torch.autograd.grad(loss, (x, layer.weight, layer.raw_shifts))]
                        np.testing.assert_allclose(y.detach().numpy(), forward(case, *arrays), rtol=2e-11, atol=2e-12)
                        check_gradients(self, case, arrays, grads, exhaustive=kind == "full")
                        results.append(grads)
                for a, b in zip(*results):
                    np.testing.assert_allclose(a, b, rtol=2e-10, atol=2e-11)

    def test_state_functional_frozen_meta_and_autocast(self):
        for rank, transposed in ((1, False), (2, False), (1, True), (2, True)):
            with self.subTest(rank=rank, transpose=transposed):
                case = Case(rank, transposed)
                arrays = case.data()
                layer = make(case)
                assign(layer, arrays)
                x = torch.tensor(arrays[0], requires_grad=True)
                original = layer(x).detach()
                names = set(layer.state_dict())
                replacement = dict(layer.named_parameters())
                replacement["weight"] = layer.weight + .05
                supplied = torch.func.functional_call(layer, replacement, (x,))
                self.assertGreater(float((supplied.detach() - original).abs().max()), 1e-7)
                torch.testing.assert_close(layer(x), original)
                with torch.no_grad():
                    layer.weight.add_(.05)
                torch.testing.assert_close(layer(x), supplied)
                buffer = io.BytesIO()
                torch.save(layer.state_dict(), buffer)
                buffer.seek(0)
                clone = make(case)
                clone.load_state_dict(torch.load(buffer, weights_only=True))
                torch.testing.assert_close(clone(x), supplied)
                self.assertEqual(set(clone.state_dict()), names)
                clone.eval().requires_grad_(False)
                self.assertIsNotNone(torch.autograd.grad(clone(x).sum(), x)[0])
                meta = make(case, device="meta")
                meta.to_empty(device="cpu")
                meta.reset_parameters()
                self.assertTrue(torch.isfinite(meta(x)).all())
                single = make(case, dtype=torch.float32)
                assign(single, arrays)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    y = single(x.float())
                    loss = y.float().square().sum()
                self.assertEqual(y.dtype, torch.bfloat16)
                torch.testing.assert_close(y.float(), single(x.float()), rtol=.02, atol=.002)
                self.assertTrue(all(torch.isfinite(g).all() for g in
                                    torch.autograd.grad(loss, [x, *single.parameters()])))

    def test_q1_native_convolution_and_compile_capture(self):
        for rank, transposed in ((1, False), (2, False), (1, True), (2, True)):
            cls = getattr(sonn, CLASSES[rank - 1 + 2 * transposed])
            layer = cls(2, 4, 3, groups=2, padding=1, q=1, max_shift=0,
                        dtype=torch.float64, recompute=True)
            x = torch.randn((1, 2, *((4,) * rank)), dtype=torch.float64, requires_grad=True)
            weight = layer.weight[:, :, 0]
            op = getattr(torch.nn.functional, ("conv_transpose" if transposed else "conv") + str(rank) + "d")
            if transposed:
                weight = weight.reshape(2, 2, 1, *((3,) * rank)).transpose(1, 2).reshape(2, 2, *((3,) * rank))
            expected = op(x, weight, layer.bias, padding=1, groups=2)
            torch.testing.assert_close(layer(x), expected)
            # Dynamo full-graph capture only: not a claim about Inductor kernels.
            compiled = torch.compile(layer, backend="eager", fullgraph=True)
            torch.testing.assert_close(compiled(x), expected)
            torch.testing.assert_close(torch.autograd.grad(compiled(x).sum(), x)[0],
                                       torch.autograd.grad(expected.sum(), x)[0])

    def test_checkpoint_policy_and_rng(self):
        from unittest.mock import patch
        import superonn.layers as implementation
        for rank, transposed in ((1, False), (2, False), (1, True), (2, True)):
            case = Case(rank, transposed)
            arrays = case.data()
            layer = make(case)
            assign(layer, arrays)
            x = torch.tensor(arrays[0], requires_grad=True)
            state = torch.random.get_rng_state().clone()
            with patch.object(implementation, "checkpoint", wraps=implementation.checkpoint) as wrapper:
                training = layer(x)
                torch.autograd.grad(training.sum(), (x, *layer.parameters()))
                called = wrapper.call_count
                self.assertGreater(called, 0)
                layer.eval().requires_grad_(False)
                evaluation = layer(x)
                torch.autograd.grad(evaluation.sum(), x)
                self.assertEqual(wrapper.call_count, called)
                torch.testing.assert_close(evaluation, training)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), state))
