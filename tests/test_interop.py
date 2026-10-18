"""Optional explicit weight conversion; run in its own CPU process."""
import importlib.util
import unittest
import numpy as np

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch", "tensorflow"))
if AVAILABLE:
    import torch
    if importlib.util.find_spec("triton") is not None:
        import triton
    import tensorflow as tf
    from superonn.interop import to_tensorflow, to_pytorch

from .reference import Case, forward, target_for


@unittest.skipUnless(AVAILABLE, "both backends are required only for conversion checks")
class InteropTests(unittest.TestCase):
    def test_round_trip_outputs_and_gradients(self):
        from .test_torch import make, assign
        from .test_tensorflow import tensor, canonical
        for rank, transposed in ((1, False), (2, False), (1, True), (2, True)):
            for kind in ("grouped", "shared"):
                with self.subTest(rank=rank, transpose=transposed, kind=kind):
                    case = Case(rank, transposed, kind)
                    arrays = case.data()
                    pt = make(case)
                    assign(pt, arrays)
                    native_tf = to_tensorflow(pt)
                    clone = to_pytorch(native_tf)
                    x = torch.tensor(arrays[0], requires_grad=True)
                    y = pt(x)
                    torch.testing.assert_close(clone(x), y, rtol=2e-11, atol=2e-12)
                    tx = tf.constant(tensor(arrays[0]))
                    with tf.GradientTape() as tape:
                        tape.watch(tx)
                        ty = native_tf(tx)
                        tloss = tf.reduce_sum(ty * tensor(target_for(case, arrays)))
                    tg = tape.gradient(tloss, (tx, native_tf.kernel, native_tf.shift_parameter))
                    pg = torch.autograd.grad((y * torch.tensor(target_for(case, arrays))).sum(),
                                             (x, pt.weight, pt.raw_shifts))
                    np.testing.assert_allclose(canonical(ty), forward(case, *arrays), rtol=2e-11, atol=2e-12)
                    grads = (canonical(tg[0]), tg[1].numpy().transpose(rank + 2, rank, rank + 1, *range(rank)),
                             tg[2].numpy().reshape(arrays[2].shape))
                    for a, b in zip(pg, grads):
                        np.testing.assert_allclose(a.detach().numpy(), b, rtol=3e-10, atol=3e-11)
