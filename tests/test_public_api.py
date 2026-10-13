"""Public PyTorch API, compiler, precision and checkpoint regression tests."""
import io
import pytest
import torch
from superonn import SuperONN1d, SuperONN2d, SuperONNTranspose1d, SuperONNTranspose2d, SuperONNConfig

torch.set_num_threads(2)


@pytest.mark.parametrize("cls",(SuperONN1d,SuperONN2d,SuperONNTranspose1d,SuperONNTranspose2d))
@pytest.mark.parametrize("sharing",("shared","full"))
def test_fullgraph_backward_and_ordinary_checkpoint(cls,sharing):
    torch.manual_seed(37)
    a = cls(2,4,3,padding=1,q=3,shift_sharing=sharing,shift_init="random",initialization="all_orders")
    b = cls(2,4,3,padding=1,q=3,shift_sharing=sharing,shift_init="random",initialization="all_orders")
    b.load_state_dict(a.state_dict())
    b.compile(backend="eager",fullgraph=True,dynamic=False)
    shape = (1,2,13) if cls.spatial_rank == 1 else (1,2,5,7)
    x = torch.rand(shape,requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected,actual = a(x),b(y)
    torch.testing.assert_close(actual,expected)
    expected.square().mean().backward();actual.square().mean().backward()
    torch.testing.assert_close(x.grad,y.grad)
    for va,vb in zip(a.parameters(),b.parameters()):
        assert va.grad is not None and vb.grad is not None
        torch.testing.assert_close(va.grad,vb.grad)
    assert set(a.state_dict()) == set(b.state_dict())
    buffer=io.BytesIO();torch.save(b.state_dict(),buffer);buffer.seek(0)
    a.load_state_dict(torch.load(buffer,weights_only=True))


@pytest.mark.parametrize("groups",(1,4))
@pytest.mark.parametrize("precision",("stable","autocast"))
def test_cpu_bfloat16_native_convolution_policy(groups,precision):
    m=SuperONN1d(4,4,3,padding=1,q=3,groups=groups,shift_sharing="shared",conv_precision=precision)
    x=torch.rand(2,4,17,requires_grad=True)
    reference=m(x).detach()
    with torch.autocast("cpu",dtype=torch.bfloat16):
        actual=m(x)
        actual.float().square().mean().backward()
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.float(),reference,atol=.008,rtol=.03)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_profile_1400_bin_decoder_and_invalid_precision():
    options=SuperONNConfig().layer_kwargs()
    encoder=SuperONN1d(1,4,4,stride=2,padding=1,**options)
    decoder=SuperONNTranspose1d(4,1,4,stride=2,padding=1,**options)
    assert decoder(encoder(torch.rand(2,1,1400))).shape == (2,1,1400)
    with pytest.raises(ValueError,match="conv_precision"):
        SuperONN1d(1,4,3,conv_precision="fp8")
