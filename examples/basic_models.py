"""Compose small generators, regressors and discriminators with SuperONN layers.

Run after installing the package: python examples/basic_models.py
"""
import torch
from torch import nn

from superonn import SuperONN1d, SuperONNTranspose1d


def generator(mode="learned", sharing="full"):
    options = dict(
        q=3, max_shift=2, shift_mode=mode, shift_sharing=sharing,
        input_bound=1, output_chunk_size=4, recompute=True,
    )
    return nn.Sequential(
        SuperONN1d(1, 8, 5, padding=2, **options),
        nn.GELU(),
        SuperONN1d(8, 8, 4, stride=2, padding=1, groups=8, **options),
        nn.GELU(),
        SuperONNTranspose1d(8, 1, 4, stride=2, padding=1, **options),
        nn.Tanh(),
    )


def regressor(mode="learned"):
    return nn.Sequential(
        SuperONN1d(1, 8, 7, stride=2, padding=3, q=3, shift_mode=mode, input_bound=1),
        nn.GELU(),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(8, 1),
    )


def discriminator(mode="learned"):
    return nn.Sequential(
        SuperONN1d(1, 8, 7, stride=2, padding=3, q=3, shift_mode=mode, input_bound=1),
        nn.LeakyReLU(0.2),
        SuperONN1d(8, 1, 3, padding=1, q=1, max_shift=0),
    )


if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.manual_seed(37)
    inputs = torch.randn(2, 1, 256)
    for mode in ("random_fixed", "learned"):
        for factory in (generator, regressor, discriminator):
            model = factory(mode)
            output = model(inputs)
            output.mean().backward()
            print(f"{factory.__name__}: mode={mode}, output_shape={tuple(output.shape)}")
