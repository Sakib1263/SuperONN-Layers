"""Differentiable sampling. Coordinates follow tensor spatial order (L or H,W)."""
from __future__ import annotations

import math
import torch


def sample_nd(x, coordinates, ndim, mode="zeros"):
    """Linear sampling; reflection is about integer centers, as in torch pad."""
    lead = x.shape[:-ndim]
    spatial = x.shape[-ndim:]
    axis = len(lead)
    output = coordinates.shape[axis + 1:]
    if coordinates.shape[:axis] != lead or coordinates.shape[axis] != ndim:
        raise ValueError("Sampling coordinates do not match input leading dimensions")
    coords = list(coordinates.unbind(axis))
    if mode == "replicate":
        coords = [c.clamp(0, n - 1) for c, n in zip(coords, spatial)]
    elif mode == "reflect":
        coords = [torch.zeros_like(c) if n == 1 else
                  (n - 1) - (torch.remainder(c, 2 * (n - 1)) - (n - 1)).abs()
                  for c, n in zip(coords, spatial)]
    elif mode == "circular":
        coords = [torch.remainder(c, n) for c, n in zip(coords, spatial)]
    elif mode != "zeros":
        raise ValueError(f"Unknown boundary mode: {mode}")
    lower = [torch.floor(c) for c in coords]
    fractions = [c - lo for c, lo in zip(coords, lower)]
    strides = [math.prod(spatial[d + 1:]) for d in range(ndim)]
    flat = x.reshape(*lead, math.prod(spatial))
    result = torch.zeros((*lead, *output), device=x.device, dtype=x.dtype)
    # Static integer loops are captured by torch.compile on the minimum
    # supported PyTorch as well as current releases. Preserve corner order.
    for corner_index in range(1 << ndim):
        corner = tuple((corner_index >> (ndim - 1 - axis)) & 1 for axis in range(ndim))
        index = torch.zeros_like(coords[0], dtype=torch.long)
        valid = torch.ones_like(coords[0], dtype=torch.bool)
        coefficient = torch.ones_like(coords[0])
        for lo, fraction, bit, size, step in zip(lower, fractions, corner, spatial, strides):
            cell = lo.to(torch.long) + bit
            coefficient = coefficient * (fraction if bit else 1 - fraction)
            if mode == "circular":
                cell = cell.remainder(size)
            else:
                valid = valid & (cell >= 0) & (cell < size)
            index = index + cell.clamp(0, size - 1) * step
        values = flat.gather(-1, index.reshape(*lead, math.prod(output))).reshape(*lead, *output)
        result = result + values * coefficient * valid.to(x.dtype)
    return result


def polynomial_pack(x, q, groups=1, include_constant=False, channel_major=True):
    """Pack powers within ORIGINAL channel groups; never split a neuron's powers."""
    n, c = x.shape[:2]
    if c % groups:
        raise ValueError("Input channels must be divisible by groups")
    z = x.reshape(n, groups, c // groups, *x.shape[2:])
    start = 0 if include_constant else 1
    powers = torch.stack([z ** p for p in range(start, q + 1)], dim=3 if channel_major else 2)
    return powers.reshape(n, c * (q + int(include_constant)), *x.shape[2:])
