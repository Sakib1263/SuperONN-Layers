"""Independent scalar NumPy oracle: no runtime sampler or backend imports.

Canonical tensors use [N,C,*spatial], [O,I,order,*kernel] and raw offsets.
Numeric symmetric padding makes every case equivalent between backends.
"""
from dataclasses import dataclass
import itertools
import numpy as np


@dataclass
class Case:
    rank: int
    transpose: bool
    kind: str = "full"

    def __post_init__(self):
        self.channels = 1 if self.kind == "full" else 2
        self.groups = 2 if self.kind in ("grouped", "depthwise") else 1
        self.outputs = {"full": 3, "grouped": 6, "shared": 3, "depthwise": 2}[self.kind]
        self.sharing = "shared" if self.kind == "shared" else "full"
        self.sizes = (4,) if self.rank == 1 else (3, 3)
        self.kernel = (2,) * self.rank
        self.stride = (2,) * self.rank
        self.dilation = (1 if self.transpose else 2,) * self.rank
        self.padding = (1,) * self.rank
        self.output_padding = (1,) * self.rank if self.transpose else (0,) * self.rank
        self.q = 2
        self.constant = not self.transpose
        self.bound = .7 if self.kind == "full" else None
        self.limit = .8
        self.boundary = "zeros"

    def data(self):
        rng = np.random.default_rng(310 + self.rank + 10 * self.transpose)
        x = rng.normal(0, .3, (1, self.channels, *self.sizes))
        w = rng.normal(0, .15, (self.outputs, self.channels // self.groups,
                                      self.q + self.constant, *self.kernel))
        shape = ((self.channels, self.rank) if self.sharing == "shared" else
                 (self.outputs, self.channels // self.groups, self.rank))
        # Strictly fractional coordinates away from floor/clipping kinks.
        raw = rng.uniform(.13, .28, shape)
        bias = rng.normal(0, .05, self.outputs)
        return x, w, raw, bias


def interpolate(source, point, boundary):
    point = list(point)
    for d, size in enumerate(source.shape):
        if boundary == "border":
            point[d] = min(max(point[d], 0.), size - 1.)
        elif boundary == "reflection":
            a = point[d] % (2 * (size - 1)) if size > 1 else 0.
            point[d] = min(a, 2 * (size - 1) - a)
        elif boundary == "circular":
            point[d] %= size
    lo = np.floor(point).astype(int)
    frac = np.asarray(point) - lo
    result = 0.
    for bits in itertools.product((0, 1), repeat=source.ndim):
        idx = lo + bits
        factor = np.prod([f if b else 1 - f for f, b in zip(frac, bits)])
        if boundary == "zeros" and any(i < 0 or i >= n for i, n in zip(idx, source.shape)):
            continue
        idx = tuple(i % n if boundary == "circular" else min(max(i, 0), n - 1)
                    for i, n in zip(idx, source.shape))
        result += source[idx] * factor
    return result


def forward(case, x, weights, raw, bias):
    values = np.tanh(x / case.bound) if case.bound is not None else x
    offsets = case.limit * np.tanh(raw)
    effective = tuple((k - 1) * d + 1 for k, d in zip(case.kernel, case.dilation))
    if case.transpose:
        sizes = tuple((n - 1) * s + 1 for n, s in zip(x.shape[2:], case.stride))
        lattice = np.zeros((*x.shape[:2], *sizes))
        lattice[(slice(None), slice(None), *(slice(None, None, s) for s in case.stride))] = values
        values = lattice
        weights = np.flip(weights, tuple(range(3, 3 + case.rank)))
        offsets = offsets * case.stride
        counts = tuple((n - 1) * s - 2 * p + k + op for n, s, p, k, op in
                       zip(x.shape[2:], case.stride, case.padding, effective, case.output_padding))
        origins = tuple(p - (k - 1) for p, k in zip(case.padding, effective))
        steps = (1,) * case.rank
    else:
        counts = tuple((n + 2 * p - k) // s + 1 for n, p, k, s in
                       zip(x.shape[2:], case.padding, effective, case.stride))
        origins = tuple(-p for p in case.padding)
        steps = case.stride
    result = np.zeros((x.shape[0], case.outputs, *counts))
    ci = case.channels // case.groups
    for n, out in itertools.product(range(x.shape[0]), range(case.outputs)):
        group = out // (case.outputs // case.groups)
        for pos in itertools.product(*(range(c) for c in counts)):
            total = bias[out]
            for i, tap in itertools.product(range(ci), itertools.product(*(range(k) for k in case.kernel))):
                channel = group * ci + i
                delta = offsets[channel] if case.sharing == "shared" else offsets[out, i]
                point = tuple(t * s + o + r * d + shift for t, s, o, r, d, shift in
                              zip(pos, steps, origins, tap, case.dilation, delta))
                sampled = interpolate(values[n, channel], point, case.boundary)
                for degree in range(case.q + case.constant):
                    total += weights[(out, i, degree, *tap)] * sampled ** (degree + (not case.constant))
            result[(n, out, *pos)] = total
    return result


def check_gradients(test, case, arrays, gradients, exhaustive=False):
    """Double precision central differences, with no coordinate at a kink."""
    x, w, raw, bias = arrays
    target = np.random.default_rng(19).normal(size=forward(case, *arrays).shape)
    rng = np.random.default_rng(23)
    for which, grad in enumerate(gradients):
        value = arrays[which]
        directions = []
        if exhaustive:
            for index in np.ndindex(value.shape):
                direction = np.zeros_like(value)
                direction[index] = 1.
                directions.append(direction)
        else:
            directions = [rng.normal(size=value.shape) for _ in range(2)]
        for direction in directions:
            epsilon = 1e-6
            plus, minus = list(arrays), list(arrays)
            plus[which] = value + epsilon * direction
            minus[which] = value - epsilon * direction
            expected = np.sum((forward(case, *plus) - forward(case, *minus)) * target) / (2 * epsilon)
            actual = np.sum(grad * direction)
            test.assertAlmostEqual(actual, expected, delta=2e-7 + 2e-6 * abs(expected))
    return target


def target_for(case, arrays):
    return np.random.default_rng(19).normal(size=forward(case, *arrays).shape)
