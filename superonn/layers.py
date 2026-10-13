"""SuperONNs with conventional convolution connectivity and explicit shifts.

Weights have shape (Cout, Cin/groups, Q, *kernel), unlike legacy flattened
weights. Learned shifts are bounded by max_shift*tanh(raw_shifts). Positive
shifts read larger input coordinates. Powers are evaluated AFTER sampling.
"""
from __future__ import annotations

import itertools
import math
from numbers import Integral, Real

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .functional import polynomial_pack, sample_nd


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _tuple(value, ndim, name, minimum=1):
    seq = (value,) * ndim if isinstance(value, Integral) else tuple(value)
    if len(seq) != ndim:
        raise ValueError(f"{name} must have {ndim} entries")
    return tuple(_integer(v, name, minimum) for v in seq)


class _SuperONN(nn.Module):
    spatial_rank = 1
    transposed = False

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode="zeros", device=None,
                 dtype=None, *, q=3, shift_mode="learned", shift_sharing="full",
                 max_shift=2.0, shift_init=None, output_chunk_size=4,
                 recompute=True, initialization="linear", input_bound=None,
                 include_constant=False, output_padding=0,
                 order_regularization=0.0, order_decay=1.5, conv_precision="stable"):
        super().__init__()
        d = self.spatial_rank
        self.in_channels = _integer(in_channels, "in_channels")
        self.out_channels = _integer(out_channels, "out_channels")
        self.groups = _integer(groups, "groups")
        if in_channels % groups or out_channels % groups:
            raise ValueError("Both in_channels and out_channels must be divisible by groups")
        self.kernel_size = _tuple(kernel_size, d, "kernel_size")
        self._kernel_taps = tuple(itertools.product(*(range(k) for k in self.kernel_size)))
        self.stride = _tuple(stride, d, "stride")
        self.dilation = _tuple(dilation, d, "dilation")
        if isinstance(padding, str):
            if self.transposed or padding not in ("same", "valid"):
                raise ValueError("Forward padding strings must be 'same' or 'valid'")
            if padding == "same" and any(s != 1 for s in self.stride):
                raise ValueError("padding='same' requires stride=1, as in torch Conv")
            self.padding = str(padding)
        else:
            self.padding = _tuple(padding, d, "padding", 0)
        if padding_mode not in ("zeros", "reflect", "replicate", "circular"):
            raise ValueError("Invalid padding_mode")
        if self.transposed and padding_mode != "zeros":
            raise ValueError("Transpose layers support padding_mode='zeros'")
        self.padding_mode = str(padding_mode)
        self.output_padding = _tuple(output_padding, d, "output_padding", 0)
        if not self.transposed and any(self.output_padding):
            raise ValueError("output_padding belongs to transpose layers")
        if any(op >= max(s, dil) for op, s, dil in zip(self.output_padding, self.stride, self.dilation)):
            raise ValueError("output_padding must be smaller than stride or dilation")
        self.q = _integer(q, "q")
        self.include_constant = bool(include_constant)
        if self.transposed and include_constant:
            raise ValueError("A per-tap constant is undefined on a sparse transpose lattice; use bias")
        self.orders = self.q + int(self.include_constant)
        if shift_mode not in ("learned", "random_fixed"):
            raise ValueError("shift_mode must be 'learned' or 'random_fixed'")
        if shift_sharing not in ("shared", "full"):
            raise ValueError("shift_sharing must be 'shared' or 'full'")
        self.shift_mode, self.shift_sharing = str(shift_mode), str(shift_sharing)
        limits = (max_shift,) * d if isinstance(max_shift, Real) else tuple(max_shift)
        if len(limits) != d or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or v < 0 for v in limits):
            raise ValueError("max_shift must contain finite nonnegative numbers")
        self.max_shift = tuple(float(v) for v in limits)
        shift_init = shift_init or ("random" if shift_mode == "random_fixed" else "zeros")
        if shift_init not in ("zeros", "random", "random_int", "half", "half_int"):
            raise ValueError("Unknown shift_init")
        if initialization not in ("linear", "all_orders", "tanh", "selu"):
            raise ValueError("Unknown initialization")
        if input_bound is not None and (isinstance(input_bound, bool) or not isinstance(input_bound, Real) or not math.isfinite(input_bound) or input_bound <= 0):
            raise ValueError("input_bound must be positive and finite")
        self.shift_init, self.initialization = str(shift_init), str(initialization)
        self.input_bound = None if input_bound is None else float(input_bound)
        self.output_chunk_size = _integer(output_chunk_size, "output_chunk_size")
        self.recompute = bool(recompute)
        if conv_precision not in ("stable", "autocast"):
            raise ValueError("conv_precision must be stable or autocast")
        self.conv_precision = conv_precision
        if isinstance(order_regularization, bool) or not isinstance(order_regularization, Real) or not math.isfinite(order_regularization) or order_regularization < 0:
            raise ValueError("order_regularization must be finite and nonnegative")
        if isinstance(order_decay, bool) or not isinstance(order_decay, Real) or not math.isfinite(order_decay) or order_decay < 1:
            raise ValueError("order_decay must be finite and >= 1")
        self.order_regularization, self.order_decay = float(order_regularization), float(order_decay)
        factory = {"device": device, "dtype": dtype}
        if dtype is not None and (not dtype.is_floating_point):
            raise ValueError("Only real floating-point dtypes are supported")
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, self.orders, *self.kernel_size, **factory))
        self.bias = nn.Parameter(torch.empty(out_channels, **factory)) if bias else None
        shape = (in_channels, d) if shift_sharing == "shared" else (out_channels, in_channels // groups, d)
        self.register_buffer("shift_limits", torch.tensor(self.max_shift, **factory))
        if shift_mode == "learned":
            self.raw_shifts = nn.Parameter(torch.empty(shape, **factory), requires_grad=any(self.max_shift))
            self.register_buffer("fixed_shifts", None)
        else:
            self.register_parameter("raw_shifts", None)
            self.register_buffer("fixed_shifts", torch.empty(shape, **factory))
        self.reset_parameters()

    @property
    def shifts(self):
        """Effective offsets in input bins (H,W for 2D), suitable for inspection."""
        if self.raw_shifts is None:
            return self.fixed_shifts
        # Compute bounds in FP32 under explicit half precision as well as AMP.
        value = self.raw_shifts.float() if self.raw_shifts.dtype in (torch.float16, torch.bfloat16) else self.raw_shifts
        return value.tanh() * value.new_tensor(self.max_shift)

    @torch.no_grad()
    def set_shifts_(self, offsets):
        # Preserve user-provided precision during validation; converting to
        # half/float32 first can turn a noninteger fixed offset into an integer.
        dtype = (offsets.dtype if isinstance(offsets, torch.Tensor) and offsets.dtype in
                 (torch.float32, torch.float64) else torch.float64)
        device = self.weight.device if self.weight.device.type != "mps" else "cpu"
        value = torch.as_tensor(offsets, device=device, dtype=dtype)
        limits = value.new_tensor(self.max_shift)
        if value.shape != self.shifts.shape:
            raise ValueError(f"Offsets must have shape {tuple(self.shifts.shape)}")
        if not torch.isfinite(value).all() or (value.abs() > limits).any():
            raise ValueError("Offsets exceed max_shift or contain nonfinite values")
        if self.raw_shifts is None:
            if not torch.equal(value, value.round()):
                raise ValueError("random_fixed offsets must be integers")
            self.fixed_shifts.copy_(value)
        else:
            ratio = value / limits.clamp_min(torch.finfo(value.dtype).tiny)
            eps = torch.finfo(self.raw_shifts.dtype).eps
            self.raw_shifts.copy_(torch.atanh(ratio.clamp(-1 + eps, 1 - eps)))
        return self

    def reset_parameters(self):
        with torch.no_grad():
            self.weight.zero_()
            first = int(self.include_constant)
            if self.initialization == "linear":
                nn.init.kaiming_uniform_(self.weight[:, :, first], a=math.sqrt(5))
            else:
                gain = nn.init.calculate_gain("selu" if self.initialization == "selu" else "tanh")
                nn.init.xavier_uniform_(self.weight.reshape(self.out_channels, -1, *self.kernel_size), gain=gain)
            if self.bias is not None:
                bound = 1 / math.sqrt(self.in_channels // self.groups * math.prod(self.kernel_size))
                nn.init.uniform_(self.bias, -bound, bound)
            shape = self.fixed_shifts.shape if self.raw_shifts is None else self.raw_shifts.shape
            shift_dtype = torch.float64 if self.weight.dtype == torch.float64 else torch.float32
            offsets = torch.empty(shape, device=self.weight.device, dtype=shift_dtype)
            for axis, limit in enumerate(self.max_shift):
                span = limit / 2 if self.shift_init.startswith("half") else limit
                if self.shift_init == "zeros":
                    offsets[..., axis].zero_()
                elif self.shift_mode == "random_fixed" or self.shift_init.endswith("int"):
                    offsets[..., axis].random_(-math.floor(span), math.floor(span) + 1)
                else:
                    offsets[..., axis].uniform_(-span, span)
            # Generated offsets already obey the initialization contract.
            # Avoid value-dependent validation here: meta tensors have no
            # values yet, and concrete device initialization needs no sync.
            if self.raw_shifts is None:
                self.fixed_shifts.copy_(offsets)
            else:
                limits = offsets.new_tensor(self.max_shift)
                ratio = offsets / limits.clamp_min(torch.finfo(offsets.dtype).tiny)
                eps = torch.finfo(self.raw_shifts.dtype).eps
                self.raw_shifts.copy_(torch.atanh(ratio.clamp(-1 + eps, 1 - eps)))
            self.shift_limits.copy_(self.shift_limits.new_tensor(self.max_shift))

    def _config(self):
        return {"format": 1, "layout": "output,input,order,kernel", "spatial_rank": self.spatial_rank,
                "transposed": self.transposed, "in_channels": self.in_channels,
                "out_channels": self.out_channels, "groups": self.groups, "q": self.q,
                "include_constant": self.include_constant, "kernel_size": self.kernel_size,
                "stride": self.stride, "padding": self.padding, "dilation": self.dilation,
                "padding_mode": self.padding_mode, "output_padding": self.output_padding,
                "shift_mode": self.shift_mode, "shift_sharing": self.shift_sharing,
                "max_shift": self.max_shift, "input_bound": self.input_bound,
                "order_regularization": self.order_regularization, "order_decay": self.order_decay}

    def get_extra_state(self):
        return self._config()

    def set_extra_state(self, state):
        if self._checkpoint_config(state) != self._config():
            raise RuntimeError("SuperONN checkpoint configuration differs from this layer; construct a matching layer")

    @staticmethod
    def _checkpoint_config(state):
        if not isinstance(state, dict):
            return None
        result = dict(state)
        # Version 0.1.0 did not expose order-aware penalties. Its mathematical
        # operator is compatible when the new training penalty is disabled.
        result.setdefault("order_regularization", 0.0)
        result.setdefault("order_decay", 1.5)
        return result

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        config = state_dict.get(prefix + "_extra_state")
        if config is None:
            error_msgs.append(prefix + "SuperONN checkpoint is missing configuration metadata; use an explicit conversion")
            return
        if config is not None and self._checkpoint_config(config) != self._config():
            error_msgs.append(prefix + "SuperONN checkpoint configuration differs; construct a matching layer")
            return  # prevent mutation of this layer's tensors on mismatch
        for name, expected in tuple(self._parameters.items()) + tuple(self._buffers.items()):
            incoming = state_dict.get(prefix + name)
            if expected is not None and incoming is not None:
                if not isinstance(incoming, torch.Tensor) or incoming.shape != expected.shape or not incoming.is_floating_point():
                    error_msgs.append(prefix + "SuperONN checkpoint contains incompatible tensor " + name)
                    return
        key = "fixed_shifts" if self.raw_shifts is None else "raw_shifts"
        offsets = state_dict.get(prefix + key)
        if isinstance(offsets, torch.Tensor):
            bad = not bool(torch.isfinite(offsets).all())
            if key == "fixed_shifts":
                limits = offsets.new_tensor(self.max_shift)
                bad = bad or bool((offsets.abs() > limits).any()) or not torch.equal(offsets, offsets.round())
            if bad:
                error_msgs.append(prefix + "SuperONN checkpoint contains invalid offsets")
                return
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def regularization_loss(self):
        """lambda*sum_p gamma**(p-1)*||W_p||²; caller adds it to task loss.

        Exact Frobenius-sum convention, no hidden mean or parameter projection.
        The optional constant basis, bias and offsets are excluded.
        """
        weight = self.weight if self.weight.dtype == torch.float64 else self.weight.float()
        if self.order_regularization == 0:
            return weight.new_zeros(())
        weights = weight[:, :, int(self.include_constant):]
        factors = self.order_decay ** torch.arange(self.q, device=weight.device, dtype=weight.dtype)
        shape = (1, 1, self.q) + (1,) * self.spatial_rank
        return self.order_regularization * (weights.square() * factors.reshape(shape)).sum()

    def _pads(self, spatial):
        if self.padding == "valid":
            return [(0, 0)] * self.spatial_rank
        if self.padding == "same":
            return [(total // 2, total - total // 2) for total in
                    ((k - 1) * d for k, d in zip(self.kernel_size, self.dilation))]
        return [(p, p) for p in self.padding]

    def _full_chunk(self, x, weight, delta, indices, base):
        d = self.spatial_rank
        selected = x[:, indices]  # N, output chunk, connected inputs, *spatial
        out_spatial = base.shape[1:]
        lead = selected.shape[:3]
        result = x.new_zeros(lead[0], lead[1], *out_spatial)
        for tap in self._kernel_taps:
            displacement = x.new_tensor([r * dil for r, dil in zip(tap, self.dilation)])
            positions = base.reshape(1, 1, 1, d, *out_spatial) + (delta + displacement).reshape(1, lead[1], lead[2], d, *((1,) * d))
            positions = positions.expand(*lead, d, *out_spatial)
            sampled = sample_nd(selected, positions, d, self.padding_mode)
            # Horner evaluation preserves the polynomial exactly while avoiding
            # separately materializing each sampled power and contribution.
            coefficient_shape = (1, lead[1], lead[2]) + (1,) * d
            polynomial = weight[(slice(None), slice(None), self.orders - 1, *tap)].reshape(coefficient_shape)
            for order in range(self.orders - 2, -1, -1):
                coefficient = weight[(slice(None), slice(None), order, *tap)].reshape(coefficient_shape)
                polynomial = polynomial * sampled + coefficient
            if not self.include_constant:
                polynomial = polynomial * sampled
            result = result + polynomial.sum(2)
        return result

    def _sampled_conv(self, x, weight, delta, grid, conv_dtype=None):
        d = self.spatial_rank
        extent = grid.shape[1:]
        coords = grid.reshape(1, 1, d, *extent) + delta.reshape(1, self.in_channels, d, *((1,) * d))
        shifted = sample_nd(x, coords.expand(x.shape[0], self.in_channels, d, *extent), d, self.padding_mode)
        packed = polynomial_pack(shifted, self.q, self.groups, self.include_constant)
        conv = F.conv1d if d == 1 else F.conv2d
        stride = (1,) * d if self.transposed else self.stride
        # Sampling and powers remain stable; only the native convolution may
        # use the caller's AMP dtype. Full connection Horner paths stay stable.
        with torch.autocast(device_type=x.device.type, enabled=conv_dtype is not None,
                            dtype=conv_dtype):
            return conv(packed, weight.reshape(self.out_channels, -1, *self.kernel_size), None,
                        stride, 0, self.dilation, self.groups).to(x.dtype)

    def _forward_core(self, x, output_padding=None, conv_dtype=None):
        d = self.spatial_rank
        weight, delta = self.weight.to(x.dtype), self.shifts.to(x.dtype)
        spatial = x.shape[-d:]
        pads = self._pads(spatial)
        if self.transposed:
            # Transpose is a sparse upsampled lattice followed by the flipped
            # polynomial kernel. Offsets retain INPUT-bin units via stride.
            lattice_shape = [(n - 1) * s + 1 for n, s in zip(spatial, self.stride)]
            lattice = x.new_zeros(*x.shape[:2], *lattice_shape)
            lattice[(slice(None), slice(None), *(slice(None, None, s) for s in self.stride))] = x
            x = lattice
            op = self.output_padding if output_padding is None else output_padding
            sizes = [(n - 1) * s - 2 * p[0] + dil * (k - 1) + 1 + extra for n, s, p, dil, k, extra in zip(spatial, self.stride, pads, self.dilation, self.kernel_size, op)]
            origins = [p[0] - dil * (k - 1) for p, dil, k in zip(pads, self.dilation, self.kernel_size)]
            steps = (1,) * d
            weight = weight.flip(tuple(range(3, 3 + d)))
            delta = delta * x.new_tensor(self.stride)
        else:
            sizes = [(n + a + b - dil * (k - 1) - 1) // s + 1 for n, (a, b), dil, k, s in zip(spatial, pads, self.dilation, self.kernel_size, self.stride)]
            origins = [-a for a, _ in pads]
            steps = self.stride
        if min(sizes) < 1:
            raise ValueError("Kernel/padding configuration produces an empty output")
        if self.shift_sharing == "shared" or self.out_channels // self.groups == 1 or not any(self.max_shift):
            # Extend the source FIRST. Sampling an already cropped shifted map
            # loses edge values which kernels with padding should still see.
            # One output per original group makes full/shared mathematically
            # identical, including depthwise layers, so use the native kernel.
            delta = x.new_zeros(self.in_channels, d) if not any(self.max_shift) else delta.reshape(self.in_channels, d)
            extended = [(n - 1) * step + dil * (k - 1) + 1 for n, step, dil, k in zip(sizes, steps, self.dilation, self.kernel_size)]
            axes = [torch.arange(n, device=x.device, dtype=x.dtype) + origin for n, origin in zip(extended, origins)]
            grid = torch.stack(torch.meshgrid(*axes, indexing="ij"))
            if self.recompute and self.training and torch.is_grad_enabled():
                return checkpoint(self._sampled_conv, x, weight, delta, grid, conv_dtype,
                                  use_reentrant=False, preserve_rng_state=False)
            return self._sampled_conv(x, weight, delta, grid, conv_dtype)
        axes = [torch.arange(n, device=x.device, dtype=x.dtype) * s + origin for n, s, origin in zip(sizes, steps, origins)]
        base = torch.stack(torch.meshgrid(*axes, indexing="ij"))
        outputs = []
        cin = self.in_channels // self.groups
        per_group = self.out_channels // self.groups
        for start in range(0, self.out_channels, self.output_chunk_size):
            end = min(start + self.output_chunk_size, self.out_channels)
            groups = torch.arange(start, end, device=x.device) // per_group
            indices = groups[:, None] * cin + torch.arange(cin, device=x.device)[None, :]
            offsets = delta[indices] if self.shift_sharing == "shared" else delta[start:end]
            args = (x, weight[start:end], offsets, indices, base)
            if self.recompute and self.training and torch.is_grad_enabled():
                outputs.append(checkpoint(self._full_chunk, *args, use_reentrant=False, preserve_rng_state=False))
            else:
                outputs.append(self._full_chunk(*args))
        return torch.cat(outputs, dim=1)

    def forward(self, x, output_size=None):
        d = self.spatial_rank
        unbatched = x.ndim == d + 1
        if unbatched:
            x = x.unsqueeze(0)
        if x.ndim != d + 2 or x.shape[1] != self.in_channels or min(x.shape[-d:]) < 1:
            raise ValueError(f"Expected (N,{self.in_channels},spatial...) or unbatched channel-first input")
        if not x.is_floating_point():
            raise ValueError("Only real floating-point inputs are supported")
        if x.device != self.weight.device:
            raise ValueError("Move layer and input to the same device")
        if not torch.is_autocast_enabled(x.device.type) and x.dtype != self.weight.dtype:
            raise ValueError("Input and layer dtypes differ; use layer.to(dtype=...) or autocast")
        op = None
        if output_size is not None:
            if not self.transposed:
                raise ValueError("output_size belongs to transpose layers")
            target = tuple(output_size)
            if unbatched and len(target) == d + 1:
                if target[0] != self.out_channels:
                    raise ValueError("output_size channel count does not match this decoder")
                target = target[1:]
            if len(target) != d:
                if len(target) != d + 2 or target[:2] != tuple(x.shape[:1]) + (self.out_channels,):
                    raise ValueError("output_size must be spatial shape or complete output shape")
                target = target[-d:]
            base = [(n - 1) * s - 2 * p + dil * (k - 1) + 1 for n, s, p, dil, k in zip(x.shape[-d:], self.stride, self.padding, self.dilation, self.kernel_size)]
            op = tuple(_integer(t, "output_size") - n for t, n in zip(target, base))
            if any(v < 0 or v >= s for v, s in zip(op, self.stride)):
                raise ValueError("Requested output_size is outside the valid stride ambiguity")
        compute_dtype = torch.float64 if x.dtype == torch.float64 or self.weight.dtype == torch.float64 else torch.float32
        output_dtype = (torch.float64 if compute_dtype == torch.float64 else
                        torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype)
        conv_dtype = (output_dtype if self.conv_precision == "autocast" and
                      output_dtype in (torch.float16, torch.bfloat16) else None)
        with torch.autocast(device_type=x.device.type, enabled=False):
            values = x.to(compute_dtype)
            if self.input_bound is not None:
                values = torch.tanh(values / self.input_bound)
            result = self._forward_core(values, op, conv_dtype)
            if self.bias is not None:
                result = result + self.bias.to(compute_dtype).reshape(1, -1, *((1,) * d))
        result = result.to(output_dtype)
        return result.squeeze(0) if unbatched else result

    @classmethod
    def from_conv(cls, conv, *, q=3, **kwargs):
        """Copy a real torch convolution into order 1, higher orders zeroed.

        Starts with exactly zero shifts. Outputs/input gradients match the
        supplied Conv when input_bound=None; shifts can subsequently learn.
        """
        expected = (nn.ConvTranspose1d if cls.transposed else nn.Conv1d) if cls.spatial_rank == 1 else (nn.ConvTranspose2d if cls.transposed else nn.Conv2d)
        if not isinstance(conv, expected):
            raise TypeError(f"Expected {expected.__name__}")
        kwargs = {**kwargs, "shift_init": "zeros"}
        extra = {"output_padding": conv.output_padding} if cls.transposed else {}
        layer = cls(conv.in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride,
                    padding=conv.padding, dilation=conv.dilation, groups=conv.groups,
                    bias=conv.bias is not None, padding_mode=conv.padding_mode,
                    device=conv.weight.device, dtype=conv.weight.dtype, q=q, **extra, **kwargs)
        with torch.no_grad():
            source = conv.weight
            if cls.transposed:
                g, ci, co = conv.groups, conv.in_channels // conv.groups, conv.out_channels // conv.groups
                source = source.reshape(g, ci, co, *conv.kernel_size).transpose(1, 2).reshape(conv.out_channels, ci, *conv.kernel_size)
            layer.weight.zero_()
            layer.weight[:, :, int(layer.include_constant)].copy_(source)
            if conv.bias is not None:
                layer.bias.copy_(conv.bias)
        layer.train(conv.training)
        return layer

    def extra_repr(self):
        return (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding}, groups={self.groups}, q={self.q}, "
                f"shift_mode={self.shift_mode!r}, shift_sharing={self.shift_sharing!r}, max_shift={self.max_shift}")


class SuperONN1d(_SuperONN):
    """Channel-first 1D polynomial convolution, with the Conv1d argument order."""


class SuperONN2d(_SuperONN):
    """Channel-first 2D polynomial convolution, with the Conv2d argument order."""
    spatial_rank = 2


class SuperONNTranspose1d(_SuperONN):
    """Polynomial decoder on an upsampled lattice; see README for shift semantics.

    Uses ConvTranspose1d's leading argument order, including output_padding.
    """
    transposed = True
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1,
                 padding_mode="zeros", device=None, dtype=None, **superonn_options):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, bias, padding_mode, device, dtype,
                         output_padding=output_padding, **superonn_options)


class SuperONNTranspose2d(SuperONNTranspose1d):
    """2D counterpart of SuperONNTranspose1d."""
    spatial_rank = 2
    transposed = True
