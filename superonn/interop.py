"""Explicit configuration and weight transfer between native framework layers.

Conversions do not transfer optimizer slots, RNG state, or an autograd graph.
Each returned layer trains using its destination framework's native operations.
"""
from __future__ import annotations


def _classes():
    from . import layers as pt
    from .tensorflow import SuperONN1D, SuperONN2D, SuperONNTranspose1D, SuperONNTranspose2D
    return {(1, False): (pt.SuperONN1d, SuperONN1D), (2, False): (pt.SuperONN2d, SuperONN2D),
            (1, True): (pt.SuperONNTranspose1d, SuperONNTranspose1D), (2, True): (pt.SuperONNTranspose2d, SuperONNTranspose2D)}


def _tf_semantics(layer):
    if not any(isinstance(layer, classes[1]) for classes in _classes().values()):
        raise TypeError("Expected a native TensorFlow SuperONN layer")
    if not layer.built:
        raise ValueError("Build the TensorFlow layer with known input channels before conversion")
    import tensorflow as tf
    if tf.keras.activations.serialize(layer.activation) != "linear":
        raise ValueError("A built-in activation cannot be transferred to a bare PyTorch layer; use a separate activation")
    d = layer.spatial_rank
    as_tuple = lambda value: (value,) if d == 1 and not isinstance(value, (tuple, list)) else tuple(value)
    kernel, stride, dilation = as_tuple(layer.kernel_size), as_tuple(layer.strides), as_tuple(layer.dilation_rate)
    padding = layer.padding
    op = (0,) * d
    if layer.transposed:
        requested = None if layer.output_padding is None else as_tuple(layer.output_padding)
        if isinstance(padding, str):
            pads, ops = [], []
            for axis, (k, s, dil) in enumerate(zip(kernel, stride, dilation)):
                effective = (k - 1) * dil + 1
                extra = None if requested is None else requested[axis]
                if padding == "valid":
                    left = 0
                    extra = max(effective, s) - effective if extra is None else extra
                else:
                    total = effective - s if extra is None else effective - effective % 2 - extra
                    if total > 0 and total % 2:
                        raise ValueError("Asymmetric Keras transpose same padding cannot be represented by PyTorch symmetric padding; use numeric padding")
                    left = max(total, 0) // 2
                    extra = max(-total, 0)
                pads.append(left)
                ops.append(extra)
            padding, op = tuple(pads), tuple(ops)
        else:
            padding = as_tuple(padding)
            op = (0,) * d if requested is None else requested
    elif isinstance(padding, str):
        if padding == "same" and any(s != 1 for s in stride):
            raise ValueError("Keras same padding with stride > 1 is input-size dependent; use numeric padding for conversion")
    else:
        padding = as_tuple(padding)
    return dict(in_channels=layer.input_channels, out_channels=layer.filters, kernel_size=kernel,
                stride=stride, padding=padding, dilation=dilation, groups=layer.groups, bias=layer.use_bias,
                padding_mode={"zeros": "zeros", "border": "replicate", "reflection": "reflect", "circular": "circular"}[layer.boundary],
                q=layer.q, shift_mode=layer.shift_mode, shift_sharing=layer.shift_sharing,
                max_shift=as_tuple(layer.max_shift), input_bound=layer.input_bound,
                include_constant=layer.include_constant, output_padding=op,
                order_regularization=layer.order_regularization, order_decay=layer.order_decay)


def _pt_semantics(layer):
    if not any(isinstance(layer, classes[0]) for classes in _classes().values()):
        raise TypeError("Expected a native PyTorch SuperONN layer")
    config = layer._config()
    names = ("in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups",
             "padding_mode", "q", "shift_mode", "shift_sharing", "max_shift", "input_bound", "include_constant",
             "output_padding", "order_regularization", "order_decay")
    return {**{name: config[name] for name in names}, "bias": layer.bias is not None}


def _validate_pair(pytorch_layer, tensorflow_layer):
    source, target = _pt_semantics(pytorch_layer), _tf_semantics(tensorflow_layer)
    if (pytorch_layer.spatial_rank, pytorch_layer.transposed) != (tensorflow_layer.spatial_rank, tensorflow_layer.transposed):
        raise ValueError("Spatial rank and transpose direction must match")
    # 'valid' and numeric zero padding are mathematically identical.
    for config in (source, target):
        if config["padding"] == "valid":
            config["padding"] = (0,) * pytorch_layer.spatial_rank
        elif config["padding"] == "same" and not pytorch_layer.transposed:
            effective = tuple((k - 1) * dil + 1 for k, dil in zip(config["kernel_size"], config["dilation"]))
            if all(k % 2 for k in effective):
                config["padding"] = tuple(k // 2 for k in effective)
    mismatches = [name for name in source if source[name] != target[name]]
    if mismatches:
        raise ValueError("Incompatible layer semantics: " + ", ".join(mismatches))
    # Validate declared state layouts before any reshape, cast, or assignment.
    d = pytorch_layer.spatial_rank
    ci = pytorch_layer.in_channels // pytorch_layer.groups
    orders = pytorch_layer.q + int(pytorch_layer.include_constant)
    pt_offsets = pytorch_layer.raw_shifts if pytorch_layer.raw_shifts is not None else pytorch_layer.fixed_shifts
    expected_offsets = (pytorch_layer.in_channels, d) if pytorch_layer.shift_sharing == "shared" else (pytorch_layer.out_channels, ci, d)
    tf_rows = tensorflow_layer.groups if tensorflow_layer.shift_sharing == "shared" else tensorflow_layer.filters
    tf_offsets = (tf_rows, ci) if d == 1 and not tensorflow_layer.transposed else (tf_rows, ci, d)
    expected = [(pytorch_layer.weight, (pytorch_layer.out_channels, ci, orders, *pytorch_layer.kernel_size)),
                (pt_offsets, expected_offsets),
                (tensorflow_layer.kernel, (*pytorch_layer.kernel_size, ci, orders, pytorch_layer.out_channels)),
                (tensorflow_layer.shift_parameter, tf_offsets)]
    if pytorch_layer.bias is not None:
        expected.extend(((pytorch_layer.bias, (pytorch_layer.out_channels,)),
                         (tensorflow_layer.bias, (pytorch_layer.out_channels,))))
    if any(tuple(variable.shape) != shape for variable, shape in expected):
        raise ValueError("State tensor shape differs from the declared layer configuration")


def _validate_offsets(values, config):
    import numpy as np
    if not np.isfinite(values).all():
        raise ValueError("Offsets must be finite")
    if config["shift_mode"] == "random_fixed":
        reshaped = values.reshape(-1, len(config["max_shift"]))
        if not np.array_equal(values, np.round(values)) or (np.abs(reshaped) > np.asarray(config["max_shift"])).any():
            raise ValueError("Fixed offsets must be integers within max_shift")


def copy_to_tensorflow(pytorch_layer, tensorflow_layer):
    """Copy coefficients, raw/fixed offsets, and bias to a matching built layer.

    Validates all semantic settings before mutation. TensorFlow assignment uses
    the target variable dtype; choose matching dtypes for precision preservation.
    """
    _validate_pair(pytorch_layer, tensorflow_layer)
    d = pytorch_layer.spatial_rank
    weight = pytorch_layer.weight.detach().cpu()
    # NumPy does not expose bfloat16; promote only that storage format.
    import torch
    array = lambda v: (v.float() if v.dtype == torch.bfloat16 else v).numpy()
    weight = array(weight.permute(*range(3, 3 + d), 1, 2, 0))
    offsets = pytorch_layer.raw_shifts if pytorch_layer.raw_shifts is not None else pytorch_layer.fixed_shifts
    offsets = array(offsets.detach().cpu()).reshape(tensorflow_layer.shift_parameter.shape)
    _validate_offsets(offsets, _pt_semantics(pytorch_layer))
    assignments = [(tensorflow_layer.kernel, weight), (tensorflow_layer.shift_parameter, offsets)]
    if pytorch_layer.bias is not None:
        assignments.append((tensorflow_layer.bias, array(pytorch_layer.bias.detach().cpu())))
    import tensorflow as tf
    # Materialize and validate every cast before assigning any variable.
    tensors = [(variable, tf.cast(tf.convert_to_tensor(value), variable.dtype)) for variable, value in assignments]
    if any(tuple(variable.shape) != tuple(value.shape) for variable, value in tensors):
        raise ValueError("State tensor shape differs from the destination variable")
    if any(not bool(tf.reduce_all(tf.math.is_finite(value))) for _, value in tensors):
        raise ValueError("State contains nonfinite values or overflows the destination dtype")
    _validate_offsets(tensors[1][1].numpy(), _tf_semantics(tensorflow_layer))
    for variable, value in tensors:
        variable.assign(value)
    return tensorflow_layer


def copy_to_pytorch(tensorflow_layer, pytorch_layer):
    """Copy state to a matching native PyTorch layer, retaining its device/dtype."""
    _validate_pair(pytorch_layer, tensorflow_layer)
    import torch
    d = pytorch_layer.spatial_rank
    import tensorflow as tf
    array = lambda variable: tf.cast(variable, tf.float32).numpy() if str(variable.dtype) == "bfloat16" else variable.numpy()
    weight = array(tensorflow_layer.kernel).transpose(d + 2, d, d + 1, *range(d))
    offsets = array(tensorflow_layer.shift_parameter)
    _validate_offsets(offsets, _tf_semantics(tensorflow_layer))
    destination = pytorch_layer.raw_shifts if pytorch_layer.raw_shifts is not None else pytorch_layer.fixed_shifts
    tensors = [(pytorch_layer.weight, weight), (destination, offsets.reshape(destination.shape))]
    if pytorch_layer.bias is not None:
        tensors.append((pytorch_layer.bias, array(tensorflow_layer.bias)))
    converted = [(target, torch.as_tensor(value, device=target.device, dtype=target.dtype)) for target, value in tensors]
    if any(tuple(target.shape) != tuple(value.shape) for target, value in converted):
        raise ValueError("State tensor shape differs from the destination variable")
    if any(not bool(torch.isfinite(value).all()) for _, value in converted):
        raise ValueError("State contains nonfinite values or overflows the destination dtype")
    cast_offsets = converted[1][1].detach().cpu()
    _validate_offsets((cast_offsets.float() if cast_offsets.dtype == torch.bfloat16 else cast_offsets).numpy(), _pt_semantics(pytorch_layer))
    with torch.no_grad():
        for target, value in converted:
            target.copy_(value)
    return pytorch_layer


def to_tensorflow(layer, *, data_format="channels_last", dtype=None, jit_compile=False):
    """Construct a native Keras counterpart and transfer this PyTorch layer.

    Uses numeric padding when supplied by PyTorch, preserving exact output
    geometry. The default input layout becomes [N,*spatial,C].
    """
    import torch
    config = _pt_semantics(layer)
    trainable = [p.requires_grad for name, p in layer.named_parameters() if name != "raw_shifts" or any(layer.max_shift)]
    if len(set(trainable)) > 1:
        raise ValueError("Partial parameter freezing requires an explicit destination layer and copy_to_tensorflow")
    _, cls = _classes()[layer.spatial_rank, layer.transposed]
    dtype = dtype or {torch.float64: "float64", torch.float32: "float32", torch.float16: "float16", torch.bfloat16: "bfloat16"}[layer.weight.dtype]
    arguments = dict(filters=layer.out_channels, kernel_size=layer.kernel_size, strides=layer.stride,
                     dilation_rate=layer.dilation, padding=layer.padding, use_bias=layer.bias is not None,
                     boundary={"zeros": "zeros", "replicate": "border", "reflect": "reflection", "circular": "circular"}[layer.padding_mode],
                     data_format=data_format, dtype=dtype, jit_compile=jit_compile,
                     output_chunk_size=layer.output_chunk_size, recompute=layer.recompute,
                     conv_precision=layer.conv_precision, trainable=all(trainable))
    for name in ("groups", "q", "shift_mode", "shift_sharing", "input_bound", "include_constant", "order_regularization", "order_decay"):
        arguments[name] = config[name]
    arguments["max_shift"] = layer.max_shift[0] if layer.spatial_rank == 1 else layer.max_shift
    if layer.transposed:
        arguments["output_padding"] = layer.output_padding
    result = cls(**arguments)
    shape = (None, *(None,) * layer.spatial_rank, layer.in_channels) if data_format == "channels_last" else (None, layer.in_channels, *(None,) * layer.spatial_rank)
    result.build(shape)
    return copy_to_tensorflow(layer, result)


def to_pytorch(layer, *, device=None, dtype=None):
    """Construct a native PyTorch counterpart and transfer this built Keras layer.

    Rejects input-size-dependent/asymmetric Keras padding and inline activations
    that cannot be represented exactly by a bare PyTorch SuperONN layer.
    """
    import torch
    config = _tf_semantics(layer)
    variables = [layer.kernel] + ([layer.bias] if layer.bias is not None else [])
    if layer.shift_mode == "learned" and any(config["max_shift"]):
        variables.append(layer.shift_parameter)
    if layer.trainable and len({v.trainable for v in variables}) > 1:
        raise ValueError("Partial parameter freezing requires an explicit destination layer and copy_to_pytorch")
    unsupported = [name for name in ("kernel_regularizer", "bias_regularizer", "kernel_constraint", "bias_constraint") if getattr(layer, name) is not None]
    if unsupported:
        raise ValueError("Backend-specific training settings need an explicit destination and copy_to_pytorch: " + ", ".join(unsupported))
    cls, _ = _classes()[layer.spatial_rank, layer.transposed]
    dtype = dtype or {"float64": torch.float64, "float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[str(layer.variable_dtype)]
    result = cls(**config, device=device, dtype=dtype, recompute=layer.recompute,
                 output_chunk_size=layer.output_chunk_size, conv_precision=layer.conv_precision)
    copy_to_pytorch(layer, result)
    if not layer.trainable or not any(v.trainable for v in variables):
        result.requires_grad_(False)
    return result


__all__ = ["to_tensorflow", "to_pytorch", "copy_to_tensorflow", "copy_to_pytorch"]
