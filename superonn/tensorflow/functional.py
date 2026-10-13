"""Differentiable channels-last sampling and sparse transpose lattices."""
from __future__ import annotations

import itertools
from numbers import Integral

import tensorflow as tf


def spatial_tuple(value, rank, name, minimum=1):
    try:
        values = (value,) * rank if isinstance(value, Integral) else tuple(value)
    except TypeError as error:
        raise ValueError(f"{name} must have {rank} integer entries >= {minimum}") from error
    if len(values) != rank or any(isinstance(v, bool) or not isinstance(v, Integral) or v < minimum for v in values):
        raise ValueError(f"{name} must have {rank} integer entries >= {minimum}")
    return tuple(int(v) for v in values)


def normalize_padding(value, rank):
    if isinstance(value, str):
        value = value.lower()
        if value not in ("same", "valid"):
            raise ValueError("padding must be same, valid, or nonnegative spatial integers")
        return value
    values = spatial_tuple(value, rank, "padding", 0)
    return values[0] if rank == 1 else values


def output_shape_1d(input_shape, filters, kernel, stride, padding, dilation, data_format):
    shape = tf.TensorShape(input_shape)
    if shape.rank != 3:
        raise ValueError("A 1D layer requires rank-3 input")
    length = shape[1 if data_format == "channels_last" else 2]
    if length is not None:
        effective = (kernel - 1) * dilation + 1
        if padding == "same":
            length = (length + stride - 1) // stride
        else:
            length = (length + 2 * (0 if padding == "valid" else padding) - effective) // stride + 1
        if length < 1:
            raise ValueError("valid kernel must fit the input")
    return tf.TensorShape((shape[0], length, filters) if data_format == "channels_last" else (shape[0], filters, length))


def sample_nd(inputs, coordinates, channels, boundary, rank):
    """Sample [N,*spatial,C] at [P,O,I,D], returning [N,P,O,I].

    Each zero-boundary interpolation neighbor is masked independently. Powers
    must be evaluated after this operation, including at fractional boundaries.
    """
    sizes = tf.shape(inputs)[1:1 + rank]
    sizes_float = tf.cast(sizes, inputs.dtype)
    last = sizes_float - 1
    if boundary == "border":
        coordinates = tf.clip_by_value(coordinates, 0, last)
    elif boundary == "reflection":
        period = tf.maximum(2 * last, 1)
        folded = tf.math.floormod(coordinates, period)
        coordinates = tf.where(folded <= last, folded, period - folded)
        coordinates = tf.where(sizes > 1, coordinates, tf.zeros_like(coordinates))
    elif boundary == "circular":
        coordinates = tf.math.floormod(coordinates, sizes_float)
    lower_float = tf.floor(coordinates)
    fraction = coordinates - lower_float
    lower = tf.cast(lower_float, tf.int32)
    selected = tf.gather(inputs, channels, axis=-1)
    values = tf.transpose(tf.reshape(selected, (tf.shape(inputs)[0], tf.reduce_prod(sizes), tf.size(channels))), (2, 1, 0))
    index_strides = tf.math.cumprod(sizes, exclusive=True, reverse=True)
    result = tf.zeros(tf.concat(([tf.shape(inputs)[0]], tf.shape(coordinates)[:-1]), 0), inputs.dtype)
    for corner in itertools.product((0, 1), repeat=rank):
        index = lower + tf.constant(corner, tf.int32)
        weight = tf.reduce_prod(tf.where(tf.constant(tuple(bool(v) for v in corner)), fraction, 1 - fraction), axis=-1)
        valid = tf.reduce_all((index >= 0) & (index < sizes), axis=-1)
        index = tf.math.floormod(index, sizes) if boundary == "circular" else tf.clip_by_value(index, 0, sizes - 1)
        flat = tf.reduce_sum(index * index_strides, axis=-1)
        gathered = tf.gather(values, tf.transpose(flat, (2, 0, 1)), axis=1, batch_dims=1)
        gathered = tf.transpose(gathered, (3, 1, 2, 0))
        if boundary == "zeros":
            weight = weight * tf.cast(valid, inputs.dtype)
        result = result + gathered * weight[None]
    return result


def sparse_upsample(inputs, strides):
    """Insert zeros between samples without inserting trailing source points."""
    result = inputs
    rank = len(strides)
    for axis, stride in enumerate(strides, 1):
        if stride == 1:
            continue
        shape = tf.shape(result)
        expanded = tf.expand_dims(result, axis + 1)
        paddings = [[0, 0] for _ in range(rank + 3)]
        paddings[axis + 1] = [0, stride - 1]
        expanded = tf.pad(expanded, paddings)
        new_shape = tf.tensor_scatter_nd_update(shape, [[axis]], [shape[axis] * stride])
        result = tf.reshape(expanded, new_shape)
        size = tf.tensor_scatter_nd_update(new_shape, [[axis]], [(shape[axis] - 1) * stride + 1])
        result = tf.slice(result, tf.zeros(rank + 2, tf.int32), size)
    return result


def position_grid(sizes, steps, origins, dtype):
    axes = [tf.cast(tf.range(sizes[i]) * steps[i] + origins[i], dtype) for i in range(len(steps))]
    return tf.reshape(tf.stack(tf.meshgrid(*axes, indexing="ij"), axis=-1), (-1, len(steps)))
