"""Train a native Keras regressor with optional whole-model XLA compilation.

Requires the TensorFlow extra. Run: python examples/tensorflow_training.py [--compile]
"""
import argparse

import tensorflow as tf

from superonn.tensorflow import conv1d, separable_conv1d


def regressor(length=256, *, jit_compile=False):
    return tf.keras.Sequential([
        tf.keras.Input((length, 1)),
        conv1d(8, 5, padding="same", operator="superonn", activation="gelu",
               order_regularization=1e-5, jit_compile=jit_compile),
        separable_conv1d(8, 3, padding="same", operator="superonn", activation="gelu",
                         jit_compile=jit_compile),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(1),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true", help="Enable whole-model XLA")
    args = parser.parse_args()
    tf.keras.utils.set_random_seed(37)
    model = regressor(jit_compile=args.compile)
    model.compile(optimizer="adam", loss="mse", jit_compile=args.compile)
    inputs = tf.random.normal((4, 256, 1))
    targets = tf.reduce_mean(inputs, axis=1)
    for step in range(3):
        loss = model.train_on_batch(inputs, targets)
        print(f"step={step + 1} loss={float(loss):.6f}")
    print(f"compiled={args.compile} tensorflow={tf.__version__}")


if __name__ == "__main__":
    main()
