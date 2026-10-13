"""Run a native 2D encoder/decoder in each framework and transfer a layer."""
import importlib.util
import torch
if importlib.util.find_spec("triton") is not None:
    import triton
import tensorflow as tf
from torch import nn

from superonn import SuperONN2d, SuperONNTranspose2d
from superonn.interop import to_tensorflow, to_pytorch
from superonn.tensorflow import SuperONN2D, SuperONNTranspose2D


def main():
    # Each model uses its own native framework operations and optimizer.
    pytorch_model = nn.Sequential(
        SuperONN2d(3, 8, 4, stride=2, padding=1, input_bound=1),
        nn.GELU(),
        SuperONNTranspose2d(8, 3, 4, stride=2, padding=1, input_bound=1),
    )
    keras_model = tf.keras.Sequential([
        tf.keras.Input((12, 10, 3)),
        SuperONN2D(8, 4, strides=2, padding=1, input_bound=1, activation="gelu"),
        SuperONNTranspose2D(3, 4, strides=2, padding=1, input_bound=1),
    ])
    x = torch.rand(2, 3, 12, 10)
    tx = tf.convert_to_tensor(x.permute(0, 2, 3, 1).numpy())
    optimizer = torch.optim.AdamW(pytorch_model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    pytorch_model(x).square().mean().backward()
    optimizer.step()
    keras_model.compile(optimizer="adam", loss="mse")
    loss = keras_model.train_on_batch(tx, tf.zeros_like(tx))
    print(f"Keras training loss: {float(loss):.6f}")

    # Convert individual compatible layers; whole models are built natively.
    transferred = to_tensorflow(pytorch_model[0])
    keras_features = transferred(tx)
    roundtrip = to_pytorch(transferred)
    pytorch_features = roundtrip(x)
    print(f"Transferred Keras feature shape: {tuple(keras_features.shape)}")
    print(f"Transferred PyTorch feature shape: {tuple(pytorch_features.shape)}")



if __name__ == "__main__":
    main()
