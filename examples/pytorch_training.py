"""Train a small regressor with AMP and optional PyTorch compilation.

Run after installing the package: python examples/pytorch_training.py [--compile]
"""
import argparse

import torch
from torch import nn

from superonn import SuperONN1d, SuperONNConfig, superonn_regularization_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true", help="Enable the PyTorch compiler")
    args = parser.parse_args()

    torch.set_num_threads(2)
    torch.manual_seed(37)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    options = SuperONNConfig(order_regularization=1e-5).layer_kwargs()
    model = nn.Sequential(
        SuperONN1d(1, 8, 5, padding=2, **options),
        nn.GELU(),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(8, 1),
    ).to(device)
    if args.compile:
        model.compile(fullgraph=True, dynamic=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    inputs = torch.randn(4, 1, 256, device=device)
    targets = inputs.mean(dim=-1)
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=amp_dtype):
            predictions = model(inputs)
            task_loss = nn.functional.mse_loss(predictions.float(), targets)
            loss = task_loss + superonn_regularization_loss(model)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        print(f"step={step + 1} loss={loss.detach().item():.6f}")

    print(f"device={device} compiled={args.compile} torch={torch.__version__}")


if __name__ == "__main__":
    main()
