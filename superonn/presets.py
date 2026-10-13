"""Reusable hyperparameter presets and PyTorch regularization helpers."""
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class SuperONNConfig:
    q: int = 3
    max_shift: float = 2.0
    shift_mode: str = "learned"
    shift_sharing: str = "full"
    initialization: str = "linear"
    input_bound: float | None = 1.0
    output_chunk_size: int = 4
    recompute: bool = True
    order_regularization: float = 0.0
    order_decay: float = 1.5
    conv_precision: str = "stable"

    def layer_kwargs(self):
        """Pass to any standalone SuperONN layer constructor."""
        return asdict(self)


def superonn_regularization_loss(model):
    """Aggregate each registered SuperONN once; add to the task loss in PyTorch."""
    from .layers import _SuperONN
    import torch
    losses = [module.regularization_loss() for module in model.modules() if isinstance(module, _SuperONN)]
    if losses:
        return sum(losses[1:], losses[0])
    parameter = next(model.parameters(), None)
    return torch.tensor(0.) if parameter is None else parameter.new_zeros(())
