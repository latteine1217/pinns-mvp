import torch
import torch.nn as nn

from pinnx.train.trainer import Trainer


class DummyModel(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class DummyPhysics:
    pass


def _base_config(tmp_path, out_dim: int):
    return {
        'training': {
            'epochs': 1,
            'lr': 1e-3,
            'optimizer': 'adam',
            'log_interval': 1,
            'checkpoint_interval': 5,
            'early_stopping': {'enabled': False},
        },
        'losses': {},
        'physics': {
            'type': 'ns_2d' if out_dim <= 3 else 'vs_pinn_channel_flow'
        },
        'output': {
            'checkpoint_dir': str(tmp_path / "ckpt")
        },
        'normalization': {
            'type': 'none'
        }
    }


def test_variable_order_three_outputs(tmp_path):
    device = torch.device('cpu')
    model = DummyModel(in_dim=2, out_dim=3).to(device)
    physics = DummyPhysics()
    config = _base_config(tmp_path, out_dim=3)

    trainer = Trainer(model, physics, losses={}, config=config, device=device)

    inferred = trainer._infer_variable_order(3, context='unit-test')
    assert inferred == ['u', 'v', 'p']


def test_variable_order_four_outputs(tmp_path):
    device = torch.device('cpu')
    model = DummyModel(in_dim=3, out_dim=4).to(device)
    physics = DummyPhysics()
    config = _base_config(tmp_path, out_dim=4)

    trainer = Trainer(model, physics, losses={}, config=config, device=device)

    inferred = trainer._infer_variable_order(4, context='unit-test')
    assert inferred == ['u', 'v', 'w', 'p']
