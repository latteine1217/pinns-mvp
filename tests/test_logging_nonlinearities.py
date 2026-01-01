"""
log_nonlinearities integration test
"""

import torch

from pinnx.models.fourier_mlp import PINNNet
from pinnx.train.training_loop_manager import TrainingLoopManager


class DummyWriter:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))


def test_log_nonlinearities_records_alpha():
    writer = DummyWriter()
    loop_helper = TrainingLoopManager(config={}, writer=writer)

    model = PINNNet(
        in_dim=2,
        out_dim=1,
        width=16,
        depth=2,
        block_type='piratenet',
        use_input_projection=True,
    )

    loop_helper.log_nonlinearities(model, epoch=0)

    assert writer.scalars, "Expected alpha scalars to be logged"
    tags = [tag for tag, _, _ in writer.scalars]
    assert all(tag.startswith('Nonlinearity/alpha_') for tag in tags)
