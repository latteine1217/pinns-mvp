import numpy as np

from pinnx.sensors.qr_pivot import QRPivotSelector


def test_qr_pivot_column_mode_returns_row_indices():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((8, 5))

    selector = QRPivotSelector(mode='column', pivoting=True)
    indices, metrics = selector.select_sensors(data, n_sensors=4)

    assert indices.shape == (4,)
    assert np.all(indices < data.shape[0])
    assert 'condition_number' in metrics


def test_qr_pivot_row_mode_returns_row_indices_on_pod_matrix():
    # POD-like matrix: more rows than columns
    rng = np.random.default_rng(1)
    data = rng.standard_normal((10, 3))

    selector = QRPivotSelector(mode='row', pivoting=True)
    indices, _ = selector.select_sensors(data, n_sensors=3)

    assert indices.shape == (3,)
    assert np.all(indices < data.shape[0])
