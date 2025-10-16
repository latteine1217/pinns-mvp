import numpy as np

from pinnx.sensors import FieldSensorSelector


def _create_3d_field(nx: int = 6, ny: int = 4, nz: int = 5, n_snapshots: int = 3):
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    z = np.linspace(0.0, np.pi, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    snapshots = []
    for k in range(n_snapshots):
        phase = 0.1 * k
        snapshots.append(
            {
                'u': np.sin(np.pi * X + phase) * np.cos(np.pi * Y),
                'v': np.cos(np.pi * X) * np.sin(np.pi * Y + phase),
                'w': np.sin(np.pi * Z + phase) * np.cos(np.pi * X),
                'p': np.cos(np.pi * X) * np.cos(np.pi * Y) * np.cos(np.pi * Z + phase),
            }
        )

    field = {
        'u': np.stack([snap['u'] for snap in snapshots], axis=0),
        'v': np.stack([snap['v'] for snap in snapshots], axis=0),
        'w': np.stack([snap['w'] for snap in snapshots], axis=0),
        'p': np.stack([snap['p'] for snap in snapshots], axis=0),
        'x': x,
        'y': y,
        'z': z,
    }
    return field, (x, y, z)


def _create_2d_field(nx: int = 8, ny: int = 6, n_snapshots: int = 2):
    x = np.linspace(0.0, 2.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    snapshots = []
    for k in range(n_snapshots):
        phase = 0.2 * k
        snapshots.append(
            {
                'u': np.sin(np.pi * X + phase),
                'v': np.cos(np.pi * Y + phase),
                'p': np.sin(np.pi * X) * np.sin(np.pi * Y + phase),
            }
        )

    field = {
        'u': np.stack([snap['u'] for snap in snapshots], axis=0),
        'v': np.stack([snap['v'] for snap in snapshots], axis=0),
        'p': np.stack([snap['p'] for snap in snapshots], axis=0),
        # w intentionally omitted to ensure graceful handling
        'x': x,
        'y': y,
    }
    return field, (x, y)


def test_field_sensor_selector_3d():
    field, axes = _create_3d_field()
    selector = FieldSensorSelector(feature_scaling='standard', nan_policy='raise')

    selection = selector.select(field, n_sensors=12)

    assert selection.indices.shape == (12,)
    assert selection.metadata.spatial_shape == tuple(len(axis) for axis in axes)
    assert selection.coordinates().shape == (12, 3)

    for comp in ('u', 'v', 'w', 'p'):
        assert comp in selection.component_values
        values = selection.component_values[comp]
        assert values.shape == (12, field[comp].shape[0])

    # ensure indices are unique and within bounds
    assert len(np.unique(selection.indices)) == len(selection.indices)
    assert selection.indices.max() < selection.metadata.n_locations


def test_field_sensor_selector_2d_slice():
    field, axes = _create_2d_field()
    selector = FieldSensorSelector(strategy='qr_pivot', feature_scaling='l2')

    selection = selector.select(field, n_sensors=10)

    assert selection.indices.shape == (10,)
    assert selection.metadata.spatial_shape == tuple(len(axis) for axis in axes)
    assert selection.coordinates().shape == (10, 2)

    assert 'w' not in selection.component_values  # not provided in field input
    for comp in ('u', 'v', 'p'):
        values = selection.component_values[comp]
        assert values.shape == (10, field[comp].shape[0])

    # verify unravelled indices reside within expected grid bounds
    unravelled = selection.unravel()
    assert unravelled.shape == (10, 2)
    nx, ny = selection.metadata.spatial_shape
    assert np.all((unravelled[:, 0] >= 0) & (unravelled[:, 0] < nx))
    assert np.all((unravelled[:, 1] >= 0) & (unravelled[:, 1] < ny))
