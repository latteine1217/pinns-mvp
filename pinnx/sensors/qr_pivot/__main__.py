"""Module test entrypoint for qr_pivot package."""

import numpy as np

from .evaluation import evaluate_sensor_placement
from .optimizer import SensorOptimizer
from .selectors.greedy import GreedySelector
from .selectors.multi_objective import MultiObjectiveSelector
from .selectors.pod_based import PODBasedSelector
from .selectors.qr_pivot import QRPivotSelector


def _fmt_metric(value) -> str:
    if isinstance(value, (int, float, np.floating)):
        return f"{value:.2f}"
    return str(value)


def _run_smoke_test() -> None:
    print("🧪 測試感測點選擇模組...")

    np.random.seed(42)
    n_locations = 100
    n_snapshots = 50

    t = np.linspace(0, 2 * np.pi, n_snapshots)
    x = np.linspace(0, 1, n_locations)

    data_matrix = np.zeros((n_locations, n_snapshots))
    for i in range(3):
        mode = np.sin((i + 1) * np.pi * x[:, np.newaxis])
        coeff = np.cos((i + 1) * t) * np.exp(-0.1 * i)
        data_matrix += mode @ coeff[np.newaxis, :]

    data_matrix += 0.01 * np.random.randn(n_locations, n_snapshots)

    n_sensors = 8

    strategies = {
        "QR-Pivot": QRPivotSelector(),
        "POD-based": PODBasedSelector(n_modes=5),
        "Greedy": GreedySelector(objective="info_gain"),
        "Multi-objective": MultiObjectiveSelector(objectives=["accuracy", "robustness"]),
    }

    results = {}

    for name, selector in strategies.items():
        print(f"\n測試 {name} 策略...")
        try:
            indices, metrics = selector.select_sensors(data_matrix, n_sensors)
            results[name] = {
                "indices": indices,
                "condition_number": metrics.get("condition_number", np.inf),
                "energy_ratio": metrics.get("energy_ratio", 0.0),
                "n_selected": len(indices),
            }
            print(f"  選擇感測點: {len(indices)} 個")
            print(f"  條件數: {_fmt_metric(metrics.get('condition_number', 'N/A'))}")
            print(f"  能量比例: {_fmt_metric(metrics.get('energy_ratio', 0.0))}")
        except Exception as exc:
            print(f"  ❌ 失敗: {exc}")
            results[name] = {"error": str(exc)}

    print("\n測試自動策略選擇...")
    optimizer = SensorOptimizer(strategy="auto")
    auto_indices, auto_metrics = optimizer.optimize_sensor_placement(data_matrix, n_sensors)
    print(f"  自動選擇策略: {auto_metrics.get('auto_selected_strategy', 'unknown')}")
    print(f"  選擇感測點: {len(auto_indices)} 個")

    print("\n綜合評估...")
    for name, result in results.items():
        if "error" not in result:
            eval_metrics = evaluate_sensor_placement(data_matrix, result["indices"])
            print(
                f"  {name}: 條件數={eval_metrics.get('condition_number', 'N/A'):.2f}, "
                f"覆蓋率={eval_metrics.get('subspace_coverage', 0.0):.3f}"
            )

    print("✅ 感測點選擇模組測試完成！")


if __name__ == "__main__":
    _run_smoke_test()
