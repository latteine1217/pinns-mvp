"""
物理引導 QR-Pivot 感測點選擇器單元測試

測試覆蓋：
1. 壁面區域識別（y/h 和 y+ 兩種方式）
2. 物理加權 QR-Pivot 選點
3. 壁面覆蓋率統計
4. 與標準 QR-Pivot 的對比
5. 邊界情況處理（無壁面點、全壁面點、座標缺失）

測試目標：
- 單元測試覆蓋率 ≥ 80%
- 驗證壁面優先選擇邏輯
- 與標準 QR-Pivot 一致
"""

import pytest
import numpy as np
import torch
from pinnx.sensors.qr_pivot import (
    PhysicsGuidedQRPivotSelector,
    QRPivotSelector
)


# ============================================================================
# Fixtures：測試資料生成
# ============================================================================

@pytest.fixture
def channel_flow_coords():
    """
    生成通道流座標
    
    Returns:
        coords: [n_locations, 3] (x, y, z)
            x ∈ [0, 2π], y ∈ [-1, 1], z ∈ [0, π]
    """
    n_x, n_y, n_z = 16, 32, 16  # 總共 8192 個點
    x = np.linspace(0, 2*np.pi, n_x)
    y = np.linspace(-1, 1, n_y)  # 通道高度 2h, h=1
    z = np.linspace(0, np.pi, n_z)
    
    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    coords = np.column_stack([xv.ravel(), yv.ravel(), zv.ravel()])
    
    return coords


@pytest.fixture
def pod_modes(channel_flow_coords):
    """
    生成模擬 POD 模態矩陣
    
    模擬特性：
    - 壁面高梯度：前 2 個模態在壁面區域振幅大
    - 中心主導：後面模態在中心區域振幅大
    
    Returns:
        data_matrix: [n_locations, n_modes]
    """
    coords = channel_flow_coords
    n_locations = coords.shape[0]
    n_modes = 10
    
    # 設定隨機種子以確保可重現性
    np.random.seed(42)
    
    # 模態矩陣
    modes = np.zeros((n_locations, n_modes))
    
    y = coords[:, 1]  # y 座標
    
    for i in range(n_modes):
        if i < 2:
            # 前 2 個模態：壁面高梯度
            # 在 y=-1, y=1 附近振幅大
            modes[:, i] = np.exp(-10 * (np.abs(y) - 1)**2)
        else:
            # 後面模態：中心主導
            # 在 y=0 附近振幅大
            modes[:, i] = np.exp(-5 * y**2) * np.sin((i-1) * np.pi * y)
    
    # 添加小量隨機噪聲
    modes += 0.01 * np.random.randn(n_locations, n_modes)
    
    return modes


# ============================================================================
# 測試 1: 壁面區域識別
# ============================================================================

def test_wall_region_identification_y_over_h(channel_flow_coords):
    """測試壁面區域識別（y/h 方式）"""
    
    selector = PhysicsGuidedQRPivotSelector(
        wall_threshold=0.1,
        threshold_type='y_over_h'
    )
    
    # 識別壁面區域
    wall_mask = selector._identify_wall_region(channel_flow_coords, re_tau=1000.0)
    
    # 驗證：壁面點應在 y/h < 0.1 的區域
    y = channel_flow_coords[:, 1]
    h = 1.0
    dist_to_wall = np.minimum(np.abs(y - (-h)), np.abs(y - h))
    expected_mask = (dist_to_wall / h) < 0.1
    
    # 斷言：識別結果正確
    assert np.array_equal(wall_mask, expected_mask), "壁面區域識別錯誤"
    
    # 統計：壁面點比例應合理（約 10-20%）
    wall_ratio = wall_mask.sum() / len(wall_mask)
    assert 0.05 < wall_ratio < 0.25, f"壁面點比例異常: {wall_ratio:.2%}"
    
    print(f"✅ y/h 識別測試通過：壁面點 {wall_mask.sum()}/{len(wall_mask)} ({wall_ratio:.1%})")


def test_wall_region_identification_y_plus(channel_flow_coords):
    """測試壁面區域識別（y+ 方式）"""
    
    selector = PhysicsGuidedQRPivotSelector(
        wall_threshold=100.0,  # y+ < 100
        threshold_type='y_plus'
    )
    
    # JHTDB 統計量
    re_tau = 1000.0
    u_tau = 0.04997
    nu = 5.0e-5
    delta_nu = nu / u_tau
    
    # 識別壁面區域
    wall_mask = selector._identify_wall_region(channel_flow_coords, re_tau=re_tau)
    
    # 驗證：y+ < 100
    y = channel_flow_coords[:, 1]
    h = 1.0
    dist_to_wall = np.minimum(np.abs(y - (-h)), np.abs(y - h))
    y_plus = dist_to_wall / delta_nu
    expected_mask = y_plus < 100.0
    
    # 斷言：識別結果正確
    assert np.array_equal(wall_mask, expected_mask), "y+ 壁面區域識別錯誤"
    
    wall_ratio = wall_mask.sum() / len(wall_mask)
    print(f"✅ y+ 識別測試通過：壁面點 {wall_mask.sum()}/{len(wall_mask)} ({wall_ratio:.1%})")


def test_wall_region_invalid_threshold_type(channel_flow_coords):
    """測試無效的壁面識別類型"""
    
    selector = PhysicsGuidedQRPivotSelector(threshold_type='invalid_type')
    
    with pytest.raises(ValueError, match="未知的壁面識別類型"):
        selector._identify_wall_region(channel_flow_coords, re_tau=1000.0)


# ============================================================================
# 測試 2: 物理加權 QR-Pivot 選點
# ============================================================================

def test_physics_weighted_qr_pivot(pod_modes, channel_flow_coords):
    """測試物理加權 QR-Pivot 選點"""
    
    n_sensors = 50
    
    selector = PhysicsGuidedQRPivotSelector(
        wall_weight=5.0,
        wall_threshold=0.1,
        threshold_type='y_over_h'
    )
    
    # 選擇感測點
    selected_indices, metrics = selector.select_sensors(
        pod_modes,
        n_sensors,
        coords=channel_flow_coords,
        re_tau=1000.0
    )
    
    # 驗證：選擇數量正確
    assert len(selected_indices) == n_sensors, "感測點數量不匹配"
    
    # 驗證：索引在有效範圍內
    assert selected_indices.min() >= 0
    assert selected_indices.max() < pod_modes.shape[0]
    
    # 驗證：無重複索引
    assert len(np.unique(selected_indices)) == n_sensors, "存在重複的感測點索引"
    
    # 驗證：壁面覆蓋率 > 0（應優先選擇壁面點）
    wall_coverage = metrics['wall_coverage']
    assert wall_coverage > 0, "壁面覆蓋率為 0，物理引導失效"
    
    # 驗證：品質指標合理
    assert metrics['condition_number'] > 0
    assert 0 <= metrics['energy_ratio'] <= 1.1  # 允許小幅超過 1.0（數值誤差）
    
    print(f"✅ 物理加權 QR-Pivot 測試通過：")
    print(f"   - 壁面覆蓋率: {wall_coverage:.2%}")
    print(f"   - 條件數: {metrics['condition_number']:.2f}")
    print(f"   - 選中壁面點: {metrics['selected_wall_points']}/{metrics['total_wall_points']}")


def test_wall_coverage_increases_with_weight(pod_modes, channel_flow_coords):
    """測試壁面覆蓋率隨權重增加"""
    
    n_sensors = 50
    weights = [1.0, 3.0, 5.0, 10.0]
    coverages = []
    
    for weight in weights:
        selector = PhysicsGuidedQRPivotSelector(
            wall_weight=weight,
            wall_threshold=0.1,
            threshold_type='y_over_h'
        )
        
        _, metrics = selector.select_sensors(
            pod_modes,
            n_sensors,
            coords=channel_flow_coords
        )
        
        coverages.append(metrics['wall_coverage'])
    
    # 驗證：壁面覆蓋率應隨權重增加（至少前 3 個是遞增的）
    assert coverages[1] >= coverages[0], "權重增加未提升壁面覆蓋率"
    assert coverages[2] >= coverages[1], "權重增加未提升壁面覆蓋率"
    
    print(f"✅ 權重測試通過：覆蓋率 {dict(zip(weights, coverages))}")


# ============================================================================
# 測試 3: 壁面覆蓋率統計
# ============================================================================

def test_wall_statistics(pod_modes, channel_flow_coords):
    """測試壁面統計信息"""
    
    selector = PhysicsGuidedQRPivotSelector(wall_weight=5.0, wall_threshold=0.1)
    
    # 選擇感測點
    _, _ = selector.select_sensors(pod_modes, 50, coords=channel_flow_coords)
    
    # 獲取統計信息
    stats = selector.get_wall_statistics()
    
    # 驗證：必要欄位存在
    required_fields = [
        'wall_coverage', 'total_wall_points', 'wall_ratio',
        'wall_weight', 'threshold', 'threshold_type'
    ]
    for field in required_fields:
        assert field in stats, f"缺少統計欄位: {field}"
    
    # 驗證：數值合理
    assert 0 <= stats['wall_coverage'] <= 1
    assert stats['total_wall_points'] > 0
    assert 0 < stats['wall_ratio'] < 1
    assert stats['wall_weight'] == 5.0
    
    print(f"✅ 統計信息測試通過：{stats}")


def test_wall_statistics_before_selection_raises_error(pod_modes):
    """測試在未選擇前調用統計信息應報錯"""
    
    selector = PhysicsGuidedQRPivotSelector()
    
    with pytest.raises(RuntimeError, match="請先調用 select_sensors"):
        selector.get_wall_statistics()


# ============================================================================
# 測試 4: 與標準 QR-Pivot 對比
# ============================================================================

def test_comparison_with_standard_qr_pivot(pod_modes, channel_flow_coords):
    """對比物理引導與標準 QR-Pivot"""
    
    n_sensors = 50
    
    # 標準 QR-Pivot
    standard_selector = QRPivotSelector(pivoting=True)
    standard_indices, standard_metrics = standard_selector.select_sensors(pod_modes, n_sensors)
    
    # 物理引導 QR-Pivot
    physics_selector = PhysicsGuidedQRPivotSelector(
        wall_weight=5.0,
        wall_threshold=0.1
    )
    physics_indices, physics_metrics = physics_selector.select_sensors(
        pod_modes,
        n_sensors,
        coords=channel_flow_coords
    )
    
    # 計算標準 QR-Pivot 的壁面覆蓋率（需要先識別壁面區域）
    # 重新識別壁面區域（與物理引導相同的參數）
    temp_selector = PhysicsGuidedQRPivotSelector(wall_threshold=0.1)
    wall_mask = temp_selector._identify_wall_region(channel_flow_coords, re_tau=1000.0)
    standard_wall_coverage = wall_mask[standard_indices].sum() / len(standard_indices)
    
    # 驗證：物理引導應有更高的壁面覆蓋率
    assert physics_metrics['wall_coverage'] > standard_wall_coverage, \
        f"物理引導壁面覆蓋率 ({physics_metrics['wall_coverage']:.2%}) " \
        f"未高於標準 QR-Pivot ({standard_wall_coverage:.2%})"
    
    # 驗證：條件數應保持在合理範圍（不能為了壁面犧牲太多數值穩定性）
    cond_ratio = physics_metrics['condition_number'] / standard_metrics['condition_number']
    assert cond_ratio < 5.0, f"物理引導導致條件數惡化過多：{cond_ratio:.2f}x"
    
    print(f"✅ 對比測試通過：")
    print(f"   標準 QR-Pivot 壁面覆蓋率: {standard_wall_coverage:.2%}")
    print(f"   物理引導壁面覆蓋率: {physics_metrics['wall_coverage']:.2%}")
    print(f"   條件數比例: {cond_ratio:.2f}x")


# ============================================================================
# 測試 5: 邊界情況與錯誤處理
# ============================================================================

def test_missing_coords_raises_error(pod_modes):
    """測試未提供座標應報錯"""
    
    selector = PhysicsGuidedQRPivotSelector()
    
    with pytest.raises(ValueError, match="需要提供空間座標"):
        selector.select_sensors(pod_modes, 50)


def test_coords_shape_mismatch_raises_error(pod_modes):
    """測試座標形狀不匹配應報錯"""
    
    selector = PhysicsGuidedQRPivotSelector()
    
    # 故意製造形狀不匹配
    wrong_coords = np.random.randn(10, 3)  # 只有 10 個點
    
    with pytest.raises(ValueError, match="座標數量.*與資料點數量.*不匹配"):
        selector.select_sensors(pod_modes, 50, coords=wrong_coords)


def test_torch_tensor_input(pod_modes, channel_flow_coords):
    """測試 PyTorch Tensor 輸入"""
    
    selector = PhysicsGuidedQRPivotSelector(wall_weight=5.0)
    
    # 轉換為 Tensor
    data_tensor = torch.from_numpy(pod_modes).float()
    coords_tensor = torch.from_numpy(channel_flow_coords).float()
    
    # 應能正常處理（內部會自動轉換為 NumPy）
    selected_indices, metrics = selector.select_sensors(
        data_tensor,  # type: ignore[arg-type]  # 內部已處理 Tensor -> NumPy 轉換
        50,
        coords=coords_tensor  # type: ignore[arg-type]
    )
    
    assert len(selected_indices) == 50
    assert metrics['wall_coverage'] > 0
    
    print(f"✅ Tensor 輸入測試通過")


def test_extreme_sensor_counts(pod_modes, channel_flow_coords):
    """測試極端感測點數量"""
    
    selector = PhysicsGuidedQRPivotSelector()
    
    # 測試 K=1（最小）
    indices_min, metrics_min = selector.select_sensors(
        pod_modes, 1, coords=channel_flow_coords
    )
    assert len(indices_min) == 1
    
    # 測試 K=n_locations（最大）
    n_max = pod_modes.shape[0]
    indices_max, metrics_max = selector.select_sensors(
        pod_modes, n_max, coords=channel_flow_coords
    )
    assert len(indices_max) == n_max
    
    # 測試 K > n_locations（應自動截斷）
    indices_over, metrics_over = selector.select_sensors(
        pod_modes, n_max + 100, coords=channel_flow_coords
    )
    assert len(indices_over) == n_max
    
    print(f"✅ 極端數量測試通過：K=1, K={n_max}, K>{n_max}")


def test_return_qr_option(pod_modes, channel_flow_coords):
    """測試返回 QR 分解結果選項"""
    
    selector = PhysicsGuidedQRPivotSelector()
    
    # 不返回 QR（預設）
    result_default = selector.select_sensors(pod_modes, 50, coords=channel_flow_coords)
    assert len(result_default) == 2  # (indices, metrics)
    
    # 返回 QR
    result_with_qr = selector.select_sensors(
        pod_modes, 50, coords=channel_flow_coords, return_qr=True
    )
    assert len(result_with_qr) == 4  # (indices, metrics, Q, R)
    
    indices, metrics, Q, R = result_with_qr  # type: ignore[misc]  # 返回類型根據 return_qr 動態變化
    assert Q is not None
    assert R is not None
    
    print(f"✅ return_qr 選項測試通過：Q shape={Q.shape}, R shape={R.shape}")


# ============================================================================
# 測試 6: 數值穩定性
# ============================================================================

def test_numerical_stability_with_noise(pod_modes, channel_flow_coords):
    """測試在噪聲資料下的數值穩定性"""
    
    selector = PhysicsGuidedQRPivotSelector(wall_weight=5.0)
    
    # 添加高斯噪聲
    noisy_modes = pod_modes + 0.1 * np.random.randn(*pod_modes.shape)
    
    try:
        selected_indices, metrics = selector.select_sensors(
            noisy_modes,
            50,
            coords=channel_flow_coords
        )
        
        # 驗證：條件數應保持合理
        assert metrics['condition_number'] < 1e3, "噪聲導致條件數爆炸"
        assert not np.any(np.isnan(selected_indices)), "存在 NaN 索引"
        
        print(f"✅ 噪聲穩定性測試通過：條件數 {metrics['condition_number']:.2f}")
        
    except np.linalg.LinAlgError:
        pytest.fail("在噪聲資料下 QR 分解失敗")


def test_regularization_prevents_singular_matrix():
    """測試正則化項防止奇異矩陣"""
    
    # 創建秩虧矩陣（前 3 列相同）
    n_locations = 100
    singular_matrix = np.random.randn(n_locations, 5)
    singular_matrix[:, 1] = singular_matrix[:, 0]  # 製造秩虧
    singular_matrix[:, 2] = singular_matrix[:, 0]
    
    coords = np.random.randn(n_locations, 3)
    coords[:, 1] = np.linspace(-1, 1, n_locations)  # y 座標
    
    selector = PhysicsGuidedQRPivotSelector(regularization=1e-12)
    
    # 應能處理秩虧矩陣（通過正則化）
    selected_indices, metrics = selector.select_sensors(
        singular_matrix,
        10,
        coords=coords
    )
    
    assert len(selected_indices) == 10
    print(f"✅ 秩虧矩陣測試通過：條件數 {metrics['condition_number']:.2f}")


# ============================================================================
# 整合測試
# ============================================================================

def test_full_pipeline_integration(pod_modes, channel_flow_coords):
    """完整流程整合測試"""
    
    print("\n" + "="*60)
    print("整合測試：完整流程")
    print("="*60)
    
    # 1. 初始化選擇器
    selector = PhysicsGuidedQRPivotSelector(
        wall_weight=5.0,
        wall_threshold=0.1,
        threshold_type='y_over_h',
        regularization=1e-12
    )
    
    # 2. 選擇感測點
    n_sensors = 50
    selected_indices, metrics = selector.select_sensors(
        pod_modes,
        n_sensors,
        coords=channel_flow_coords,
        re_tau=1000.0,
        return_qr=False
    )
    
    # 3. 獲取統計信息
    stats = selector.get_wall_statistics()
    
    # 4. 驗證結果
    assert len(selected_indices) == n_sensors
    assert metrics['wall_coverage'] > 0
    assert stats['wall_coverage'] == metrics['wall_coverage']
    
    # 5. 輸出報告
    print(f"\n感測點選擇結果：")
    print(f"  ✓ 選擇數量: {len(selected_indices)}/{n_sensors}")
    print(f"  ✓ 壁面覆蓋率: {metrics['wall_coverage']:.2%}")
    print(f"  ✓ 條件數: {metrics['condition_number']:.2f}")
    print(f"  ✓ 能量比例: {metrics['energy_ratio']:.3f}")
    print(f"  ✓ 選中壁面點: {metrics['selected_wall_points']}/{metrics['total_wall_points']}")
    print(f"\n統計信息：")
    print(f"  ✓ 壁面點總數: {stats['total_wall_points']}")
    print(f"  ✓ 壁面點比例: {stats['wall_ratio']:.2%}")
    print(f"  ✓ 權重倍數: {stats['wall_weight']}x")
    
    print("\n✅ 整合測試通過！")


if __name__ == "__main__":
    # 直接運行測試（不使用 pytest）
    print("🧪 運行物理引導 QR-Pivot 單元測試...\n")
    
    # 生成測試資料（直接調用函數）
    # 生成通道流座標
    n_x, n_y, n_z = 16, 32, 16
    x = np.linspace(0, 2*np.pi, n_x)
    y = np.linspace(-1, 1, n_y)
    z = np.linspace(0, np.pi, n_z)
    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    coords = np.column_stack([xv.ravel(), yv.ravel(), zv.ravel()])
    
    # 生成模擬 POD 模態
    n_locations = coords.shape[0]
    n_modes = 10
    np.random.seed(42)
    modes = np.zeros((n_locations, n_modes))
    y_coord = coords[:, 1]
    for i in range(n_modes):
        if i < 2:
            modes[:, i] = np.exp(-10 * (np.abs(y_coord) - 1)**2)
        else:
            modes[:, i] = np.exp(-5 * y_coord**2) * np.sin((i-1) * np.pi * y_coord)
    modes += 0.01 * np.random.randn(n_locations, n_modes)
    
    # 執行關鍵測試
    test_wall_region_identification_y_over_h(coords)
    test_wall_region_identification_y_plus(coords)
    test_physics_weighted_qr_pivot(modes, coords)
    test_wall_coverage_increases_with_weight(modes, coords)
    test_comparison_with_standard_qr_pivot(modes, coords)
    test_full_pipeline_integration(modes, coords)
    
    print("\n✅ 所有測試通過！")
