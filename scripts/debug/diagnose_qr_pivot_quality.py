#!/usr/bin/env python3
"""
QR-Pivot 品質異常診斷與修正
診斷為何條件數達到 1.43e13，並提供正確的實現方案
"""

import numpy as np
import sys
from pathlib import Path
from scipy.linalg import qr, svd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from pinnx.sensors.qr_pivot import QRPivotSelector


def diagnose_current_sensor_file():
    """診斷現有感測點檔案的問題"""
    print("="*80)
    print("🔍 診斷現有 K=50 感測點檔案")
    print("="*80)
    
    sensor_file = project_root / "data/jhtdb/channel_flow_re1000/sensors_K50_qr_pivot.npz"
    
    if not sensor_file.exists():
        print(f"❌ 檔案不存在: {sensor_file}")
        return
    
    data = np.load(sensor_file, allow_pickle=True)
    
    print("\n📊 檔案內容：")
    for key in data.files:
        item = data[key]
        if hasattr(item, 'shape'):
            print(f"  {key}: shape={item.shape}, dtype={item.dtype}")
        else:
            print(f"  {key}: {type(item)}")
    
    # 讀取品質指標
    if 'selection_info' in data:
        info = data['selection_info'].item()
        print("\n⚠️  儲存的品質指標（異常）：")
        print(f"  條件數: {info['condition_number']:.2e}")
        print(f"  能量比例: {info['energy_ratio']:.2%}")
        print(f"  子空間覆蓋: {info['subspace_coverage']:.2%}")
        print(f"  log行列式: {info['log_determinant']:.4f}")
        
        # 診斷
        print("\n🔬 問題診斷：")
        if info['condition_number'] > 1e10:
            print("  ❌ 條件數異常高（> 1e10）")
            print("     → 可能原因：資料矩陣秩不足或尺度問題")
        
        if info['energy_ratio'] < 0.5:
            print("  ❌ 能量比例過低（< 50%）")
            print("     → 可能原因：資料矩陣特徵數不足（僅 3 個）")
        
        if info['subspace_coverage'] < 0.1:
            print("  ❌ 子空間覆蓋度接近 0")
            print("     → 可能原因：演算法實現錯誤或資料格式不正確")


def demonstrate_problem():
    """演示問題：單快照 vs 多快照的差異"""
    print("\n" + "="*80)
    print("🧪 實驗：單快照 vs 多快照的 QR-Pivot 品質比較")
    print("="*80)
    
    np.random.seed(42)
    n_locations = 100
    K = 10
    
    # ========================================
    # 情境 1: 單快照（當前錯誤做法）
    # ========================================
    print("\n📉 情境 1: 單快照（3個特徵：u, v, w）")
    print("-"*80)
    
    # 模擬單時間步的速度場 [n_locations, 3]
    single_snapshot = np.random.randn(n_locations, 3)
    
    print(f"  資料矩陣形狀: {single_snapshot.shape}")
    print(f"  特徵數: {single_snapshot.shape[1]}")
    print(f"  秩: {np.linalg.matrix_rank(single_snapshot)}")
    
    # 使用 QR-Pivot 選點
    selector = QRPivotSelector(mode='row', pivoting=True)
    indices_1, metrics_1 = selector.select_sensors(single_snapshot, K)
    
    print(f"\n  品質指標：")
    print(f"    條件數: {metrics_1['condition_number']:.2e}")
    print(f"    能量比例: {metrics_1['energy_ratio']:.2%}")
    print(f"    子空間覆蓋: {metrics_1['subspace_coverage']:.2%}")
    
    # ========================================
    # 情境 2: 多快照（正確做法）
    # ========================================
    print("\n📈 情境 2: 多快照（50個時間步）")
    print("-"*80)
    
    # 模擬多時間步的速度場快照 [n_locations, n_snapshots]
    n_snapshots = 50
    
    # 創建含有空間結構的資料（使用 Fourier 模態）
    x = np.linspace(0, 2*np.pi, n_locations)
    t = np.linspace(0, 2*np.pi, n_snapshots)
    
    multi_snapshot = np.zeros((n_locations, n_snapshots))
    for k in range(1, 6):  # 5 個主要模態
        spatial_mode = np.sin(k * x) + 0.5 * np.cos(k * x)
        temporal_coeff = np.cos(k * t) * np.exp(-0.1 * k)
        multi_snapshot += spatial_mode[:, np.newaxis] * temporal_coeff[np.newaxis, :]
    
    # 添加雜訊
    multi_snapshot += 0.01 * np.random.randn(n_locations, n_snapshots)
    
    print(f"  資料矩陣形狀: {multi_snapshot.shape}")
    print(f"  特徵數: {multi_snapshot.shape[1]}")
    print(f"  秩: {np.linalg.matrix_rank(multi_snapshot)}")
    
    # 使用 QR-Pivot 選點
    indices_2, metrics_2 = selector.select_sensors(multi_snapshot, K)
    
    print(f"\n  品質指標：")
    print(f"    條件數: {metrics_2['condition_number']:.2e}")
    print(f"    能量比例: {metrics_2['energy_ratio']:.2%}")
    print(f"    子空間覆蓋: {metrics_2['subspace_coverage']:.2%}")
    
    # ========================================
    # 情境 3: POD 模態（推薦做法）
    # ========================================
    print("\n⭐ 情境 3: POD 模態（最佳實踐）")
    print("-"*80)
    
    # 從多快照中提取 POD 模態
    U, s, Vt = svd(multi_snapshot, full_matrices=False)
    
    # 選擇前 20 個模態（保留 >99% 能量）
    n_modes = 20
    pod_modes = U[:, :n_modes]
    energy_retained = np.sum(s[:n_modes]**2) / np.sum(s**2)
    
    print(f"  POD 模態形狀: {pod_modes.shape}")
    print(f"  保留模態數: {n_modes}")
    print(f"  保留能量: {energy_retained:.2%}")
    
    # 使用加權 POD 模態（使用奇異值作為權重）
    weighted_modes = pod_modes * s[:n_modes][np.newaxis, :]
    
    indices_3, metrics_3 = selector.select_sensors(weighted_modes, K)
    
    print(f"\n  品質指標：")
    print(f"    條件數: {metrics_3['condition_number']:.2e}")
    print(f"    能量比例: {metrics_3['energy_ratio']:.2%}")
    print(f"    子空間覆蓋: {metrics_3['subspace_coverage']:.2%}")
    
    # ========================================
    # 比較總結
    # ========================================
    print("\n" + "="*80)
    print("📊 三種方法比較總結")
    print("="*80)
    
    comparison_table = f"""
    {"方法":<20} {"條件數":<15} {"能量比例":<12} {"子空間覆蓋":<12} {"推薦度"}
    {"-"*80}
    {"單快照 (錯誤)":<20} {metrics_1['condition_number']:>12.2e}   {metrics_1['energy_ratio']:>9.1%}   {metrics_1['subspace_coverage']:>9.1%}     {"❌"}
    {"多快照 (可用)":<20} {metrics_2['condition_number']:>12.2e}   {metrics_2['energy_ratio']:>9.1%}   {metrics_2['subspace_coverage']:>9.1%}     {"✅"}
    {"POD 模態 (最佳)":<20} {metrics_3['condition_number']:>12.2e}   {metrics_3['energy_ratio']:>9.1%}   {metrics_3['subspace_coverage']:>9.1%}     {"⭐"}
    """
    
    print(comparison_table)
    
    # 理想指標參考
    print("\n📋 理想品質指標參考：")
    print("  條件數: < 100 (警告 > 1000)")
    print("  能量比例: > 95% (警告 < 90%)")
    print("  子空間覆蓋: > 90% (警告 < 80%)")


def propose_fix():
    """提出修正方案"""
    print("\n" + "="*80)
    print("🔧 修正方案")
    print("="*80)
    
    print("""
修正 visualize_qr_sensors.py 中的 compute_sensors_from_jhtdb() 函數：

當前錯誤實現（第 207 行）：
-----------------------------------
    # ❌ 錯誤：僅使用單快照的速度場
    data_matrix = velocities  # [n_locations, 3]

建議修正方案 1（多快照）：
-----------------------------------
    # ✅ 方案 1: 如果有時間序列資料，使用多快照
    if 'time' in data and len(data['time']) > 1:
        # 假設資料是 [n_time, n_locations, 3]
        u_snapshots = data['u']  # [n_time, nx, ny, nz]
        
        # 重塑為 [n_locations, n_time]
        n_snapshots = u_snapshots.shape[0]
        u_flat = u_snapshots.reshape(n_snapshots, -1).T  # [n_locations, n_time]
        
        # 可選：同時考慮 u, v, w 的快照
        data_matrix = u_flat  # 或堆疊 [u_flat, v_flat, w_flat]
    
    else:
        # 單快照情況：回退到 POD 模態
        print("⚠️  僅有單快照，建議使用 POD-based 選點策略")
        # 使用 PODBasedSelector 代替 QRPivotSelector

建議修正方案 2（POD 模態，推薦）：
-----------------------------------
    from pinnx.sensors.qr_pivot import PODBasedSelector
    
    # ✅ 方案 2: 如果資料有空間相關性，先提取 POD 模態
    if 'u' in data:
        # 建立增強資料矩陣（包含多個物理量）
        u = data['u'].flatten()[:, np.newaxis]
        v = data['v'].flatten()[:, np.newaxis]
        w = data['w'].flatten()[:, np.newaxis]
        
        # 如果有 RANS 基線或歷史資料，可作為快照
        if 'u_baseline' in data:
            u_snapshots = np.stack([data[f'u_t{i}'] for i in range(n_time)], axis=1)
            data_matrix = u_snapshots  # [n_locations, n_time]
        else:
            # 使用梯度資訊增強特徵
            # 計算空間梯度（如果是結構化網格）
            # 這樣可以從 3 個特徵擴展到更多
            pass
    
    # 使用 POD-based 選點器
    selector = PODBasedSelector(n_modes=min(20, n_sensors), 
                                energy_threshold=0.99,
                                mode_weighting='energy')

建議修正方案 3（使用現有 JHTDB 資料）：
-----------------------------------
    # ✅ 方案 3: 如果有完整的 JHTDB 資料庫存取
    # 下載多個時間步的速度場快照
    
    from pinnx.dataio.jhtdb_client import fetch_cutout_temporal
    
    # 獲取時間序列資料
    time_range = np.linspace(0, 10, 50)  # 50 個時間步
    snapshots = []
    
    for t in time_range:
        u_t = fetch_cutout_temporal(time=t, field='u', ...)
        snapshots.append(u_t.flatten())
    
    data_matrix = np.stack(snapshots, axis=1)  # [n_locations, n_time]
    
    # 現在可以安全地使用 QR-Pivot
    selector = QRPivotSelector(mode='row', pivoting=True)
    selected_indices, metrics = selector.select_sensors(data_matrix, n_sensors)

""")
    
    print("\n💡 建議的執行步驟：")
    print("  1. 檢查現有 JHTDB 資料是否包含多時間步")
    print("  2. 如果有：使用多快照矩陣 [n_locations, n_time]")
    print("  3. 如果沒有：改用 PODBasedSelector 或下載時間序列資料")
    print("  4. 重新生成 sensors_K50_qr_pivot.npz")
    print("  5. 驗證品質指標（條件數 < 1000，能量比例 > 90%）")


def main():
    """主函數"""
    print("🔬 QR-Pivot 品質異常診斷工具\n")
    
    # 步驟 1: 診斷現有檔案
    diagnose_current_sensor_file()
    
    # 步驟 2: 演示問題
    demonstrate_problem()
    
    # 步驟 3: 提出修正方案
    propose_fix()
    
    print("\n" + "="*80)
    print("✅ 診斷完成")
    print("="*80)
    print("\n下一步：請根據修正方案更新 visualize_qr_sensors.py 中的")
    print("        compute_sensors_from_jhtdb() 函數，並重新生成感測點檔案。")


if __name__ == "__main__":
    main()
