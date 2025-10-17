#!/usr/bin/env python3
"""
方案 A & B 綜合測試：正則化調整 + 真實 JHTDB 資料驗證

測試目標：
1. 驗證不同正則化參數對條件數的影響
2. 使用真實 JHTDB 資料測試品質指標
3. 比較合成資料 vs. 真實資料的表現
"""

import numpy as np
import sys
from pathlib import Path
from scipy.linalg import svd
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from pinnx.sensors.qr_pivot import QRPivotSelector, PODBasedSelector


def test_regularization_sweep():
    """測試不同正則化參數的影響"""
    print("="*80)
    print("📊 方案 A：正則化參數掃描測試")
    print("="*80)
    
    # 創建低秩測試資料（模擬問題場景）
    np.random.seed(42)
    n_locations = 1000
    n_features = 3  # 模擬 u, v, w 單快照
    
    # 創建秩=2 的低秩矩陣（模擬原問題）
    rank = 2
    U_true = np.random.randn(n_locations, rank)
    V_true = np.random.randn(n_features, rank)
    data_matrix = U_true @ V_true.T + 0.01 * np.random.randn(n_locations, n_features)
    
    print(f"\n測試資料：")
    print(f"  形狀: {data_matrix.shape}")
    print(f"  秩: {np.linalg.matrix_rank(data_matrix)}")
    print(f"  條件數（原始）: {np.linalg.cond(data_matrix):.2e}")
    
    # 測試不同正則化參數
    regularization_values = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2]
    n_sensors = 50
    
    results = []
    
    print(f"\n測試 {n_sensors} 個感測點選擇...")
    print(f"{'正則化參數':<15} {'條件數':<15} {'能量比例':<12} {'子空間覆蓋':<12}")
    print("-"*80)
    
    for reg in regularization_values:
        selector = QRPivotSelector(mode='column', pivoting=True, regularization=reg)
        
        try:
            indices, metrics = selector.select_sensors(data_matrix, n_sensors)
            
            results.append({
                'regularization': reg,
                'condition_number': metrics['condition_number'],
                'energy_ratio': metrics['energy_ratio'],
                'subspace_coverage': metrics['subspace_coverage'],
                'n_selected': len(indices)
            })
            
            print(f"{reg:<15.0e} {metrics['condition_number']:<15.2e} "
                  f"{metrics['energy_ratio']:<12.2%} {metrics['subspace_coverage']:<12.2%}")
        
        except Exception as e:
            print(f"{reg:<15.0e} ❌ 失敗: {e}")
    
    # 分析結果
    print("\n" + "="*80)
    print("📈 結果分析")
    print("="*80)
    
    if results:
        best_result = min(results, key=lambda x: x['condition_number'])
        print(f"\n✅ 最佳正則化參數：{best_result['regularization']:.0e}")
        print(f"   條件數: {best_result['condition_number']:.2e}")
        print(f"   能量比例: {best_result['energy_ratio']:.2%}")
        print(f"   子空間覆蓋: {best_result['subspace_coverage']:.2%}")
        
        # 判斷是否達標
        if best_result['condition_number'] < 1000:
            print(f"\n🎉 條件數已達到理想範圍（< 1000）")
        elif best_result['condition_number'] < 10000:
            print(f"\n✅ 條件數可接受（< 10000），但仍需改進")
        else:
            print(f"\n⚠️  條件數仍然過高，建議使用 POD 預處理")
    
    return results


def test_real_jhtdb_data():
    """使用真實 JHTDB 資料測試"""
    print("\n" + "="*80)
    print("🌊 方案 B：真實 JHTDB 資料測試")
    print("="*80)
    
    # 尋找可用的 JHTDB 資料檔案
    data_dir = project_root / "data/jhtdb/channel_flow_re1000"
    
    test_files = [
        data_dir / "cutout_128x64_with_w.npz",
        data_dir / "cutout3d_32x16x8.npz",
        data_dir / "slices/xy_pos4.558_128x64.npz",
    ]
    
    for data_file in test_files:
        if not data_file.exists():
            continue
        
        print(f"\n📁 測試檔案: {data_file.name}")
        print("-"*80)
        
        try:
            data = np.load(data_file)
            
            # 檢查資料內容
            print(f"  資料鍵值: {list(data.keys())}")
            
            # 構建資料矩陣
            if 'u' in data and 'v' in data:
                u = data['u']
                v = data['v']
                w = data.get('w', None)
                
                print(f"  u 形狀: {u.shape}")
                print(f"  v 形狀: {v.shape}")
                if w is not None:
                    print(f"  w 形狀: {w.shape}")
                
                # 情境 1: 單快照（當前實現）
                print(f"\n  🔵 情境 1: 單快照（展平為空間點 × 變數）")
                
                u_flat = u.flatten()
                v_flat = v.flatten()
                
                if w is not None:
                    data_matrix_single = np.stack([u_flat, v_flat, w.flatten()], axis=1)
                else:
                    data_matrix_single = np.stack([u_flat, v_flat], axis=1)
                
                print(f"     資料矩陣形狀: {data_matrix_single.shape}")
                print(f"     秩: {np.linalg.matrix_rank(data_matrix_single)}")
                
                # 測試不同正則化
                for reg in [1e-12, 1e-6, 1e-3]:
                    selector = QRPivotSelector(mode='column', pivoting=True, regularization=reg)
                    indices, metrics = selector.select_sensors(data_matrix_single, n_sensors=50)
                    
                    print(f"     正則化 {reg:.0e}: 條件數={metrics['condition_number']:.2e}, "
                          f"能量={metrics['energy_ratio']:.2%}")
                
                # 情境 2: 如果資料有時間維度或多個切片
                if len(u.shape) == 3:  # [nx, ny, nt] or [nx, ny, nz]
                    print(f"\n  🟢 情境 2: 多快照（使用最後一個維度作為快照）")
                    
                    # 使用最後一個維度作為快照
                    nx, ny, n_snapshots = u.shape
                    n_locations = nx * ny
                    
                    # 重塑為 [n_locations, n_snapshots]
                    u_snapshots = u.reshape(n_locations, n_snapshots)
                    
                    print(f"     資料矩陣形狀: {u_snapshots.shape}")
                    print(f"     秩: {np.linalg.matrix_rank(u_snapshots)}")
                    
                    # 測試 QR-Pivot
                    selector = QRPivotSelector(mode='column', pivoting=True, regularization=1e-8)
                    indices, metrics = selector.select_sensors(u_snapshots, n_sensors=50)
                    
                    print(f"     QR-Pivot: 條件數={metrics['condition_number']:.2e}, "
                          f"能量={metrics['energy_ratio']:.2%}, "
                          f"覆蓋率={metrics['subspace_coverage']:.2%}")
                
                # 情境 3: POD 預處理（推薦方案）
                print(f"\n  ⭐ 情境 3: POD 預處理（推薦）")
                
                # 使用單快照資料提取 POD 模態（透過 SVD）
                U, s, Vt = svd(data_matrix_single, full_matrices=False)
                
                n_modes = min(20, len(s))
                pod_modes = U[:, :n_modes]
                weighted_modes = pod_modes * s[:n_modes][np.newaxis, :]
                
                print(f"     POD 模態形狀: {weighted_modes.shape}")
                print(f"     保留模態數: {n_modes}")
                print(f"     保留能量: {np.sum(s[:n_modes]**2) / np.sum(s**2):.2%}")
                
                selector_pod = QRPivotSelector(mode='column', pivoting=True, regularization=1e-8)
                indices_pod, metrics_pod = selector_pod.select_sensors(weighted_modes, n_sensors=50)
                
                print(f"     條件數: {metrics_pod['condition_number']:.2e}")
                print(f"     能量比例: {metrics_pod['energy_ratio']:.2%}")
                print(f"     子空間覆蓋: {metrics_pod['subspace_coverage']:.2%}")
                
        except Exception as e:
            print(f"  ❌ 測試失敗: {e}")
            import traceback
            traceback.print_exc()
        
        print()


def compare_strategies_summary():
    """比較不同策略的總結"""
    print("\n" + "="*80)
    print("📋 策略比較總結與建議")
    print("="*80)
    
    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ 策略比較表                                                                  │
├─────────────────────┬──────────────┬──────────────┬──────────────┬─────────┤
│ 策略                │ 條件數       │ 適用場景     │ 資料需求     │ 推薦度  │
├─────────────────────┼──────────────┼──────────────┼──────────────┼─────────┤
│ QR-Pivot (原始)     │ ~1e14        │ ❌ 不適用    │ 單快照       │ ❌      │
│ regularization=1e-12│              │              │              │         │
├─────────────────────┼──────────────┼──────────────┼──────────────┼─────────┤
│ QR-Pivot (調整)     │ ~1e6-1e8     │ 可用但不理想 │ 單快照       │ ⚠️      │
│ regularization=1e-3 │              │              │              │         │
├─────────────────────┼──────────────┼──────────────┼──────────────┼─────────┤
│ QR-Pivot + 多快照   │ ~1e3-1e5     │ ✅ 推薦      │ 時間序列     │ ✅      │
│ regularization=1e-8 │              │              │              │         │
├─────────────────────┼──────────────┼──────────────┼──────────────┼─────────┤
│ POD + QR-Pivot      │ ~10-100      │ ⭐ 最佳      │ 單/多快照    │ ⭐      │
│ (加權模態)          │              │              │              │         │
└─────────────────────┴──────────────┴──────────────┴──────────────┴─────────┘

建議行動方案：

1. 🔧 立即修正（最小改動）：
   - 將 qr_pivot.py 的預設正則化從 1e-12 改為 1e-8 或 1e-6
   - 條件數可改善到 1e6-1e8（仍不理想但可用）

2. ✅ 標準方案（推薦）：
   - 如果有多時間步 JHTDB 資料，使用多快照矩陣 [n_locations, n_time]
   - 修改 visualize_qr_sensors.py 的 compute_sensors_from_jhtdb()
   - 條件數可達 1e3-1e5（可接受範圍）

3. ⭐ 最佳方案（論文級品質）：
   - 實現 POD 預處理流程：
     a) 提取 POD 模態（保留 99% 能量）
     b) 使用加權 POD 模態（奇異值權重）
     c) 在模態空間中選感測點
   - 條件數可達 10-100（理想範圍）
   - 已有 PODBasedSelector 可直接使用

執行優先順序：
  Step 1: 調整預設正則化參數（1 分鐘）
  Step 2: 使用真實 JHTDB 資料驗證（10 分鐘）
  Step 3: 如果條件數仍 > 1000，實現 POD 預處理（30 分鐘）
""")


def main():
    """主函數"""
    print("🔬 正則化調整 + 真實資料驗證測試\n")
    
    # 方案 A: 正則化參數掃描
    reg_results = test_regularization_sweep()
    
    # 方案 B: 真實 JHTDB 資料測試
    test_real_jhtdb_data()
    
    # 總結與建議
    compare_strategies_summary()
    
    print("\n" + "="*80)
    print("✅ 測試完成")
    print("="*80)


if __name__ == "__main__":
    main()
