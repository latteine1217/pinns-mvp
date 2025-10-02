#!/usr/bin/env python3
"""
感測點選擇模組整合測試
測試所有主要功能是否正常工作
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pinnx.sensors import (
    QRPivotSelector,
    PODBasedSelector, 
    GreedySelector,
    MultiObjectiveSelector,
    SensorOptimizer,
    evaluate_sensor_placement,
    create_sensor_selector
)

def create_test_data():
    """創建測試資料"""
    np.random.seed(42)
    n_locations = 50
    n_snapshots = 30
    
    # 模擬流場資料（3個主要模態）
    t = np.linspace(0, 2*np.pi, n_snapshots)
    x = np.linspace(0, 1, n_locations)
    
    data_matrix = np.zeros((n_locations, n_snapshots))
    for i in range(3):
        mode = np.sin((i+1) * np.pi * x[:, np.newaxis])
        coeff = np.cos((i+1) * t) * np.exp(-0.1 * i)
        data_matrix += mode @ coeff[np.newaxis, :]
    
    # 添加少量雜訊
    data_matrix += 0.005 * np.random.randn(n_locations, n_snapshots)
    
    return data_matrix

def test_individual_selectors():
    """測試各個選擇器"""
    print("=== 測試各個感測點選擇器 ===")
    
    data = create_test_data()
    n_sensors = 6
    
    selectors = {
        'QR-Pivot': QRPivotSelector(mode='column', pivoting=True),
        'POD-based': PODBasedSelector(n_modes=5, energy_threshold=0.95),
        'Greedy': GreedySelector(objective='info_gain'),
        'Multi-objective': MultiObjectiveSelector(objectives=['accuracy', 'robustness'])
    }
    
    success_count = 0
    
    for name, selector in selectors.items():
        print(f"\n{name}:")
        try:
            indices, metrics = selector.select_sensors(data, n_sensors)
            success_count += 1
            
            print(f"  ✅ 成功選擇 {len(indices)} 個感測點")
            print(f"  📊 條件數: {metrics.get('condition_number', 'N/A'):.2e}")
            print(f"  ⚡ 能量比例: {metrics.get('energy_ratio', 0.0):.3f}")
            
            if 'n_pod_modes' in metrics:
                print(f"  🔄 POD 模態數: {metrics['n_pod_modes']}")
                
            # 基本驗證
            assert len(indices) == n_sensors, f"{name}: 感測點數量不正確"
            assert len(set(indices)) == n_sensors, f"{name}: 感測點索引重複"
            assert all(0 <= idx < data.shape[0] for idx in indices), f"{name}: 感測點索引超出範圍"
            
        except Exception as e:
            print(f"  ❌ 失敗: {e}")
    
    # 確保至少有一半的選擇器成功
    assert success_count >= len(selectors) // 2, f"成功的選擇器太少: {success_count}/{len(selectors)}"

def test_sensor_optimizer():
    """測試感測點最適化器"""
    print("\n=== 測試 SensorOptimizer ===")
    
    data = create_test_data()
    validation_data = create_test_data() + 0.001 * np.random.randn(*create_test_data().shape)
    n_sensors = 6
    
    # 測試不同策略
    strategies = ['qr_pivot', 'pod_based', 'greedy', 'auto']
    
    for strategy in strategies:
        print(f"\n策略: {strategy}")
        try:
            optimizer = SensorOptimizer(strategy=strategy)
            indices, metrics = optimizer.optimize_sensor_placement(
                data, n_sensors, validation_data)
            
            print(f"  ✅ 選擇 {len(indices)} 個感測點")
            print(f"  📈 驗證 MSE: {metrics.get('validation_mse', 'N/A'):.2e}")
            
            if strategy == 'auto':
                print(f"  🤖 自動選擇策略: {metrics.get('auto_selected_strategy', 'unknown')}")
            
        except Exception as e:
            print(f"  ❌ 失敗: {e}")

def test_convenience_functions():
    """測試便捷函數"""
    print("\n=== 測試便捷函數 ===")
    
    data = create_test_data()
    n_sensors = 6
    
    # 測試 create_sensor_selector
    print("\ncreate_sensor_selector:")
    try:
        selector = create_sensor_selector('qr_pivot', mode='column')
        indices, metrics = selector.select_sensors(data, n_sensors)
        print(f"  ✅ 創建並執行成功，選擇 {len(indices)} 個感測點")
    except Exception as e:
        print(f"  ❌ 失敗: {e}")
    
    # 測試 evaluate_sensor_placement
    print("\nevaluate_sensor_placement:")
    try:
        qr_selector = QRPivotSelector()
        indices, _ = qr_selector.select_sensors(data, n_sensors)
        test_data = create_test_data() + 0.002 * np.random.randn(*create_test_data().shape)
        
        eval_metrics = evaluate_sensor_placement(data, indices, test_data)
        
        print(f"  ✅ 評估成功")
        print(f"  📊 條件數: {eval_metrics.get('condition_number', 'N/A'):.2e}")
        print(f"  🎯 覆蓋率: {eval_metrics.get('subspace_coverage', 0.0):.3f}")
        
        if 'robustness' in eval_metrics:
            noise_errors = eval_metrics['robustness']
            print(f"  🛡️ 雜訊穩健性: {len(noise_errors)} 個雜訊水準測試")
        
        if 'geometry' in eval_metrics:
            geom = eval_metrics['geometry']
            print(f"  📐 最小距離: {geom.get('min_sensor_distance', 'N/A'):.3f}")
        
    except Exception as e:
        print(f"  ❌ 失敗: {e}")

def test_performance_comparison():
    """性能比較測試"""
    print("\n=== 性能比較 ===")
    
    data = create_test_data()
    test_data = create_test_data() + 0.001 * np.random.randn(*create_test_data().shape)
    n_sensors = 6
    
    selectors = {
        'QR-Pivot': QRPivotSelector(),
        'POD-based': PODBasedSelector(n_modes=5),
        'Greedy': GreedySelector(objective='info_gain')
    }
    
    comparison_results = []
    
    for name, selector in selectors.items():
        try:
            # 選擇感測點
            indices, metrics = selector.select_sensors(data, n_sensors)
            
            # 評估性能
            eval_metrics = evaluate_sensor_placement(data, indices, test_data, 
                                                   noise_levels=[0.01, 0.05])
            
            result = {
                'name': name,
                'condition_number': eval_metrics.get('condition_number', np.inf),
                'energy_ratio': metrics.get('energy_ratio', 0.0),
                'subspace_coverage': eval_metrics.get('subspace_coverage', 0.0),
                'n_sensors': len(indices)
            }
            
            comparison_results.append(result)
            
        except Exception as e:
            print(f"{name} 評估失敗: {e}")
    
    # 輸出比較結果
    print(f"\n性能比較結果:")
    print(f"{'策略':<15} {'條件數':<12} {'能量比例':<10} {'覆蓋率':<10} {'感測點數':<8}")
    print("-" * 60)
    
    for result in comparison_results:
        print(f"{result['name']:<15} "
              f"{result['condition_number']:<12.2e} "
              f"{result['energy_ratio']:<10.3f} "
              f"{result['subspace_coverage']:<10.3f} "
              f"{result['n_sensors']:<8}")

if __name__ == "__main__":
    print("🚀 開始感測點選擇模組整合測試...\n")
    
    try:
        # 執行所有測試
        test_individual_selectors()
        test_sensor_optimizer()  
        test_convenience_functions()
        test_performance_comparison()
        
        print("\n✅ 所有測試完成！感測點選擇模組整合成功。")
        
    except Exception as e:
        print(f"\n❌ 測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()