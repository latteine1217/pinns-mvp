#!/usr/bin/env python3
"""
TASK-008 數據載入驗證腳本
驗證修復後的 prepare_training_data() 不會用 Mock 數據覆蓋真實 JHTDB 數據

驗證項目：
1. 感測點數據統計量（應為湍流特徵）
2. 數據載入流程正確性
3. prior_type 參數行為

執行方式：
    python scripts/debug/verify_task008_data_loading.py
"""

import numpy as np
import torch
import sys
from pathlib import Path

# 添加專案根目錄到路徑
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from pinnx.dataio.channel_flow_loader import prepare_training_data

def print_header(title: str):
    """印出格式化標題"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def verify_turbulence_statistics(data: dict, field: str) -> bool:
    """
    驗證數據是否具有湍流特徵
    
    湍流特徵：
    - 流向速度 U: 均值 > 0，波動顯著
    - 法向/展向速度 V, W: 均值 ≈ 0，標準差 > 0
    - 壓力 P: 應有合理分佈
    """
    if field not in data:
        print(f"  ⚠️  欄位 {field} 不存在")
        return False
    
    values = data[field]
    mean = np.mean(values)
    std = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)
    
    print(f"\n  {field} 統計量:")
    print(f"    範圍: [{min_val:.4f}, {max_val:.4f}]")
    print(f"    均值: {mean:.4f}")
    print(f"    標準差: {std:.4f}")
    print(f"    變異係數: {std/abs(mean) if mean != 0 else np.inf:.4f}")
    
    # 驗證規則
    is_valid = True
    
    if field == 'u':
        # 流向速度：應該 > 0，且有湍流波動
        if mean <= 0:
            print(f"  ❌ U 均值應為正值（實際: {mean:.4f}）")
            is_valid = False
        if std < 0.01 * abs(mean):
            print(f"  ❌ U 標準差過小，疑似層流（實際: {std:.4f}）")
            is_valid = False
        if mean < 1.0 or mean > 20.0:
            print(f"  ⚠️  U 均值範圍異常（預期: 1-20，實際: {mean:.4f}）")
        if is_valid:
            print(f"  ✅ U 場統計量正常（湍流特徵）")
    
    elif field in ['v', 'w']:
        # 法向/展向速度：均值 ≈ 0，但標準差應顯著
        if abs(mean) > 1.0:
            print(f"  ⚠️  {field.upper()} 均值偏大（預期 ≈ 0，實際: {mean:.4f}）")
        if std < 0.01:
            print(f"  ❌ {field.upper()} 標準差過小，疑似層流（實際: {std:.4f}）")
            is_valid = False
        if is_valid:
            print(f"  ✅ {field.upper()} 場統計量正常（湍流波動）")
    
    elif field == 'p':
        # 壓力：應有合理分佈
        if std < 0.01:
            print(f"  ⚠️  壓力場標準差過小（實際: {std:.4f}）")
        else:
            print(f"  ✅ 壓力場統計量正常")
    
    return is_valid

def main():
    print_header("TASK-008 數據載入驗證")
    print("📋 驗證修復後的 prepare_training_data() 函數行為")
    print("🎯 目標：確認不會用 Mock 數據覆蓋真實 JHTDB 數據")
    
    # ========================================
    # 測試 1: 預設行為（prior_type='none'）
    # ========================================
    print_header("測試 1: 預設行為（prior_type='none'）")
    print("📍 調用：prepare_training_data(strategy='qr_pivot', K=500)")
    
    try:
        training_data = prepare_training_data(
            strategy='qr_pivot',
            K=500,
            sensor_file='sensors_K500_qr_pivot_3d_wall_enhanced.npz',
            target_fields=['u', 'v', 'w', 'p']  # 明確指定所有欄位
        )
        
        print("\n✅ 數據載入成功")
        print(f"  感測點數量: {training_data['coordinates'].shape[0]}")
        print(f"  可用欄位: {list(training_data['sensor_data'].keys())}")
        
        # 驗證統計量
        all_valid = True
        for field in ['u', 'v', 'w', 'p']:
            if field in training_data['sensor_data']:
                is_valid = verify_turbulence_statistics(training_data['sensor_data'], field)
                all_valid = all_valid and is_valid
        
        if all_valid:
            print("\n" + "="*80)
            print("✅ 測試 1 通過：數據具有湍流特徵（未被 Mock 覆蓋）")
            print("="*80)
        else:
            print("\n" + "="*80)
            print("❌ 測試 1 失敗：數據統計量異常")
            print("="*80)
            return False
        
    except Exception as e:
        print(f"\n❌ 測試 1 失敗：{e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========================================
    # 測試 2: 明確指定 prior_type='none'
    # ========================================
    print_header("測試 2: 明確指定 prior_type='none'")
    print("📍 調用：prepare_training_data(..., prior_type='none')")
    
    try:
        training_data_explicit = prepare_training_data(
            strategy='qr_pivot',
            K=500,
            sensor_file='sensors_K500_qr_pivot_3d_wall_enhanced.npz',
            target_fields=['u', 'v', 'w', 'p'],  # 保持一致
            prior_type='none'
        )
        
        print("\n✅ 數據載入成功")
        
        # 驗證與測試 1 結果一致
        coords_match = np.allclose(
            training_data['coordinates'], 
            training_data_explicit['coordinates']
        )
        data_match = all(
            np.allclose(
                training_data['sensor_data'][field],
                training_data_explicit['sensor_data'][field]
            )
            for field in ['u', 'v', 'w', 'p']
        )
        
        if coords_match and data_match:
            print("✅ 測試 2 通過：明確指定與預設行為一致")
        else:
            print("❌ 測試 2 失敗：明確指定與預設行為不一致")
            return False
        
    except Exception as e:
        print(f"\n❌ 測試 2 失敗：{e}")
        return False
    
    # ========================================
    # 測試 3: 對比 prior_type='mock'（應覆蓋為層流）
    # ========================================
    print_header("測試 3: 對比 prior_type='mock'（驗證覆蓋行為）")
    print("📍 調用：prepare_training_data(..., prior_type='mock')")
    print("⚠️  預期：數據應被覆蓋為層流（標準差極小）")
    
    try:
        training_data_mock = prepare_training_data(
            strategy='qr_pivot',
            K=500,
            sensor_file='sensors_K500_qr_pivot_3d_wall_enhanced.npz',
            target_fields=['u', 'v', 'w', 'p'],  # 保持一致
            prior_type='mock'
        )
        
        print("\n✅ 數據載入成功")
        
        # 驗證 Mock 數據特徵（層流）
        u_data = training_data_mock['sensor_data']['u']
        u_mean = np.mean(u_data)
        u_std = np.std(u_data)
        
        print(f"\n  Mock U 統計量:")
        print(f"    均值: {u_mean:.4f}")
        print(f"    標準差: {u_std:.4f}")
        print(f"    變異係數: {u_std/u_mean:.6f}")
        
        # Mock 應該是層流（標準差極小）
        if u_std < 0.01 * u_mean:
            print("✅ 測試 3 通過：Mock 數據確實覆蓋為層流")
        else:
            print("⚠️  Mock 數據標準差未如預期變小")
        
        # 驗證與真實數據不同
        u_real = training_data['sensor_data']['u']
        if not np.allclose(u_real, u_data):
            print("✅ Mock 數據與真實數據不同（驗證覆蓋機制）")
        else:
            print("❌ Mock 數據與真實數據相同（覆蓋失敗）")
            return False
        
    except Exception as e:
        print(f"\n❌ 測試 3 失敗：{e}")
        return False
    
    # ========================================
    # 最終總結
    # ========================================
    print_header("🎉 驗證完成")
    print("✅ 所有測試通過")
    print("\n關鍵結論：")
    print("  1. prepare_training_data() 預設不添加 prior (prior_type='none')")
    print("  2. 感測點數據保持原始 JHTDB 湍流特徵")
    print("  3. prior_type='mock' 僅在明確指定時才覆蓋數據")
    print("\n🚀 可以安全開始重新訓練 TASK-008")
    print("="*80)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
