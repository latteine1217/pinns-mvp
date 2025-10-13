#!/usr/bin/env python3
"""
驗證混合感測點與 JHTDB 的一致性
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any


def load_sensor_data(cache_dir: Path) -> Dict[str, Any]:
    """載入感測點數據"""
    
    sensor_file = cache_dir / "sensors_K80_hybrid.npz"
    
    if not sensor_file.exists():
        raise FileNotFoundError(f"感測點文件不存在: {sensor_file}")
    
    data = np.load(sensor_file, allow_pickle=True)
    
    return {
        'points': data['sensor_points'],
        'u': data['sensor_u'],
        'v': data['sensor_v'],
        'p': data['sensor_p'],
        'indices': data['sensor_indices'],
        'info': data['selection_info'].item()
    }


def load_jhtdb_full(cache_dir: Path) -> Dict[str, Any]:
    """載入完整 JHTDB 數據"""
    
    cutout_file = cache_dir / "cutout_128x64.npz"
    
    if not cutout_file.exists():
        raise FileNotFoundError(f"JHTDB cutout 文件不存在: {cutout_file}")
    
    data = np.load(cutout_file, allow_pickle=True)
    coords = data['coordinates'].item()
    
    return {
        'coordinates': coords,
        'u': data['u'],
        'v': data['v'],
        'p': data['p']
    }


def validate_consistency(sensor_data: Dict, jhtdb_data: Dict):
    """驗證感測點與 JHTDB 的一致性"""
    
    print("=" * 80)
    print("🔬 混合感測點與 JHTDB 一致性驗證")
    print("=" * 80)
    
    coords = jhtdb_data['coordinates']
    x, y = coords['x'], coords['y']
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    u_full = jhtdb_data['u'].ravel()
    v_full = jhtdb_data['v'].ravel()
    p_full = jhtdb_data['p'].ravel()
    
    sensor_indices = sensor_data['indices']
    
    # 驗證索引是否正確
    u_sensor_from_index = u_full[sensor_indices]
    v_sensor_from_index = v_full[sensor_indices]
    p_sensor_from_index = p_full[sensor_indices]
    
    # 與保存的感測點數據比較
    u_diff = np.abs(u_sensor_from_index - sensor_data['u'])
    v_diff = np.abs(v_sensor_from_index - sensor_data['v'])
    p_diff = np.abs(p_sensor_from_index - sensor_data['p'])
    
    print(f"\n📊 索引一致性檢查:")
    print(f"   U 最大差異: {u_diff.max():.2e}")
    print(f"   V 最大差異: {v_diff.max():.2e}")
    print(f"   P 最大差異: {p_diff.max():.2e}")
    
    if u_diff.max() < 1e-10 and v_diff.max() < 1e-10 and p_diff.max() < 1e-10:
        print("   ✅ 索引正確：感測點數據與 JHTDB 完全一致")
    else:
        print("   ⚠️  索引可能有誤：存在數值差異")
    
    # 統計特性驗證
    print(f"\n📊 統計特性比較:")
    print(f"\n   U 統計:")
    print(f"      JHTDB:  mean={u_full.mean():.4f}, std={u_full.std():.4f}")
    print(f"      Sensor: mean={sensor_data['u'].mean():.4f}, std={sensor_data['u'].std():.4f}")
    
    print(f"\n   V 統計 (關鍵):")
    print(f"      JHTDB:  mean={v_full.mean():.4f}, std={v_full.std():.4f}")
    print(f"      Sensor: mean={sensor_data['v'].mean():.4f}, std={sensor_data['v'].std():.4f}")
    print(f"      均值誤差: {abs(sensor_data['v'].mean() - v_full.mean()):.4f}")
    
    print(f"\n   P 統計:")
    print(f"      JHTDB:  mean={p_full.mean():.4f}, std={p_full.std():.4f}")
    print(f"      Sensor: mean={sensor_data['p'].mean():.4f}, std={sensor_data['p'].std():.4f}")
    
    # 空間分佈驗證
    sensor_points = sensor_data['points']
    
    print(f"\n📍 空間分佈:")
    print(f"   X 範圍: [{sensor_points[:, 0].min():.2f}, {sensor_points[:, 0].max():.2f}]")
    print(f"   Y 範圍: [{sensor_points[:, 1].min():.2f}, {sensor_points[:, 1].max():.2f}]")
    
    y_upper = (sensor_points[:, 1] > 0).sum()
    y_lower = (sensor_points[:, 1] < 0).sum()
    y_center = (np.abs(sensor_points[:, 1]) < 0.1).sum()
    
    print(f"\n   Y 分佈:")
    print(f"      上半通道 (y>0):   {y_upper} ({y_upper/len(sensor_points)*100:.1f}%)")
    print(f"      下半通道 (y<0):   {y_lower} ({y_lower/len(sensor_points)*100:.1f}%)")
    print(f"      中心區 (|y|<0.1): {y_center} ({y_center/len(sensor_points)*100:.1f}%)")
    
    # 選點策略驗證
    info = sensor_data['info']
    print(f"\n🔧 混合選點策略:")
    print(f"   方法: {info['method']}")
    print(f"   總點數: {info['K']}")
    print(f"   - V 極值點: {info['K_extrema']} ({info['K_extrema']/info['K']*100:.1f}%)")
    print(f"   - V 分層點: {info['K_stratified']} ({info['K_stratified']/info['K']*100:.1f}%)")
    print(f"   - QR 選點:  {info['K_qr']} ({info['K_qr']/info['K']*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("✅ 驗證完成")
    print("=" * 80)


def main():
    """主函數"""
    
    cache_dir = Path("data/jhtdb/channel_flow_re1000")
    
    try:
        # 1. 載入數據
        print("📂 載入數據...")
        sensor_data = load_sensor_data(cache_dir)
        jhtdb_data = load_jhtdb_full(cache_dir)
        
        # 2. 驗證一致性
        validate_consistency(sensor_data, jhtdb_data)
        
    except Exception as e:
        print(f"\n❌ 驗證失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
