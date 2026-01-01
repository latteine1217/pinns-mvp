#!/usr/bin/env python3
"""
驗證 Normalization 統計量來源
目標：確認 checkpoint 中的統計量與 DNS sensor 是否一致
"""

import sys
import numpy as np
import torch
import h5py
from pathlib import Path

# 設定路徑
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

def load_checkpoint_normalization(ckpt_path):
    """從 checkpoint 載入 normalization metadata"""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    if 'normalization' not in ckpt:
        print("❌ Checkpoint 缺少 normalization metadata")
        return None
    
    norm_meta = ckpt['normalization']
    print(f"\n📦 Checkpoint Normalization:")
    print(f"   Type: {norm_meta.get('norm_type', 'unknown')}")
    print(f"   Means: {norm_meta.get('means', {})}")
    print(f"   Stds: {norm_meta.get('stds', {})}")
    
    return norm_meta

def load_dns_full_field(dns_path):
    """載入 DNS 全場資料並計算統計量"""
    with h5py.File(dns_path, 'r') as f:
        # 讀取時間範圍
        time_all = np.array(f['time'])
        print(f"\n📊 DNS Full Field:")
        print(f"   Time range: [{time_all.min():.1f}, {time_all.max():.1f}]")
        
        # 選擇時間範圍 [15.0, 35.0]（與訓練相同）
        t_start, t_end = 15.0, 35.0
        time_mask = (time_all >= t_start) & (time_all <= t_end)
        time_selected = time_all[time_mask]
        T = len(time_selected)
        
        print(f"   Selected: [{t_start:.1f}, {t_end:.1f}], T={T} steps")
        
        # 讀取場域
        u_slice = f['u'][time_mask]  # [T, N, N]
        v_slice = f['v'][time_mask]
        
        if 'p' in f:
            p_slice = f['p'][time_mask]
        else:
            p_slice = None
        
        # 計算統計量（全時空域）
        u_mean = float(u_slice.mean())
        u_std = float(u_slice.std())
        v_mean = float(v_slice.mean())
        v_std = float(v_slice.std())
        
        if p_slice is not None:
            p_mean = float(p_slice.mean())
            p_std = float(p_slice.std())
        else:
            p_mean, p_std = 0.0, 0.0
        
        stats = {
            'u': {'mean': u_mean, 'std': u_std},
            'v': {'mean': v_mean, 'std': v_std},
            'p': {'mean': p_mean, 'std': p_std}
        }
        
        print(f"\n   Full Field Statistics (T={T} × N×N points):")
        for var in ['u', 'v', 'p']:
            print(f"   {var}: μ={stats[var]['mean']:+.6f}, σ={stats[var]['std']:.6f}")
        
        return stats, u_slice, v_slice, p_slice

def load_dns_sensors(dns_path, sensor_path):
    """載入 DNS sensor 資料並計算統計量"""
    # 讀取 sensor 索引
    sensor_data = np.load(sensor_path, allow_pickle=True)
    spatial_indices = np.array(sensor_data['indices'])
    K = len(spatial_indices)
    
    print(f"\n🎯 DNS Sensors:")
    print(f"   K = {K} sensors")
    print(f"   Spatial indices: [{spatial_indices.min()}, {spatial_indices.max()}]")
    
    # 讀取 DNS 全場
    with h5py.File(dns_path, 'r') as f:
        time_all = np.array(f['time'])
        t_start, t_end = 15.0, 35.0
        time_mask = (time_all >= t_start) & (time_all <= t_end)
        T = time_mask.sum()
        
        u_slice = f['u'][time_mask]  # [T, N, N]
        v_slice = f['v'][time_mask]
        if 'p' in f:
            p_slice = f['p'][time_mask]
        else:
            p_slice = None
        
        # 展平空間維度
        u_flat = u_slice.reshape(T, -1)  # [T, N*N]
        v_flat = v_slice.reshape(T, -1)
        p_flat = p_slice.reshape(T, -1) if p_slice is not None else None
        
        # 提取 sensor 點
        u_sensors = u_flat[:, spatial_indices]  # [T, K]
        v_sensors = v_flat[:, spatial_indices]
        p_sensors = p_flat[:, spatial_indices] if p_flat is not None else None
        
        # 計算統計量（T×K 個樣本）
        u_mean = float(u_sensors.mean())
        u_std = float(u_sensors.std())
        v_mean = float(v_sensors.mean())
        v_std = float(v_sensors.std())
        
        if p_sensors is not None:
            p_mean = float(p_sensors.mean())
            p_std = float(p_sensors.std())
        else:
            p_mean, p_std = 0.0, 0.0
        
        stats = {
            'u': {'mean': u_mean, 'std': u_std},
            'v': {'mean': v_mean, 'std': v_std},
            'p': {'mean': p_mean, 'std': p_std}
        }
        
        print(f"\n   Sensor Statistics (T={T} × K={K} = {T*K} samples):")
        for var in ['u', 'v', 'p']:
            print(f"   {var}: μ={stats[var]['mean']:+.6f}, σ={stats[var]['std']:.6f}")
        
        return stats

def load_rans_prior(rans_path):
    """載入 RANS prior 並計算統計量"""
    print(f"\n🔍 RANS Prior:")
    
    with h5py.File(rans_path, 'r') as f:
        group = f['/mean_field']
        
        u_rans = np.array(group['u'])  # [N, N]
        v_rans = np.array(group['v'])
        
        u_mean = float(u_rans.mean())
        u_std = float(u_rans.std())
        v_mean = float(v_rans.mean())
        v_std = float(v_rans.std())
        
        # Leith 無壓力場
        p_mean, p_std = 0.0, 0.0
        
        stats = {
            'u': {'mean': u_mean, 'std': u_std},
            'v': {'mean': v_mean, 'std': v_std},
            'p': {'mean': p_mean, 'std': p_std}
        }
        
        print(f"   RANS Statistics (N×N spatial points, time-averaged):")
        for var in ['u', 'v']:
            print(f"   {var}: μ={stats[var]['mean']:+.6f}, σ={stats[var]['std']:.6f}")
        print(f"   p: (not available in Leith model)")
        
        return stats

def compare_stats(ckpt_stats, dns_full_stats, dns_sensor_stats, rans_stats):
    """比較三種統計量來源"""
    print(f"\n" + "="*80)
    print(f"📊 統計量比較")
    print(f"="*80)
    
    for var in ['u', 'v', 'p']:
        print(f"\n{var.upper()} 變量:")
        print(f"{'Source':<20} {'Mean':>12} {'Std':>12} {'Mean Diff':>12} {'Std Ratio':>12}")
        print(f"-"*80)
        
        # Checkpoint
        ckpt_mean = ckpt_stats['means'][var]
        ckpt_std = ckpt_stats['stds'][var]
        print(f"{'Checkpoint':<20} {ckpt_mean:+12.6f} {ckpt_std:12.6f} {'-':>12} {'-':>12}")
        
        # DNS Full Field
        dns_full_mean = dns_full_stats[var]['mean']
        dns_full_std = dns_full_stats[var]['std']
        mean_diff_full = ckpt_mean - dns_full_mean
        std_ratio_full = (ckpt_std / dns_full_std) * 100 if dns_full_std > 0 else 0
        print(f"{'DNS Full Field':<20} {dns_full_mean:+12.6f} {dns_full_std:12.6f} "
              f"{mean_diff_full:+12.6f} {std_ratio_full:12.1f}%")
        
        # DNS Sensors
        dns_sens_mean = dns_sensor_stats[var]['mean']
        dns_sens_std = dns_sensor_stats[var]['std']
        mean_diff_sens = ckpt_mean - dns_sens_mean
        std_ratio_sens = (ckpt_std / dns_sens_std) * 100 if dns_sens_std > 0 else 0
        print(f"{'DNS Sensors':<20} {dns_sens_mean:+12.6f} {dns_sens_std:12.6f} "
              f"{mean_diff_sens:+12.6f} {std_ratio_sens:12.1f}%")
        
        # RANS Prior (only u, v)
        if var in ['u', 'v']:
            rans_mean = rans_stats[var]['mean']
            rans_std = rans_stats[var]['std']
            mean_diff_rans = ckpt_mean - rans_mean
            std_ratio_rans = (ckpt_std / rans_std) * 100 if rans_std > 0 else 0
            print(f"{'RANS Prior':<20} {rans_mean:+12.6f} {rans_std:12.6f} "
                  f"{mean_diff_rans:+12.6f} {std_ratio_rans:12.1f}%")
    
    print(f"\n" + "="*80)
    print(f"🔍 診斷結果:")
    print(f"="*80)
    
    # 判斷最接近的來源
    for var in ['u', 'v', 'p']:
        ckpt_mean = ckpt_stats['means'][var]
        ckpt_std = ckpt_stats['stds'][var]
        
        dns_full_mean = dns_full_stats[var]['mean']
        dns_full_std = dns_full_stats[var]['std']
        
        dns_sens_mean = dns_sensor_stats[var]['mean']
        dns_sens_std = dns_sensor_stats[var]['std']
        
        # 計算歐式距離（標準化）
        def normalized_distance(m1, s1, m2, s2):
            return np.sqrt(((m1 - m2) / max(abs(m2), 1e-6))**2 + 
                          ((s1 - s2) / max(s2, 1e-6))**2)
        
        dist_full = normalized_distance(ckpt_mean, ckpt_std, dns_full_mean, dns_full_std)
        dist_sens = normalized_distance(ckpt_mean, ckpt_std, dns_sens_mean, dns_sens_std)
        
        print(f"\n{var.upper()}:")
        print(f"  Distance to DNS Full:   {dist_full:.4f}")
        print(f"  Distance to DNS Sensor: {dist_sens:.4f}")
        
        if dist_sens < dist_full * 0.5:
            print(f"  ✅ 最接近 DNS Sensor（符合預期）")
        elif dist_sens < dist_full * 2.0:
            print(f"  ⚠️  與 DNS Sensor 和 DNS Full 都接近")
        else:
            print(f"  ❌ 不符合任何已知來源！")
        
        # RANS 比較（僅 u, v）
        if var in ['u', 'v']:
            rans_mean = rans_stats[var]['mean']
            rans_std = rans_stats[var]['std']
            dist_rans = normalized_distance(ckpt_mean, ckpt_std, rans_mean, rans_std)
            print(f"  Distance to RANS Prior: {dist_rans:.4f}")
            
            if dist_rans < min(dist_full, dist_sens):
                print(f"  🚨 最接近 RANS Prior！可能被污染")

def main():
    # 檔案路徑
    ckpt_path = project_root / 'checkpoints/loss_balance/B1_joint_optimization/epoch_1000.pth'
    dns_path = project_root / 'data/kolmogorov_dns/dns_re50_t100.h5'
    sensor_path = project_root / 'data/sensors/kolmogorov_re50_k100_qr_sensors.npz'
    rans_path = project_root / 'data/lowfi/kolmogorov_leith/rans_re50_kf4_leith.h5'
    
    print("="*80)
    print("🔬 Normalization 統計量來源驗證")
    print("="*80)
    
    # 1. 載入 checkpoint normalization
    ckpt_stats = load_checkpoint_normalization(ckpt_path)
    if ckpt_stats is None:
        return
    
    # 2. 載入 DNS 全場統計量
    dns_full_stats, _, _, _ = load_dns_full_field(dns_path)
    
    # 3. 載入 DNS sensor 統計量
    dns_sensor_stats = load_dns_sensors(dns_path, sensor_path)
    
    # 4. 載入 RANS prior 統計量
    rans_stats = load_rans_prior(rans_path)
    
    # 5. 比較統計量
    compare_stats(ckpt_stats, dns_full_stats, dns_sensor_stats, rans_stats)

if __name__ == '__main__':
    main()
