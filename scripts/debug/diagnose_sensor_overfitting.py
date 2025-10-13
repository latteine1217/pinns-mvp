"""
診斷感測點過擬合問題
分析感測點分佈與全場的統計差異
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_sensor_distribution(sensors, full_field):
    """分析感測點與全場的統計分佈"""
    print("\n" + "="*80)
    print("📊 感測點 vs. 全場統計分佈")
    print("="*80)
    
    # 提取座標與變數
    sensor_coords = sensors[:, :3]  # x, y, z
    sensor_vals = sensors[:, 3:]     # u, v, p
    
    full_coords = full_field[:, :3]
    full_vals = full_field[:, 3:]
    
    var_names = ['U', 'V', 'P']
    
    print(f"\n感測點數量: {len(sensors)}")
    print(f"全場點數量: {len(full_field)}")
    print(f"採樣比例: {len(sensors)/len(full_field)*100:.2f}%")
    
    print("\n" + "-"*80)
    print("變數統計對比")
    print("-"*80)
    print(f"{'Variable':<10} {'Metric':<15} {'Sensors':<20} {'Full Field':<20} {'Diff %'}")
    print("-"*80)
    
    stats_diff = {}
    
    for i, var in enumerate(var_names):
        sensor_data = sensor_vals[:, i]
        full_data = full_vals[:, i]
        
        # 計算統計量
        sensor_mean = np.mean(sensor_data)
        full_mean = np.mean(full_data)
        mean_diff = abs(sensor_mean - full_mean) / (abs(full_mean) + 1e-10) * 100
        
        sensor_std = np.std(sensor_data)
        full_std = np.std(full_data)
        std_diff = abs(sensor_std - full_std) / (full_std + 1e-10) * 100
        
        sensor_min = np.min(sensor_data)
        full_min = np.min(full_data)
        
        sensor_max = np.max(sensor_data)
        full_max = np.max(full_data)
        
        range_coverage = (sensor_max - sensor_min) / (full_max - full_min) * 100
        
        print(f"{var:<10} {'Mean':<15} {sensor_mean:>10.4f} {' '*9} {full_mean:>10.4f} {' '*9} {mean_diff:>6.2f}%")
        print(f"{'':<10} {'Std':<15} {sensor_std:>10.4f} {' '*9} {full_std:>10.4f} {' '*9} {std_diff:>6.2f}%")
        print(f"{'':<10} {'Min':<15} {sensor_min:>10.4f} {' '*9} {full_min:>10.4f}")
        print(f"{'':<10} {'Max':<15} {sensor_max:>10.4f} {' '*9} {full_max:>10.4f}")
        print(f"{'':<10} {'Range Cov':<15} {range_coverage:>6.2f}%")
        print("-"*80)
        
        stats_diff[var] = {
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'range_coverage': range_coverage
        }
    
    return stats_diff


def analyze_spatial_distribution(sensors, full_field):
    """分析空間分佈"""
    print("\n" + "="*80)
    print("📍 空間分佈分析")
    print("="*80)
    
    sensor_y = sensors[:, 1]
    full_y = full_field[:, 1]
    
    # Y 方向分層
    y_bins = np.linspace(-1, 1, 11)
    sensor_y_hist, _ = np.histogram(sensor_y, bins=y_bins)
    full_y_hist, _ = np.histogram(full_y, bins=y_bins)
    
    sensor_y_density = sensor_y_hist / len(sensors)
    full_y_density = full_y_hist / len(full_field)
    
    print(f"\n{'Y Range':<20} {'Sensor %':<15} {'Full %':<15} {'Diff %'}")
    print("-"*80)
    
    max_diff = 0
    for i in range(len(y_bins)-1):
        diff = abs(sensor_y_density[i] - full_y_density[i]) * 100
        max_diff = max(max_diff, diff)
        print(f"[{y_bins[i]:>6.2f}, {y_bins[i+1]:>6.2f}]  {sensor_y_density[i]*100:>6.2f}%        {full_y_density[i]*100:>6.2f}%        {diff:>6.2f}%")
    
    print(f"\n最大分層密度差異: {max_diff:.2f}%")
    
    # 檢查壁面附近採樣
    wall_threshold = 0.05
    sensor_near_wall = np.sum((np.abs(sensor_y) > (1 - wall_threshold)))
    full_near_wall = np.sum((np.abs(full_y) > (1 - wall_threshold)))
    
    sensor_wall_ratio = sensor_near_wall / len(sensors) * 100
    full_wall_ratio = full_near_wall / len(full_field) * 100
    
    print(f"\n壁面附近採樣 (|y| > {1-wall_threshold}):")
    print(f"  感測點: {sensor_near_wall}/{len(sensors)} ({sensor_wall_ratio:.2f}%)")
    print(f"  全場:   {full_near_wall}/{len(full_field)} ({full_wall_ratio:.2f}%)")
    print(f"  差異:   {abs(sensor_wall_ratio - full_wall_ratio):.2f}%")
    
    return max_diff, sensor_wall_ratio, full_wall_ratio


def create_diagnostic_plots(sensors, full_field, save_dir):
    """創建診斷圖表"""
    print("\n" + "="*80)
    print("📊 生成診斷圖表")
    print("="*80)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Sensor Distribution vs. Full Field Diagnosis', fontsize=16, fontweight='bold')
    
    var_names = ['U', 'V', 'P']
    y_vals_sensor = sensors[:, 1]
    y_vals_full = full_field[:, 1]
    
    # 第一行：直方圖對比
    for i, var in enumerate(var_names):
        ax = axes[0, i]
        
        sensor_vals = sensors[:, 3+i]
        full_vals = full_field[:, 3+i]
        
        bins = 30
        ax.hist(full_vals, bins=bins, alpha=0.5, label='Full Field', density=True, color='blue')
        ax.hist(sensor_vals, bins=bins, alpha=0.7, label='Sensors (K=80)', density=True, color='red')
        
        ax.set_xlabel(f'{var}', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{var} Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加統計信息
        sensor_mean = np.mean(sensor_vals)
        full_mean = np.mean(full_vals)
        ax.axvline(sensor_mean, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Sensor μ={sensor_mean:.2f}')
        ax.axvline(full_mean, color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'Full μ={full_mean:.2f}')
    
    # 第二行：Y 方向剖面
    for i, var in enumerate(var_names):
        ax = axes[1, i]
        
        # 分層統計
        y_bins = np.linspace(-1, 1, 21)
        sensor_vals = sensors[:, 3+i]
        full_vals = full_field[:, 3+i]
        
        sensor_profile = []
        full_profile = []
        y_centers = []
        
        for j in range(len(y_bins)-1):
            y_center = (y_bins[j] + y_bins[j+1]) / 2
            y_centers.append(y_center)
            
            sensor_mask = (y_vals_sensor >= y_bins[j]) & (y_vals_sensor < y_bins[j+1])
            full_mask = (y_vals_full >= y_bins[j]) & (y_vals_full < y_bins[j+1])
            
            if np.sum(sensor_mask) > 0:
                sensor_profile.append(np.mean(sensor_vals[sensor_mask]))
            else:
                sensor_profile.append(np.nan)
            
            if np.sum(full_mask) > 0:
                full_profile.append(np.mean(full_vals[full_mask]))
            else:
                full_profile.append(np.nan)
        
        ax.plot(y_centers, full_profile, 'b-o', label='Full Field', markersize=4, linewidth=2)
        ax.plot(y_centers, sensor_profile, 'r-s', label='Sensors (K=80)', markersize=6, linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Y', fontsize=12)
        ax.set_ylabel(f'{var}', fontsize=12)
        ax.set_title(f'{var} Profile (Y-direction)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        
        # 標示壁面位置
        ax.axvline(x=-1, color='gray', linestyle='-', linewidth=2, alpha=0.5)
        ax.axvline(x=1, color='gray', linestyle='-', linewidth=2, alpha=0.5)
    
    plt.tight_layout()
    
    save_path = save_dir / 'sensor_overfitting_diagnosis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 診斷圖表已保存: {save_path}")
    plt.close()


def main():
    # 設定路徑
    project_root = Path(__file__).resolve().parents[2]
    sensor_file = project_root / 'data/jhtdb/channel_flow_re1000/sensors_K80_hybrid.npz'
    full_field_file = project_root / 'data/jhtdb/channel_flow_re1000/field_data_8192.npz'
    save_dir = project_root / 'evaluation_results_k80_hybrid'
    
    # 載入數據
    print("📥 載入數據...")
    sensor_data = np.load(sensor_file)
    sensors = sensor_data['data']
    
    full_data = np.load(full_field_file)
    full_field = full_data['data']
    
    print(f"✅ 感測點: {sensors.shape}")
    print(f"✅ 全場: {full_field.shape}")
    
    # 執行診斷
    stats_diff = analyze_sensor_distribution(sensors, full_field)
    max_diff, sensor_wall_ratio, full_wall_ratio = analyze_spatial_distribution(sensors, full_field)
    
    # 創建圖表
    create_diagnostic_plots(sensors, full_field, save_dir)
    
    # 總結
    print("\n" + "="*80)
    print("📋 診斷總結")
    print("="*80)
    
    print("\n🔍 關鍵發現:")
    for var, diff in stats_diff.items():
        print(f"\n{var}:")
        print(f"  - 均值差異: {diff['mean_diff']:.2f}%")
        print(f"  - 標準差差異: {diff['std_diff']:.2f}%")
        print(f"  - 範圍覆蓋: {diff['range_coverage']:.2f}%")
    
    print(f"\n空間分層最大差異: {max_diff:.2f}%")
    print(f"壁面採樣差異: {abs(sensor_wall_ratio - full_wall_ratio):.2f}%")
    
    # 診斷結論
    print("\n🎯 診斷結論:")
    
    issues = []
    
    # 檢查統計差異
    for var, diff in stats_diff.items():
        if diff['mean_diff'] > 10:
            issues.append(f"  ⚠️ {var} 均值偏差過大 ({diff['mean_diff']:.1f}%)")
        if diff['range_coverage'] < 80:
            issues.append(f"  ⚠️ {var} 範圍覆蓋不足 ({diff['range_coverage']:.1f}%)")
    
    # 檢查空間分佈
    if max_diff > 5:
        issues.append(f"  ⚠️ 空間分層不均勻 (最大差異 {max_diff:.1f}%)")
    
    if abs(sensor_wall_ratio - full_wall_ratio) > 5:
        issues.append(f"  ⚠️ 壁面採樣不足 (差異 {abs(sensor_wall_ratio - full_wall_ratio):.1f}%)")
    
    if issues:
        print("\n發現以下問題：")
        for issue in issues:
            print(issue)
    else:
        print("  ✅ 感測點分佈與全場統計一致")
    
    print("\n✅ 診斷完成")
    print(f"結果保存至: {save_dir}")


if __name__ == '__main__':
    main()
