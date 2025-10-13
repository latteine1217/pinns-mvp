"""
診斷訓練數據範圍與感測點分佈
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # 載入訓練感測器數據
    sensor_file = Path("data/jhtdb/channel_flow_re1000/sensors_K80_adaptive_coverage.npz")
    jhtdb_file = Path("data/jhtdb/channel_flow_re1000/cutout_128x64.npz")
    
    if not sensor_file.exists():
        print(f"❌ 找不到感測器文件: {sensor_file}")
        return
    
    if not jhtdb_file.exists():
        print(f"❌ 找不到 JHTDB 文件: {jhtdb_file}")
        return
    
    # 載入數據
    sensor_data = np.load(sensor_file)
    jhtdb_data = np.load(jhtdb_file)
    
    print("=" * 80)
    print("📊 訓練感測器數據分析")
    print("=" * 80)
    
    # 顯示感測器數據內容
    print("\n🔍 感測器文件包含的鍵:")
    for key in sensor_data.keys():
        print(f"  - {key}: shape={sensor_data[key].shape}, dtype={sensor_data[key].dtype}")
    
    # 提取座標和物理量
    x_sensor = sensor_data['x']
    y_sensor = sensor_data['y']
    u_sensor = sensor_data['u']
    v_sensor = sensor_data['v']
    p_sensor = sensor_data['p']
    
    print(f"\n📍 感測點數量: {len(x_sensor)}")
    
    print("\n" + "-" * 80)
    print("📏 訓練數據實際範圍 (感測器)")
    print("-" * 80)
    print(f"  X: [{x_sensor.min():.4f}, {x_sensor.max():.4f}]")
    print(f"  Y: [{y_sensor.min():.4f}, {y_sensor.max():.4f}]")
    print(f"  U: [{u_sensor.min():.4f}, {u_sensor.max():.4f}]")
    print(f"  V: [{v_sensor.min():.4f}, {v_sensor.max():.4f}]")
    print(f"  P: [{p_sensor.min():.4f}, {p_sensor.max():.4f}]")
    
    # 載入 JHTDB 完整數據
    x_jhtdb = jhtdb_data['x']
    y_jhtdb = jhtdb_data['y']
    u_jhtdb = jhtdb_data['u']
    v_jhtdb = jhtdb_data['v']
    p_jhtdb = jhtdb_data['p']
    
    print("\n" + "-" * 80)
    print("📏 JHTDB 完整數據範圍")
    print("-" * 80)
    print(f"  X: [{x_jhtdb.min():.4f}, {x_jhtdb.max():.4f}]  shape={x_jhtdb.shape}")
    print(f"  Y: [{y_jhtdb.min():.4f}, {y_jhtdb.max():.4f}]  shape={y_jhtdb.shape}")
    print(f"  U: [{u_jhtdb.min():.4f}, {u_jhtdb.max():.4f}]")
    print(f"  V: [{v_jhtdb.min():.4f}, {v_jhtdb.max():.4f}]")
    print(f"  P: [{p_jhtdb.min():.4f}, {p_jhtdb.max():.4f}]")
    
    # 配置文件中的標準化範圍
    config_ranges = {
        'x': (0, 25.13),
        'y': (-1, 1),
        'u': (0, 16.5),
        'v': (-0.6, 0.6),
        'p': (-85, 3)
    }
    
    print("\n" + "-" * 80)
    print("⚙️  配置文件中的標準化範圍")
    print("-" * 80)
    for var, (vmin, vmax) in config_ranges.items():
        print(f"  {var.upper()}: [{vmin:.4f}, {vmax:.4f}]")
    
    # 檢查範圍一致性
    print("\n" + "=" * 80)
    print("🔬 範圍一致性檢查")
    print("=" * 80)
    
    def check_range(name, data, config_min, config_max):
        data_min, data_max = data.min(), data.max()
        in_range = (data_min >= config_min) and (data_max <= config_max)
        coverage = (data_max - data_min) / (config_max - config_min) * 100
        status = "✅" if in_range else "❌"
        print(f"{status} {name}:")
        print(f"     數據範圍: [{data_min:.4f}, {data_max:.4f}]")
        print(f"     配置範圍: [{config_min:.4f}, {config_max:.4f}]")
        print(f"     覆蓋率: {coverage:.1f}%")
        if not in_range:
            if data_min < config_min:
                print(f"     ⚠️  最小值超出配置 {(config_min - data_min):.4f}")
            if data_max > config_max:
                print(f"     ⚠️  最大值超出配置 {(data_max - config_max):.4f}")
        return in_range
    
    print("\n🎯 感測器數據 vs 配置:")
    check_range("U (sensor)", u_sensor, *config_ranges['u'])
    check_range("V (sensor)", v_sensor, *config_ranges['v'])
    check_range("P (sensor)", p_sensor, *config_ranges['p'])
    
    print("\n🌍 JHTDB 完整數據 vs 配置:")
    check_range("U (JHTDB)", u_jhtdb, *config_ranges['u'])
    check_range("V (JHTDB)", v_jhtdb, *config_ranges['v'])
    check_range("P (JHTDB)", p_jhtdb, *config_ranges['p'])
    
    # 可視化感測點分佈
    print("\n" + "=" * 80)
    print("📈 生成可視化...")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 感測點空間分佈
    ax = axes[0, 0]
    # 重塑 JHTDB 數據為網格
    if len(x_jhtdb.shape) == 1:
        # 假設是 128x64 網格
        nx, ny = 128, 64
        x_grid = x_jhtdb.reshape(nx, ny)
        y_grid = y_jhtdb.reshape(nx, ny)
        u_grid = u_jhtdb.reshape(nx, ny)
    else:
        x_grid, y_grid = x_jhtdb, y_jhtdb
        u_grid = u_jhtdb
    
    # 背景：JHTDB U 場
    im = ax.contourf(x_grid, y_grid, u_grid, levels=20, cmap='RdBu_r', alpha=0.6)
    # 感測點
    ax.scatter(x_sensor, y_sensor, c='red', s=30, marker='x', label=f'Sensors (K={len(x_sensor)})', zorder=10)
    ax.set_xlabel('X (streamwise)')
    ax.set_ylabel('Y (wall-normal)')
    ax.set_title('Sensor Spatial Distribution\n(Background: JHTDB U field)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(im, ax=ax, label='U velocity')
    
    # 2. U 分佈比較
    ax = axes[0, 1]
    ax.hist(u_jhtdb.flatten(), bins=50, alpha=0.6, label='JHTDB (full)', density=True)
    ax.hist(u_sensor, bins=20, alpha=0.8, label='Sensors (K=80)', density=True, color='red')
    ax.axvline(config_ranges['u'][0], color='green', linestyle='--', label='Config range')
    ax.axvline(config_ranges['u'][1], color='green', linestyle='--')
    ax.set_xlabel('U velocity')
    ax.set_ylabel('Probability density')
    ax.set_title('U Distribution: Sensors vs JHTDB')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. V 分佈比較
    ax = axes[1, 0]
    ax.hist(v_jhtdb.flatten(), bins=50, alpha=0.6, label='JHTDB (full)', density=True)
    ax.hist(v_sensor, bins=20, alpha=0.8, label='Sensors (K=80)', density=True, color='red')
    ax.axvline(config_ranges['v'][0], color='green', linestyle='--', label='Config range')
    ax.axvline(config_ranges['v'][1], color='green', linestyle='--')
    ax.set_xlabel('V velocity')
    ax.set_ylabel('Probability density')
    ax.set_title('V Distribution: Sensors vs JHTDB')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. P 分佈比較
    ax = axes[1, 1]
    ax.hist(p_jhtdb.flatten(), bins=50, alpha=0.6, label='JHTDB (full)', density=True)
    ax.hist(p_sensor, bins=20, alpha=0.8, label='Sensors (K=80)', density=True, color='red')
    ax.axvline(config_ranges['p'][0], color='green', linestyle='--', label='Config range')
    ax.axvline(config_ranges['p'][1], color='green', linestyle='--')
    ax.set_xlabel('Pressure')
    ax.set_ylabel('Probability density')
    ax.set_title('P Distribution: Sensors vs JHTDB')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = Path("evaluation_results_jhtdb_comparison/training_data_diagnosis.png")
    output_file.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ 圖表已保存: {output_file}")
    
    # 統計分析
    print("\n" + "=" * 80)
    print("📊 統計摘要")
    print("=" * 80)
    
    print("\n感測器數據統計:")
    print(f"  U: mean={u_sensor.mean():.4f}, std={u_sensor.std():.4f}")
    print(f"  V: mean={v_sensor.mean():.4f}, std={v_sensor.std():.4f}")
    print(f"  P: mean={p_sensor.mean():.4f}, std={p_sensor.std():.4f}")
    
    print("\nJHTDB 完整數據統計:")
    print(f"  U: mean={u_jhtdb.mean():.4f}, std={u_jhtdb.std():.4f}")
    print(f"  V: mean={v_jhtdb.mean():.4f}, std={v_jhtdb.std():.4f}")
    print(f"  P: mean={p_jhtdb.mean():.4f}, std={p_jhtdb.std():.4f}")
    
    print("\n均值偏差:")
    print(f"  ΔU: {abs(u_sensor.mean() - u_jhtdb.mean()):.4f}")
    print(f"  ΔV: {abs(v_sensor.mean() - v_jhtdb.mean()):.4f}")
    print(f"  ΔP: {abs(p_sensor.mean() - p_jhtdb.mean()):.4f}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
