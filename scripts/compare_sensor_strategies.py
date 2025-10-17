#!/usr/bin/env python3
"""
感測點策略對比分析腳本

比較 QR-Pivot 與 Wall-Clustered 兩種感測點選擇策略的差異
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, Tuple
import seaborn as sns

# 設定繪圖風格
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150


def load_sensor_data(filepath: Path) -> Dict:
    """載入感測點資料"""
    data = np.load(filepath, allow_pickle=True)
    return {
        'coords': data['coords'],
        'coords_2d': data['coords_2d'],
        'u': data['u'],
        'v': data['v'],
        'w': data['w'],
        'p': data['p'],
        'condition_number': data.get('condition_number', None),
        'energy_ratio': data.get('energy_ratio', None),
        'metadata': data.get('metadata', None)
    }


def compute_statistics(coords_2d: np.ndarray, velocity: np.ndarray) -> Dict:
    """計算統計指標（所有數值轉換為 Python 原生類型以支援 JSON 序列化）"""
    x, y = coords_2d[:, 0], coords_2d[:, 1]
    
    # 空間分佈統計
    stats = {
        'x_mean': float(np.mean(x)),
        'x_std': float(np.std(x)),
        'x_min': float(np.min(x)),
        'x_max': float(np.max(x)),
        'y_mean': float(np.mean(y)),
        'y_std': float(np.std(y)),
        'y_min': float(np.min(y)),
        'y_max': float(np.max(y)),
    }
    
    # 速度場統計
    stats['u_mean'] = float(np.mean(velocity))
    stats['u_std'] = float(np.std(velocity))
    stats['u_min'] = float(np.min(velocity))
    stats['u_max'] = float(np.max(velocity))
    
    # 分層統計（基於 y 座標）
    wall_mask = np.abs(y) > 0.95
    log_mask = (np.abs(y) > 0.3) & (np.abs(y) <= 0.95)
    center_mask = np.abs(y) <= 0.3
    
    stats['n_wall'] = int(np.sum(wall_mask))
    stats['n_log'] = int(np.sum(log_mask))
    stats['n_center'] = int(np.sum(center_mask))
    stats['pct_wall'] = float(stats['n_wall'] / len(y) * 100)
    stats['pct_log'] = float(stats['n_log'] / len(y) * 100)
    stats['pct_center'] = float(stats['n_center'] / len(y) * 100)
    
    # 最小距離（避免點過度聚集）
    from scipy.spatial.distance import pdist
    distances = pdist(coords_2d)
    stats['min_distance'] = float(np.min(distances))
    stats['mean_distance'] = float(np.mean(distances))
    
    return stats


def plot_comparison(data1: Dict, data2: Dict, 
                   name1: str, name2: str,
                   output_dir: Path):
    """繪製對比圖表"""
    
    # 1. 空間分佈對比（2D XY 平面）
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, data, name in zip(axes, [data1, data2], [name1, name2]):
        x, y = data['coords_2d'][:, 0], data['coords_2d'][:, 1]
        u = data['u']
        
        scatter = ax.scatter(x, y, c=u, s=50, cmap='RdBu_r', alpha=0.7, edgecolor='k', linewidth=0.5)
        ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='Wall boundary')
        ax.axhline(y=-0.95, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=0.3, color='gray', linestyle=':', alpha=0.5, label='Log-law boundary')
        ax.axhline(y=-0.3, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{name} Sensor Distribution (K={len(x)})')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim(-1.05, 1.05)
        plt.colorbar(scatter, ax=ax, label='u velocity')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_spatial_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. 統計指標對比
    stats1 = compute_statistics(data1['coords_2d'], data1['u'])
    stats2 = compute_statistics(data2['coords_2d'], data2['u'])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 2.1 分層分佈對比
    ax = axes[0, 0]
    categories = ['Wall\n(|y|>0.95)', 'Log-law\n(0.3<|y|<0.95)', 'Center\n(|y|<0.3)']
    x_pos = np.arange(len(categories))
    width = 0.35
    
    values1 = [stats1['pct_wall'], stats1['pct_log'], stats1['pct_center']]
    values2 = [stats2['pct_wall'], stats2['pct_log'], stats2['pct_center']]
    
    ax.bar(x_pos - width/2, values1, width, label=name1, alpha=0.8, color='steelblue')
    ax.bar(x_pos + width/2, values2, width, label=name2, alpha=0.8, color='coral')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Sensor Distribution by Region')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 2.2 空間標準差對比
    ax = axes[0, 1]
    metrics = ['x_std', 'y_std']
    values1 = [stats1[m] for m in metrics]
    values2 = [stats2[m] for m in metrics]
    x_pos = np.arange(len(metrics))
    
    ax.bar(x_pos - width/2, values1, width, label=name1, alpha=0.8, color='steelblue')
    ax.bar(x_pos + width/2, values2, width, label=name2, alpha=0.8, color='coral')
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Spatial Spread (Higher = More Uniform)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['X-direction', 'Y-direction'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 2.3 距離指標對比
    ax = axes[1, 0]
    metrics = ['min_distance', 'mean_distance']
    values1 = [stats1[m] for m in metrics]
    values2 = [stats2[m] for m in metrics]
    x_pos = np.arange(len(metrics))
    
    ax.bar(x_pos - width/2, values1, width, label=name1, alpha=0.8, color='steelblue')
    ax.bar(x_pos + width/2, values2, width, label=name2, alpha=0.8, color='coral')
    ax.set_ylabel('Distance')
    ax.set_title('Inter-Sensor Distances')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Min Distance', 'Mean Distance'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 2.4 品質指標對比（如果有）
    ax = axes[1, 1]
    if data1.get('condition_number') is not None and data2.get('condition_number') is not None:
        metrics_available = True
        cond1 = float(data1['condition_number'])
        cond2 = float(data2['condition_number']) if data2['condition_number'] is not None else np.nan
        energy1 = float(data1['energy_ratio'])
        energy2 = float(data2['energy_ratio']) if data2['energy_ratio'] is not None else np.nan
        
        # 使用對數尺度顯示條件數
        if not np.isnan(cond2):
            x_pos = np.arange(2)
            ax.bar(x_pos - width/2, [np.log10(cond1), energy1], width, 
                   label=name1, alpha=0.8, color='steelblue')
            ax.bar(x_pos + width/2, [np.log10(cond2), energy2], width,
                   label=name2, alpha=0.8, color='coral')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(['log10(Condition Number)', 'Energy Ratio'])
        else:
            x_pos = [0]
            ax.bar([0 - width/2], [energy1], width, label=name1, alpha=0.8, color='steelblue')
            ax.bar([0 + width/2], [energy2] if not np.isnan(energy2) else [0], width,
                   label=name2, alpha=0.8, color='coral')
            ax.set_xticks([0])
            ax.set_xticklabels(['Energy Ratio'])
            ax.text(0, 0.5, f'{name1} Cond#: {cond1:.2e}\n{name2}: N/A', 
                   ha='center', va='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_ylabel('Value')
        ax.set_title('Quality Metrics')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Quality metrics not available', 
                ha='center', va='center', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Y 方向分佈直方圖
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    y1 = data1['coords_2d'][:, 1]
    y2 = data2['coords_2d'][:, 1]
    
    ax.hist(y1, bins=20, alpha=0.5, label=name1, color='steelblue', edgecolor='black')
    ax.hist(y2, bins=20, alpha=0.5, label=name2, color='coral', edgecolor='black')
    
    ax.axvline(x=0.95, color='gray', linestyle='--', alpha=0.5, label='Wall boundary')
    ax.axvline(x=-0.95, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.3, color='gray', linestyle=':', alpha=0.5, label='Log-law boundary')
    ax.axvline(x=-0.3, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('y coordinate')
    ax.set_ylabel('Number of sensors')
    ax.set_title('Sensor Distribution along Wall-Normal Direction')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_y_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return stats1, stats2


def generate_comparison_report(stats1: Dict, stats2: Dict,
                               name1: str, name2: str,
                               output_dir: Path):
    """生成對比報告"""
    
    report = f"""
================================================================================
  感測點策略對比分析報告
================================================================================

策略 1: {name1}
策略 2: {name2}

--------------------------------------------------------------------------------
1. 空間分佈統計
--------------------------------------------------------------------------------

{name1:20s} | {name2:20s} | 指標
{'-'*20} | {'-'*20} | {'-'*30}
{stats1['x_mean']:20.4f} | {stats2['x_mean']:20.4f} | X 平均值
{stats1['x_std']:20.4f} | {stats2['x_std']:20.4f} | X 標準差
{stats1['y_mean']:20.4f} | {stats2['y_mean']:20.4f} | Y 平均值
{stats1['y_std']:20.4f} | {stats2['y_std']:20.4f} | Y 標準差

--------------------------------------------------------------------------------
2. 分層分佈
--------------------------------------------------------------------------------

區域              | {name1:>12s} | {name2:>12s} | 差異
{'-'*17} | {'-'*12} | {'-'*12} | {'-'*12}
壁面層 (|y|>0.95)  | {stats1['n_wall']:4d} ({stats1['pct_wall']:5.1f}%) | {stats2['n_wall']:4d} ({stats2['pct_wall']:5.1f}%) | {stats2['n_wall']-stats1['n_wall']:+4d} ({stats2['pct_wall']-stats1['pct_wall']:+5.1f}%)
對數層 (0.3<|y|<0.95) | {stats1['n_log']:4d} ({stats1['pct_log']:5.1f}%) | {stats2['n_log']:4d} ({stats2['pct_log']:5.1f}%) | {stats2['n_log']-stats1['n_log']:+4d} ({stats2['pct_log']-stats1['pct_log']:+5.1f}%)
中心層 (|y|<0.3)    | {stats1['n_center']:4d} ({stats1['pct_center']:5.1f}%) | {stats2['n_center']:4d} ({stats2['pct_center']:5.1f}%) | {stats2['n_center']-stats1['n_center']:+4d} ({stats2['pct_center']-stats1['pct_center']:+5.1f}%)

--------------------------------------------------------------------------------
3. 距離指標
--------------------------------------------------------------------------------

{name1:20s} | {name2:20s} | 指標
{'-'*20} | {'-'*20} | {'-'*30}
{stats1['min_distance']:20.4f} | {stats2['min_distance']:20.4f} | 最小距離
{stats1['mean_distance']:20.4f} | {stats2['mean_distance']:20.4f} | 平均距離

--------------------------------------------------------------------------------
4. 速度場統計
--------------------------------------------------------------------------------

{name1:20s} | {name2:20s} | 指標
{'-'*20} | {'-'*20} | {'-'*30}
{stats1['u_mean']:20.4f} | {stats2['u_mean']:20.4f} | 平均速度
{stats1['u_std']:20.4f} | {stats2['u_std']:20.4f} | 速度標準差
{stats1['u_min']:20.4f} | {stats2['u_min']:20.4f} | 最小速度
{stats1['u_max']:20.4f} | {stats2['u_max']:20.4f} | 最大速度

--------------------------------------------------------------------------------
5. 關鍵差異總結
--------------------------------------------------------------------------------

✓ {name2} 相較於 {name1}:
  - 壁面層點數：{stats2['n_wall']-stats1['n_wall']:+d} 點 ({stats2['pct_wall']-stats1['pct_wall']:+.1f}%)
  - 對數層點數：{stats2['n_log']-stats1['n_log']:+d} 點 ({stats2['pct_log']-stats1['pct_log']:+.1f}%)
  - 中心層點數：{stats2['n_center']-stats1['n_center']:+d} 點 ({stats2['pct_center']-stats1['pct_center']:+.1f}%)
  - X 方向分散度：{'更均勻' if stats2['x_std'] > stats1['x_std'] else '更集中'} (Δσ={stats2['x_std']-stats1['x_std']:+.3f})
  - Y 方向分散度：{'更均勻' if stats2['y_std'] > stats1['y_std'] else '更集中'} (Δσ={stats2['y_std']-stats1['y_std']:+.3f})
  - 最小距離：{stats2['min_distance']-stats1['min_distance']:+.4f} ({'較遠' if stats2['min_distance'] > stats1['min_distance'] else '較近'})

--------------------------------------------------------------------------------
6. 物理意義解讀
--------------------------------------------------------------------------------

{name1}:
  - 集中於對數層（{stats1['pct_log']:.1f}%），適合捕捉湍流主體結構
  - 較高的條件數可能影響數值穩定性
  - 數學優化策略（最小化條件數）

{name2}:
  - 壁面層與中心層覆蓋更均衡
  - 重視物理先驗（壁面剪應力、中心線速度）
  - 計算成本低（無需矩陣分解）

================================================================================
"""
    
    # 保存報告
    with open(output_dir / 'comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 保存 JSON 格式統計
    comparison_data = {
        name1: stats1,
        name2: stats2,
        'differences': {
            'n_wall': int(stats2['n_wall'] - stats1['n_wall']),
            'n_log': int(stats2['n_log'] - stats1['n_log']),
            'n_center': int(stats2['n_center'] - stats1['n_center']),
            'x_std_diff': float(stats2['x_std'] - stats1['x_std']),
            'y_std_diff': float(stats2['y_std'] - stats1['y_std']),
            'min_distance_diff': float(stats2['min_distance'] - stats1['min_distance'])
        }
    }
    
    with open(output_dir / 'comparison_data.json', 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    return report


def main():
    """主程式"""
    # 檔案路徑
    qr_pivot_file = Path('data/jhtdb/channel_flow_re1000/sensors_K50_velocity_qr_pivot.npz')
    wall_clustered_file = Path('data/jhtdb/channel_flow_re1000/sensors_K50_wall_clustered.npz')
    output_dir = Path('results/sensor_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("  感測點策略對比分析")
    print("="*80)
    print()
    
    # 載入資料
    print(f"📂 載入 QR-Pivot 感測點: {qr_pivot_file}")
    data_qr = load_sensor_data(qr_pivot_file)
    print(f"  ✅ 載入成功 (K={len(data_qr['coords'])})")
    
    print(f"📂 載入 Wall-Clustered 感測點: {wall_clustered_file}")
    data_wc = load_sensor_data(wall_clustered_file)
    print(f"  ✅ 載入成功 (K={len(data_wc['coords'])})")
    print()
    
    # 繪製對比圖表
    print("📊 生成對比圖表...")
    stats_qr, stats_wc = plot_comparison(
        data_qr, data_wc,
        'QR-Pivot', 'Wall-Clustered',
        output_dir
    )
    print(f"  ✅ 已保存: {output_dir}/comparison_spatial_distribution.png")
    print(f"  ✅ 已保存: {output_dir}/comparison_statistics.png")
    print(f"  ✅ 已保存: {output_dir}/comparison_y_distribution.png")
    print()
    
    # 生成報告
    print("📝 生成對比報告...")
    report = generate_comparison_report(
        stats_qr, stats_wc,
        'QR-Pivot', 'Wall-Clustered',
        output_dir
    )
    print(f"  ✅ 已保存: {output_dir}/comparison_report.txt")
    print(f"  ✅ 已保存: {output_dir}/comparison_data.json")
    print()
    
    # 顯示報告
    print(report)
    
    print("="*80)
    print("✅ 對比分析完成")
    print("="*80)
    print(f"\n結果保存至: {output_dir.resolve()}")


if __name__ == '__main__':
    main()
