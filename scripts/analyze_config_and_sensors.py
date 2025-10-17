#!/usr/bin/env python3
"""
分析訓練配置差異與感測點分佈
=================================

功能：
1. 比較 Wall-Clustered 與 QR-Pivot 的訓練配置差異
2. 可視化感測點在各物理層的分佈
3. 分析感測點的統計特性

作者：PINNx Team
日期：2025-10-17
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from collections import defaultdict

# 設定中文字型（支援 matplotlib 顯示中文）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 物理層定義（與 evaluate_layer_wise_errors.py 相同）
# =============================================================================
LAYER_DEFINITIONS = {
    'wall': {
        'name': '壁面層',
        'range': [0.8, 1.0],
        'description': 'High shear, y+ < 100'
    },
    'log': {
        'name': '對數層',
        'range': [0.2, 0.8],
        'description': 'Turbulent core, 100 < y+ < 800'
    },
    'center': {
        'name': '中心層',
        'range': [0.0, 0.2],
        'description': 'Low gradient, y+ > 800'
    }
}


# =============================================================================
# 配置比較函數
# =============================================================================
def load_config(config_path: Path) -> Dict:
    """載入 YAML 配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def compare_loss_weights(config1: Dict, config2: Dict, name1: str, name2: str) -> Dict:
    """比較損失函數權重"""
    print(f"\n{'='*80}")
    print(f"  損失函數權重比較")
    print(f"{'='*80}\n")
    
    losses1 = config1.get('losses', {})
    losses2 = config2.get('losses', {})
    
    # 收集所有權重鍵
    all_keys = set(losses1.keys()) | set(losses2.keys())
    weight_keys = [k for k in all_keys if 'weight' in k or k in ['data_variables', 'adaptive_weighting', 'causal_weighting']]
    
    comparison = {}
    print(f"{'損失項':<30} {name1:<20} {name2:<20} {'差異':<15}")
    print(f"{'-'*90}")
    
    for key in sorted(weight_keys):
        val1 = losses1.get(key, 'N/A')
        val2 = losses2.get(key, 'N/A')
        
        if val1 == val2:
            diff_str = '✅ 相同'
        elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            diff = abs(val1 - val2)
            diff_str = f'⚠️ Δ={diff:.2e}'
        else:
            diff_str = '❌ 不同'
        
        print(f"{key:<30} {str(val1):<20} {str(val2):<20} {diff_str:<15}")
        comparison[key] = {'wall': val1, 'qr': val2, 'diff': diff_str}
    
    return comparison


def compare_physics_config(config1: Dict, config2: Dict, name1: str, name2: str) -> Dict:
    """比較物理設定"""
    print(f"\n{'='*80}")
    print(f"  物理設定比較")
    print(f"{'='*80}\n")
    
    physics1 = config1.get('physics', {})
    physics2 = config2.get('physics', {})
    
    comparison = {}
    
    # 基本物理參數
    print("【基本參數】")
    basic_keys = ['type', 'nu', 'rho']
    for key in basic_keys:
        val1 = physics1.get(key, 'N/A')
        val2 = physics2.get(key, 'N/A')
        match = '✅' if val1 == val2 else '❌'
        print(f"  {key:<20} {str(val1):<20} {str(val2):<20} {match}")
        comparison[key] = {'wall': val1, 'qr': val2}
    
    # VS-PINN 設定
    print("\n【VS-PINN 設定】")
    vs1 = physics1.get('vs_pinn', {})
    vs2 = physics2.get('vs_pinn', {})
    
    # 縮放因子
    print("  縮放因子:")
    scaling1 = vs1.get('scaling_factors', {})
    scaling2 = vs2.get('scaling_factors', {})
    for key in ['N_x', 'N_y', 'N_z']:
        val1 = scaling1.get(key, 'N/A')
        val2 = scaling2.get(key, 'N/A')
        match = '✅' if val1 == val2 else '❌'
        print(f"    {key:<18} {str(val1):<20} {str(val2):<20} {match}")
    
    # RANS 啟用狀態
    enable_rans1 = vs1.get('enable_rans', False)
    enable_rans2 = vs2.get('enable_rans', False)
    match = '✅' if enable_rans1 == enable_rans2 else '❌'
    print(f"\n  enable_rans:        {str(enable_rans1):<20} {str(enable_rans2):<20} {match}")
    comparison['enable_rans'] = {'wall': enable_rans1, 'qr': enable_rans2}
    
    return comparison


def compare_training_config(config1: Dict, config2: Dict, name1: str, name2: str) -> Dict:
    """比較訓練設定"""
    print(f"\n{'='*80}")
    print(f"  訓練設定比較")
    print(f"{'='*80}\n")
    
    train1 = config1.get('training', {})
    train2 = config2.get('training', {})
    
    comparison = {}
    
    key_params = ['optimizer', 'lr', 'weight_decay', 'epochs', 'batch_size', 'gradient_clip']
    
    print(f"{'參數':<25} {name1:<20} {name2:<20} {'狀態':<15}")
    print(f"{'-'*85}")
    
    for key in key_params:
        val1 = train1.get(key, 'N/A')
        val2 = train2.get(key, 'N/A')
        match = '✅' if val1 == val2 else '❌'
        print(f"{key:<25} {str(val1):<20} {str(val2):<20} {match}")
        comparison[key] = {'wall': val1, 'qr': val2}
    
    # 學習率調度器
    print("\n【學習率調度器】")
    sched1 = train1.get('lr_scheduler', {})
    sched2 = train2.get('lr_scheduler', {})
    for key in ['type', 'min_lr', 'warmup_epochs', 'T_max']:
        val1 = sched1.get(key, 'N/A')
        val2 = sched2.get(key, 'N/A')
        match = '✅' if val1 == val2 else '❌'
        print(f"  {key:<23} {str(val1):<20} {str(val2):<20} {match}")
    
    return comparison


# =============================================================================
# 感測點分佈分析
# =============================================================================
def load_sensor_data(sensor_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """載入感測點數據"""
    print(f"📂 載入感測點: {sensor_file.name}")
    
    if not sensor_file.exists():
        raise FileNotFoundError(f"感測點文件不存在: {sensor_file}")
    
    data = np.load(sensor_file)
    coords = data['coords']  # (K, 3) - x, y, z
    values = data.get('values', None)  # (K, n_vars) - u, v, w, (p)
    
    print(f"  點數: {coords.shape[0]}")
    print(f"  維度: {coords.shape[1]}D")
    if values is not None:
        print(f"  變量數: {values.shape[1]}")
    
    return coords, values


def analyze_layer_distribution(coords: np.ndarray, strategy_name: str) -> Dict:
    """分析感測點在各物理層的分佈"""
    print(f"\n📊 分析 {strategy_name} 感測點分層分佈...")
    
    y_coords = coords[:, 1]
    y_abs = np.abs(y_coords)  # 對稱處理
    
    layer_stats = {}
    
    for layer_name, layer_info in LAYER_DEFINITIONS.items():
        y_min, y_max = layer_info['range']
        
        # 計算該層的點數
        mask = (y_abs >= y_min) & (y_abs < y_max)
        n_points = mask.sum()
        fraction = n_points / len(y_coords)
        
        layer_stats[layer_name] = {
            'n_points': int(n_points),
            'fraction': float(fraction),
            'y_range': layer_info['range'],
            'name': layer_info['name'],
            'description': layer_info['description']
        }
        
        print(f"  {layer_info['name']} ({layer_name}):")
        print(f"    點數: {n_points} / {len(y_coords)} ({fraction*100:.1f}%)")
        print(f"    y 範圍: [{y_min:.2f}, {y_max:.2f}]")
    
    return layer_stats


def compute_sensor_quality_metrics(coords: np.ndarray, values: Optional[np.ndarray] = None) -> Dict:
    """計算感測點品質指標"""
    metrics = {}
    
    # 空間覆蓋
    for i, axis in enumerate(['x', 'y', 'z']):
        coord_range = coords[:, i].max() - coords[:, i].min()
        coord_std = coords[:, i].std()
        metrics[f'{axis}_range'] = float(coord_range)
        metrics[f'{axis}_std'] = float(coord_std)
    
    # 最近鄰距離
    from scipy.spatial.distance import pdist
    distances = pdist(coords)
    metrics['min_distance'] = float(distances.min())
    metrics['mean_distance'] = float(distances.mean())
    metrics['max_distance'] = float(distances.max())
    
    # 條件數（如果有速度場數據）
    if values is not None:
        U, S, Vt = np.linalg.svd(values, full_matrices=False)
        cond_number = S.max() / (S.min() + 1e-10)
        energy_ratio = S[:10].sum() / S.sum()  # 前 10 個奇異值佔比
        
        metrics['condition_number'] = float(cond_number)
        metrics['energy_ratio_top10'] = float(energy_ratio)
        metrics['singular_values'] = S.tolist()
    
    return metrics


# =============================================================================
# 視覺化函數
# =============================================================================
def plot_sensor_layer_distribution(coords_wall: np.ndarray, coords_qr: np.ndarray,
                                   stats_wall: Dict, stats_qr: Dict, 
                                   output_path: Path):
    """繪製感測點分層分佈對比圖"""
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # === 子圖 1: 分層點數對比柱狀圖 ===
    ax1 = fig.add_subplot(gs[0, 0])
    
    layers = list(LAYER_DEFINITIONS.keys())
    layer_names = [LAYER_DEFINITIONS[l]['name'] for l in layers]
    
    wall_counts = [stats_wall[l]['n_points'] for l in layers]
    qr_counts = [stats_qr[l]['n_points'] for l in layers]
    
    x = np.arange(len(layers))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, wall_counts, width, label='Wall-Clustered', color='#FF6B6B', alpha=0.8)
    bars2 = ax1.bar(x + width/2, qr_counts, width, label='QR-Pivot', color='#4ECDC4', alpha=0.8)
    
    ax1.set_xlabel('Physical Layer', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Sensors', fontsize=12, fontweight='bold')
    ax1.set_title('Sensor Count by Layer', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(layer_names, fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加數值標籤
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)
    
    # === 子圖 2: 分層比例對比餅圖（Wall-Clustered） ===
    ax2 = fig.add_subplot(gs[0, 1])
    
    wall_fractions = [stats_wall[l]['fraction'] for l in layers]
    colors_wall = ['#FF6B6B', '#FFA07A', '#FFB6C1']
    
    wedges, texts, autotexts = ax2.pie(wall_fractions, labels=layer_names, autopct='%1.1f%%',
                                        colors=colors_wall, startangle=90,
                                        textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax2.set_title('Wall-Clustered: Layer Fraction', fontsize=12, fontweight='bold')
    
    # === 子圖 3: 分層比例對比餅圖（QR-Pivot） ===
    ax3 = fig.add_subplot(gs[0, 2])
    
    qr_fractions = [stats_qr[l]['fraction'] for l in layers]
    colors_qr = ['#4ECDC4', '#45B7AF', '#3A9D9A']
    
    wedges, texts, autotexts = ax3.pie(qr_fractions, labels=layer_names, autopct='%1.1f%%',
                                        colors=colors_qr, startangle=90,
                                        textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax3.set_title('QR-Pivot: Layer Fraction', fontsize=12, fontweight='bold')
    
    # === 子圖 4: Wall-Clustered 空間分佈（y 方向） ===
    ax4 = fig.add_subplot(gs[1, 0])
    
    y_wall = coords_wall[:, 1]
    x_wall = coords_wall[:, 0]
    
    # 根據層次著色
    colors_wall_scatter = []
    for y in y_wall:
        y_abs = abs(y)
        if y_abs >= 0.8:
            colors_wall_scatter.append('#FF6B6B')  # 壁面層
        elif y_abs >= 0.2:
            colors_wall_scatter.append('#FFA07A')  # 對數層
        else:
            colors_wall_scatter.append('#FFB6C1')  # 中心層
    
    ax4.scatter(x_wall, y_wall, c=colors_wall_scatter, s=80, alpha=0.7, edgecolors='black', linewidths=0.5)
    
    # 標註層次分界線
    for layer_name, layer_info in LAYER_DEFINITIONS.items():
        y_min, y_max = layer_info['range']
        ax4.axhline(y=y_max, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax4.axhline(y=-y_max, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax4.set_xlabel('x', fontsize=12, fontweight='bold')
    ax4.set_ylabel('y', fontsize=12, fontweight='bold')
    ax4.set_title('Wall-Clustered: Spatial Distribution', fontsize=12, fontweight='bold')
    ax4.set_ylim(-1.05, 1.05)
    ax4.grid(alpha=0.3)
    
    # === 子圖 5: QR-Pivot 空間分佈（y 方向） ===
    ax5 = fig.add_subplot(gs[1, 1])
    
    y_qr = coords_qr[:, 1]
    x_qr = coords_qr[:, 0]
    
    # 根據層次著色
    colors_qr_scatter = []
    for y in y_qr:
        y_abs = abs(y)
        if y_abs >= 0.8:
            colors_qr_scatter.append('#4ECDC4')  # 壁面層
        elif y_abs >= 0.2:
            colors_qr_scatter.append('#45B7AF')  # 對數層
        else:
            colors_qr_scatter.append('#3A9D9A')  # 中心層
    
    ax5.scatter(x_qr, y_qr, c=colors_qr_scatter, s=80, alpha=0.7, edgecolors='black', linewidths=0.5)
    
    # 標註層次分界線
    for layer_name, layer_info in LAYER_DEFINITIONS.items():
        y_min, y_max = layer_info['range']
        ax5.axhline(y=y_max, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax5.axhline(y=-y_max, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax5.set_xlabel('x', fontsize=12, fontweight='bold')
    ax5.set_ylabel('y', fontsize=12, fontweight='bold')
    ax5.set_title('QR-Pivot: Spatial Distribution', fontsize=12, fontweight='bold')
    ax5.set_ylim(-1.05, 1.05)
    ax5.grid(alpha=0.3)
    
    # === 子圖 6: y 方向直方圖對比 ===
    ax6 = fig.add_subplot(gs[1, 2])
    
    ax6.hist(np.abs(y_wall), bins=20, alpha=0.6, label='Wall-Clustered', 
             color='#FF6B6B', edgecolor='black', linewidth=0.5)
    ax6.hist(np.abs(y_qr), bins=20, alpha=0.6, label='QR-Pivot', 
             color='#4ECDC4', edgecolor='black', linewidth=0.5)
    
    # 標註層次分界線
    for layer_name, layer_info in LAYER_DEFINITIONS.items():
        y_min, y_max = layer_info['range']
        ax6.axvline(x=y_max, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax6.set_xlabel('|y|', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax6.set_title('y-Direction Histogram (|y|)', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ 已儲存: {output_path}")
    plt.close()


def plot_quality_metrics_comparison(metrics_wall: Dict, metrics_qr: Dict, output_path: Path):
    """繪製感測點品質指標對比"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Sensor Quality Metrics Comparison', fontsize=16, fontweight='bold')
    
    # === 子圖 1: 空間覆蓋範圍 ===
    ax1 = axes[0, 0]
    axes_labels = ['x', 'y', 'z']
    wall_ranges = [metrics_wall[f'{ax}_range'] for ax in axes_labels]
    qr_ranges = [metrics_qr[f'{ax}_range'] for ax in axes_labels]
    
    x = np.arange(len(axes_labels))
    width = 0.35
    
    ax1.bar(x - width/2, wall_ranges, width, label='Wall-Clustered', color='#FF6B6B', alpha=0.8)
    ax1.bar(x + width/2, qr_ranges, width, label='QR-Pivot', color='#4ECDC4', alpha=0.8)
    
    ax1.set_ylabel('Range', fontsize=11, fontweight='bold')
    ax1.set_title('Spatial Coverage', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(axes_labels)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # === 子圖 2: 空間標準差 ===
    ax2 = axes[0, 1]
    wall_stds = [metrics_wall[f'{ax}_std'] for ax in axes_labels]
    qr_stds = [metrics_qr[f'{ax}_std'] for ax in axes_labels]
    
    ax2.bar(x - width/2, wall_stds, width, label='Wall-Clustered', color='#FF6B6B', alpha=0.8)
    ax2.bar(x + width/2, qr_stds, width, label='QR-Pivot', color='#4ECDC4', alpha=0.8)
    
    ax2.set_ylabel('Standard Deviation', fontsize=11, fontweight='bold')
    ax2.set_title('Spatial Distribution (Std)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(axes_labels)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # === 子圖 3: 點間距離統計 ===
    ax3 = axes[1, 0]
    distance_metrics = ['min_distance', 'mean_distance', 'max_distance']
    distance_labels = ['Min', 'Mean', 'Max']
    wall_distances = [metrics_wall[m] for m in distance_metrics]
    qr_distances = [metrics_qr[m] for m in distance_metrics]
    
    x = np.arange(len(distance_labels))
    ax3.bar(x - width/2, wall_distances, width, label='Wall-Clustered', color='#FF6B6B', alpha=0.8)
    ax3.bar(x + width/2, qr_distances, width, label='QR-Pivot', color='#4ECDC4', alpha=0.8)
    
    ax3.set_ylabel('Distance', fontsize=11, fontweight='bold')
    ax3.set_title('Inter-Sensor Distance', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(distance_labels)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # === 子圖 4: 條件數與能量比例（如果有） ===
    ax4 = axes[1, 1]
    
    if 'condition_number' in metrics_wall and 'condition_number' in metrics_qr:
        # 條件數對比
        strategies = ['Wall-Clustered', 'QR-Pivot']
        cond_numbers = [metrics_wall['condition_number'], metrics_qr['condition_number']]
        energy_ratios = [metrics_wall['energy_ratio_top10'], metrics_qr['energy_ratio_top10']]
        
        x = np.arange(len(strategies))
        
        # 使用雙 y 軸
        ax4_twin = ax4.twinx()
        
        bars1 = ax4.bar(x - 0.2, cond_numbers, 0.4, label='Condition Number', 
                        color='#FF6B6B', alpha=0.8)
        bars2 = ax4_twin.bar(x + 0.2, energy_ratios, 0.4, label='Energy Ratio (Top 10)', 
                             color='#4ECDC4', alpha=0.8)
        
        ax4.set_ylabel('Condition Number', fontsize=11, fontweight='bold', color='#FF6B6B')
        ax4_twin.set_ylabel('Energy Ratio', fontsize=11, fontweight='bold', color='#4ECDC4')
        ax4.set_title('Condition Number & Energy Ratio', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(strategies)
        ax4.tick_params(axis='y', labelcolor='#FF6B6B')
        ax4_twin.tick_params(axis='y', labelcolor='#4ECDC4')
        ax4.grid(alpha=0.3)
        
        # 添加數值標籤
        for bar, val in zip(bars1, cond_numbers):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2e}', ha='center', va='bottom', fontsize=9, color='#FF6B6B')
        
        for bar, val in zip(bars2, energy_ratios):
            height = bar.get_height()
            ax4_twin.text(bar.get_x() + bar.get_width()/2., height,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=9, color='#4ECDC4')
    else:
        ax4.text(0.5, 0.5, 'No velocity data available\nfor quality metrics', 
                ha='center', va='center', fontsize=12, color='gray')
        ax4.set_xticks([])
        ax4.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ 已儲存: {output_path}")
    plt.close()


# =============================================================================
# 主函數
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='分析訓練配置差異與感測點分佈')
    parser.add_argument('--wall-config', type=str, required=True, help='Wall-Clustered 配置文件')
    parser.add_argument('--qr-config', type=str, required=True, help='QR-Pivot 配置文件')
    parser.add_argument('--wall-sensors', type=str, required=True, help='Wall-Clustered 感測點文件')
    parser.add_argument('--qr-sensors', type=str, required=True, help='QR-Pivot 感測點文件')
    parser.add_argument('--output', type=str, default='results/config_sensor_analysis', 
                       help='輸出目錄')
    
    args = parser.parse_args()
    
    # 創建輸出目錄
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("  配置與感測點分析")
    print("="*80)
    
    # =========================================================================
    # 步驟 1: 載入配置文件
    # =========================================================================
    print(f"\n{'='*80}")
    print("步驟 1: 載入配置文件")
    print(f"{'='*80}\n")
    
    config_wall = load_config(Path(args.wall_config))
    config_qr = load_config(Path(args.qr_config))
    
    print(f"  ✅ Wall-Clustered: {args.wall_config}")
    print(f"  ✅ QR-Pivot: {args.qr_config}")
    
    # =========================================================================
    # 步驟 2: 比較訓練配置
    # =========================================================================
    print(f"\n{'='*80}")
    print("步驟 2: 比較訓練配置")
    print(f"{'='*80}")
    
    loss_comparison = compare_loss_weights(config_wall, config_qr, 
                                          'Wall-Clustered', 'QR-Pivot')
    physics_comparison = compare_physics_config(config_wall, config_qr, 
                                               'Wall-Clustered', 'QR-Pivot')
    training_comparison = compare_training_config(config_wall, config_qr, 
                                                 'Wall-Clustered', 'QR-Pivot')
    
    # =========================================================================
    # 步驟 3: 載入感測點數據
    # =========================================================================
    print(f"\n{'='*80}")
    print("步驟 3: 載入感測點數據")
    print(f"{'='*80}\n")
    
    coords_wall, values_wall = load_sensor_data(Path(args.wall_sensors))
    coords_qr, values_qr = load_sensor_data(Path(args.qr_sensors))
    
    # =========================================================================
    # 步驟 4: 分析感測點分層分佈
    # =========================================================================
    print(f"\n{'='*80}")
    print("步驟 4: 分析感測點分層分佈")
    print(f"{'='*80}")
    
    stats_wall = analyze_layer_distribution(coords_wall, 'Wall-Clustered')
    stats_qr = analyze_layer_distribution(coords_qr, 'QR-Pivot')
    
    # =========================================================================
    # 步驟 5: 計算品質指標
    # =========================================================================
    print(f"\n{'='*80}")
    print("步驟 5: 計算感測點品質指標")
    print(f"{'='*80}\n")
    
    print("Wall-Clustered:")
    metrics_wall = compute_sensor_quality_metrics(coords_wall, values_wall)
    for key, val in metrics_wall.items():
        if key != 'singular_values':
            print(f"  {key}: {val}")
    
    print("\nQR-Pivot:")
    metrics_qr = compute_sensor_quality_metrics(coords_qr, values_qr)
    for key, val in metrics_qr.items():
        if key != 'singular_values':
            print(f"  {key}: {val}")
    
    # =========================================================================
    # 步驟 6: 視覺化
    # =========================================================================
    print(f"\n{'='*80}")
    print("步驟 6: 生成視覺化圖表")
    print(f"{'='*80}\n")
    
    plot_sensor_layer_distribution(coords_wall, coords_qr, stats_wall, stats_qr,
                                   output_dir / 'sensor_layer_distribution.png')
    
    plot_quality_metrics_comparison(metrics_wall, metrics_qr,
                                    output_dir / 'sensor_quality_metrics.png')
    
    # =========================================================================
    # 步驟 7: 儲存結果
    # =========================================================================
    print(f"\n{'='*80}")
    print("步驟 7: 儲存分析結果")
    print(f"{'='*80}\n")
    
    # 儲存 JSON 報告
    report = {
        'configuration_comparison': {
            'losses': loss_comparison,
            'physics': physics_comparison,
            'training': training_comparison
        },
        'sensor_distribution': {
            'wall_clustered': stats_wall,
            'qr_pivot': stats_qr
        },
        'sensor_quality': {
            'wall_clustered': {k: v for k, v in metrics_wall.items() if k != 'singular_values'},
            'qr_pivot': {k: v for k, v in metrics_qr.items() if k != 'singular_values'}
        }
    }
    
    json_path = output_dir / 'config_sensor_analysis.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  ✅ JSON 報告已儲存: {json_path}")
    
    # 儲存 Markdown 報告
    md_path = output_dir / 'config_sensor_report.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 配置與感測點分析報告\n\n")
        
        f.write("## 1. 訓練配置比較\n\n")
        f.write("### 損失函數權重\n\n")
        f.write("| 損失項 | Wall-Clustered | QR-Pivot | 差異 |\n")
        f.write("|--------|----------------|----------|------|\n")
        for key, vals in loss_comparison.items():
            f.write(f"| {key} | {vals['wall']} | {vals['qr']} | {vals['diff']} |\n")
        
        f.write("\n### 物理設定\n\n")
        f.write(f"- **RANS 啟用**: Wall-Clustered = {physics_comparison['enable_rans']['wall']}, ")
        f.write(f"QR-Pivot = {physics_comparison['enable_rans']['qr']}\n")
        
        f.write("\n## 2. 感測點分層分佈\n\n")
        f.write("| 物理層 | Wall-Clustered | QR-Pivot |\n")
        f.write("|--------|----------------|----------|\n")
        for layer in ['wall', 'log', 'center']:
            wall_n = stats_wall[layer]['n_points']
            wall_f = stats_wall[layer]['fraction'] * 100
            qr_n = stats_qr[layer]['n_points']
            qr_f = stats_qr[layer]['fraction'] * 100
            f.write(f"| {LAYER_DEFINITIONS[layer]['name']} | {wall_n} ({wall_f:.1f}%) | {qr_n} ({qr_f:.1f}%) |\n")
        
        f.write("\n## 3. 感測點品質指標\n\n")
        f.write("| 指標 | Wall-Clustered | QR-Pivot |\n")
        f.write("|------|----------------|----------|\n")
        for key in ['min_distance', 'mean_distance', 'condition_number', 'energy_ratio_top10']:
            if key in metrics_wall and key in metrics_qr:
                f.write(f"| {key} | {metrics_wall[key]:.4e} | {metrics_qr[key]:.4e} |\n")
    
    print(f"  ✅ Markdown 報告已儲存: {md_path}")
    
    print(f"\n{'='*80}")
    print("✅ 分析完成！")
    print(f"輸出目錄: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
