#!/usr/bin/env python3
"""
視覺化比較：完整梯度特徵的時間序列 QR-Pivot 感測器
比較不同 K 值和不同方法的感測器分布
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def load_sensor_data(sensor_file: str):
    """載入感測器數據"""
    with open(sensor_file, 'r') as f:
        data = json.load(f)
    return data


def visualize_feature_comparison():
    """
    比較不同特徵集合的感測器分布
    - v1.0: 84 特徵（u, v, p, vorticity）
    - v2.0: 210 特徵（完整梯度）
    """
    K = 100  # 使用 K=100 作為範例
    
    print(f"📊 生成特徵比較圖 (K={K})...")
    
    # 載入數據（假設舊版本檔案已被覆蓋，我們從新檔案讀取）
    temporal_file = f"./data/sensors/kolmogorov/sensors_temporal_K{K}_re50_256x256_t15-35.json"
    
    try:
        temporal_data = load_sensor_data(temporal_file)
        print(f"   ✓ 已載入: {temporal_file}")
    except FileNotFoundError:
        print(f"   ✗ 找不到檔案: {temporal_file}")
        return
    
    # 創建圖表
    fig = plt.figure(figsize=(16, 10))
    
    # === 1. 特徵組成餅圖 ===
    ax1 = plt.subplot(2, 3, 1)
    features = temporal_data['features']
    n_features = len(features)
    n_time = temporal_data['time_steps']
    
    # 分組特徵
    feature_groups = {
        'Velocities (u, v)': 2,
        'Pressure (p)': 1,
        'Velocity Gradients': 4,
        'Pressure Gradients': 2,
        'Vorticity': 1
    }
    
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
    explode = (0.05, 0.05, 0, 0, 0.05)
    
    ax1.pie(feature_groups.values(), labels=feature_groups.keys(), autopct='%1.0f%%',
            colors=colors, explode=explode, startangle=90)
    ax1.set_title(f'Feature Composition\n(K={K}, 10 features per time step)', 
                  fontsize=12, fontweight='bold')
    
    # === 2. 條件數比較 ===
    ax2 = plt.subplot(2, 3, 2)
    K_values = [30, 50, 80, 100]
    cond_numbers = []
    coverages = []
    
    for K in K_values:
        try:
            data = load_sensor_data(f"./data/sensors/kolmogorov/sensors_temporal_K{K}_re50_256x256_t15-35.json")
            cond_numbers.append(data['condition_number'])
            coverages.append(data['subspace_coverage'])
        except:
            cond_numbers.append(0)
            coverages.append(0)
    
    ax2_twin = ax2.twinx()
    
    bars1 = ax2.bar(np.arange(len(K_values)) - 0.2, cond_numbers, 0.4, 
                    label='Condition Number', color='steelblue', alpha=0.8)
    bars2 = ax2_twin.bar(np.arange(len(K_values)) + 0.2, 
                         [c*100 for c in coverages], 0.4,
                         label='Coverage (%)', color='coral', alpha=0.8)
    
    ax2.set_xlabel('K (Number of Sensors)', fontweight='bold')
    ax2.set_ylabel('Condition Number', color='steelblue', fontweight='bold')
    ax2_twin.set_ylabel('Subspace Coverage (%)', color='coral', fontweight='bold')
    ax2.set_xticks(range(len(K_values)))
    ax2.set_xticklabels(K_values)
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor='steelblue')
    ax2_twin.tick_params(axis='y', labelcolor='coral')
    ax2.set_title('Condition Number & Coverage vs K', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 圖例
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    
    # === 3. 時間-特徵矩陣結構 ===
    ax3 = plt.subplot(2, 3, 3)
    
    # 繪製特徵矩陣結構示意圖
    n_features_per_time = 10
    n_time_steps = 21
    
    # 創建示意矩陣
    matrix = np.zeros((n_features_per_time, n_time_steps))
    feature_names_short = ['u', 'v', 'p', 'du/dx', 'du/dy', 'dv/dx', 'dv/dy', 'dp/dx', 'dp/dy', 'ω']
    
    # 填充不同顏色區塊
    for i in range(n_features_per_time):
        if i < 2:  # u, v
            matrix[i, :] = 1
        elif i == 2:  # p
            matrix[i, :] = 2
        elif i < 7:  # gradients
            matrix[i, :] = 3
        elif i < 9:  # pressure gradients
            matrix[i, :] = 4
        else:  # vorticity
            matrix[i, :] = 5
    
    im = ax3.imshow(matrix, cmap='tab10', aspect='auto', interpolation='nearest')
    ax3.set_yticks(range(n_features_per_time))
    ax3.set_yticklabels(feature_names_short, fontsize=9)
    ax3.set_xlabel('Time Steps (15→35s)', fontweight='bold')
    ax3.set_ylabel('Features', fontweight='bold')
    ax3.set_title('Spatio-Temporal Feature Matrix\n(210 features = 10 × 21)', 
                  fontsize=12, fontweight='bold')
    
    # 顯示時間範圍
    time_ticks = [0, 5, 10, 15, 20]
    time_labels = ['15s', '20s', '25s', '30s', '35s']
    ax3.set_xticks(time_ticks)
    ax3.set_xticklabels(time_labels)
    
    # === 4. 感測器空間分布 (K=100) ===
    ax4 = plt.subplot(2, 3, 4)
    K = 100
    data = load_sensor_data(f"./data/sensors/kolmogorov/sensors_temporal_K{K}_re50_256x256_t15-35.json")
    indices = np.array(data['indices'])
    nx, ny = 256, 256
    
    i_indices = indices // ny
    j_indices = indices % ny
    
    grid = np.zeros((nx, ny))
    grid[i_indices, j_indices] = 1
    
    im4 = ax4.imshow(grid.T, origin='lower', cmap='YlOrRd', interpolation='nearest')
    ax4.scatter(i_indices, j_indices, c='darkred', s=15, alpha=0.7, 
               edgecolors='black', linewidths=0.5)
    ax4.set_title(f'Sensor Distribution (K={K})\nFull Gradient Features (210D)', 
                  fontsize=12, fontweight='bold')
    ax4.set_xlabel('x index')
    ax4.set_ylabel('y index')
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    # === 5. 不同 K 值的空間分布比較 ===
    ax5 = plt.subplot(2, 3, 5)
    
    K_values_plot = [30, 50, 80, 100]
    colors_k = ['blue', 'green', 'orange', 'red']
    markers = ['o', 's', '^', 'D']
    
    for idx, K in enumerate(K_values_plot):
        try:
            data = load_sensor_data(f"./data/sensors/kolmogorov/sensors_temporal_K{K}_re50_256x256_t15-35.json")
            indices = np.array(data['indices'])
            i_indices = indices // ny
            j_indices = indices % ny
            
            ax5.scatter(i_indices, j_indices, c=colors_k[idx], s=20, 
                       alpha=0.6, label=f'K={K}', marker=markers[idx],
                       edgecolors='black', linewidths=0.5)
        except:
            pass
    
    ax5.set_title('Sensor Distributions Comparison\n(All K values)', 
                  fontsize=12, fontweight='bold')
    ax5.set_xlabel('x index')
    ax5.set_ylabel('y index')
    ax5.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.set_xlim(-5, 260)
    ax5.set_ylim(-5, 260)
    
    # === 6. 統計信息表格 ===
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # 創建表格數據
    table_data = [
        ['K', 'Cond. Num', 'Coverage', 'Energy'],
    ]
    
    for K in [30, 50, 80, 100]:
        try:
            data = load_sensor_data(f"./data/sensors/kolmogorov/sensors_temporal_K{K}_re50_256x256_t15-35.json")
            cond = f"{data['condition_number']:.2e}"
            cov = f"{data['subspace_coverage']*100:.1f}%"
            eng = f"{data['energy_ratio']*100:.0f}%"
            table_data.append([str(K), cond, cov, eng])
        except:
            table_data.append([str(K), 'N/A', 'N/A', 'N/A'])
    
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.15, 0.3, 0.25, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # 設置表頭樣式
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 設置交替行顏色
    for i in range(1, len(table_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
            else:
                table[(i, j)].set_facecolor('#F2F2F2')
    
    ax6.set_title('Sensor Quality Metrics\n(210 Features: 10 × 21 time steps)', 
                  fontsize=12, fontweight='bold', pad=20)
    
    # 添加方法說明
    method_text = (
        "Method: QR-Pivot from DNS Time Series (Full Gradients)\n"
        "Features: u, v, p, ∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y, ∂p/∂x, ∂p/∂y, ω_z\n"
        "Time Range: [15, 35] seconds (21 snapshots @ 1s interval)\n"
        "Total Feature Dimension: 10 × 21 = 210"
    )
    ax6.text(0.5, 0.15, method_text, transform=ax6.transAxes,
            fontsize=9, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存圖片
    output_dir = Path('./results/sensor_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'temporal_qr_full_gradient_comparison.png'
    
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"\n💾 圖片已保存: {output_file}")
    plt.close()


def visualize_k_comparison():
    """
    生成 K 值比較的詳細圖表
    """
    print(f"\n📊 生成 K 值比較圖...")
    
    K_values = [30, 50, 80, 100]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.ravel()
    
    for idx, K in enumerate(K_values):
        ax = axes[idx]
        
        try:
            data = load_sensor_data(f"./data/sensors/kolmogorov/sensors_temporal_K{K}_re50_256x256_t15-35.json")
            indices = np.array(data['indices'])
            nx, ny = 256, 256
            
            i_indices = indices // ny
            j_indices = indices % ny
            
            # 創建 2D 網格
            grid = np.zeros((nx, ny))
            grid[i_indices, j_indices] = 1
            
            # 繪製
            im = ax.imshow(grid.T, origin='lower', cmap='YlOrRd', 
                          interpolation='nearest', alpha=0.6)
            ax.scatter(i_indices, j_indices, c='darkred', s=30, alpha=0.8, 
                      edgecolors='black', linewidths=0.8)
            
            # 標題與統計信息
            cond_num = data['condition_number']
            coverage = data['subspace_coverage']
            n_features = data['total_features']
            
            title_text = (
                f"K = {K} Sensors\n"
                f"Cond. Num: {cond_num:.2e} | Coverage: {coverage*100:.1f}%"
            )
            ax.set_title(title_text, fontsize=13, fontweight='bold', pad=10)
            ax.set_xlabel('x index', fontsize=11)
            ax.set_ylabel('y index', fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # 添加信息框
            info_text = (
                f"Features: {n_features}\n"
                f"Time steps: {data['time_steps']}\n"
                f"Spatial coverage: {len(indices)}/{nx*ny}"
            )
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
            
        except FileNotFoundError:
            ax.text(0.5, 0.5, f'K={K}\nFile not found', 
                   ha='center', va='center', fontsize=14)
            ax.set_title(f'K = {K} (Missing)', fontsize=13, fontweight='bold')
    
    plt.suptitle('Temporal QR-Pivot Sensor Comparison (Full Gradient Features)\n'
                 '210 Features: [u, v, p, ∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y, ∂p/∂x, ∂p/∂y, ω_z] × 21 time steps',
                 fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    
    # 保存圖片
    output_dir = Path('./results/sensor_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'temporal_qr_k_comparison_grid.png'
    
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"💾 圖片已保存: {output_file}")
    plt.close()


def main():
    """生成所有比較圖"""
    print("=" * 70)
    print("🎨 生成時間序列 QR-Pivot 感測器視覺化比較圖")
    print("=" * 70)
    
    try:
        # 1. 特徵比較圖（綜合）
        visualize_feature_comparison()
        
        # 2. K 值比較圖（詳細）
        visualize_k_comparison()
        
        print(f"\n{'='*70}")
        print("✅ 所有比較圖生成完成！")
        print(f"{'='*70}")
        print("\n📁 生成的圖片:")
        print("   1. temporal_qr_full_gradient_comparison.png（特徵與統計綜合比較）")
        print("   2. temporal_qr_k_comparison_grid.png（不同 K 值的空間分布）")
        
    except Exception as e:
        print(f"❌ 生成失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
