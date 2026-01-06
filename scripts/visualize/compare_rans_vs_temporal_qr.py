#!/usr/bin/env python3
"""
視覺化比較：原始 QR-Pivot（RANS）vs 時間序列 QR-Pivot（DNS）
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


def visualize_comparison(K: int = 100):
    """
    視覺化比較原始 vs 時間序列 QR-Pivot
    
    Args:
        K: 感測器數量
    """
    # 載入兩種方法的感測器數據
    rans_sensor_file = f"./data/sensors/kolmogorov/sensors_K{K}_re50_256x256.json"
    temporal_sensor_file = f"./data/sensors/kolmogorov/sensors_temporal_K{K}_re50_256x256_t15-35.json"
    
    print(f"📂 載入感測器數據 (K={K})...")
    
    try:
        rans_data = load_sensor_data(rans_sensor_file)
        print(f"   ✓ 原始 RANS/Leith QR-Pivot: {rans_sensor_file}")
    except FileNotFoundError:
        print(f"   ✗ 找不到原始感測器檔案: {rans_sensor_file}")
        rans_data = None
    
    try:
        temporal_data = load_sensor_data(temporal_sensor_file)
        print(f"   ✓ 時間序列 DNS QR-Pivot: {temporal_sensor_file}")
    except FileNotFoundError:
        print(f"   ✗ 找不到時間序列感測器檔案: {temporal_sensor_file}")
        temporal_data = None
    
    if rans_data is None and temporal_data is None:
        print("❌ 兩種感測器檔案都不存在，無法進行比較！")
        return
    
    # 創建圖表
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # === 左圖：原始 RANS/Leith QR-Pivot ===
    if rans_data is not None:
        ax = axes[0]
        indices = np.array(rans_data['indices'])
        nx, ny = 256, 256
        
        # 轉換索引為 (x, y) 座標
        i_indices = indices // ny
        j_indices = indices % ny
        
        # 創建 2D 網格
        grid = np.zeros((nx, ny))
        grid[i_indices, j_indices] = 1
        
        # 繪製
        im = ax.imshow(grid.T, origin='lower', cmap='Reds', interpolation='nearest')
        ax.scatter(i_indices, j_indices, c='red', s=20, alpha=0.8, edgecolors='black', linewidths=0.5)
        ax.set_title(f'Original QR-Pivot (RANS/Leith)\nK={K}', fontsize=14, fontweight='bold')
        ax.set_xlabel('x index')
        ax.set_ylabel('y index')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 添加統計信息
        method = rans_data.get('method', 'Unknown')
        cond_num = rans_data.get('condition_number', -1)
        ax.text(0.02, 0.98, f"Method: {method}\nCond. Num: {cond_num:.2e}", 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        axes[0].text(0.5, 0.5, 'Original sensor file not found', 
                     ha='center', va='center', fontsize=14)
        axes[0].axis('off')
    
    # === 中圖：時間序列 DNS QR-Pivot ===
    if temporal_data is not None:
        ax = axes[1]
        indices = np.array(temporal_data['indices'])
        nx, ny = 256, 256
        
        # 轉換索引為 (x, y) 座標
        i_indices = indices // ny
        j_indices = indices % ny
        
        # 創建 2D 網格
        grid = np.zeros((nx, ny))
        grid[i_indices, j_indices] = 1
        
        # 繪製
        im = ax.imshow(grid.T, origin='lower', cmap='Blues', interpolation='nearest')
        ax.scatter(i_indices, j_indices, c='blue', s=20, alpha=0.8, edgecolors='black', linewidths=0.5)
        ax.set_title(f'Temporal QR-Pivot (DNS Time Series)\nK={K}', fontsize=14, fontweight='bold')
        ax.set_xlabel('x index')
        ax.set_ylabel('y index')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 添加統計信息
        time_range = temporal_data.get('time_range', [0, 0])
        n_time = temporal_data.get('time_steps', 0)
        n_features = temporal_data.get('total_features', 0)
        cond_num = temporal_data.get('condition_number', -1)
        coverage = temporal_data.get('subspace_coverage', -1)
        
        info_text = (
            f"Time: t∈[{time_range[0]:.0f}, {time_range[1]:.0f}]s (N={n_time})\n"
            f"Features: {n_features} (4×{n_time})\n"
            f"Cond. Num: {cond_num:.2e}\n"
            f"Coverage: {coverage:.4f}"
        )
        ax.text(0.02, 0.98, info_text, 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    else:
        axes[1].text(0.5, 0.5, 'Temporal sensor file not found', 
                     ha='center', va='center', fontsize=14)
        axes[1].axis('off')
    
    # === 右圖：重疊比較 ===
    ax = axes[2]
    if rans_data is not None and temporal_data is not None:
        # 原始感測器
        rans_indices = np.array(rans_data['indices'])
        rans_i = rans_indices // ny
        rans_j = rans_indices % ny
        
        # 時間序列感測器
        temporal_indices = np.array(temporal_data['indices'])
        temporal_i = temporal_indices // ny
        temporal_j = temporal_indices % ny
        
        # 繪製
        ax.scatter(rans_i, rans_j, c='red', s=50, alpha=0.6, label='Original (RANS)', 
                   edgecolors='darkred', linewidths=1, marker='s')
        ax.scatter(temporal_i, temporal_j, c='blue', s=30, alpha=0.7, label='Temporal (DNS)', 
                   edgecolors='darkblue', linewidths=1, marker='o')
        
        ax.set_title(f'Overlay Comparison\nK={K}', fontsize=14, fontweight='bold')
        ax.set_xlabel('x index')
        ax.set_ylabel('y index')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(-5, 260)
        ax.set_ylim(-5, 260)
        
        # 計算重疊比例
        rans_set = set(rans_indices)
        temporal_set = set(temporal_indices)
        overlap = rans_set & temporal_set
        overlap_ratio = len(overlap) / K
        
        ax.text(0.02, 0.98, f"Overlap: {len(overlap)}/{K} ({overlap_ratio*100:.1f}%)", 
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'Cannot compare:\nOne or both files missing', 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
    
    plt.tight_layout()
    
    # 保存圖片
    output_dir = Path('./results/sensor_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'qr_comparison_K{K}_rans_vs_temporal.png'
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n💾 圖片已保存: {output_file}")
    plt.close()


def main():
    """批次生成比較圖"""
    K_values = [30, 50, 80, 100]
    
    print("=" * 70)
    print("🎨 視覺化比較：原始 vs 時間序列 QR-Pivot")
    print("=" * 70)
    
    for K in K_values:
        print(f"\n{'='*70}")
        print(f"生成 K={K} 的比較圖...")
        print(f"{'='*70}")
        
        try:
            visualize_comparison(K=K)
        except Exception as e:
            print(f"❌ K={K} 生成失敗: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*70}")
    print("✅ 所有比較圖生成完成！")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
