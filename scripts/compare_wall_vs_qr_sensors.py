#!/usr/bin/env python3
"""
Wall-Clustered vs QR-Pivot 感測點策略完整比較

比較內容：
1. 空間分佈（2D/3D）
2. 品質指標（條件數、能量比例、覆蓋率）
3. 物理區域覆蓋（壁面層、對數層、中心層）
4. 訓練結果（損失曲線、收斂速度、最終誤差）

使用方式：
    python scripts/compare_wall_vs_qr_sensors.py --output results/sensor_comparison/
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import json
from typing import Dict, Tuple, List, Optional
import sys

# 添加 pinnx 路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinnx.sensors.qr_pivot import evaluate_sensor_placement


def load_sensor_data(sensor_file: Path) -> Dict:
    """載入感測點資料"""
    data = np.load(sensor_file, allow_pickle=True)
    
    # 提取品質指標（只提取標量值）
    metrics = {}
    scalar_keys = ['condition_number', 'energy_ratio', 'subspace_coverage']
    for key in scalar_keys:
        if key in data and data[key].shape == ():
            value = float(data[key])
            if not np.isnan(value):
                metrics[key] = value
    
    return {
        'indices': data['indices'],
        'coords': data['coords'],
        'metrics': metrics
    }


def load_jhtdb_field(cache_dir: Path) -> Optional[np.ndarray]:
    """載入 JHTDB 2D 切片資料（用於計算品質指標）
    
    注意：必須使用 2D 切片而非 3D 全場，因為感測點索引對應切片位置
    """
    # 嘗試從 HDF5 文件載入 3D 速度場並提取 z=4.71 切片
    h5_file = cache_dir / "raw" / "JHU Turbulence Channel_velocity_t1.h5"
    if h5_file.exists():
        try:
            import h5py
            with h5py.File(h5_file, 'r') as f:
                velocity_full = np.array(f['Velocity_0001'])  # type: ignore  # shape: [Ny, Nx, Nz, 3]
                zcoor = np.array(f['zcoor'])  # type: ignore
                
                # 找到 z ≈ 4.71 的索引
                z_target = 4.71
                z_idx = np.argmin(np.abs(zcoor - z_target))
                
                # 提取 2D 切片: [Ny, Nx, 3]
                slice_2d = velocity_full[:, :, z_idx, :]
                
                # Reshape 為 [N_points, 3]：按照 (y, x) 順序展平
                field = slice_2d.reshape(-1, 3)
                print(f"  ✅ 成功載入 2D 切片 (z={zcoor[z_idx]:.2f}): shape={field.shape}")
                return field
        except Exception as e:
            print(f"⚠️  載入 HDF5 失敗: {e}")
            return None
    
    # 備用：嘗試載入預存的 2D 切片資料
    slice_file = cache_dir / "channel_flow_2d_slice_z4.71.npz"
    if slice_file.exists():
        data = np.load(slice_file)
        u = data['u'].flatten()
        v = data['v'].flatten()
        w = data['w'].flatten()
        field = np.column_stack([u, v, w])
        print(f"  ✅ 成功載入預存 2D 切片: shape={field.shape}")
        return field
    
    print(f"⚠️  找不到 JHTDB 資料: {h5_file} 或 {slice_file}")
    return None


def analyze_spatial_distribution(coords: np.ndarray, name: str) -> Dict:
    """分析空間分佈特性"""
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    
    # 基本統計
    stats = {
        'x_mean': float(np.mean(x)),
        'x_std': float(np.std(x)),
        'y_mean': float(np.mean(y)),
        'y_std': float(np.std(y)),
        'z_mean': float(np.mean(z)),
        'z_std': float(np.std(z)),
        'min_distance': float(np.min(np.diff(np.sort(y)))),  # y 方向最小間距
    }
    
    # 物理區域分佈（通道流特定）
    y_abs = np.abs(y)
    wall_layer = np.sum(y_abs > 0.8)  # |y| > 0.8
    log_layer = np.sum((y_abs > 0.2) & (y_abs <= 0.8))  # 0.2 < |y| <= 0.8
    center = np.sum(y_abs <= 0.2)  # |y| <= 0.2
    
    stats['wall_layer_count'] = int(wall_layer)
    stats['log_layer_count'] = int(log_layer)
    stats['center_count'] = int(center)
    stats['wall_layer_fraction'] = float(wall_layer / len(y))
    stats['log_layer_fraction'] = float(log_layer / len(y))
    stats['center_fraction'] = float(center / len(y))
    
    return stats


def compute_quality_metrics(field_data: Optional[np.ndarray], indices: np.ndarray) -> Dict:
    """使用修正後的方法計算品質指標"""
    if field_data is None:
        return {}
    
    try:
        metrics = evaluate_sensor_placement(field_data, indices)
        return metrics
    except Exception as e:
        print(f"⚠️  計算品質指標失敗: {e}")
        return {}


def plot_2d_distribution_comparison(
    coords_wall: np.ndarray,
    coords_qr: np.ndarray,
    output_dir: Path
):
    """繪製 2D 空間分佈對比圖"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # XY 平面
    axes[0, 0].scatter(coords_wall[:, 0], coords_wall[:, 1], 
                       c='red', s=100, alpha=0.6, edgecolors='black', 
                       label='Wall-Clustered')
    axes[0, 0].set_xlabel('x (Streamwise)', fontsize=12)
    axes[0, 0].set_ylabel('y (Wall-normal)', fontsize=12)
    axes[0, 0].set_title('Wall-Clustered: XY Plane', fontsize=14, fontweight='bold')
    axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].axhline(y=1.0, color='black', linewidth=2, label='Wall')
    axes[0, 0].axhline(y=-1.0, color='black', linewidth=2)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].scatter(coords_qr[:, 0], coords_qr[:, 1], 
                       c='blue', s=100, alpha=0.6, edgecolors='black', 
                       label='QR-Pivot')
    axes[0, 1].set_xlabel('x (Streamwise)', fontsize=12)
    axes[0, 1].set_ylabel('y (Wall-normal)', fontsize=12)
    axes[0, 1].set_title('QR-Pivot: XY Plane', fontsize=14, fontweight='bold')
    axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].axhline(y=1.0, color='black', linewidth=2, label='Wall')
    axes[0, 1].axhline(y=-1.0, color='black', linewidth=2)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Y 方向分佈直方圖
    axes[1, 0].hist(coords_wall[:, 1], bins=20, color='red', alpha=0.6, 
                    edgecolor='black', label='Wall-Clustered')
    axes[1, 0].axvline(x=0.8, color='orange', linestyle='--', label='Log layer boundary')
    axes[1, 0].axvline(x=-0.8, color='orange', linestyle='--')
    axes[1, 0].axvline(x=0.2, color='green', linestyle='--', label='Center boundary')
    axes[1, 0].axvline(x=-0.2, color='green', linestyle='--')
    axes[1, 0].set_xlabel('y (Wall-normal)', fontsize=12)
    axes[1, 0].set_ylabel('Count', fontsize=12)
    axes[1, 0].set_title('Wall-Clustered: Y Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(coords_qr[:, 1], bins=20, color='blue', alpha=0.6, 
                    edgecolor='black', label='QR-Pivot')
    axes[1, 1].axvline(x=0.8, color='orange', linestyle='--', label='Log layer boundary')
    axes[1, 1].axvline(x=-0.8, color='orange', linestyle='--')
    axes[1, 1].axvline(x=0.2, color='green', linestyle='--', label='Center boundary')
    axes[1, 1].axvline(x=-0.2, color='green', linestyle='--')
    axes[1, 1].set_xlabel('y (Wall-normal)', fontsize=12)
    axes[1, 1].set_ylabel('Count', fontsize=12)
    axes[1, 1].set_title('QR-Pivot: Y Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "spatial_distribution_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 已保存 2D 分佈對比圖: {output_file}")
    plt.close()


def plot_quality_metrics_comparison(
    metrics_wall: Dict,
    metrics_qr: Dict,
    output_dir: Path
):
    """繪製品質指標對比圖"""
    # 提取共同指標
    metric_names = ['condition_number', 'energy_ratio', 'subspace_coverage']
    metric_labels = ['Condition Number\n(κ, lower is better)', 
                     'Energy Ratio\n(higher is better)', 
                     'Subspace Coverage\n(higher is better)']
    
    wall_values = [metrics_wall.get(m, np.nan) for m in metric_names]
    qr_values = [metrics_qr.get(m, np.nan) for m in metric_names]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (ax, name, label) in enumerate(zip(axes, metric_names, metric_labels)):
        x = [0, 1]
        y = [wall_values[i], qr_values[i]]
        colors = ['red', 'blue']
        labels = ['Wall-Clustered', 'QR-Pivot']
        
        bars = ax.bar(x, y, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(label, fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        
        # 添加數值標籤
        for j, (bar, val) in enumerate(zip(bars, y)):
            if not np.isnan(val):
                if name == 'condition_number':
                    text = f'{val:.2e}'
                else:
                    text = f'{val:.4f}'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       text, ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / "quality_metrics_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 已保存品質指標對比圖: {output_file}")
    plt.close()


def plot_layer_distribution_comparison(
    stats_wall: Dict,
    stats_qr: Dict,
    output_dir: Path
):
    """繪製物理層分佈對比圖"""
    layers = ['Wall Layer\n(|y| > 0.8)', 'Log Layer\n(0.2 < |y| ≤ 0.8)', 'Center\n(|y| ≤ 0.2)']
    wall_fractions = [
        stats_wall['wall_layer_fraction'],
        stats_wall['log_layer_fraction'],
        stats_wall['center_fraction']
    ]
    qr_fractions = [
        stats_qr['wall_layer_fraction'],
        stats_qr['log_layer_fraction'],
        stats_qr['center_fraction']
    ]
    
    x = np.arange(len(layers))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, wall_fractions, width, label='Wall-Clustered', 
                   color='red', alpha=0.7, edgecolor='black', linewidth=2)
    bars2 = ax.bar(x + width/2, qr_fractions, width, label='QR-Pivot', 
                   color='blue', alpha=0.7, edgecolor='black', linewidth=2)
    
    ax.set_xlabel('Physical Layer', fontsize=14)
    ax.set_ylabel('Fraction of Sensors', fontsize=14)
    ax.set_title('Physical Layer Coverage Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layers, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    
    # 添加百分比標籤
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                   f'{height*100:.1f}%', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / "layer_distribution_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 已保存物理層分佈對比圖: {output_file}")
    plt.close()


def load_training_logs(checkpoint_dir: Path) -> Dict:
    """載入訓練日誌"""
    # 嘗試從 nohup.out 解析損失曲線
    log_file = checkpoint_dir.parent.parent / "log" / checkpoint_dir.name / "nohup.out"
    
    if not log_file.exists():
        print(f"⚠️  找不到訓練日誌: {log_file}")
        return {}
    
    epochs = []
    total_losses = []
    data_losses = []
    pde_losses = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # 解析新格式: "Epoch 10/100 | total_loss: 0.1234 | data_loss: 0.0567 | pde_loss: 0.0890 ..."
            if 'Epoch' in line and 'total_loss:' in line:
                try:
                    # 提取 epoch 數字
                    epoch_part = line.split('|')[0]
                    epoch_str = epoch_part.split('Epoch')[-1].split('/')[0].strip()
                    epoch = int(epoch_str)
                    
                    # 提取 total_loss
                    if 'total_loss:' in line:
                        total_loss_str = line.split('total_loss:')[1].split('|')[0].strip()
                        total_loss = float(total_loss_str)
                    else:
                        continue
                    
                    # 提取 data_loss 和 pde_loss（可選）
                    data_loss = None
                    pde_loss = None
                    if 'data_loss:' in line:
                        data_loss_str = line.split('data_loss:')[1].split('|')[0].strip()
                        data_loss = float(data_loss_str)
                    if 'pde_loss:' in line:
                        pde_loss_str = line.split('pde_loss:')[1].split('|')[0].strip()
                        pde_loss = float(pde_loss_str)
                    
                    epochs.append(epoch)
                    total_losses.append(total_loss)
                    if data_loss is not None:
                        data_losses.append(data_loss)
                    if pde_loss is not None:
                        pde_losses.append(pde_loss)
                    
                except Exception as e:
                    continue
    
    result = {
        'epochs': np.array(epochs),
        'total_loss': np.array(total_losses),
    }
    
    if len(data_losses) == len(epochs):
        result['data_loss'] = np.array(data_losses)
    if len(pde_losses) == len(epochs):
        result['pde_loss'] = np.array(pde_losses)
    
    return result


def plot_training_comparison(
    logs_wall: Dict,
    logs_qr: Dict,
    output_dir: Path
):
    """繪製訓練過程對比圖"""
    if not logs_wall or not logs_qr:
        print("⚠️  缺少訓練日誌，跳過訓練對比圖")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if 'total_loss' in logs_wall and len(logs_wall['total_loss']) > 0:
        ax.plot(logs_wall['epochs'], logs_wall['total_loss'], 
               'r-', linewidth=2, label='Wall-Clustered', alpha=0.8)
    
    if 'total_loss' in logs_qr and len(logs_qr['total_loss']) > 0:
        ax.plot(logs_qr['epochs'], logs_qr['total_loss'], 
               'b-', linewidth=2, label='QR-Pivot', alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Total Loss', fontsize=14)
    ax.set_title('Training Loss Comparison', fontsize=16, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "training_loss_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 已保存訓練損失對比圖: {output_file}")
    plt.close()


def generate_comparison_report(
    stats_wall: Dict,
    stats_qr: Dict,
    metrics_wall: Dict,
    metrics_qr: Dict,
    output_dir: Path,
    logs_wall: Optional[Dict] = None,
    logs_qr: Optional[Dict] = None
):
    """生成完整比較報告"""
    report = {
        'wall_clustered': {
            'spatial_distribution': stats_wall,
            'quality_metrics': metrics_wall
        },
        'qr_pivot': {
            'spatial_distribution': stats_qr,
            'quality_metrics': metrics_qr
        },
        'comparison': {
            'condition_number_ratio': metrics_wall.get('condition_number', np.nan) / 
                                     metrics_qr.get('condition_number', np.nan),
            'energy_ratio_diff': metrics_wall.get('energy_ratio', np.nan) - 
                                metrics_qr.get('energy_ratio', np.nan),
            'wall_layer_fraction_diff': stats_wall['wall_layer_fraction'] - 
                                       stats_qr['wall_layer_fraction']
        }
    }
    
    # 添加訓練日誌數據（如果可用）
    if logs_wall and logs_qr:
        report['training_logs'] = {
            'wall_clustered': {
                'epochs': logs_wall.get('epochs', []).tolist() if isinstance(logs_wall.get('epochs'), np.ndarray) else logs_wall.get('epochs', []),
                'total_loss': logs_wall.get('total_loss', []).tolist() if isinstance(logs_wall.get('total_loss'), np.ndarray) else logs_wall.get('total_loss', []),
                'data_loss': logs_wall.get('data_loss', []).tolist() if isinstance(logs_wall.get('data_loss'), np.ndarray) else logs_wall.get('data_loss', []),
                'pde_loss': logs_wall.get('pde_loss', []).tolist() if isinstance(logs_wall.get('pde_loss'), np.ndarray) else logs_wall.get('pde_loss', [])
            },
            'qr_pivot': {
                'epochs': logs_qr.get('epochs', []).tolist() if isinstance(logs_qr.get('epochs'), np.ndarray) else logs_qr.get('epochs', []),
                'total_loss': logs_qr.get('total_loss', []).tolist() if isinstance(logs_qr.get('total_loss'), np.ndarray) else logs_qr.get('total_loss', []),
                'data_loss': logs_qr.get('data_loss', []).tolist() if isinstance(logs_qr.get('data_loss'), np.ndarray) else logs_qr.get('data_loss', []),
                'pde_loss': logs_qr.get('pde_loss', []).tolist() if isinstance(logs_qr.get('pde_loss'), np.ndarray) else logs_qr.get('pde_loss', [])
            }
        }
    
    # 保存 JSON
    json_file = output_dir / "comparison_report.json"
    with open(json_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✅ 已保存 JSON 報告: {json_file}")
    
    # 生成 Markdown 報告
    md_file = output_dir / "comparison_report.md"
    with open(md_file, 'w') as f:
        f.write("# Wall-Clustered vs QR-Pivot 感測點策略比較報告\n\n")
        
        f.write("## 1. 空間分佈統計\n\n")
        f.write("| 指標 | Wall-Clustered | QR-Pivot | 差異 |\n")
        f.write("|------|----------------|----------|------|\n")
        f.write(f"| 壁面層比例 | {stats_wall['wall_layer_fraction']:.2%} | "
                f"{stats_qr['wall_layer_fraction']:.2%} | "
                f"{stats_wall['wall_layer_fraction'] - stats_qr['wall_layer_fraction']:+.2%} |\n")
        f.write(f"| 對數層比例 | {stats_wall['log_layer_fraction']:.2%} | "
                f"{stats_qr['log_layer_fraction']:.2%} | "
                f"{stats_wall['log_layer_fraction'] - stats_qr['log_layer_fraction']:+.2%} |\n")
        f.write(f"| 中心層比例 | {stats_wall['center_fraction']:.2%} | "
                f"{stats_qr['center_fraction']:.2%} | "
                f"{stats_wall['center_fraction'] - stats_qr['center_fraction']:+.2%} |\n")
        f.write(f"| Y 方向標準差 | {stats_wall['y_std']:.4f} | "
                f"{stats_qr['y_std']:.4f} | "
                f"{stats_wall['y_std'] - stats_qr['y_std']:+.4f} |\n\n")
        
        f.write("## 2. 品質指標比較\n\n")
        f.write("| 指標 | Wall-Clustered | QR-Pivot | 比值 |\n")
        f.write("|------|----------------|----------|------|\n")
        
        cond_wall = metrics_wall.get('condition_number', np.nan)
        cond_qr = metrics_qr.get('condition_number', np.nan)
        f.write(f"| 條件數 | {cond_wall:.2e} | {cond_qr:.2e} | "
                f"{cond_wall/cond_qr:.2f}× |\n")
        
        energy_wall = metrics_wall.get('energy_ratio', np.nan)
        energy_qr = metrics_qr.get('energy_ratio', np.nan)
        f.write(f"| 能量比例 | {energy_wall:.4f} | {energy_qr:.4f} | "
                f"{energy_wall - energy_qr:+.4f} |\n")
        
        coverage_wall = metrics_wall.get('subspace_coverage', np.nan)
        coverage_qr = metrics_qr.get('subspace_coverage', np.nan)
        f.write(f"| 子空間覆蓋 | {coverage_wall:.4f} | {coverage_qr:.4f} | "
                f"{coverage_wall - coverage_qr:+.4f} |\n\n")
        
        f.write("## 3. 訓練結果比較\n\n")
        if logs_wall and logs_qr and len(logs_wall.get('epochs', [])) > 0 and len(logs_qr.get('epochs', [])) > 0:
            # 最終損失值
            final_loss_wall = logs_wall['total_loss'][-1]
            final_loss_qr = logs_qr['total_loss'][-1]
            
            # 損失下降率
            initial_loss_wall = logs_wall['total_loss'][0]
            initial_loss_qr = logs_qr['total_loss'][0]
            reduction_wall = (initial_loss_wall - final_loss_wall) / initial_loss_wall * 100
            reduction_qr = (initial_loss_qr - final_loss_qr) / initial_loss_qr * 100
            
            f.write("| 指標 | Wall-Clustered | QR-Pivot | 差異 |\n")
            f.write("|------|----------------|----------|------|\n")
            f.write(f"| 訓練輪數 | {len(logs_wall['epochs'])} | {len(logs_qr['epochs'])} | - |\n")
            f.write(f"| 初始損失 | {initial_loss_wall:.4e} | {initial_loss_qr:.4e} | "
                    f"{initial_loss_wall/initial_loss_qr:.2f}× |\n")
            f.write(f"| 最終損失 | {final_loss_wall:.4e} | {final_loss_qr:.4e} | "
                    f"{final_loss_wall/final_loss_qr:.2f}× |\n")
            f.write(f"| 損失下降率 | {reduction_wall:.1f}% | {reduction_qr:.1f}% | "
                    f"{reduction_wall - reduction_qr:+.1f}% |\n\n")
            
            # 添加詳細損失項（如果可用）
            if 'data_loss' in logs_wall and 'data_loss' in logs_qr:
                f.write("### 損失項分解（最終值）\n\n")
                f.write("| 損失項 | Wall-Clustered | QR-Pivot | 比值 |\n")
                f.write("|--------|----------------|----------|------|\n")
                f.write(f"| Data Loss | {logs_wall['data_loss'][-1]:.4e} | "
                        f"{logs_qr['data_loss'][-1]:.4e} | "
                        f"{logs_wall['data_loss'][-1]/logs_qr['data_loss'][-1]:.2f}× |\n")
                f.write(f"| PDE Loss | {logs_wall['pde_loss'][-1]:.4e} | "
                        f"{logs_qr['pde_loss'][-1]:.4e} | "
                        f"{logs_wall['pde_loss'][-1]/logs_qr['pde_loss'][-1]:.2f}× |\n\n")
        else:
            f.write("⚠️  訓練日誌數據不可用\n\n")
        
        f.write("## 4. 策略特性總結\n\n")
        f.write("### Wall-Clustered 策略\n")
        f.write("- **優勢**: 物理先驗（強制壁面層覆蓋）、生成成本低\n")
        f.write("- **劣勢**: 條件數較高（數值穩定性較差）、可能遺漏關鍵模態\n\n")
        
        f.write("### QR-Pivot 策略\n")
        f.write("- **優勢**: 條件數優秀（數值穩定）、最大化資訊熵\n")
        f.write("- **劣勢**: 計算成本高、可能偏向高梯度區域\n\n")
    
    print(f"✅ 已保存 Markdown 報告: {md_file}")


def main():
    parser = argparse.ArgumentParser(description='比較 Wall-Clustered 和 QR-Pivot 感測點策略')
    parser.add_argument('--wall-sensors', type=str, 
                       default='./data/jhtdb/channel_flow_re1000/sensors_K50_wall_clustered.npz',
                       help='Wall-Clustered 感測點文件路徑')
    parser.add_argument('--qr-sensors', type=str, 
                       default='./data/jhtdb/channel_flow_re1000/sensors_K50_qr_pivot.npz',
                       help='QR-Pivot 感測點文件路徑')
    parser.add_argument('--jhtdb-cache', type=str, 
                       default='./data/jhtdb/channel_flow_re1000',
                       help='JHTDB 資料快取目錄')
    parser.add_argument('--wall-checkpoint', type=str, 
                       default='./checkpoints/wall_clustered_random_K50_validation',
                       help='Wall-Clustered 訓練檢查點目錄')
    parser.add_argument('--qr-checkpoint', type=str, 
                       default='./checkpoints/velocity_qr_pivot_K50_validation',
                       help='QR-Pivot 訓練檢查點目錄')
    parser.add_argument('--output', type=str, 
                       default='./results/sensor_comparison',
                       help='輸出目錄')
    
    args = parser.parse_args()
    
    # 創建輸出目錄
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Wall-Clustered vs QR-Pivot 感測點策略完整比較")
    print("=" * 80)
    
    # 1. 載入感測點資料
    print("\n[1/6] 載入感測點資料...")
    sensor_wall = load_sensor_data(Path(args.wall_sensors))
    sensor_qr = load_sensor_data(Path(args.qr_sensors))
    print(f"  ✅ Wall-Clustered: {len(sensor_wall['coords'])} 個感測點")
    print(f"  ✅ QR-Pivot: {len(sensor_qr['coords'])} 個感測點")
    
    # 2. 分析空間分佈
    print("\n[2/6] 分析空間分佈特性...")
    stats_wall = analyze_spatial_distribution(sensor_wall['coords'], 'Wall-Clustered')
    stats_qr = analyze_spatial_distribution(sensor_qr['coords'], 'QR-Pivot')
    print(f"  ✅ Wall-Clustered: 壁面層 {stats_wall['wall_layer_count']}, "
          f"對數層 {stats_wall['log_layer_count']}, 中心層 {stats_wall['center_count']}")
    print(f"  ✅ QR-Pivot: 壁面層 {stats_qr['wall_layer_count']}, "
          f"對數層 {stats_qr['log_layer_count']}, 中心層 {stats_qr['center_count']}")
    
    # 3. 計算品質指標
    print("\n[3/6] 計算品質指標...")
    field_data = load_jhtdb_field(Path(args.jhtdb_cache))
    if field_data is not None:
        metrics_wall = compute_quality_metrics(field_data, sensor_wall['indices'])
        metrics_qr = compute_quality_metrics(field_data, sensor_qr['indices'])
        print(f"  ✅ Wall-Clustered: κ={metrics_wall.get('condition_number', 'N/A'):.2e}, "
              f"E={metrics_wall.get('energy_ratio', 'N/A'):.4f}")
        print(f"  ✅ QR-Pivot: κ={metrics_qr.get('condition_number', 'N/A'):.2e}, "
              f"E={metrics_qr.get('energy_ratio', 'N/A'):.4f}")
    else:
        metrics_wall = sensor_wall['metrics']
        metrics_qr = sensor_qr['metrics']
        print("  ⚠️  使用預存的品質指標")
    
    # 4. 生成視覺化圖表
    print("\n[4/6] 生成視覺化圖表...")
    plot_2d_distribution_comparison(sensor_wall['coords'], sensor_qr['coords'], output_dir)
    plot_quality_metrics_comparison(metrics_wall, metrics_qr, output_dir)
    plot_layer_distribution_comparison(stats_wall, stats_qr, output_dir)
    
    # 5. 載入訓練日誌並比較
    print("\n[5/6] 分析訓練結果...")
    logs_wall = load_training_logs(Path(args.wall_checkpoint))
    logs_qr = load_training_logs(Path(args.qr_checkpoint))
    plot_training_comparison(logs_wall, logs_qr, output_dir)
    
    # 6. 生成完整報告
    print("\n[6/6] 生成完整報告...")
    generate_comparison_report(stats_wall, stats_qr, metrics_wall, metrics_qr, output_dir, logs_wall, logs_qr)
    
    print("\n" + "=" * 80)
    print(f"✅ 比較分析完成！所有結果已保存至: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
