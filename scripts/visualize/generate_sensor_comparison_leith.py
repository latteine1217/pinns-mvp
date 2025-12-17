#!/usr/bin/env python3
"""
生成 Random vs QR-Pivot 感測器對比圖（基於 DNS v7 標準版本）
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import h5py
from pathlib import Path

# 配置
DNS_FILE = "./data/kolmogorov_dns/dns_re50_t100.h5"
QR_SENSOR_FILE = "./data/kolmogorov_dns/qr_sensors_K100_v7_standard.npz"  # 最新 v7 版本
RANDOM_SENSOR_FILE = "./data/sensors/kolmogorov/sensors_K100_re50_256x256_random_seed42.json"
OUTPUT_FILE = "./thesis/result_figures/sensors/sensor_comparison_re50_K100.png"

def load_dns_vorticity():
    """載入 DNS 渦度場作為背景"""
    print(f"📂 載入 DNS 數據: {DNS_FILE}")
    with h5py.File(DNS_FILE, 'r') as f:
        # DNS 時間序列，取最後一個時刻 (statistically stationary)
        u = f['u'][-1, :, :]  # (256, 256)
        v = f['v'][-1, :, :]
        
        # 重建座標 (2D Kolmogorov flow domain: [0, 2π] × [0, 2π])
        nx, ny = u.shape
        L = 2 * np.pi
        x = np.linspace(0, L, nx, endpoint=False)
        y = np.linspace(0, L, ny, endpoint=False)
        
        # 計算渦度
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dvdx = np.gradient(v, dx, axis=0)
        dudy = np.gradient(u, dy, axis=1)
        vorticity = dvdx - dudy
        
        X, Y = np.meshgrid(x, y, indexing='ij')
        
    print(f"   ✅ DNS 解析度: {u.shape}")
    print(f"   ✅ 域大小: [{0:.2f}, {L:.2f}] × [{0:.2f}, {L:.2f}]")
    return X, Y, vorticity

def load_sensor_data(file_path):
    """載入感測器數據（支持 .json 和 .npz 格式）"""
    path = Path(file_path)
    
    if path.suffix == '.json':
        # JSON 格式：只有索引
        with open(file_path, 'r') as f:
            data = json.load(f)
        return {'indices': np.array(data['indices']), 'has_coords': False}
    
    elif path.suffix == '.npz':
        # NPZ 格式：有座標
        data = np.load(file_path, allow_pickle=True)
        if 'sensor_x' in data and 'sensor_y' in data:
            # v7 格式
            return {
                'x': data['sensor_x'],
                'y': data['sensor_y'],
                'indices': data['sensor_indices'],
                'has_coords': True
            }
        else:
            # 舊格式：只有索引
            return {'indices': data['indices'], 'has_coords': False}
    
    else:
        raise ValueError(f"不支援的檔案格式: {path.suffix}")

def indices_to_coords(indices, nx, ny):
    """將 1D 索引轉換為 2D 座標"""
    i = indices // ny
    j = indices % ny
    return i, j

def main():
    # 1. 載入 DNS 渦度場
    X, Y, vorticity = load_dns_vorticity()
    nx, ny = X.shape
    
    # 2. 載入感測器數據
    print(f"\n📍 載入感測器數據...")
    qr_data = load_sensor_data(QR_SENSOR_FILE)
    random_data = load_sensor_data(RANDOM_SENSOR_FILE)
    
    print(f"   QR-Pivot: {len(qr_data['indices'])} 個感測點 (v7 標準版本)")
    print(f"   Random:   {len(random_data['indices'])} 個感測點")
    
    # 3. 獲取座標
    if qr_data['has_coords']:
        # v7 格式：直接使用座標
        qr_x = qr_data['x']
        qr_y = qr_data['y']
    else:
        # 舊格式：從索引轉換
        qr_i, qr_j = indices_to_coords(qr_data['indices'], nx, ny)
        qr_x = X[qr_i, qr_j]
        qr_y = Y[qr_i, qr_j]
    
    if random_data['has_coords']:
        random_x = random_data['x']
        random_y = random_data['y']
    else:
        random_i, random_j = indices_to_coords(random_data['indices'], nx, ny)
        random_x = X[random_i, random_j]
        random_y = Y[random_i, random_j]
    
    # 4. 繪圖
    print(f"\n🎨 生成對比圖...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 統一色階
    vmin, vmax = np.percentile(vorticity, [1, 99])
    
    # 左圖：Random sensors
    ax = axes[0]
    im = ax.contourf(X, Y, vorticity, levels=50, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.scatter(random_x, random_y, c='yellow', s=40, alpha=0.8, 
               edgecolors='black', linewidths=1.0, label='Random Sensors', zorder=10)
    ax.set_title('(a) Random Sensor Placement (K=100)', fontsize=13, fontweight='bold')
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=10)
    
    # 右圖：QR-Pivot sensors
    ax = axes[1]
    im = ax.contourf(X, Y, vorticity, levels=50, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.scatter(qr_x, qr_y, c='lime', s=40, alpha=0.8,
               edgecolors='black', linewidths=1.0, label='QR-Pivot Sensors', zorder=10)
    ax.set_title('(b) QR-Pivot Sensor Placement (K=100)', fontsize=13, fontweight='bold')
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=10)
    
    # 調整佈局為 colorbar 留出空間
    plt.tight_layout(rect=[0, 0.08, 1, 1])  # 底部留 8% 空間給 colorbar
    
    # Colorbar（放在底部，兩個子圖之間）
    cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.03])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Vorticity $\\omega$ (DNS)', fontsize=11)
    
    # 5. 保存
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 圖表已保存: {output_path}")
    print(f"   大小: {output_path.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    main()
