#!/usr/bin/env python3
"""
診斷散度的空間分佈：檢查是否存在局部高散度區域
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
sys.path.insert(0, '/Users/latteine/Documents/coding/pinns-mvp')

from pinnx.models.fourier_mlp import FourierMLP
from pinnx.physics.kolmogorov_flow_2d import KolmogorovFlow2D

# 載入檢查點
checkpoint = torch.load('checkpoints/kolmogorov_2d_baseline/best_model.pth', 
                       map_location='cpu')

# 載入配置
with open('configs/kolmogorov_2d_baseline.yml', 'r') as f:
    config = yaml.safe_load(f)

# 重建模型
model = FourierMLP(
    input_dim=2,
    output_dim=3,
    hidden_layers=[256] * 6,
    activation='sine',
    fourier_features=config['model']['fourier_features']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 重建物理模組
physics = KolmogorovFlow2D(
    forcing_params=config['physics']['forcing'],
    physics_params={'nu': config['physics']['nu'], 'rho': config['physics']['rho']},
    domain_bounds={'x': (0, 2*np.pi), 'y': (0, 2*np.pi)}
)

# 生成均勻網格
n_grid = 100
x = torch.linspace(0, 2*np.pi, n_grid)
y = torch.linspace(0, 2*np.pi, n_grid)
X, Y = torch.meshgrid(x, y, indexing='ij')
coords = torch.stack([X.flatten(), Y.flatten()], dim=1)
coords.requires_grad_(True)

# 前向傳播
with torch.no_grad():
    predictions = model(coords)

# 重新啟用梯度計算散度
coords_grad = coords.detach().requires_grad_(True)
predictions_grad = model(coords_grad)

u = predictions_grad[:, 0:1]
v = predictions_grad[:, 1:2]

# 計算散度
u_grad = torch.autograd.grad(u.sum(), coords_grad, create_graph=True, retain_graph=True)[0]
v_grad = torch.autograd.grad(v.sum(), coords_grad, create_graph=True, retain_graph=True)[0]

du_dx = u_grad[:, 0:1]
dv_dy = v_grad[:, 1:2]
divergence = (du_dx + dv_dy).detach().numpy().reshape(n_grid, n_grid)

# 統計分析
print("=" * 60)
print("散度空間分佈診斷")
print("=" * 60)
print(f"平均散度（絕對值）: {np.mean(np.abs(divergence)):.6e}")
print(f"最大散度（絕對值）: {np.max(np.abs(divergence)):.6e}")
print(f"散度標準差: {np.std(divergence):.6e}")
print(f"散度均方根: {np.sqrt(np.mean(divergence**2)):.6e}")
print()
print(f"Continuity Loss (MSE): {np.mean(divergence**2):.6e}")
print(f"Mass Conservation Error (L∞): {np.max(np.abs(divergence)):.6e}")
print()
print(f"散度 > 1.0 的點數: {np.sum(np.abs(divergence) > 1.0)} / {n_grid**2}")
print(f"散度 > 0.1 的點數: {np.sum(np.abs(divergence) > 0.1)} / {n_grid**2}")
print(f"散度 > 0.01 的點數: {np.sum(np.abs(divergence) > 0.01)} / {n_grid**2}")

# 視覺化
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. 散度場
im1 = axes[0, 0].contourf(X.numpy(), Y.numpy(), divergence, levels=20, cmap='RdBu_r')
axes[0, 0].set_title('Divergence Field')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('y')
plt.colorbar(im1, ax=axes[0, 0])

# 2. 散度絕對值
im2 = axes[0, 1].contourf(X.numpy(), Y.numpy(), np.abs(divergence), levels=20, cmap='hot')
axes[0, 1].set_title('|Divergence| Field')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('y')
plt.colorbar(im2, ax=axes[0, 1])

# 3. 散度直方圖
axes[1, 0].hist(divergence.flatten(), bins=100, alpha=0.7, edgecolor='black')
axes[1, 0].axvline(0, color='r', linestyle='--', label='Zero')
axes[1, 0].set_xlabel('Divergence')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Divergence Distribution')
axes[1, 0].legend()
axes[1, 0].set_yscale('log')

# 4. 累積分佈
sorted_div = np.sort(np.abs(divergence.flatten()))
cumulative = np.arange(1, len(sorted_div) + 1) / len(sorted_div)
axes[1, 1].plot(sorted_div, cumulative, linewidth=2)
axes[1, 1].axhline(0.95, color='r', linestyle='--', label='95th percentile')
axes[1, 1].axvline(0.01, color='g', linestyle='--', label='Threshold (0.01)')
axes[1, 1].set_xlabel('|Divergence|')
axes[1, 1].set_ylabel('Cumulative Probability')
axes[1, 1].set_title('CDF of |Divergence|')
axes[1, 1].set_xscale('log')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/divergence_spatial_analysis.png', dpi=150)
print("\n視覺化已保存: results/divergence_spatial_analysis.png")

# 檢查感測點附近的散度
sensor_data = np.load('data/kolmogorov_qr_sensors_K50.npz')
sensor_coords = sensor_data['coords']  # [K, 2]

# 找最近鄰散度
from scipy.spatial import cKDTree
tree = cKDTree(coords.detach().numpy())
distances, indices = tree.query(sensor_coords, k=1)
sensor_divergence = divergence.flatten()[indices]

print("\n感測點附近散度統計:")
print(f"  平均散度（絕對值）: {np.mean(np.abs(sensor_divergence)):.6e}")
print(f"  最大散度（絕對值）: {np.max(np.abs(sensor_divergence)):.6e}")
print(f"  與全域比較: {np.mean(np.abs(sensor_divergence)) / np.mean(np.abs(divergence)):.2f}x")
