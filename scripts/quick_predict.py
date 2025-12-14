#!/usr/bin/env python3
"""
快速預測腳本：直接從 checkpoint 生成預測結果
"""
import sys
sys.path.insert(0, '/Users/latteine/Documents/coding/pinns-mvp')

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

# 載入 checkpoint
checkpoint_path = "checkpoints/kolmogorov_re50_kf4_K100_rans_prior/epoch_6500.pth"
print(f"📥 載入 checkpoint: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location='cpu')
config = checkpoint['config']

# 從配置重建模型
from pinnx.models.fourier_mlp import PINNNet

model_cfg = config['model']

# 從 checkpoint 檢測實際輸入維度
state_dict = checkpoint['model_state_dict']
if 'fourier.B' in state_dict:
    actual_in_dim = state_dict['fourier.B'].shape[0]
    print(f"🔍 檢測到實際輸入維度: {actual_in_dim}")
else:
    actual_in_dim = model_cfg.get('in_dim', 2)

in_dim = actual_in_dim
out_dim = model_cfg.get('out_dim', 3)
width = model_cfg.get('width', 256)
depth = model_cfg.get('depth', 8)
fourier_m = model_cfg.get('fourier_m', 12)
fourier_sigma = model_cfg.get('fourier_sigma', 4.0)

print(f"🏗️  創建模型: in_dim={in_dim}, {depth} layers × {width} neurons, Fourier m={fourier_m}")

model = PINNNet(
    in_dim=in_dim,
    out_dim=out_dim,
    width=width,
    depth=depth,
    fourier_m=fourier_m,
    fourier_sigma=fourier_sigma,
    block_type='resnet'
)

# 載入權重
model.load_state_dict(state_dict, strict=False)
model.eval()

print(f"✅ 模型載入成功 (Epoch {checkpoint['epoch']})")

# 生成預測網格
N = 128  # 解析度
x = np.linspace(0, 2*np.pi, N)
y = np.linspace(0, 2*np.pi, N)
X, Y = np.meshgrid(x, y)

# 準備輸入 (根據實際維度)
if in_dim == 2:
    xy = np.stack([X.ravel(), Y.ravel()], axis=1)
elif in_dim == 3:
    # 假設第三維是時間，使用 t=0
    t = np.zeros_like(X.ravel())
    xy = np.stack([X.ravel(), Y.ravel(), t], axis=1)
else:
    raise ValueError(f"Unsupported in_dim: {in_dim}")

xy_tensor = torch.tensor(xy, dtype=torch.float32)

print(f"🔮 生成預測 (網格: {N}×{N}={N*N} 點)")

# 批次預測
batch_size = 4096
predictions = []

with torch.no_grad():
    for i in range(0, len(xy_tensor), batch_size):
        batch = xy_tensor[i:i+batch_size]
        pred = model(batch)
        predictions.append(pred.cpu().numpy())

predictions = np.concatenate(predictions, axis=0)

# 重塑為場
u_pred = predictions[:, 0].reshape(N, N)
v_pred = predictions[:, 1].reshape(N, N)
p_pred = predictions[:, 2].reshape(N, N)

print(f"✅ 預測完成")

# 統計資訊
print("\n" + "="*60)
print("📊 預測場統計")
print("="*60)
print(f"u: min={u_pred.min():.4f}, max={u_pred.max():.4f}, mean={u_pred.mean():.4f}, std={u_pred.std():.4f}")
print(f"v: min={v_pred.min():.4f}, max={v_pred.max():.4f}, mean={v_pred.mean():.4f}, std={v_pred.std():.4f}")
print(f"p: min={p_pred.min():.4f}, max={p_pred.max():.4f}, mean={p_pred.mean():.4f}, std={p_pred.std():.4f}")

# 計算散度 (質量守恆檢查)
dx = x[1] - x[0]
dy = y[1] - y[0]
du_dx = np.gradient(u_pred, dx, axis=1)
dv_dy = np.gradient(v_pred, dy, axis=0)
div = du_dx + dv_dy

print(f"\n散度 (∇·u): mean={np.abs(div).mean():.6f}, max={np.abs(div).max():.6f}")
print("="*60)

# 保存結果
output_dir = Path("results/quick_prediction")
output_dir.mkdir(parents=True, exist_ok=True)

np.savez(
    output_dir / "predictions_epoch6500.npz",
    x=x, y=y,
    u=u_pred, v=v_pred, p=p_pred,
    divergence=div,
    X=X, Y=Y
)
print(f"\n💾 數據已保存: {output_dir}/predictions_epoch6500.npz")

# 生成可視化
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# u 速度
im1 = axes[0, 0].contourf(X, Y, u_pred, levels=50, cmap='RdBu_r')
axes[0, 0].set_title('u velocity', fontsize=14)
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('y')
plt.colorbar(im1, ax=axes[0, 0])

# v 速度
im2 = axes[0, 1].contourf(X, Y, v_pred, levels=50, cmap='RdBu_r')
axes[0, 1].set_title('v velocity', fontsize=14)
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('y')
plt.colorbar(im2, ax=axes[0, 1])

# 壓力
im3 = axes[1, 0].contourf(X, Y, p_pred, levels=50, cmap='viridis')
axes[1, 0].set_title('Pressure', fontsize=14)
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('y')
plt.colorbar(im3, ax=axes[1, 0])

# 散度
im4 = axes[1, 1].contourf(X, Y, np.abs(div), levels=50, cmap='Reds')
axes[1, 1].set_title('|Divergence| (mass conservation error)', fontsize=14)
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('y')
plt.colorbar(im4, ax=axes[1, 1])

plt.tight_layout()
plt.savefig(output_dir / "predictions_epoch6500.png", dpi=150, bbox_inches='tight')
print(f"📊 圖表已保存: {output_dir}/predictions_epoch6500.png")

# 生成速度向量場
fig2, ax = plt.subplots(figsize=(10, 10))
skip = 8
Q = ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
              u_pred[::skip, ::skip], v_pred[::skip, ::skip],
              np.sqrt(u_pred[::skip, ::skip]**2 + v_pred[::skip, ::skip]**2),
              cmap='jet', scale=20)
ax.set_title('Velocity Field (Epoch 6500)', fontsize=16)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
plt.colorbar(Q, ax=ax, label='Velocity Magnitude')
plt.savefig(output_dir / "velocity_field_epoch6500.png", dpi=150, bbox_inches='tight')
print(f"🌊 向量場已保存: {output_dir}/velocity_field_epoch6500.png")

print("\n✅ 全部完成！")
