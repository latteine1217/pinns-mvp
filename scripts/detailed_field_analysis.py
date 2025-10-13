#!/usr/bin/env python3
"""
詳細場分析腳本 - Phase 3 深度診斷
分析壁面速度剖面、數據點分布、邊界條件失效模式
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import sys

# 添加專案路徑
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pinnx
from pinnx.models.fourier_mlp import PINNNet

# ============================================================
# 配置參數
# ============================================================
CHECKPOINT_PATH = "checkpoints/pinnx_channel_flow_re1000_fix6_k50_phase3_quick_latest.pth"
CONFIG_PATH = "configs/channel_flow_re1000_fix6_k50_phase3_quick.yml"
SENSOR_DATA_PATH = "data/jhtdb/channel_flow_re1000/sensors_K50_qr_pivot.npz"
OUTPUT_DIR = Path("evaluation_results_phase3_quick")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# 載入模型
# ============================================================
print(f"載入檢查點: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')

# 從 checkpoint 中提取模型配置
model_cfg = checkpoint['config']['model']

# 創建模型
model = PINNNet(
    in_dim=model_cfg['in_dim'],
    out_dim=model_cfg['out_dim'],
    width=model_cfg['width'],
    depth=model_cfg['depth'],
    activation=model_cfg['activation'],
    use_fourier=True,
    fourier_m=model_cfg['fourier_m'],
    fourier_sigma=model_cfg['fourier_sigma']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ 模型載入成功 (Epoch {checkpoint['epoch']})")

# 載入訓練數據（K=50 個點）
sensor_data = np.load(SENSOR_DATA_PATH, allow_pickle=True)
sensor_points = sensor_data['sensor_points']  # (K, 2)
sensor_vals_dict = sensor_data['sensor_data'].item()  # 字典格式
sensor_values_u = sensor_vals_dict['u']  # (K,)
sensor_values_v = sensor_vals_dict['v']  # (K,)

print(f"\n訓練數據點數量: K = {sensor_points.shape[0]}")
print(f"數據點範圍: X ∈ [{sensor_points[:, 0].min():.3f}, {sensor_points[:, 0].max():.3f}]")
print(f"             Y ∈ [{sensor_points[:, 1].min():.3f}, {sensor_points[:, 1].max():.3f}]")

# ============================================================
# 1. 壁面附近速度剖面分析
# ============================================================
print("\n" + "="*60)
print("🔬 1. 壁面附近速度剖面分析")
print("="*60)

# 在不同流向位置 (x) 取垂直剖面
x_positions = np.linspace(0, 2*np.pi, 5)  # 5 個流向位置
y_profile = np.linspace(-1, 1, 200)  # 200 個法向點（高解析度）

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Wall Velocity Profiles Analysis', fontsize=16, fontweight='bold')

wall_errors = {'lower': [], 'upper': []}

for idx, x_pos in enumerate(x_positions):
    # 創建剖面網格
    xy_profile = np.stack([
        np.full_like(y_profile, x_pos),
        y_profile
    ], axis=1)
    
    # 模型預測
    with torch.no_grad():
        xy_tensor = torch.FloatTensor(xy_profile)
        pred = model(xy_tensor).numpy()  # (200, 3): [u, v, p]
    
    u_profile = pred[:, 0]
    v_profile = pred[:, 1]
    
    # 繪製 U 速度剖面
    ax = axes[0, idx] if idx < 3 else axes[1, idx-3]
    ax.plot(u_profile, y_profile, 'b-', linewidth=2, label='U(y) - PINN')
    ax.axhline(y=-1, color='r', linestyle='--', alpha=0.5, label='Lower wall')
    ax.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Upper wall')
    ax.axvline(x=0, color='k', linestyle=':', alpha=0.3, label='U=0 (Theory)')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('U velocity')
    ax.set_ylabel('y')
    ax.set_title(f'x = {x_pos:.2f}')
    ax.legend(fontsize=8)
    
    # 計算壁面誤差
    lower_wall_u = u_profile[0]   # y = -1
    upper_wall_u = u_profile[-1]  # y = +1
    wall_errors['lower'].append(abs(lower_wall_u))
    wall_errors['upper'].append(abs(upper_wall_u))
    
    # 標註壁面速度值
    ax.text(0.05, 0.95, f'Lower: U={lower_wall_u:.3f}', 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(0.05, 0.05, f'Upper: U={upper_wall_u:.3f}', 
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 移除多餘的子圖
if len(x_positions) == 5:
    fig.delaxes(axes[1, 2])

plt.tight_layout()
profile_path = OUTPUT_DIR / "wall_velocity_profiles.png"
plt.savefig(profile_path, dpi=300, bbox_inches='tight')
print(f"✅ Wall velocity profiles saved: {profile_path}")

# 統計壁面誤差
print(f"\nWall no-slip error statistics:")
print(f"  Lower wall (y=-1): Mean error = {np.mean(wall_errors['lower']):.4f} ± {np.std(wall_errors['lower']):.4f}")
print(f"  Upper wall (y=+1): Mean error = {np.mean(wall_errors['upper']):.4f} ± {np.std(wall_errors['upper']):.4f}")

# ============================================================
# 2. K=50 數據點空間分布分析
# ============================================================
print("\n" + "="*60)
print("🔬 2. Training Data Spatial Distribution")
print("="*60)

# 分析數據點在 Y 方向的分布
y_coords = sensor_points[:, 1]
y_bins = np.linspace(-1, 1, 21)  # 20 個區間
y_hist, _ = np.histogram(y_coords, bins=y_bins)

# 檢查邊界附近的覆蓋度
wall_threshold = 0.1  # 距離壁面 0.1 以內
near_lower_wall = np.sum(np.abs(y_coords - (-1)) < wall_threshold)
near_upper_wall = np.sum(np.abs(y_coords - 1) < wall_threshold)

print(f"\nY-direction distribution:")
print(f"  Total points: {len(y_coords)}")
print(f"  Y range: [{y_coords.min():.3f}, {y_coords.max():.3f}]")
print(f"  Near lower wall (y < -0.9): {near_lower_wall} points ({near_lower_wall/len(y_coords)*100:.1f}%)")
print(f"  Near upper wall (y > 0.9): {near_upper_wall} points ({near_upper_wall/len(y_coords)*100:.1f}%)")
print(f"  Core region (|y| < 0.9): {len(y_coords) - near_lower_wall - near_upper_wall} points")

# 可視化數據點分布
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('K=50 Training Data Distribution', fontsize=16, fontweight='bold')

# 2.1 散點圖（空間位置）
ax = axes[0]
scatter = ax.scatter(sensor_points[:, 0], sensor_points[:, 1], 
                     c=sensor_values_u, cmap='RdBu_r', 
                     s=100, edgecolors='k', linewidths=0.5)
ax.axhline(y=-1, color='r', linestyle='--', linewidth=2, label='Wall')
ax.axhline(y=1, color='r', linestyle='--', linewidth=2)
ax.axhline(y=-0.9, color='orange', linestyle=':', alpha=0.5, label='Boundary layer')
ax.axhline(y=0.9, color='orange', linestyle=':', alpha=0.5)
ax.set_xlabel('X (streamwise)', fontsize=12)
ax.set_ylabel('Y (wall-normal)', fontsize=12)
ax.set_title('Spatial Distribution (Color=U)', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='U velocity')

# 2.2 Y 方向直方圖
ax = axes[1]
ax.barh(y_bins[:-1], y_hist, height=np.diff(y_bins), 
        align='edge', color='steelblue', edgecolor='k', alpha=0.7)
ax.axhline(y=-1, color='r', linestyle='--', linewidth=2, label='Wall')
ax.axhline(y=1, color='r', linestyle='--', linewidth=2)
ax.axhline(y=-0.9, color='orange', linestyle=':', alpha=0.5)
ax.axhline(y=0.9, color='orange', linestyle=':', alpha=0.5)
ax.set_ylabel('Y (wall-normal)', fontsize=12)
ax.set_xlabel('Number of points', fontsize=12)
ax.set_title('Y-direction Density', fontsize=12)
ax.grid(True, alpha=0.3, axis='x')

# 2.3 數據值分布
ax = axes[2]
ax.hist(sensor_values_u, bins=20, color='steelblue', edgecolor='k', alpha=0.7)
ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='U=0 (Wall theory)')
ax.set_xlabel('U velocity', fontsize=12)
ax.set_ylabel('Number of points', fontsize=12)
ax.set_title('Training Data U Distribution', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
distribution_path = OUTPUT_DIR / "sensor_points_distribution.png"
plt.savefig(distribution_path, dpi=300, bbox_inches='tight')
print(f"✅ Sensor distribution plot saved: {distribution_path}")

# ============================================================
# 3. 邊界條件失效模式分析
# ============================================================
print("\n" + "="*60)
print("🔬 3. Boundary Condition Failure Analysis")
print("="*60)

# 3.1 壁面邊界 2D 熱圖
nx, ny = 100, 50
x_grid = np.linspace(0, 2*np.pi, nx)
y_grid = np.linspace(-1, 1, ny)
X, Y = np.meshgrid(x_grid, y_grid)
xy_full = np.stack([X.ravel(), Y.ravel()], axis=1)

with torch.no_grad():
    xy_tensor = torch.FloatTensor(xy_full)
    pred_full = model(xy_tensor).numpy()

U_field = pred_full[:, 0].reshape(ny, nx)
V_field = pred_full[:, 1].reshape(ny, nx)
P_field = pred_full[:, 2].reshape(ny, nx)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Boundary Condition Failure Analysis', fontsize=16, fontweight='bold')

# 3.1.1 U 速度場
ax = axes[0, 0]
im = ax.contourf(X, Y, U_field, levels=20, cmap='RdBu_r')
ax.scatter(sensor_points[:, 0], sensor_points[:, 1], 
           c='lime', s=30, marker='o', edgecolors='k', linewidths=0.5, 
           label=f'Training (K={len(sensor_points)})', zorder=5)
ax.axhline(y=-1, color='yellow', linestyle='--', linewidth=2, label='Wall')
ax.axhline(y=1, color='yellow', linestyle='--', linewidth=2)
ax.set_xlabel('X (streamwise)')
ax.set_ylabel('Y (wall-normal)')
ax.set_title('U Velocity (Should be 0 at walls)')
ax.legend()
plt.colorbar(im, ax=ax, label='U velocity')

# 3.1.2 壁面 U 誤差分布
ax = axes[0, 1]
lower_wall_u = U_field[0, :]   # y = -1
upper_wall_u = U_field[-1, :]  # y = +1
ax.plot(x_grid, lower_wall_u, 'b-', linewidth=2, label='Lower wall (y=-1)')
ax.plot(x_grid, upper_wall_u, 'r-', linewidth=2, label='Upper wall (y=+1)')
ax.axhline(y=0, color='k', linestyle='--', linewidth=1, label='Theory: U=0')
ax.fill_between(x_grid, 0, lower_wall_u, alpha=0.3, color='blue')
ax.fill_between(x_grid, 0, upper_wall_u, alpha=0.3, color='red')
ax.set_xlabel('X (streamwise)')
ax.set_ylabel('Wall U velocity')
ax.set_title('No-Slip Error Spatial Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# 3.1.3 週期性邊界誤差
ax = axes[1, 0]
left_boundary_u = U_field[:, 0]   # x = 0
right_boundary_u = U_field[:, -1]  # x = 2π
periodicity_error = np.abs(left_boundary_u - right_boundary_u)
ax.plot(y_grid, left_boundary_u, 'b-', linewidth=2, label='Left (x=0)')
ax.plot(y_grid, right_boundary_u, 'r--', linewidth=2, label='Right (x=2π)')
ax.set_xlabel('Y (wall-normal)')
ax.set_ylabel('U velocity')
ax.set_title('Periodicity Boundary Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# 3.1.4 週期性誤差剖面
ax = axes[1, 1]
ax.plot(y_grid, periodicity_error, 'g-', linewidth=2)
ax.axhline(y=0.02, color='orange', linestyle='--', linewidth=1, label='Target (0.02)')
ax.fill_between(y_grid, 0, periodicity_error, alpha=0.3, color='green')
ax.set_xlabel('Y (wall-normal)')
ax.set_ylabel('|U_left - U_right|')
ax.set_title(f'Periodicity Error Profile (Mean={periodicity_error.mean():.4f})')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
boundary_path = OUTPUT_DIR / "boundary_condition_analysis.png"
plt.savefig(boundary_path, dpi=300, bbox_inches='tight')
print(f"✅ Boundary condition analysis saved: {boundary_path}")

# ============================================================
# 4. 總結報告
# ============================================================
print("\n" + "="*60)
print("📋 4. Diagnostic Summary")
print("="*60)

print(f"""
【Key Findings】

1️⃣ Wall Velocity Profiles:
   - Lower wall mean error: {np.mean(wall_errors['lower']):.4f} (Theory=0)
   - Upper wall mean error: {np.mean(wall_errors['upper']):.4f} (Theory=0)
   - Conclusion: Wall condition completely violated ❌

2️⃣ Sensor Distribution:
   - Near lower wall (<-0.9): {near_lower_wall} points ({near_lower_wall/len(y_coords)*100:.1f}%)
   - Near upper wall (>0.9): {near_upper_wall} points ({near_upper_wall/len(y_coords)*100:.1f}%)
   - Conclusion: {'Insufficient boundary coverage ⚠️' if (near_lower_wall + near_upper_wall) < 5 else 'Adequate boundary coverage ✅'}

3️⃣ Periodicity Boundary:
   - Mean error: {periodicity_error.mean():.4f}
   - Max error: {periodicity_error.max():.4f}
   - Target threshold: 0.02
   - Conclusion: {'Needs improvement ❌' if periodicity_error.mean() > 0.1 else 'Partially satisfied 🟡'}

【Root Cause Analysis】
""")

# 根因判斷邏輯
if (near_lower_wall + near_upper_wall) < 5:
    print("⚠️  Cause 1: QR-pivot insufficient boundary sampling")
    print("   → Suggestion: Enforce wall-stratified sampling")
else:
    print("✅ Data coverage is sufficient, issue is in weight configuration")

if np.mean(wall_errors['lower']) > 3.0:
    print("⚠️  Cause 2: boundary_weight too low relative to data_weight")
    print("   → Suggestion: Increase boundary_weight from 15.0 to 50-100")

if periodicity_error.mean() > 0.1:
    print("⚠️  Cause 3: periodicity_weight insufficient")
    print("   → Suggestion: Increase periodicity_weight from 5.0 to 20-30")

print("\n" + "="*60)
print("✅ Detailed Field Analysis Complete!")
print("="*60)
print(f"\nGenerated plots:")
print(f"  1. {profile_path}")
print(f"  2. {distribution_path}")
print(f"  3. {boundary_path}")
