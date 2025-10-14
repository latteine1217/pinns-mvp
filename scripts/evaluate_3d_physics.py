"""
3D VS-PINN 物理驗證與可視化腳本

功能：
1. 載入訓練完成的模型
2. 計算物理驗證指標（壁面剪應力、能譜、統計量）
3. 生成 3D 流場可視化（速度剖面、切面圖、誤差分布）
4. 輸出評估報告
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import logging
from typing import Dict, Tuple, Optional
import argparse

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 導入專案模組
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import pinnx
from pinnx.models.fourier_mlp import PINNNet
from pinnx.models.wrappers import ManualScalingWrapper
from pinnx.physics.vs_pinn_channel_flow import create_vs_pinn_channel_flow
from pinnx.dataio.channel_flow_loader import ChannelFlowLoader


def load_trained_model(checkpoint_path: Path, config: Dict, device: torch.device):
    """載入訓練完成的模型"""
    logger.info(f"Loading model from {checkpoint_path}")
    
    # 創建模型架構（與訓練時相同）
    from scripts.train import create_model
    model = create_model(config, device, statistics=None)
    
    # 載入權重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    logger.info("Model loaded successfully")
    return model


def compute_wall_shear_stress(model, physics, x_range, z_range, n_points=64, device='cpu'):
    """計算壁面剪應力分布"""
    logger.info("Computing wall shear stress...")
    
    # 生成壁面網格
    x = torch.linspace(x_range[0], x_range[1], n_points, device=device)
    z = torch.linspace(z_range[0], z_range[1], n_points, device=device)
    X, Z = torch.meshgrid(x, z, indexing='ij')
    
    # 下壁面 (y = -1)
    y_wall = torch.full_like(X, -1.0)
    wall_points = torch.stack([X.flatten(), y_wall.flatten(), Z.flatten()], dim=1)
    wall_points.requires_grad_(True)
    
    # 計算預測值和梯度（需要啟用梯度）
    u_pred = model(wall_points)
    
    # 計算 du/dy
    u_component = u_pred[:, 0]  # 流向速度
    du_dy = torch.autograd.grad(u_component.sum(), wall_points, create_graph=False)[0][:, 1]
    
    # 壁面剪應力 τ_w = μ * du/dy
    nu = physics.nu.item()
    tau_w = nu * du_dy
    
    tau_w_grid = tau_w.reshape(n_points, n_points).cpu().numpy()
    
    # 統計量
    tau_w_mean = np.mean(tau_w_grid)
    tau_w_std = np.std(tau_w_grid)
    
    logger.info(f"Wall shear stress: mean={tau_w_mean:.6f}, std={tau_w_std:.6f}")
    
    return {
        'tau_w': tau_w_grid,
        'x': x.cpu().numpy(),
        'z': z.cpu().numpy(),
        'mean': tau_w_mean,
        'std': tau_w_std
    }


def compute_velocity_profile(model, x_pos, z_pos, y_range, n_points=128, device='cpu'):
    """計算指定位置的速度剖面"""
    logger.info(f"Computing velocity profile at x={x_pos}, z={z_pos}")
    
    y = torch.linspace(y_range[0], y_range[1], n_points, device=device)
    x = torch.full_like(y, x_pos)
    z = torch.full_like(y, z_pos)
    
    points = torch.stack([x, y, z], dim=1)
    
    with torch.no_grad():
        u_pred = model(points)
    
    profile = {
        'y': y.cpu().numpy(),
        'u': u_pred[:, 0].cpu().numpy(),
        'v': u_pred[:, 1].cpu().numpy(),
        'w': u_pred[:, 2].cpu().numpy(),
        'p': u_pred[:, 3].cpu().numpy()
    }
    
    return profile


def compute_energy_spectrum_1d(velocity_field, axis=0):
    """計算 1D 能譜"""
    # FFT
    fft_result = np.fft.rfft(velocity_field, axis=axis)
    energy = np.abs(fft_result) ** 2
    
    # 平均其他維度
    if axis == 0:
        energy = np.mean(energy, axis=(1, 2))
    elif axis == 1:
        energy = np.mean(energy, axis=(0, 2))
    else:
        energy = np.mean(energy, axis=(0, 1))
    
    return energy


def evaluate_full_field(model, grid_resolution, domain, device='cpu'):
    """評估完整流場"""
    logger.info(f"Evaluating full field at resolution {grid_resolution}")
    
    nx, ny, nz = grid_resolution
    x = torch.linspace(domain['x'][0], domain['x'][1], nx, device=device)
    y = torch.linspace(domain['y'][0], domain['y'][1], ny, device=device)
    z = torch.linspace(domain['z'][0], domain['z'][1], nz, device=device)
    
    # 分批評估以避免記憶體溢出
    batch_size = 10000
    all_points = []
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                all_points.append([x[i].item(), y[j].item(), z[k].item()])
    
    all_points = torch.tensor(all_points, device=device, dtype=torch.float32)
    n_total = all_points.shape[0]
    
    u_all = []
    v_all = []
    w_all = []
    p_all = []
    
    with torch.no_grad():
        for i in range(0, n_total, batch_size):
            batch = all_points[i:i+batch_size]
            pred = model(batch)
            u_all.append(pred[:, 0].cpu().numpy())
            v_all.append(pred[:, 1].cpu().numpy())
            w_all.append(pred[:, 2].cpu().numpy())
            p_all.append(pred[:, 3].cpu().numpy())
    
    # 重塑為 3D 網格
    u_field = np.concatenate(u_all).reshape(nx, ny, nz)
    v_field = np.concatenate(v_all).reshape(nx, ny, nz)
    w_field = np.concatenate(w_all).reshape(nx, ny, nz)
    p_field = np.concatenate(p_all).reshape(nx, ny, nz)
    
    return {
        'u': u_field,
        'v': v_field,
        'w': w_field,
        'p': p_field,
        'x': x.cpu().numpy(),
        'y': y.cpu().numpy(),
        'z': z.cpu().numpy()
    }


def plot_wall_shear_stress(wall_data, save_path):
    """繪製壁面剪應力分布"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.contourf(wall_data['x'], wall_data['z'], wall_data['tau_w'].T, levels=20, cmap='viridis')
    ax.set_xlabel('Streamwise (x)')
    ax.set_ylabel('Spanwise (z)')
    ax.set_title(f"Wall Shear Stress Distribution (mean={wall_data['mean']:.4f})")
    plt.colorbar(im, ax=ax, label=r'$\tau_w$')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    logger.info(f"Saved wall shear stress plot to {save_path}")
    plt.close()


def plot_velocity_profiles(profile, save_path):
    """繪製速度剖面"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 流向速度
    axes[0].plot(profile['u'], profile['y'], 'b-', linewidth=2)
    axes[0].set_xlabel('u')
    axes[0].set_ylabel('y')
    axes[0].set_title('Streamwise Velocity Profile')
    axes[0].grid(True, alpha=0.3)
    
    # 法向速度
    axes[1].plot(profile['v'], profile['y'], 'r-', linewidth=2)
    axes[1].set_xlabel('v')
    axes[1].set_ylabel('y')
    axes[1].set_title('Wall-Normal Velocity Profile')
    axes[1].grid(True, alpha=0.3)
    
    # 展向速度
    axes[2].plot(profile['w'], profile['y'], 'g-', linewidth=2)
    axes[2].set_xlabel('w')
    axes[2].set_ylabel('y')
    axes[2].set_title('Spanwise Velocity Profile')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    logger.info(f"Saved velocity profiles to {save_path}")
    plt.close()


def plot_field_slices(field_data, save_dir):
    """繪製流場切面"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # XY 平面切面（中間 z）
    z_mid = len(field_data['z']) // 2
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # u 分量
    im0 = axes[0, 0].contourf(field_data['x'], field_data['y'], field_data['u'][:, :, z_mid].T, levels=20, cmap='RdBu_r')
    axes[0, 0].set_title('Streamwise Velocity (u)')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # v 分量
    im1 = axes[0, 1].contourf(field_data['x'], field_data['y'], field_data['v'][:, :, z_mid].T, levels=20, cmap='RdBu_r')
    axes[0, 1].set_title('Wall-Normal Velocity (v)')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # w 分量
    im2 = axes[1, 0].contourf(field_data['x'], field_data['y'], field_data['w'][:, :, z_mid].T, levels=20, cmap='RdBu_r')
    axes[1, 0].set_title('Spanwise Velocity (w)')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # p 分量
    im3 = axes[1, 1].contourf(field_data['x'], field_data['y'], field_data['p'][:, :, z_mid].T, levels=20, cmap='viridis')
    axes[1, 1].set_title('Pressure (p)')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(save_dir / 'field_slices_xy.png', dpi=150)
    logger.info(f"Saved field slices to {save_dir / 'field_slices_xy.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate 3D VS-PINN physics')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default='configs/vs_pinn_3d_full_training.yml', help='Config file')
    parser.add_argument('--output_dir', type=str, default='results/3d_evaluation', help='Output directory')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/mps/auto)')
    
    args = parser.parse_args()
    
    # 設置輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 載入配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 設置設備
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # 載入模型
    checkpoint_path = Path(args.checkpoint)
    model = load_trained_model(checkpoint_path, config, device)
    
    # 創建物理模組 - 從配置中提取正確的參數
    physics_config = config['physics']
    physics_params = {
        'nu': physics_config.get('nu', 5e-5),
        'dP_dx': physics_config.get('channel_flow', {}).get('pressure_gradient', 0.0025),
        'rho': physics_config.get('rho', 1.0),
        'N_x': physics_config.get('vs_pinn', {}).get('scaling_factors', {}).get('N_x', 2.0),
        'N_y': physics_config.get('vs_pinn', {}).get('scaling_factors', {}).get('N_y', 12.0),
        'N_z': physics_config.get('vs_pinn', {}).get('scaling_factors', {}).get('N_z', 2.0),
        'enable_rans': physics_config.get('vs_pinn', {}).get('enable_rans', False),
    }
    physics = create_vs_pinn_channel_flow(**physics_params)
    
    # 域範圍
    domain = config['physics']['domain']
    
    # 轉換 device 為字串格式
    device_str = str(device)
    
    # ========== 1. 壁面剪應力 ==========
    wall_data = compute_wall_shear_stress(
        model, physics,
        x_range=domain['x_range'],
        z_range=domain['z_range'],
        n_points=64,
        device=device_str
    )
    plot_wall_shear_stress(wall_data, output_dir / 'wall_shear_stress.png')
    
    # ========== 2. 速度剖面 ==========
    profile = compute_velocity_profile(
        model,
        x_pos=(domain['x_range'][0] + domain['x_range'][1]) / 2,
        z_pos=(domain['z_range'][0] + domain['z_range'][1]) / 2,
        y_range=domain['y_range'],
        n_points=128,
        device=device_str
    )
    plot_velocity_profiles(profile, output_dir / 'velocity_profiles.png')
    
    # ========== 3. 完整流場評估 ==========
    grid_resolution = config['evaluation']['grid_resolution']
    # 轉換域格式以匹配函數期望
    domain_for_eval = {
        'x': domain['x_range'],
        'y': domain['y_range'],
        'z': domain['z_range']
    }
    field_data = evaluate_full_field(model, grid_resolution, domain_for_eval, device_str)
    
    # 保存流場數據
    np.savez(
        output_dir / 'predicted_field.npz',
        u=field_data['u'],
        v=field_data['v'],
        w=field_data['w'],
        p=field_data['p'],
        x=field_data['x'],
        y=field_data['y'],
        z=field_data['z']
    )
    logger.info(f"Saved predicted field to {output_dir / 'predicted_field.npz'}")
    
    # 繪製切面
    plot_field_slices(field_data, output_dir)
    
    # ========== 4. 生成評估報告 ==========
    report = f"""
# 3D VS-PINN 物理驗證報告

## 訓練配置
- Checkpoint: {args.checkpoint}
- Config: {args.config}
- Device: {device}

## 壁面剪應力
- Mean: {wall_data['mean']:.6f}
- Std: {wall_data['std']:.6f}

## 速度剖面統計
- u (streamwise): min={np.min(profile['u']):.4f}, max={np.max(profile['u']):.4f}, mean={np.mean(profile['u']):.4f}
- v (wall-normal): min={np.min(profile['v']):.4f}, max={np.max(profile['v']):.4f}, mean={np.mean(profile['v']):.4f}
- w (spanwise): min={np.min(profile['w']):.4f}, max={np.max(profile['w']):.4f}, mean={np.mean(profile['w']):.4f}

## 流場統計
- u: min={np.min(field_data['u']):.4f}, max={np.max(field_data['u']):.4f}
- v: min={np.min(field_data['v']):.4f}, max={np.max(field_data['v']):.4f}
- w: min={np.min(field_data['w']):.4f}, max={np.max(field_data['w']):.4f}
- p: min={np.min(field_data['p']):.4f}, max={np.max(field_data['p']):.4f}

## 輸出文件
- Wall shear stress: wall_shear_stress.png
- Velocity profiles: velocity_profiles.png
- Field slices: field_slices_xy.png
- Predicted field: predicted_field.npz
"""
    
    with open(output_dir / 'evaluation_report.md', 'w') as f:
        f.write(report)
    
    logger.info(f"\n{report}")
    logger.info("Evaluation completed successfully!")


if __name__ == '__main__':
    main()
