#!/usr/bin/env python
"""
快速評估與視覺化腳本
用於評估 Kolmogorov 流場訓練結果並生成圖表
"""

import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import sys

# 加入專案路徑
sys.path.insert(0, str(Path(__file__).parent))

from pinnx.models.fourier_mlp import PINNNet
from pinnx.utils.denormalization import denormalize_output  # 統一反標準化


def load_checkpoint(ckpt_path):
    """載入檢查點"""
    print(f"📥 載入檢查點: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    epoch = checkpoint.get('epoch', 'unknown')
    config = checkpoint.get('config', {})
    print(f"   Epoch: {epoch}")

    return checkpoint, config


def create_model_from_checkpoint(checkpoint):
    """從 checkpoint 創建完全匹配的模型"""
    from pinnx.models.fourier_mlp import PINNNet

    config = checkpoint['config']
    model_cfg = config.get('model', {})
    ff_cfg = model_cfg.get('fourier_features')
    if not isinstance(ff_cfg, dict) or 'type' not in ff_cfg:
        raise KeyError("Missing required config key: model.fourier_features.type")

    # 從 checkpoint state_dict 推斷正確的參數
    state_dict = checkpoint['model_state_dict']

    # 檢查 Fourier B 矩陣尺寸 --> fourier_m
    if 'fourier.B' in state_dict:
        fourier_m = int(state_dict['fourier.B'].shape[1])
    else:
        fourier_m = 0

    # 檢查輸出層尺寸 --> out_dim
    if 'output_layer.bias' in state_dict:
        out_dim = state_dict['output_layer.bias'].shape[0]
    else:
        out_dim = 3

    # 檢查輸入維度
    if 'fourier.B' in state_dict:
        in_dim = state_dict['fourier.B'].shape[0]
    else:
        in_dim = 3

    # 檢查是否有 ResNet 結構
    has_resnet = any('fc1' in key for key in state_dict.keys())

    print(f"📐 從 checkpoint 推斷模型參數:")
    print(f"   in_dim={in_dim}, out_dim={out_dim}, fourier_m={fourier_m}")
    print(f"   ResNet blocks: {has_resnet}")

    use_fourier = ('fourier.B' in state_dict)
    fourier_sigma = float(ff_cfg.get('fourier_sigma', 0.0)) if use_fourier else 0.0

    # 創建模型
    model = PINNNet(
        in_dim=in_dim,
        out_dim=out_dim,
        width=model_cfg.get('width', 256),
        depth=model_cfg.get('depth', 6),
        activation=model_cfg.get('activation', 'swish'),
        use_fourier=use_fourier,
        fourier_m=fourier_m,
        fourier_sigma=fourier_sigma,
        use_residual=has_resnet
    )

    # 計算參數數量
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型已創建: {n_params:,} 參數")

    return model


def load_dns_data(h5_path, time_range=[15.0, 35.0]):
    """載入 DNS 數據"""
    print(f"📂 載入 DNS 數據: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        u_all = f['u'][:]  # (T, H, W)
        v_all = f['v'][:]
        p_all = f['p'][:]
        X = f['X'][:]  # (H, W)
        Y = f['Y'][:]
        t_all = f['t'][:]

    # 時間範圍過濾
    t_mask = (t_all >= time_range[0]) & (t_all <= time_range[1])
    u = u_all[t_mask]
    v = v_all[t_mask]
    p = p_all[t_mask]
    t = t_all[t_mask]

    print(f"   時間步數: {len(t)}, 網格: {X.shape}")
    print(f"   時間範圍: [{t.min():.2f}, {t.max():.2f}]")

    return {
        'u': u, 'v': v, 'p': p,
        'X': X, 'Y': Y, 't': t
    }


def evaluate_model(model, dns_data, config, checkpoint_path, device='cpu', num_samples=5):
    """評估模型
    
    Args:
        model: 訓練好的模型
        dns_data: DNS 參考數據
        config: 配置字典（用於反標準化）
        checkpoint_path: checkpoint 路徑（用於載入標準化統計量）
        device: 計算設備
        num_samples: 評估的時間點數量
    """
    model.eval()
    model.to(device)

    X = dns_data['X']
    Y = dns_data['Y']
    t = dns_data['t']

    # 選擇幾個時間點評估
    t_indices = np.linspace(0, len(t)-1, num_samples, dtype=int)

    results = {
        'predictions': [],
        'ground_truth': [],
        'errors': [],
        'time_points': []
    }

    print(f"\n🔍 評估 {num_samples} 個時間點...")

    for i, t_idx in enumerate(t_indices):
        t_val = t[t_idx]

        # 準備輸入
        H, W = X.shape
        x_flat = X.flatten()
        y_flat = Y.flatten()
        t_flat = np.full_like(x_flat, t_val)

        # 歸一化（簡單範圍歸一化）
        x_norm = x_flat / (2 * np.pi)
        y_norm = y_flat / (2 * np.pi)
        t_norm = (t_flat - t.min()) / (t.max() - t.min())

        # 組合輸入
        inputs = np.stack([x_norm, y_norm, t_norm], axis=1)
        inputs_tensor = torch.FloatTensor(inputs).to(device)

        # 預測
        with torch.no_grad():
            pred = model(inputs_tensor).cpu().numpy()
        
        # 反標準化回物理空間
        pred_physical = denormalize_output(
            pred,
            config,
            output_norm_type='training_data_norm',
            verbose=(i == 0),  # 只在第一個時間點輸出日誌
            checkpoint_path=checkpoint_path
        )

        # Reshape
        u_pred = pred_physical[:, 0].reshape(H, W)
        v_pred = pred_physical[:, 1].reshape(H, W)
        p_pred = pred_physical[:, 2].reshape(H, W)

        # Ground truth
        u_true = dns_data['u'][t_idx]
        v_true = dns_data['v'][t_idx]
        p_true = dns_data['p'][t_idx]

        # 計算誤差
        u_error = np.abs(u_pred - u_true)
        v_error = np.abs(v_pred - v_true)
        p_error = np.abs(p_pred - p_true)

        # 相對 L2 誤差
        u_l2 = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
        v_l2 = np.linalg.norm(v_pred - v_true) / np.linalg.norm(v_true)
        p_l2 = np.linalg.norm(p_pred - p_true) / np.linalg.norm(p_true)

        results['predictions'].append({
            'u': u_pred, 'v': v_pred, 'p': p_pred
        })
        results['ground_truth'].append({
            'u': u_true, 'v': v_true, 'p': p_true
        })
        results['errors'].append({
            'u': u_error, 'v': v_error, 'p': p_error,
            'u_l2': u_l2, 'v_l2': v_l2, 'p_l2': p_l2
        })
        results['time_points'].append(t_val)

        print(f"   t={t_val:.2f}: u_L2={u_l2:.4f}, v_L2={v_l2:.4f}, p_L2={p_l2:.4f}")

    return results


def plot_results(results, dns_data, output_dir):
    """繪製結果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X = dns_data['X']
    Y = dns_data['Y']

    # 繪製每個時間點的結果
    for i, t_val in enumerate(results['time_points']):
        pred = results['predictions'][i]
        truth = results['ground_truth'][i]
        error = results['errors'][i]

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f'Kolmogorov Flow at t={t_val:.2f}', fontsize=16)

        # u 速度
        im0 = axes[0, 0].contourf(X, Y, truth['u'], levels=50, cmap='RdBu_r')
        axes[0, 0].set_title('u (Ground Truth)')
        axes[0, 0].set_aspect('equal')
        plt.colorbar(im0, ax=axes[0, 0])

        im1 = axes[0, 1].contourf(X, Y, pred['u'], levels=50, cmap='RdBu_r')
        axes[0, 1].set_title('u (Prediction)')
        axes[0, 1].set_aspect('equal')
        plt.colorbar(im1, ax=axes[0, 1])

        im2 = axes[0, 2].contourf(X, Y, error['u'], levels=50, cmap='hot')
        axes[0, 2].set_title(f'u Error (L2={error["u_l2"]:.4f})')
        axes[0, 2].set_aspect('equal')
        plt.colorbar(im2, ax=axes[0, 2])

        # v 速度
        im3 = axes[1, 0].contourf(X, Y, truth['v'], levels=50, cmap='RdBu_r')
        axes[1, 0].set_title('v (Ground Truth)')
        axes[1, 0].set_aspect('equal')
        plt.colorbar(im3, ax=axes[1, 0])

        im4 = axes[1, 1].contourf(X, Y, pred['v'], levels=50, cmap='RdBu_r')
        axes[1, 1].set_title('v (Prediction)')
        axes[1, 1].set_aspect('equal')
        plt.colorbar(im4, ax=axes[1, 1])

        im5 = axes[1, 2].contourf(X, Y, error['v'], levels=50, cmap='hot')
        axes[1, 2].set_title(f'v Error (L2={error["v_l2"]:.4f})')
        axes[1, 2].set_aspect('equal')
        plt.colorbar(im5, ax=axes[1, 2])

        # 壓力
        im6 = axes[2, 0].contourf(X, Y, truth['p'], levels=50, cmap='viridis')
        axes[2, 0].set_title('p (Ground Truth)')
        axes[2, 0].set_aspect('equal')
        plt.colorbar(im6, ax=axes[2, 0])

        im7 = axes[2, 1].contourf(X, Y, pred['p'], levels=50, cmap='viridis')
        axes[2, 1].set_title('p (Prediction)')
        axes[2, 1].set_aspect('equal')
        plt.colorbar(im7, ax=axes[2, 1])

        im8 = axes[2, 2].contourf(X, Y, error['p'], levels=50, cmap='hot')
        axes[2, 2].set_title(f'p Error (L2={error["p_l2"]:.4f})')
        axes[2, 2].set_aspect('equal')
        plt.colorbar(im8, ax=axes[2, 2])

        plt.tight_layout()
        plt.savefig(output_dir / f'prediction_t{t_val:.2f}.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\n✅ 預測圖已保存至: {output_dir}")


def plot_loss_curve(log_file, output_dir):
    """從日誌文件繪製 loss curve"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📊 繪製 Loss Curve from {log_file}...")

    epochs = []
    total_losses = []
    data_losses = []
    pde_losses = []
    div_losses = []

    with open(log_file, 'r') as f:
        for line in f:
            # 解析 epoch 和 loss
            if 'Epoch' in line and 'total_loss' in line and 'kolmogorov' in line:
                parts = line.split('|')
                for part in parts:
                    part = part.strip()
                    if part.startswith('Epoch'):
                        epoch_str = part.split()[1].split('/')[0]
                        epochs.append(int(epoch_str))
                    elif part.startswith('total_loss:'):
                        total_losses.append(float(part.split(':')[1]))
                    elif part.startswith('data_loss:'):
                        data_losses.append(float(part.split(':')[1]))
                    elif part.startswith('pde_loss:'):
                        pde_losses.append(float(part.split(':')[1]))
                    elif part.startswith('div_loss:'):
                        div_losses.append(float(part.split(':')[1]))

    if not epochs:
        print("⚠️  日誌中找不到 loss 數據")
        return

    # 繪製 loss curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Loss Curves', fontsize=16)

    axes[0, 0].plot(epochs, total_losses, linewidth=1.5)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')

    axes[0, 1].plot(epochs, data_losses, linewidth=1.5, color='orange')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Data Loss')
    axes[0, 1].set_title('Data Loss')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')

    axes[1, 0].plot(epochs, pde_losses, linewidth=1.5, color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('PDE Loss')
    axes[1, 0].set_title('PDE Loss (Momentum)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')

    axes[1, 1].plot(epochs, div_losses, linewidth=1.5, color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Divergence Loss')
    axes[1, 1].set_title('Continuity Loss (Divergence)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Loss curves 已保存至: {output_dir / 'loss_curves.png'}")

    # 統計資訊
    print(f"\n📊 訓練統計:")
    print(f"   總 Epochs: {len(epochs)}")
    print(f"   初始 Total Loss: {total_losses[0]:.4f}")
    print(f"   最終 Total Loss: {total_losses[-1]:.4f}")
    print(f"   Data Loss: {data_losses[0]:.4f} → {data_losses[-1]:.4f}")
    print(f"   PDE Loss: {pde_losses[0]:.4f} → {pde_losses[-1]:.4f}")
    print(f"   Div Loss: {div_losses[0]:.4f} → {div_losses[-1]:.4f}")


def main():
    # 配置
    checkpoint_path = "checkpoints/kolmogorov_re50_kf4_K100/epoch_10000.pth"
    dns_data_path = "data/kolmogorov_dns/dns_re50_t100.h5"
    log_file = "pinnx.log"
    output_dir = "results/quick_eval_viz"

    print("="*70)
    print("  Kolmogorov Flow 訓練結果評估與視覺化")
    print("="*70)

    # 載入檢查點
    checkpoint, config = load_checkpoint(checkpoint_path)

    # 創建模型
    model = create_model_from_checkpoint(checkpoint)

    # 載入權重（使用 strict=False 允許部分匹配）
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    if missing_keys:
        print(f"⚠️  Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")

    print(f"✅ 模型權重已載入 (Epoch {checkpoint['epoch']})")

    # 載入 DNS 數據
    dns_data = load_dns_data(dns_data_path)

    # 評估模型
    results = evaluate_model(model, dns_data, config, checkpoint_path, device='cpu', num_samples=5)

    # 繪製結果
    plot_results(results, dns_data, output_dir)

    # 繪製 loss curve
    plot_loss_curve(log_file, output_dir)

    print("\n" + "="*70)
    print("✅ 評估完成！")
    print("="*70)


if __name__ == '__main__':
    main()
