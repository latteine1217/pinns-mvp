#!/usr/bin/env python3
"""
生成論文格式的 Kolmogorov Flow 圖片
為每個場 (u, v, p) 生成獨立的 3 面板圖 (DNS / PINN / Error)
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt

# 加入專案路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pinnx.models.fourier_mlp import PINNNet
from pinnx.utils.normalization import OutputTransform


def load_checkpoint(checkpoint_path, device='cpu'):
    """載入訓練好的 checkpoint"""
    print(f"📥 載入 checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    config = ckpt['config']
    epoch = ckpt['epoch']
    
    model_config = config['model']
    block_type = model_config.get('block_type', 'dense')
    
    # 處理兩種配置格式
    if 'fourier_features' in model_config and isinstance(model_config['fourier_features'], dict):
        ff_config = model_config['fourier_features']
        if 'fourier_m' in ff_config:
            use_fourier = ff_config['enabled']
            fourier_m = ff_config['fourier_m']
            fourier_sigma = ff_config['fourier_sigma']
            trainable_fourier = ff_config.get('trainable', False)
        else:
            use_fourier = model_config.get('use_fourier', False)
            fourier_m = model_config.get('fourier_m', 16)
            fourier_sigma = model_config.get('fourier_sigma', 4.0)
            trainable_fourier = model_config.get('fourier_trainable', False)
    else:
        use_fourier = model_config.get('use_fourier', False)
        fourier_m = model_config.get('fourier_m', 16)
        fourier_sigma = model_config.get('fourier_sigma', 4.0)
        trainable_fourier = model_config.get('fourier_trainable', False)
    
    model = PINNNet(
        in_dim=model_config['in_dim'],
        out_dim=model_config['out_dim'],
        width=model_config['width'],
        depth=model_config['depth'],
        activation=model_config['activation'],
        block_type=block_type,
        use_fourier=use_fourier,
        fourier_m=fourier_m,
        fourier_sigma=fourier_sigma,
        trainable_fourier=trainable_fourier
    )
    
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ 載入完成 (Epoch {epoch})")
    return model, config, ckpt


def load_dns_reference(h5_path, time_index=50):
    """從 HDF5 載入 DNS 參考資料"""
    print(f"📂 載入 DNS 參考: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        N = f['config'].attrs['N']
        L = f['config'].attrs['L']
        x = np.linspace(0, L, N, endpoint=False)
        y = np.linspace(0, L, N, endpoint=False)
        X, Y = np.meshgrid(x, y)
        
        u_dns = f['u'][time_index, :, :]
        v_dns = f['v'][time_index, :, :]
        p_dns = f['p'][time_index, :, :]
    
    print(f"✅ DNS 資料載入: {u_dns.shape}")
    return {
        'x': x, 'y': y, 'X': X, 'Y': Y,
        'u': u_dns, 'v': v_dns, 'p': p_dns
    }


def predict_field(model, X, Y, t, device, output_transform=None):
    """使用模型預測全場"""
    print(f"🔮 預測流場 (t={t})...")
    
    x_flat = X.flatten()
    y_flat = Y.flatten()
    t_arr = np.full_like(x_flat, t)
    
    coords = np.stack([t_arr, x_flat, y_flat], axis=1)
    coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        pred_normalized = model(coords_tensor)
    
    if output_transform is not None:
        pred = output_transform.denormalize_batch(pred_normalized)
        pred = pred.cpu().numpy()
    else:
        pred = pred_normalized.cpu().numpy()
    
    u_pred = pred[:, 0].reshape(X.shape)
    v_pred = pred[:, 1].reshape(X.shape)
    p_pred = pred[:, 2].reshape(X.shape)
    
    print(f"✅ 預測完成")
    return u_pred, v_pred, p_pred


def generate_field_figure(X, Y, dns_field, pred_field, field_name, output_path):
    """
    生成論文格式的 3 面板圖：DNS / PINN / Error
    
    Args:
        X, Y: 網格座標
        dns_field: DNS 參考場
        pred_field: PINN 預測場
        field_name: 場名稱 ('u', 'v', 'p')
        output_path: 輸出路徑
    """
    print(f"📊 生成 {field_name} 場圖...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 計算共用範圍（使用 DNS 的動態範圍）
    vmin, vmax = np.percentile(dns_field, [1, 99])
    
    # 1. DNS Reference
    im0 = axes[0].contourf(X, Y, dns_field, levels=100, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'{field_name.upper()} Field (DNS Reference)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('y', fontsize=12)
    axes[0].set_aspect('equal')
    cbar0 = plt.colorbar(im0, ax=axes[0])
    cbar0.ax.tick_params(labelsize=10)
    
    # 2. PINN Prediction
    im1 = axes[1].contourf(X, Y, pred_field, levels=100, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'{field_name.upper()} Field (PINN Prediction)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel('y', fontsize=12)
    axes[1].set_aspect('equal')
    cbar1 = plt.colorbar(im1, ax=axes[1])
    cbar1.ax.tick_params(labelsize=10)
    
    # 3. Absolute Error
    error = np.abs(pred_field - dns_field)
    im2 = axes[2].contourf(X, Y, error, levels=100, cmap='Reds')
    axes[2].set_title(f'{field_name.upper()} Absolute Error', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('x', fontsize=12)
    axes[2].set_ylabel('y', fontsize=12)
    axes[2].set_aspect('equal')
    cbar2 = plt.colorbar(im2, ax=axes[2])
    cbar2.ax.tick_params(labelsize=10)
    
    # 計算並顯示相對 L2 誤差
    rel_l2 = np.linalg.norm(pred_field - dns_field) / np.linalg.norm(dns_field)
    fig.suptitle(f'Kolmogorov Flow: {field_name.upper()} (Relative L2 Error: {rel_l2*100:.2f}%)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 已儲存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='生成論文格式的 Kolmogorov 場圖')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint 路徑')
    parser.add_argument('--reference', required=True, help='DNS H5 檔案路徑')
    parser.add_argument('--output-dir', required=True, help='輸出目錄')
    parser.add_argument('--device', default='auto', help='計算裝置')
    parser.add_argument('--time-index', type=int, default=50, help='DNS 時間索引')
    parser.add_argument('--t-eval', type=float, default=25.0, help='評估時間')
    
    args = parser.parse_args()
    
    # 設定裝置
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print(f"🖥️  使用裝置: {device}")
    
    # 建立輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 載入模型
    model, config, ckpt = load_checkpoint(args.checkpoint, device)
    
    # 載入 DNS 參考
    dns_data = load_dns_reference(args.reference, args.time_index)
    
    # 建立 OutputTransform
    normalization_meta = ckpt.get('normalization', None)
    output_transform = None
    if normalization_meta is not None:
        try:
            output_transform = OutputTransform.from_metadata(normalization_meta)
            print(f"✅ 已載入 OutputTransform")
        except Exception as e:
            print(f"⚠️  OutputTransform 載入失敗: {e}")
    
    # 預測流場
    pred_u, pred_v, pred_p = predict_field(
        model, dns_data['X'], dns_data['Y'], args.t_eval, device, output_transform
    )
    
    # 生成獨立圖片
    print("\n" + "="*70)
    print("📊 生成論文格式圖片...")
    print("="*70 + "\n")
    
    generate_field_figure(
        dns_data['X'], dns_data['Y'], 
        dns_data['u'], pred_u, 
        'u', 
        output_dir / 'field_u.png'
    )
    
    generate_field_figure(
        dns_data['X'], dns_data['Y'], 
        dns_data['v'], pred_v, 
        'v', 
        output_dir / 'field_v.png'
    )
    
    generate_field_figure(
        dns_data['X'], dns_data['Y'], 
        dns_data['p'], pred_p, 
        'p', 
        output_dir / 'field_p.png'
    )
    
    print("\n" + "="*70)
    print(f"✅ 所有圖片已儲存至: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
