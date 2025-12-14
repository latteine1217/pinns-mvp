#!/usr/bin/env python3
"""
簡化的 Kolmogorov Flow 2D 評估腳本
專門針對 RANS Prior 實驗進行評估
"""

import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
import yaml

# 加入專案路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pinnx.models.fourier_mlp import PINNNet
from pinnx.utils.normalization import OutputTransform


def load_checkpoint(checkpoint_path, device='cpu'):
    """載入訓練好的 checkpoint"""
    print(f"📥 載入 checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # 提取配置
    config = ckpt['config']
    epoch = ckpt['epoch']
    
    # 建立模型（使用 PINNNet 支援 block_type）
    model_config = config['model']
    block_type = model_config.get('block_type', 'dense')
    
    print(f"🏗️  使用 PINNNet 架構 (block_type={block_type})")
    # 處理兩種配置格式：嵌套結構 (RANS prior) vs 扁平結構 (vanilla)
    if 'fourier_features' in model_config and isinstance(model_config['fourier_features'], dict):
        ff_config = model_config['fourier_features']
        # 檢查是否有完整的嵌套配置
        if 'fourier_m' in ff_config:
            # 完整嵌套結構 (新格式)
            use_fourier = ff_config['enabled']
            fourier_m = ff_config['fourier_m']
            fourier_sigma = ff_config['fourier_sigma']
            trainable_fourier = ff_config.get('trainable', False)
        else:
            # 部分嵌套結構 (混合格式) - fourier_m 在頂層
            use_fourier = model_config.get('use_fourier', False)
            fourier_m = model_config.get('fourier_m', 16)
            fourier_sigma = model_config.get('fourier_sigma', 4.0)
            trainable_fourier = model_config.get('fourier_trainable', False)
    else:
        # 扁平結構 (舊格式)
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
        # 重建座標
        N = f['config'].attrs['N']
        L = f['config'].attrs['L']
        x = np.linspace(0, L, N, endpoint=False)
        y = np.linspace(0, L, N, endpoint=False)
        X, Y = np.meshgrid(x, y)
        
        # 提取場
        u_dns = f['u'][time_index, :, :]
        v_dns = f['v'][time_index, :, :]
        p_dns = f['p'][time_index, :, :]
    
    print(f"✅ DNS 資料載入: {u_dns.shape}")
    return {
        'x': x,
        'y': y,
        'X': X,
        'Y': Y,
        'u': u_dns,
        'v': v_dns,
        'p': p_dns
    }


def predict_field(model, X, Y, t, device, output_transform=None):
    """使用模型預測全場
    
    Args:
        model: PINNNet 模型
        X, Y: 空間網格座標
        t: 評估時間
        device: 計算設備
        output_transform: OutputTransform 物件（用於反歸一化）
    """
    print(f"🔮 預測流場 (t={t})...")
    
    # 準備輸入
    x_flat = X.flatten()
    y_flat = Y.flatten()
    t_arr = np.full_like(x_flat, t)
    
    coords = np.stack([t_arr, x_flat, y_flat], axis=1)
    coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
    
    # 注意：training_data_norm 只標準化輸出變數，不標準化輸入座標
    
    # 預測（標準化空間）
    with torch.no_grad():
        pred_normalized = model(coords_tensor)
    
    # 使用 OutputTransform 反歸一化
    if output_transform is not None:
        # denormalize_batch 需要 variable_order，使用 checkpoint 中的順序 ['u', 'v', 'p']
        pred = output_transform.denormalize_batch(pred_normalized)
        pred = pred.cpu().numpy()
    else:
        pred = pred_normalized.cpu().numpy()
    
    # Reshape
    u_pred = pred[:, 0].reshape(X.shape)
    v_pred = pred[:, 1].reshape(X.shape)
    p_pred = pred[:, 2].reshape(X.shape)
    
    print(f"✅ 預測完成")
    return u_pred, v_pred, p_pred


def compute_metrics(dns_data, pred_u, pred_v, pred_p):
    """計算評估指標"""
    print(f"📊 計算評估指標...")
    
    # 相對 L2 誤差
    u_l2 = np.linalg.norm(pred_u - dns_data['u']) / np.linalg.norm(dns_data['u'])
    v_l2 = np.linalg.norm(pred_v - dns_data['v']) / np.linalg.norm(dns_data['v'])
    p_l2 = np.linalg.norm(pred_p - dns_data['p']) / np.linalg.norm(dns_data['p'])
    
    overall_l2 = np.sqrt((u_l2**2 + v_l2**2 + p_l2**2) / 3)
    
    # 計算渦度誤差
    dx = dns_data['x'][1] - dns_data['x'][0]
    dy = dns_data['y'][1] - dns_data['y'][0]
    
    # DNS 渦度
    dv_dx_dns = np.gradient(dns_data['v'], dx, axis=1)
    du_dy_dns = np.gradient(dns_data['u'], dy, axis=0)
    vort_dns = dv_dx_dns - du_dy_dns
    
    # 預測渦度
    dv_dx_pred = np.gradient(pred_v, dx, axis=1)
    du_dy_pred = np.gradient(pred_u, dy, axis=0)
    vort_pred = dv_dx_pred - du_dy_pred
    
    vort_l2 = np.linalg.norm(vort_pred - vort_dns) / np.linalg.norm(vort_dns)
    
    # 連續性誤差
    du_dx = np.gradient(pred_u, dx, axis=1)
    dv_dy = np.gradient(pred_v, dy, axis=0)
    div = du_dx + dv_dy
    div_error_mean = np.abs(div).mean()
    div_error_max = np.abs(div).max()
    
    # 壓力梯度誤差
    dp_dx_dns = np.gradient(dns_data['p'], dx, axis=1)
    dp_dy_dns = np.gradient(dns_data['p'], dy, axis=0)
    dp_dx_pred = np.gradient(pred_p, dx, axis=1)
    dp_dy_pred = np.gradient(pred_p, dy, axis=0)
    
    dpdx_l2 = np.linalg.norm(dp_dx_pred - dp_dx_dns) / np.linalg.norm(dp_dx_dns)
    dpdy_l2 = np.linalg.norm(dp_dy_pred - dp_dy_dns) / np.linalg.norm(dp_dy_dns)
    
    metrics = {
        'relative_l2_overall': float(overall_l2),
        'relative_l2_u': float(u_l2),
        'relative_l2_v': float(v_l2),
        'relative_l2_p': float(p_l2),
        'relative_l2_vorticity': float(vort_l2),
        'divergence_error_mean': float(div_error_mean),
        'divergence_error_max': float(div_error_max),
        'pressure_gradient_dpdx_l2': float(dpdx_l2),
        'pressure_gradient_dpdy_l2': float(dpdy_l2),
    }
    
    print(f"✅ 指標計算完成")
    return metrics


def visualize_results(dns_data, pred_u, pred_v, pred_p, metrics, output_dir):
    """生成視覺化圖表"""
    print(f"📊 生成視覺化...")
    
    X, Y = dns_data['X'], dns_data['Y']
    
    # === 圖1: 場重建對比 (DNS / Pred / Error) ===
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    
    fields = [
        ('u', dns_data['u'], pred_u),
        ('v', dns_data['v'], pred_v),
        ('p', dns_data['p'], pred_p)
    ]
    
    for i, (name, dns, pred) in enumerate(fields):
        # DNS
        vmin, vmax = np.percentile(dns, [1, 99])
        im0 = axes[i, 0].contourf(X, Y, dns, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f'{name} (DNS)', fontsize=14, fontweight='bold')
        axes[i, 0].set_xlabel('x')
        axes[i, 0].set_ylabel('y')
        plt.colorbar(im0, ax=axes[i, 0])
        
        # Prediction
        im1 = axes[i, 1].contourf(X, Y, pred, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f'{name} (PINN)', fontsize=14, fontweight='bold')
        axes[i, 1].set_xlabel('x')
        axes[i, 1].set_ylabel('y')
        plt.colorbar(im1, ax=axes[i, 1])
        
        # Error
        error = np.abs(pred - dns)
        im2 = axes[i, 2].contourf(X, Y, error, levels=50, cmap='Reds')
        axes[i, 2].set_title(f'{name} Error', fontsize=14, fontweight='bold')
        axes[i, 2].set_xlabel('x')
        axes[i, 2].set_ylabel('y')
        plt.colorbar(im2, ax=axes[i, 2])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'field_reconstruction.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # === 圖2: 誤差統計 ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart
    vars = ['u', 'v', 'p', 'vorticity']
    errors = [
        metrics['relative_l2_u'] * 100,
        metrics['relative_l2_v'] * 100,
        metrics['relative_l2_p'] * 100,
        metrics['relative_l2_vorticity'] * 100
    ]
    colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12']
    
    bars = ax1.bar(vars, errors, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.axhline(15, color='red', linestyle='--', linewidth=2, label='Target (15%)')
    ax1.set_ylabel('Relative L2 Error (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Field Reconstruction Errors', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Physics metrics
    metrics_names = ['Div (mean)', 'Div (max)', '∂p/∂x', '∂p/∂y']
    metrics_values = [
        metrics['divergence_error_mean'] * 1000,  # scale to 1e-3
        metrics['divergence_error_max'] * 1000,
        metrics['pressure_gradient_dpdx_l2'] * 100,
        metrics['pressure_gradient_dpdy_l2'] * 100
    ]
    
    ax2.bar(metrics_names, metrics_values, color='#9B59B6', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Error', fontsize=12, fontweight='bold')
    ax2.set_title('Physics Metrics', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 視覺化完成")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Kolmogorov 2D PINNs checkpoint')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--reference', required=True, help='Path to DNS H5 file')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--device', default='auto', help='Device (cuda/cpu/mps/auto)')
    parser.add_argument('--time-index', type=int, default=50, help='DNS time index')
    parser.add_argument('--t-eval', type=float, default=25.0, help='Evaluation time')
    
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
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # === 1. 載入模型 ===
    model, config, ckpt = load_checkpoint(args.checkpoint, device)
    
    # === 2. 載入 DNS 參考 ===
    dns_data = load_dns_reference(args.reference, args.time_index)
    
    # === 3. 建立 OutputTransform（正確的反歸一化器）===
    normalization_meta = ckpt.get('normalization', None)
    output_transform = None
    if normalization_meta is not None:
        try:
            output_transform = OutputTransform.from_metadata(normalization_meta)
            print(f"✅ 已載入 OutputTransform: {normalization_meta['norm_type']}")
        except Exception as e:
            print(f"⚠️  OutputTransform 載入失敗: {e}")
            print("   將使用原始預測值（可能未正確反歸一化）")
    
    # === 4. 預測流場 ===
    pred_u, pred_v, pred_p = predict_field(
        model, dns_data['X'], dns_data['Y'], args.t_eval, device, output_transform
    )
    
    # === 5. 計算指標 ===
    metrics = compute_metrics(dns_data, pred_u, pred_v, pred_p)
    
    # === 6. 列印結果 ===
    print("\n" + "="*70)
    print("📊 評估結果總結")
    print("="*70)
    print(f"\n場重建誤差:")
    print(f"  Overall L2: {metrics['relative_l2_overall']*100:.2f}%")
    print(f"  u L2:       {metrics['relative_l2_u']*100:.2f}% {'✅' if metrics['relative_l2_u'] < 0.15 else '❌'}")
    print(f"  v L2:       {metrics['relative_l2_v']*100:.2f}% {'✅' if metrics['relative_l2_v'] < 0.15 else '❌'}")
    print(f"  p L2:       {metrics['relative_l2_p']*100:.2f}% {'✅' if metrics['relative_l2_p'] < 0.20 else '❌'}")
    print(f"  ω L2:       {metrics['relative_l2_vorticity']*100:.2f}%")
    
    print(f"\n物理守恆:")
    print(f"  Div (mean): {metrics['divergence_error_mean']:.2e} {'✅' if metrics['divergence_error_mean'] < 1e-3 else '❌'}")
    print(f"  Div (max):  {metrics['divergence_error_max']:.2e}")
    
    print(f"\n壓力梯度誤差 (RANS Prior 改善目標):")
    print(f"  ∂p/∂x L2:   {metrics['pressure_gradient_dpdx_l2']*100:.2f}% {'✅' if metrics['pressure_gradient_dpdx_l2'] < 0.30 else '❌'}")
    print(f"  ∂p/∂y L2:   {metrics['pressure_gradient_dpdy_l2']*100:.2f}% {'✅' if metrics['pressure_gradient_dpdy_l2'] < 0.30 else '❌'}")
    
    print("\n" + "="*70)
    
    # === 7. 儲存指標 ===
    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 指標已儲存: {metrics_file}")
    
    # === 8. 生成視覺化 ===
    visualize_results(dns_data, pred_u, pred_v, pred_p, metrics, output_dir)
    print(f"📊 視覺化已儲存: {output_dir}")
    
    print("\n✅ 評估完成！")


if __name__ == '__main__':
    main()
