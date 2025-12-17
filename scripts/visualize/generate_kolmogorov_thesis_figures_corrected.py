#!/usr/bin/env python3
"""
生成論文格式的 Kolmogorov Flow 圖片（修正版）
- 使用時間平均場作為 DNS 參考（符合湍流研究標準）
- 模型預測使用時間範圍的中點
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


def load_dns_time_averaged(h5_path, time_range=(15.0, 35.0)):
    """從 HDF5 載入時間平均 DNS 場（統計穩態）"""
    print(f"📂 載入 DNS 時間平均場: {h5_path}")
    print(f"   時間範圍: [{time_range[0]}, {time_range[1]}]")
    
    with h5py.File(h5_path, 'r') as f:
        # 讀取時間軸
        time_all = np.array(f['time'])
        t_start, t_end = time_range
        time_mask = (time_all >= t_start) & (time_all <= t_end)
        time_selected = time_all[time_mask]
        
        print(f"   使用 {len(time_selected)} 個時間步進行平均")
        
        # 重建座標
        N = f['config'].attrs['N']
        L = f['config'].attrs['L']
        x = np.linspace(0, L, N, endpoint=False)
        y = np.linspace(0, L, N, endpoint=False)
        X, Y = np.meshgrid(x, y)
        
        # 提取時間範圍內的場並計算平均
        u_slice = f['u'][time_mask]  # [T, N, N]
        v_slice = f['v'][time_mask]
        p_slice = f['p'][time_mask]
        
        # 時間平均
        u_mean = u_slice.mean(axis=0)  # [N, N]
        v_mean = v_slice.mean(axis=0)
        p_mean = p_slice.mean(axis=0)
        
        # 計算時間 RMS（用於評估時間波動）
        u_std = u_slice.std(axis=0)
        v_std = v_slice.std(axis=0)
        p_std = p_slice.std(axis=0)
    
    print(f"✅ DNS 時間平均場載入完成")
    print(f"   u: mean={u_mean.mean():.6f}, std={u_std.mean():.6f}")
    print(f"   v: mean={v_mean.mean():.6f}, std={v_std.mean():.6f}")
    print(f"   p: mean={p_mean.mean():.6f}, std={p_std.mean():.6f}")
    
    return {
        'x': x, 'y': y, 'X': X, 'Y': Y,
        'u': u_mean, 'v': v_mean, 'p': p_mean,
        'u_std': u_std, 'v_std': v_std, 'p_std': p_std,
        'time_range': time_range,
        'n_samples': len(time_selected)
    }


def predict_time_averaged(model, X, Y, time_range, n_samples, device, output_transform=None):
    """預測時間平均場（在時間範圍內採樣）"""
    print(f"🔮 預測時間平均場...")
    print(f"   時間範圍: [{time_range[0]}, {time_range[1]}]")
    print(f"   時間採樣點: {n_samples}")
    
    t_start, t_end = time_range
    t_samples = np.linspace(t_start, t_end, n_samples)
    
    x_flat = X.flatten()
    y_flat = Y.flatten()
    N_spatial = len(x_flat)
    
    # 累積預測結果
    u_accumulated = np.zeros(N_spatial)
    v_accumulated = np.zeros(N_spatial)
    p_accumulated = np.zeros(N_spatial)
    
    # 分批預測以節省記憶體
    batch_size = 20  # 每次預測 20 個時間步
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        t_batch = t_samples[start_idx:end_idx]
        
        # 構建輸入 [T_batch * N_spatial, 3]
        coords_batch = []
        for t in t_batch:
            t_arr = np.full(N_spatial, t)
            coords = np.stack([t_arr, x_flat, y_flat], axis=1)
            coords_batch.append(coords)
        
        coords_batch = np.concatenate(coords_batch, axis=0)
        coords_tensor = torch.tensor(coords_batch, dtype=torch.float32, device=device)
        
        # 預測
        with torch.no_grad():
            pred_normalized = model(coords_tensor)
        
        if output_transform is not None:
            pred = output_transform.denormalize_batch(pred_normalized)
            pred = pred.cpu().numpy()
        else:
            pred = pred_normalized.cpu().numpy()
        
        # 累積（reshape 並平均）
        pred_reshaped = pred.reshape(len(t_batch), N_spatial, 3)  # [T_batch, N_spatial, 3]
        u_accumulated += pred_reshaped[:, :, 0].sum(axis=0)
        v_accumulated += pred_reshaped[:, :, 1].sum(axis=0)
        p_accumulated += pred_reshaped[:, :, 2].sum(axis=0)
        
        if (i + 1) % 5 == 0 or (i + 1) == n_batches:
            print(f"   進度: {end_idx}/{n_samples} ({100*end_idx/n_samples:.1f}%)")
    
    # 計算平均
    u_mean = (u_accumulated / n_samples).reshape(X.shape)
    v_mean = (v_accumulated / n_samples).reshape(X.shape)
    p_mean = (p_accumulated / n_samples).reshape(X.shape)
    
    print(f"✅ 時間平均預測完成")
    return u_mean, v_mean, p_mean


def generate_field_figure(X, Y, dns_field, pred_field, field_name, output_path, dns_std=None):
    """
    生成論文格式的 3 面板圖：DNS / PINN / Error
    """
    print(f"📊 生成 {field_name} 場圖...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 計算共用範圍（使用 DNS 的動態範圍）
    vmin, vmax = np.percentile(dns_field, [1, 99])
    
    # 1. DNS Time-Averaged Reference
    im0 = axes[0].contourf(X, Y, dns_field, levels=100, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'{field_name.upper()} Field (DNS Time-Averaged)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('y', fontsize=12)
    axes[0].set_aspect('equal')
    cbar0 = plt.colorbar(im0, ax=axes[0])
    cbar0.ax.tick_params(labelsize=10)
    
    # 添加時間 RMS 資訊（如果有）
    if dns_std is not None:
        axes[0].text(0.02, 0.98, f'RMS: {dns_std.mean():.4f}', 
                    transform=axes[0].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. PINN Time-Averaged Prediction
    im1 = axes[1].contourf(X, Y, pred_field, levels=100, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'{field_name.upper()} Field (PINN Time-Averaged)', fontsize=14, fontweight='bold')
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
    fig.suptitle(f'Kolmogorov Flow: {field_name.upper()} (Time-Averaged, Relative L2 Error: {rel_l2*100:.2f}%)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 已儲存: {output_path}")
    return rel_l2


def main():
    parser = argparse.ArgumentParser(description='生成論文格式的 Kolmogorov 場圖（時間平均）')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint 路徑')
    parser.add_argument('--reference', required=True, help='DNS H5 檔案路徑')
    parser.add_argument('--output-dir', required=True, help='輸出目錄')
    parser.add_argument('--device', default='auto', help='計算裝置')
    parser.add_argument('--time-range', type=float, nargs=2, default=[15.0, 35.0], 
                        help='時間範圍 (default: 15.0 35.0)')
    parser.add_argument('--n-time-samples', type=int, default=20, 
                        help='時間採樣點數 (default: 20)')
    
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
    
    # 載入 DNS 時間平均參考場
    dns_data = load_dns_time_averaged(args.reference, tuple(args.time_range))
    
    # 建立 OutputTransform
    normalization_meta = ckpt.get('normalization', None)
    output_transform = None
    if normalization_meta is not None:
        try:
            output_transform = OutputTransform.from_metadata(normalization_meta)
            print(f"✅ 已載入 OutputTransform")
        except Exception as e:
            print(f"⚠️  OutputTransform 載入失敗: {e}")
    
    # 預測時間平均場
    pred_u, pred_v, pred_p = predict_time_averaged(
        model, dns_data['X'], dns_data['Y'], 
        args.time_range, args.n_time_samples, 
        device, output_transform
    )
    
    # 生成獨立圖片
    print("\n" + "="*70)
    print("📊 生成論文格式圖片（時間平均）...")
    print("="*70 + "\n")
    
    errors = {}
    
    errors['u'] = generate_field_figure(
        dns_data['X'], dns_data['Y'], 
        dns_data['u'], pred_u, 
        'u', 
        output_dir / 'field_u.png',
        dns_data.get('u_std')
    )
    
    errors['v'] = generate_field_figure(
        dns_data['X'], dns_data['Y'], 
        dns_data['v'], pred_v, 
        'v', 
        output_dir / 'field_v.png',
        dns_data.get('v_std')
    )
    
    errors['p'] = generate_field_figure(
        dns_data['X'], dns_data['Y'], 
        dns_data['p'], pred_p, 
        'p', 
        output_dir / 'field_p.png',
        dns_data.get('p_std')
    )
    
    # 儲存評估報告
    report_path = output_dir / 'evaluation_summary.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Kolmogorov Flow Time-Averaged Evaluation\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"DNS Reference:\n")
        f.write(f"  Time Range: [{args.time_range[0]}, {args.time_range[1]}]\n")
        f.write(f"  Time Samples: {dns_data['n_samples']}\n\n")
        f.write(f"PINN Prediction:\n")
        f.write(f"  Time Range: [{args.time_range[0]}, {args.time_range[1]}]\n")
        f.write(f"  Time Samples: {args.n_time_samples}\n\n")
        f.write(f"Relative L2 Errors:\n")
        f.write(f"  u: {errors['u']*100:.2f}%\n")
        f.write(f"  v: {errors['v']*100:.2f}%\n")
        f.write(f"  p: {errors['p']*100:.2f}%\n")
        f.write(f"  Overall: {np.sqrt(sum(e**2 for e in errors.values())/3)*100:.2f}%\n")
    
    print("\n" + "="*70)
    print(f"✅ 所有圖片已儲存至: {output_dir}")
    print(f"📄 評估報告: {report_path}")
    print("="*70)
    
    print("\n📊 相對 L2 誤差:")
    print(f"  u: {errors['u']*100:.2f}%")
    print(f"  v: {errors['v']*100:.2f}%")
    print(f"  p: {errors['p']*100:.2f}%")
    print(f"  Overall: {np.sqrt(sum(e**2 for e in errors.values())/3)*100:.2f}%")


if __name__ == '__main__':
    main()
