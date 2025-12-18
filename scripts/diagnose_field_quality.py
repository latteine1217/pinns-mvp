#!/usr/bin/env python3
"""
診斷腳本：分析場重建質量
檢查預測場與 DNS 真值之間的相關性、振幅比、頻譜特性
"""

import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from scipy.fft import fft2, fftshift


def analyze_field_quality(checkpoint_path, reference_h5, time_index=250, output_dir=None):
    """
    深入分析場重建質量
    
    檢查項目:
    1. 相關係數（空間相關性）
    2. 振幅比（預測/真值的振幅比）
    3. 頻譜分析（高頻/低頻內容）
    4. 空間統計（梯度、局部極值）
    """
    print("="*70)
    print(f"場質量診斷: {checkpoint_path}")
    print("="*70)
    
    # === 加載模型與數據 ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加載 checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # 重建模型（簡化版，假設使用 FourierMLP）
    from pinnx.models.fourier_mlp import FourierMLP
    model_config = ckpt['config']['model']
    model = FourierMLP(
        input_dim=2,
        output_dim=3,  # u, v, p
        hidden_dim=model_config.get('hidden_dim', 256),
        num_layers=model_config.get('num_layers', 6),
        activation=model_config.get('activation', 'tanh'),
        fourier_features=model_config.get('fourier_features', 16),
        fourier_std=model_config.get('fourier_std', 4.0)
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # 重建 OutputTransform
    from pinnx.utils.normalization import OutputTransform
    output_transform = OutputTransform.from_metadata(ckpt['normalization'])
    
    # 加載 DNS 參考數據
    with h5py.File(reference_h5, 'r') as f:
        u_dns = f['u'][time_index]  # [Ny, Nx]
        v_dns = f['v'][time_index]
        p_dns = f['p'][time_index]
        x = f['x'][:]
        y = f['y'][:]
        t_val = f['t'][time_index]
    
    print(f"\n📊 DNS 數據: t={t_val:.2f}, shape={u_dns.shape}")
    
    # 生成預測場
    X, Y = np.meshgrid(x, y, indexing='ij')
    coords = np.stack([X.ravel(), Y.ravel()], axis=1)
    coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        pred_norm = model(coords_tensor)
        pred = output_transform.denormalize_batch(pred_norm).cpu().numpy()
    
    u_pred = pred[:, 0].reshape(X.shape)
    v_pred = pred[:, 1].reshape(X.shape)
    p_pred = pred[:, 2].reshape(X.shape)
    
    # === 分析各場 ===
    fields = {
        'u velocity': (u_dns, u_pred),
        'v velocity': (v_dns, v_pred),
        'pressure': (p_dns, p_pred)
    }
    
    results = {}
    
    for field_name, (truth, pred) in fields.items():
        print(f"\n{'='*70}")
        print(f"🔬 {field_name.upper()}")
        print(f"{'='*70}")
        
        # 1. 基本統計
        print(f"\n📐 基本統計:")
        print(f"  DNS   - mean: {truth.mean():8.4f}, std: {truth.std():8.4f}, range: [{truth.min():8.4f}, {truth.max():8.4f}]")
        print(f"  Pred  - mean: {pred.mean():8.4f}, std: {pred.std():8.4f}, range: [{pred.min():8.4f}, {pred.max():8.4f}]")
        print(f"  Ratio - mean: {pred.mean()/truth.mean() if abs(truth.mean())>1e-8 else np.nan:8.4f}, std: {pred.std()/truth.std():8.4f}")
        
        # 2. 相關性分析
        corr, p_value = pearsonr(truth.ravel(), pred.ravel())
        print(f"\n🔗 空間相關性:")
        print(f"  Pearson r: {corr:.4f} (p={p_value:.2e})")
        
        if corr < 0.3:
            print(f"  ⚠️  相關性極低！預測場與真值幾乎不相關")
        elif corr < 0.7:
            print(f"  ⚠️  相關性較弱，預測捕捉部分結構但不準確")
        else:
            print(f"  ✅ 相關性良好")
        
        # 3. 振幅分析
        truth_amplitude = np.std(truth)
        pred_amplitude = np.std(pred)
        amplitude_ratio = pred_amplitude / truth_amplitude
        
        print(f"\n📊 振幅分析:")
        print(f"  DNS amplitude (std):  {truth_amplitude:.4f}")
        print(f"  Pred amplitude (std): {pred_amplitude:.4f}")
        print(f"  Amplitude ratio:      {amplitude_ratio:.4f}")
        
        if amplitude_ratio < 0.5:
            print(f"  ⚠️  預測場嚴重低估振幅（過於平滑）")
        elif amplitude_ratio > 2.0:
            print(f"  ⚠️  預測場嚴重高估振幅（過於嘈雜）")
        
        # 4. 梯度分析
        gy_truth, gx_truth = np.gradient(truth)
        gy_pred, gx_pred = np.gradient(pred)
        
        grad_truth = np.sqrt(gx_truth**2 + gy_truth**2)
        grad_pred = np.sqrt(gx_pred**2 + gy_pred**2)
        
        print(f"\n∇ 梯度分析:")
        print(f"  DNS |∇field|:  mean={grad_truth.mean():.4f}, std={grad_truth.std():.4f}")
        print(f"  Pred |∇field|: mean={grad_pred.mean():.4f}, std={grad_pred.std():.4f}")
        print(f"  Gradient ratio: {grad_pred.mean()/grad_truth.mean():.4f}")
        
        # 5. 頻譜分析
        fft_truth = fftshift(fft2(truth))
        fft_pred = fftshift(fft2(pred))
        
        power_truth = np.abs(fft_truth)**2
        power_pred = np.abs(fft_pred)**2
        
        # 計算低頻能量（中心 10%）
        center_y, center_x = truth.shape[0]//2, truth.shape[1]//2
        radius = min(center_y, center_x) // 10
        
        y_idx, x_idx = np.ogrid[:truth.shape[0], :truth.shape[1]]
        mask_low = (y_idx - center_y)**2 + (x_idx - center_x)**2 <= radius**2
        
        low_freq_truth = power_truth[mask_low].sum()
        low_freq_pred = power_pred[mask_low].sum()
        
        total_truth = power_truth.sum()
        total_pred = power_pred.sum()
        
        print(f"\n🌊 頻譜分析:")
        print(f"  DNS  - 低頻能量比: {low_freq_truth/total_truth:.2%}")
        print(f"  Pred - 低頻能量比: {low_freq_pred/total_pred:.2%}")
        print(f"  總能量比 (Pred/DNS): {total_pred/total_truth:.4f}")
        
        # 保存結果
        results[field_name] = {
            'correlation': corr,
            'amplitude_ratio': amplitude_ratio,
            'gradient_ratio': grad_pred.mean()/grad_truth.mean(),
            'energy_ratio': total_pred/total_truth,
            'low_freq_ratio_dns': low_freq_truth/total_truth,
            'low_freq_ratio_pred': low_freq_pred/total_pred
        }
    
    # === 生成診斷圖 ===
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        
        for idx, (field_name, (truth, pred)) in enumerate(fields.items()):
            # Truth
            im1 = axes[idx, 0].imshow(truth.T, origin='lower', cmap='RdBu_r')
            axes[idx, 0].set_title(f'{field_name} - DNS')
            plt.colorbar(im1, ax=axes[idx, 0])
            
            # Prediction
            im2 = axes[idx, 1].imshow(pred.T, origin='lower', cmap='RdBu_r')
            axes[idx, 1].set_title(f'{field_name} - Prediction')
            plt.colorbar(im2, ax=axes[idx, 1])
            
            # Error
            error = np.abs(pred - truth)
            im3 = axes[idx, 2].imshow(error.T, origin='lower', cmap='hot')
            axes[idx, 2].set_title(f'{field_name} - Absolute Error')
            plt.colorbar(im3, ax=axes[idx, 2])
        
        plt.tight_layout()
        fig_path = output_dir / 'field_quality_diagnosis.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"\n💾 診斷圖已保存: {fig_path}")
        plt.close()
        
        # 保存數值結果
        import json
        json_path = output_dir / 'field_quality_metrics.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 數值結果已保存: {json_path}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='診斷場重建質量')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint 路徑')
    parser.add_argument('--reference', required=True, help='DNS 參考數據 (.h5)')
    parser.add_argument('--time-index', type=int, default=250, help='時間索引 (default: 250 = t=25.0)')
    parser.add_argument('--output', required=True, help='輸出目錄')
    
    args = parser.parse_args()
    
    analyze_field_quality(
        checkpoint_path=args.checkpoint,
        reference_h5=args.reference,
        time_index=args.time_index,
        output_dir=args.output
    )
