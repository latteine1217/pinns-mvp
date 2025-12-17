#!/usr/bin/env python
"""生成詳細的 DNS vs Prediction vs Error 對比圖"""
import sys
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import h5py

# 添加項目路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

def load_dns_data(h5_path, t_eval=25.0):
    """載入 DNS 參考數據"""
    print(f"📂 載入 DNS 數據: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        # 從配置中讀取座標信息
        L = f['config'].attrs['L']
        N = f['config'].attrs['N']

        # 生成座標網格
        x = np.linspace(0, L, N, endpoint=False)
        y = np.linspace(0, L, N, endpoint=False)

        # 讀取時間
        t = f['time'][:]

        # 找到最接近的時間索引
        t_idx = np.argmin(np.abs(t - t_eval))
        actual_t = t[t_idx]
        print(f"  評估時間: t={actual_t:.2f} (requested: {t_eval:.2f})")

        # 讀取場數據 (shape: [time, x, y])
        u = f['u'][t_idx, :, :]
        v = f['v'][t_idx, :, :]
        p = f['p'][t_idx, :, :]

    print(f"  DNS 數據形狀: {u.shape}")
    print(f"  座標範圍: x ∈ [0, {L:.4f}], y ∈ [0, {L:.4f}]")
    return x, y, u, v, p

def load_model_and_predict(checkpoint_path, x, y):
    """載入模型並進行預測"""
    from pinnx.models.fourier_mlp import PINNNet
    from pinnx.train.factory import create_model

    print(f"📥 載入 checkpoint: {checkpoint_path}")

    # 載入檢查點
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    # 獲取配置
    if 'config' in ckpt:
        cfg = ckpt['config']
    else:
        raise ValueError("No config in checkpoint")

    # 創建模型
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = create_model(cfg, device)

    # 載入權重
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    elif 'model' in ckpt:
        model.load_state_dict(ckpt['model'], strict=False)

    model.eval()
    print(f"✅ 模型載入完成 (Epoch {ckpt.get('epoch', '?')})")

    # 準備網格
    X, Y = np.meshgrid(x, y, indexing='ij')
    coords = np.stack([X.ravel(), Y.ravel()], axis=1)

    # 添加時間維度（對於 2D Kolmogorov，時間是輸入的一部分）
    t_eval = 25.0
    t_coords = np.full((coords.shape[0], 1), t_eval)
    coords_with_t = np.concatenate([coords, t_coords], axis=1)

    # 預測
    print("🔮 模型預測中...")
    with torch.no_grad():
        coords_tensor = torch.tensor(coords_with_t, dtype=torch.float32).to(device)

        # 批次預測以避免記憶體問題
        batch_size = 10000
        predictions = []
        for i in range(0, len(coords_tensor), batch_size):
            batch = coords_tensor[i:i+batch_size]
            pred = model(batch)
            predictions.append(pred.cpu().numpy())

        pred = np.concatenate(predictions, axis=0)

    # 重塑為網格形狀
    u_pred = pred[:, 0].reshape(X.shape)
    v_pred = pred[:, 1].reshape(X.shape)
    p_pred = pred[:, 2].reshape(X.shape)

    print("✅ 預測完成")
    return u_pred, v_pred, p_pred

def plot_comparison(x, y, u_dns, v_dns, p_dns, u_pred, v_pred, p_pred,
                   title, output_path):
    """繪製 DNS vs Prediction vs Error 對比圖"""

    # 計算壓力梯度範數 |∇p|
    dy, dx = np.gradient(p_dns)
    grad_p_dns = np.sqrt(dx**2 + dy**2)

    dy_pred, dx_pred = np.gradient(p_pred)
    grad_p_pred = np.sqrt(dx_pred**2 + dy_pred**2)

    # 計算誤差
    u_error = np.abs(u_pred - u_dns)
    v_error = np.abs(v_pred - v_dns)
    grad_p_error = np.abs(grad_p_pred - grad_p_dns)

    # 計算相對誤差
    u_l2 = np.sqrt(np.mean((u_pred - u_dns)**2)) / np.sqrt(np.mean(u_dns**2)) * 100
    v_l2 = np.sqrt(np.mean((v_pred - v_dns)**2)) / np.sqrt(np.mean(v_dns**2)) * 100
    grad_p_l2 = np.sqrt(np.mean((grad_p_pred - grad_p_dns)**2)) / np.sqrt(np.mean(grad_p_dns**2)) * 100

    # 創建圖表 (只有3列: DNS, Prediction, Error)
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, wspace=0.3, hspace=0.4,
                          left=0.05, right=0.95, top=0.93, bottom=0.05)

    fig.suptitle(title, fontsize=16, fontweight='bold')

    X, Y = np.meshgrid(x, y, indexing='ij')

    # 定義變量和數據
    variables = [
        ('u', u_dns, u_pred, u_error, u_l2),
        ('v', v_dns, v_pred, v_error, v_l2),
        ('|∇p|', grad_p_dns, grad_p_pred, grad_p_error, grad_p_l2)
    ]

    for row, (var_name, dns, pred, error, l2_error) in enumerate(variables):
        # 確定顏色範圍
        vmin = min(dns.min(), pred.min())
        vmax = max(dns.max(), pred.max())

        # 使用對稱的顏色範圍（如果數據跨越零點且不是梯度範數）
        if vmin < 0 and vmax > 0 and var_name != '|∇p|':
            abs_max = max(abs(vmin), abs(vmax))
            norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
            cmap_field = 'RdBu_r'
        else:
            norm = None
            cmap_field = 'viridis'

        # DNS
        ax = fig.add_subplot(gs[row, 0])
        im = ax.contourf(X, Y, dns, levels=50, cmap=cmap_field, norm=norm)
        ax.set_title(f'{var_name} (DNS)', fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Prediction
        ax = fig.add_subplot(gs[row, 1])
        im = ax.contourf(X, Y, pred, levels=50, cmap=cmap_field, norm=norm)
        ax.set_title(f'{var_name} (Prediction)', fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Error (在標題中加入 L2 誤差)
        ax = fig.add_subplot(gs[row, 2])
        im = ax.contourf(X, Y, error, levels=50, cmap='hot_r')
        ax.set_title(f'{var_name} (|Error|) - L2: {l2_error:.1f}%', fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 圖表已保存: {output_path}")
    plt.close()

def main():
    # 設定路徑
    dns_path = './data/kolmogorov_dns/dns_re50_t100.h5'
    rans_checkpoint = './checkpoints/kolmogorov_re50_kf4_K100/epoch_1000.pth'
    vanilla_checkpoint = './checkpoints/kolmogorov_re50_kf4_K100_vanilla/epoch_1000.pth'

    output_dir = Path('./results/detailed_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("DNS vs Prediction vs Error 詳細對比視覺化")
    print("="*80)

    # 載入 DNS 數據
    x, y, u_dns, v_dns, p_dns = load_dns_data(dns_path, t_eval=25.0)

    print("\n" + "="*80)
    print("處理 RANS Prior 版本")
    print("="*80)

    # 載入 RANS Prior 模型並預測
    u_pred_rans, v_pred_rans, p_pred_rans = load_model_and_predict(
        rans_checkpoint, x, y
    )

    # 繪製 RANS Prior 對比圖
    plot_comparison(
        x, y, u_dns, v_dns, p_dns,
        u_pred_rans, v_pred_rans, p_pred_rans,
        'RANS Prior: DNS vs Prediction vs Error (t=25.0, Epoch 1000)',
        output_dir / 'rans_prior_detailed_comparison.png'
    )

    print("\n" + "="*80)
    print("處理 Vanilla 版本")
    print("="*80)

    # 載入 Vanilla 模型並預測
    u_pred_vanilla, v_pred_vanilla, p_pred_vanilla = load_model_and_predict(
        vanilla_checkpoint, x, y
    )

    # 繪製 Vanilla 對比圖
    plot_comparison(
        x, y, u_dns, v_dns, p_dns,
        u_pred_vanilla, v_pred_vanilla, p_pred_vanilla,
        'Vanilla Baseline: DNS vs Prediction vs Error (t=25.0, Epoch 1000)',
        output_dir / 'vanilla_detailed_comparison.png'
    )

    print("\n" + "="*80)
    print("✅ 所有對比圖已生成！")
    print(f"📁 輸出目錄: {output_dir}")
    print("="*80)

if __name__ == '__main__':
    main()
