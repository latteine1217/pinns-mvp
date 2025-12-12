#!/usr/bin/env python3
"""
PINNs 壓力梯度誤差論文圖表生成工具
====================================

功能：
1. 評估 PINNs（Vanilla vs Full）的壓力梯度重建精度
2. 與 DNS 真值比較 ∇P 誤差（而非 P 本身）
3. 生成論文級圖表（∂p/∂x, ∂p/∂y 誤差分布、統計比較）

使用方式：
---------
python scripts/generate_paper_figures_pinns_pressure.py

輸出：
-----
- results/paper_figures_pinns/
  ├── table_pressure_gradient_errors.png
  ├── fig_pressure_gradient_comparison.png
  ├── fig_pressure_gradient_profiles.png
  └── pressure_gradient_summary.json

作者：PINNs-MVP 團隊
日期：2025-12-12
"""

import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import json
import logging
from typing import Dict, Tuple
import sys

# 添加模組路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinnx.evals.metrics import pressure_gradient_from_finite_diff, relative_L2, rmse_metrics
from pinnx.train.config_loader import load_config
from pinnx.train.factory import create_model, create_physics

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def load_dns_reference(dns_path: Path, time_range: Tuple[float, float] = (20.0, 40.0)) -> Dict:
    """載入 DNS 參考數據"""
    logging.info(f"載入 DNS 參考數據: {dns_path}")
    
    with h5py.File(dns_path, 'r') as f:
        time = np.array(f['time'])
        mask = (time >= time_range[0]) & (time <= time_range[1])
        
        # 時間平均
        u = np.mean(np.array(f['u'])[mask], axis=0)
        v = np.mean(np.array(f['v'])[mask], axis=0)
        p = np.mean(np.array(f['p'])[mask], axis=0)
        
        # 網格
        N = u.shape[0]
        L = f['config'].attrs.get('L', 2 * np.pi)
        x = np.linspace(0, L, N, endpoint=False)
        y = np.linspace(0, L, N, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        logging.info(f"  DNS: N={N}×{N}, t∈[{time[mask][0]:.1f}, {time[mask][-1]:.1f}], {np.sum(mask)} 幀平均")
    
    return {
        'u': u, 'v': v, 'p': p,
        'x': x, 'y': y, 'X': X, 'Y': Y,
        'L': L, 'N': N
    }


def evaluate_pinn_checkpoint(checkpoint_path: Path, config_path: Path, 
                              dns_data: Dict, device: str = 'cpu') -> Dict:
    """評估 PINNs 檢查點的壓力梯度"""
    logging.info(f"評估檢查點: {checkpoint_path.name}")
    
    # 載入檢查點（優先使用檢查點內的配置）
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 優先使用檢查點內保存的配置（更準確）
    if 'config' in checkpoint:
        config = checkpoint['config']
        logging.info("  使用檢查點內保存的配置")
    else:
        config = load_config(str(config_path))
        logging.info("  使用外部配置檔案")
    
    torch_device = torch.device(device)
    model = create_model(config, device=torch_device)
    physics = create_physics(config, device=torch_device)
    
    # 載入權重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 準備評估網格（使用 DNS 網格）
    X, Y = dns_data['X'], dns_data['Y']
    x_flat = X.flatten()
    y_flat = Y.flatten()
    N = dns_data['N']
    
    # 使用穩態中點時間
    t_eval = 30.0
    t_flat = np.full_like(x_flat, t_eval)
    
    # 組合輸入 (t, x, y)
    coords = np.stack([t_flat, x_flat, y_flat], axis=1).astype(np.float32)
    coords_tensor = torch.from_numpy(coords).to(device)
    
    # 前向預測
    with torch.no_grad():
        pred_tensor = model(coords_tensor)  # [N*N, 3] -> (u, v, p)
    
    pred_np = pred_tensor.cpu().numpy()
    
    # 反標準化（如果檢查點包含標準化參數）
    if 'normalization' in checkpoint:
        norm = checkpoint['normalization']
        means = norm['means']
        stds = norm['stds']
        var_order = norm.get('variable_order', ['u', 'v', 'p'])
        
        logging.info(f"  反標準化: means={list(means.values())}, stds={list(stds.values())}")
        
        for i, var in enumerate(var_order):
            pred_np[:, i] = pred_np[:, i] * stds[var] + means[var]
    
    # 重塑為 2D 場
    u_pred = pred_np[:, 0].reshape(N, N)
    v_pred = pred_np[:, 1].reshape(N, N)
    p_pred = pred_np[:, 2].reshape(N, N)
    
    logging.info(f"  預測場範圍: u∈[{u_pred.min():.4f}, {u_pred.max():.4f}], "
                 f"p∈[{p_pred.min():.4f}, {p_pred.max():.4f}]")
    
    return {
        'u': u_pred,
        'v': v_pred,
        'p': p_pred,
    }


def compute_pressure_gradient_errors(pred: Dict, ref: Dict, coords: Dict) -> Tuple[Dict, Dict, Dict]:
    """計算壓力梯度誤差"""
    # 計算壓力梯度
    pred_grad = pressure_gradient_from_finite_diff(pred['p'], coords)
    ref_grad = pressure_gradient_from_finite_diff(ref['p'], coords)
    
    # 計算誤差
    errors = {}
    for axis, key in [('x', 'dpdx'), ('y', 'dpdy')]:
        # L2 相對誤差
        l2_err = np.linalg.norm(pred_grad[key] - ref_grad[key]) / \
                 (np.linalg.norm(ref_grad[key]) + 1e-12)
        
        # RMSE
        rmse = np.sqrt(np.mean((pred_grad[key] - ref_grad[key])**2))
        
        # 統計量
        pred_mean = float(np.mean(pred_grad[key]))
        pred_std = float(np.std(pred_grad[key]))
        ref_mean = float(np.mean(ref_grad[key]))
        ref_std = float(np.std(ref_grad[key]))
        
        errors[f'{key}_l2'] = float(l2_err)
        errors[f'{key}_rmse'] = float(rmse)
        errors[f'{key}_pred_mean'] = pred_mean
        errors[f'{key}_pred_std'] = pred_std
        errors[f'{key}_ref_mean'] = ref_mean
        errors[f'{key}_ref_std'] = ref_std
    
    # 綜合壓力梯度誤差
    errors['grad_p_l2'] = float(np.mean([errors['dpdx_l2'], errors['dpdy_l2']]))
    
    # 速度場誤差（用於對比）
    errors['u_l2'] = float(np.linalg.norm(pred['u'] - ref['u']) / 
                           (np.linalg.norm(ref['u']) + 1e-12))
    errors['v_l2'] = float(np.linalg.norm(pred['v'] - ref['v']) / 
                           (np.linalg.norm(ref['v']) + 1e-12))
    
    # 壓力場誤差（參考用，可能有常數偏移）
    errors['p_l2'] = float(np.linalg.norm(pred['p'] - ref['p']) / 
                           (np.linalg.norm(ref['p']) + 1e-12))
    
    return errors, pred_grad, ref_grad


def plot_pressure_gradient_comparison(results: Dict, dns_data: Dict, 
                                      output_dir: Path, dpi: int = 150):
    """繪製壓力梯度比較圖"""
    logging.info("繪製壓力梯度比較圖...")
    
    # 解析數據
    vanilla_grad = results['vanilla']['grad']
    full_grad = results['full']['grad']
    ref_grad = results['reference']['grad']
    X, Y = dns_data['X'], dns_data['Y']
    
    # === 圖 1: 壓力梯度場比較（2×3 佈局）===
    fig1 = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig1, hspace=0.25, wspace=0.3)
    
    # ∂p/∂x 比較
    axes_dpdx = [fig1.add_subplot(gs[0, i]) for i in range(3)]
    titles_dpdx = ['DNS ∂p/∂x (Reference)', 'Vanilla PINNs ∂p/∂x', 'Full PINNs ∂p/∂x']
    data_dpdx = [ref_grad['dpdx'], vanilla_grad['dpdx'], full_grad['dpdx']]
    
    vmin_x = min(d.min() for d in data_dpdx)
    vmax_x = max(d.max() for d in data_dpdx)
    
    for ax, title, data in zip(axes_dpdx, titles_dpdx, data_dpdx):
        im = ax.contourf(X, Y, data, levels=20, cmap='RdBu_r', vmin=vmin_x, vmax=vmax_x)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='∂p/∂x')
    
    # ∂p/∂y 比較
    axes_dpdy = [fig1.add_subplot(gs[1, i]) for i in range(3)]
    titles_dpdy = ['DNS ∂p/∂y (Reference)', 'Vanilla PINNs ∂p/∂y', 'Full PINNs ∂p/∂y']
    data_dpdy = [ref_grad['dpdy'], vanilla_grad['dpdy'], full_grad['dpdy']]
    
    vmin_y = min(d.min() for d in data_dpdy)
    vmax_y = max(d.max() for d in data_dpdy)
    
    for ax, title, data in zip(axes_dpdy, titles_dpdy, data_dpdy):
        im = ax.contourf(X, Y, data, levels=20, cmap='RdBu_r', vmin=vmin_y, vmax=vmax_y)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='∂p/∂y')
    
    fig1.suptitle('Pressure Gradient Comparison: PINNs vs DNS', 
                  fontsize=16, fontweight='bold', y=0.98)
    
    fig1_path = output_dir / 'fig_pressure_gradient_fields.png'
    fig1.savefig(fig1_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig1)
    logging.info(f"  ✅ 壓力梯度場圖已保存: {fig1_path.name}")
    
    # === 圖 2: 誤差統計比較 ===
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左圖：L2 誤差比較
    ax1 = axes[0]
    categories = ['∂p/∂x', '∂p/∂y', 'Avg ∇p']
    vanilla_errs = [
        results['vanilla']['errors']['dpdx_l2'] * 100,
        results['vanilla']['errors']['dpdy_l2'] * 100,
        results['vanilla']['errors']['grad_p_l2'] * 100
    ]
    full_errs = [
        results['full']['errors']['dpdx_l2'] * 100,
        results['full']['errors']['dpdy_l2'] * 100,
        results['full']['errors']['grad_p_l2'] * 100
    ]
    
    x_pos = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x_pos - width/2, vanilla_errs, width, label='Vanilla PINNs', 
            color='#FF6B6B', edgecolor='black', linewidth=1.2)
    ax1.bar(x_pos + width/2, full_errs, width, label='Full PINNs', 
            color='#4ECDC4', edgecolor='black', linewidth=1.2)
    
    ax1.set_xlabel('Gradient Component', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Relative L2 Error (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Pressure Gradient Reconstruction Error', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(categories)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # 右圖：梯度統計比較
    ax2 = axes[1]
    
    metrics = ['dpdx Mean', 'dpdx Std', 'dpdy Mean', 'dpdy Std']
    vanilla_stats = [
        results['vanilla']['errors']['dpdx_pred_mean'],
        results['vanilla']['errors']['dpdx_pred_std'],
        results['vanilla']['errors']['dpdy_pred_mean'],
        results['vanilla']['errors']['dpdy_pred_std']
    ]
    full_stats = [
        results['full']['errors']['dpdx_pred_mean'],
        results['full']['errors']['dpdx_pred_std'],
        results['full']['errors']['dpdy_pred_mean'],
        results['full']['errors']['dpdy_pred_std']
    ]
    ref_stats = [
        results['reference']['errors']['dpdx_ref_mean'],
        results['reference']['errors']['dpdx_ref_std'],
        results['reference']['errors']['dpdy_ref_mean'],
        results['reference']['errors']['dpdy_ref_std']
    ]
    
    x_pos2 = np.arange(len(metrics))
    width2 = 0.25
    
    ax2.bar(x_pos2 - width2, ref_stats, width2, label='DNS (Reference)', 
            color='#95E1D3', edgecolor='black', linewidth=1.2)
    ax2.bar(x_pos2, vanilla_stats, width2, label='Vanilla PINNs', 
            color='#FF6B6B', edgecolor='black', linewidth=1.2)
    ax2.bar(x_pos2 + width2, full_stats, width2, label='Full PINNs', 
            color='#4ECDC4', edgecolor='black', linewidth=1.2)
    
    ax2.set_xlabel('Statistical Metric', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax2.set_title('Pressure Gradient Statistics', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos2)
    ax2.set_xticklabels(metrics, rotation=15, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    fig2.tight_layout()
    fig2_path = output_dir / 'fig_pressure_gradient_statistics.png'
    fig2.savefig(fig2_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig2)
    logging.info(f"  ✅ 統計比較圖已保存: {fig2_path.name}")


def plot_summary_table(results: Dict, output_dir: Path, dpi: int = 150):
    """繪製誤差總結表格"""
    logging.info("生成誤差總結表格...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # 準備表格數據
    headers = ['Metric', 'Vanilla PINNs', 'Full PINNs', 'Improvement']
    
    vanilla_errs = results['vanilla']['errors']
    full_errs = results['full']['errors']
    
    def calc_improvement(v, f):
        if v == 0:
            return '-'
        return f"{(v - f) / v * 100:.1f}%"
    
    data = [
        ['∂p/∂x L2 (%)', f"{vanilla_errs['dpdx_l2']*100:.2f}", 
         f"{full_errs['dpdx_l2']*100:.2f}", 
         calc_improvement(vanilla_errs['dpdx_l2'], full_errs['dpdx_l2'])],
        
        ['∂p/∂y L2 (%)', f"{vanilla_errs['dpdy_l2']*100:.2f}", 
         f"{full_errs['dpdy_l2']*100:.2f}",
         calc_improvement(vanilla_errs['dpdy_l2'], full_errs['dpdy_l2'])],
        
        ['Avg ∇p L2 (%)', f"{vanilla_errs['grad_p_l2']*100:.2f}", 
         f"{full_errs['grad_p_l2']*100:.2f}",
         calc_improvement(vanilla_errs['grad_p_l2'], full_errs['grad_p_l2'])],
        
        ['---', '---', '---', '---'],
        
        ['u L2 (%)', f"{vanilla_errs['u_l2']*100:.2f}", 
         f"{full_errs['u_l2']*100:.2f}",
         calc_improvement(vanilla_errs['u_l2'], full_errs['u_l2'])],
        
        ['v L2 (%)', f"{vanilla_errs['v_l2']*100:.2f}", 
         f"{full_errs['v_l2']*100:.2f}",
         calc_improvement(vanilla_errs['v_l2'], full_errs['v_l2'])],
        
        ['p L2 (%)* ', f"{vanilla_errs['p_l2']*100:.2f}", 
         f"{full_errs['p_l2']*100:.2f}",
         '(Const offset)'],
    ]
    
    table = ax.table(cellText=data, colLabels=headers, cellLoc='center',
                    loc='center', colWidths=[0.3, 0.2, 0.2, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # 標題行格式
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # 數據行格式（交替顏色）
    for i in range(1, len(data) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#E7E6E6')
            else:
                cell.set_facecolor('white')
            
            # 分隔線
            if data[i-1][0] == '---':
                cell.set_facecolor('#D3D3D3')
    
    # 標題
    fig.suptitle('PINNs Pressure Gradient Error Summary\n(Vanilla vs Full Features)', 
                 fontsize=15, fontweight='bold', y=0.95)
    
    # 註解
    fig.text(0.5, 0.05, 
             '* Pressure absolute error includes arbitrary constant offset (less meaningful than gradient error)',
             ha='center', fontsize=9, style='italic', color='gray')
    
    table_path = output_dir / 'table_pressure_gradient_summary.png'
    fig.savefig(table_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"  ✅ 總結表格已保存: {table_path.name}")


def main():
    """主程式"""
    logging.info("=" * 70)
    logging.info("PINNs 壓力梯度誤差論文圖表生成")
    logging.info("=" * 70)
    
    # 配置路徑
    base_dir = Path(__file__).parent.parent
    dns_path = base_dir / 'data/kolmogorov_dns/dns_re50_t100.h5'
    
    checkpoints = {
        'vanilla': base_dir / 'checkpoints/kolmogorov_re50_kf4_K100_vanilla/best_model.pth',
        'full': base_dir / 'checkpoints/kolmogorov_re50_kf4_K100_full/best_model.pth'
    }
    
    configs = {
        'vanilla': base_dir / 'configs/kolmogorov_re50_kf4_K100_vanilla_1k.yml',
        'full': base_dir / 'configs/kolmogorov_re50_kf4_K100_full_1k.yml'
    }
    
    output_dir = base_dir / 'results/paper_figures_pinns'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 檢查檔案存在
    if not dns_path.exists():
        logging.error(f"❌ DNS 數據不存在: {dns_path}")
        return
    
    for name, path in checkpoints.items():
        if not path.exists():
            logging.error(f"❌ 檢查點不存在: {path}")
            return
    
    # 1. 載入 DNS 參考數據
    dns_data = load_dns_reference(dns_path)
    
    # 計算 DNS 壓力梯度（參考）
    coords = {'x': dns_data['x'], 'y': dns_data['y']}
    ref_grad = pressure_gradient_from_finite_diff(dns_data['p'], coords)
    
    ref_errors = {
        'dpdx_ref_mean': float(np.mean(ref_grad['dpdx'])),
        'dpdx_ref_std': float(np.std(ref_grad['dpdx'])),
        'dpdy_ref_mean': float(np.mean(ref_grad['dpdy'])),
        'dpdy_ref_std': float(np.std(ref_grad['dpdy']))
    }
    
    # 2. 評估兩個檢查點
    results = {
        'reference': {
            'p': dns_data['p'],
            'grad': ref_grad,
            'errors': ref_errors
        }
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"使用設備: {device}")
    
    for name in ['vanilla', 'full']:
        pred = evaluate_pinn_checkpoint(checkpoints[name], configs[name], 
                                        dns_data, device=device)
        
        errors, pred_grad, _ = compute_pressure_gradient_errors(
            pred, dns_data, coords
        )
        
        results[name] = {
            'pred': pred,
            'grad': pred_grad,
            'errors': errors
        }
        
        logging.info(f"  {name.upper()} 結果:")
        logging.info(f"    ∇p L2 error: {errors['grad_p_l2']*100:.2f}%")
        logging.info(f"    u L2 error: {errors['u_l2']*100:.2f}%")
    
    # 3. 生成圖表
    plot_pressure_gradient_comparison(results, dns_data, output_dir)
    plot_summary_table(results, output_dir)
    
    # 4. 保存 JSON 摘要
    summary = {
        'vanilla': results['vanilla']['errors'],
        'full': results['full']['errors'],
        'reference': results['reference']['errors']
    }
    
    json_path = output_dir / 'pressure_gradient_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info("")
    logging.info("=" * 70)
    logging.info("✅ 所有圖表已生成！")
    logging.info(f"   輸出目錄: {output_dir}")
    logging.info("   生成檔案:")
    logging.info("     - fig_pressure_gradient_fields.png")
    logging.info("     - fig_pressure_gradient_statistics.png")
    logging.info("     - table_pressure_gradient_summary.png")
    logging.info("     - pressure_gradient_summary.json")
    logging.info("=" * 70)


if __name__ == '__main__':
    main()
