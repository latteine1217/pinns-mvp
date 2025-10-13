#!/usr/bin/env python3
"""
診斷質量守恆誤差的根本原因

檢查項目：
1. 模型預測值的數值範圍
2. 參考數據的數值範圍
3. 散度的空間分布
4. 高散度區域定位
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def diagnose_conservation(checkpoint_path, config_path=None):
    """診斷質量守恆問題"""
    
    from scripts.train import create_model, get_device
    from pinnx.dataio.channel_flow_loader import ChannelFlowLoader
    from pinnx.evals.metrics import conservation_error
    import yaml
    
    device = get_device('auto')
    print(f"使用設備: {device}")
    print("="*60)
    
    # === 載入模型 ===
    print("\n[1/5] 載入模型...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'config' in checkpoint:
        config = checkpoint['config']
        print("  ✓ 從 checkpoint 載入配置")
    elif config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"  ✓ 從文件載入配置: {config_path}")
    else:
        raise ValueError("需要提供配置")
    
    model = create_model(config, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  ✓ 模型載入完成 (epoch: {checkpoint.get('epoch', 'Unknown')})")
    
    # === 載入數據 ===
    print("\n[2/5] 載入參考數據...")
    loader = ChannelFlowLoader(config_path=config_path)
    channel_data = loader.load_full_field_data()
    
    coords = torch.from_numpy(channel_data.sensor_points).float().to(device)
    ref_u = torch.from_numpy(channel_data.sensor_data['u']).float().to(device)
    ref_v = torch.from_numpy(channel_data.sensor_data['v']).float().to(device)
    ref_p = torch.from_numpy(channel_data.sensor_data['p']).float().to(device)
    
    print(f"  ✓ 數據形狀: {coords.shape}")
    print(f"  ✓ 參考數據 u 範圍: [{ref_u.min():.6f}, {ref_u.max():.6f}]")
    print(f"  ✓ 參考數據 v 範圍: [{ref_v.min():.6f}, {ref_v.max():.6f}]")
    print(f"  ✓ 參考數據 p 範圍: [{ref_p.min():.6f}, {ref_p.max():.6f}]")
    
    # === 模型預測 ===
    print("\n[3/5] 執行模型預測...")
    coords_eval = coords.clone().detach().requires_grad_(True)
    
    with torch.no_grad():
        pred_no_grad = model(coords_eval)
    
    pred_with_grad = model(coords_eval)
    
    pred_u = pred_with_grad[:, 0]
    pred_v = pred_with_grad[:, 1]
    pred_p = pred_with_grad[:, 2] if pred_with_grad.shape[1] > 2 else pred_with_grad[:, 1]
    
    print(f"  ✓ 預測 u 範圍: [{pred_u.min():.6f}, {pred_u.max():.6f}]")
    print(f"  ✓ 預測 v 範圍: [{pred_v.min():.6f}, {pred_v.max():.6f}]")
    print(f"  ✓ 預測 p 範圍: [{pred_p.min():.6f}, {pred_p.max():.6f}]")
    print(f"  ✓ u.grad_fn: {pred_u.grad_fn}")
    print(f"  ✓ v.grad_fn: {pred_v.grad_fn}")
    
    # === 計算散度 ===
    print("\n[4/5] 計算散度分布...")
    
    # 計算梯度
    u_x = torch.autograd.grad(pred_u, coords_eval, 
                               grad_outputs=torch.ones_like(pred_u),
                               create_graph=True, retain_graph=True)[0][:, 0]
    
    v_y = torch.autograd.grad(pred_v, coords_eval,
                               grad_outputs=torch.ones_like(pred_v),
                               create_graph=True, retain_graph=True)[0][:, 1]
    
    divergence = u_x + v_y
    div_array = divergence.detach().cpu().numpy()
    
    print(f"  ✓ 散度範圍: [{div_array.min():.6f}, {div_array.max():.6f}]")
    print(f"  ✓ 散度平均值: {div_array.mean():.6f}")
    print(f"  ✓ 散度標準差: {div_array.std():.6f}")
    print(f"  ✓ 散度 RMS: {np.sqrt(np.mean(div_array**2)):.6f}")
    
    # 使用官方函數驗證
    official_error = conservation_error(pred_u, pred_v, coords_eval)
    print(f"  ✓ 官方 conservation_error: {official_error:.6e}")
    
    # === 統計分析 ===
    print("\n[5/5] 散度統計分析...")
    
    # 計算百分位數
    percentiles = [50, 75, 90, 95, 99]
    abs_div = np.abs(div_array)
    for p in percentiles:
        val = np.percentile(abs_div, p)
        print(f"  • {p}% 的點 |散度| < {val:.6f}")
    
    # 找出高散度點
    threshold = np.percentile(abs_div, 95)
    high_div_indices = np.where(abs_div > threshold)[0]
    print(f"\n  高散度點（top 5%）: {len(high_div_indices)} 個點")
    
    # 檢查高散度點的位置
    coords_np = coords.cpu().numpy()
    high_div_coords = coords_np[high_div_indices]
    
    print(f"  • x 範圍: [{high_div_coords[:, 0].min():.3f}, {high_div_coords[:, 0].max():.3f}]")
    print(f"  • y 範圍: [{high_div_coords[:, 1].min():.3f}, {high_div_coords[:, 1].max():.3f}]")
    
    # 檢查是否集中在邊界
    y_coords = high_div_coords[:, 1]
    y_min, y_max = coords_np[:, 1].min(), coords_np[:, 1].max()
    near_boundary = np.sum((y_coords <= y_min + 0.1) | (y_coords >= y_max - 0.1))
    print(f"  • 靠近邊界（y ± 0.1）的點: {near_boundary} / {len(high_div_indices)} ({100*near_boundary/len(high_div_indices):.1f}%)")
    
    # === 誤差分析 ===
    print("\n" + "="*60)
    print("誤差統計:")
    print("="*60)
    
    u_error = torch.abs(pred_u - ref_u).detach().cpu().numpy()
    v_error = torch.abs(pred_v - ref_v).detach().cpu().numpy()
    
    print(f"u 絕對誤差: mean={u_error.mean():.6f}, max={u_error.max():.6f}")
    print(f"v 絕對誤差: mean={v_error.mean():.6f}, max={v_error.max():.6f}")
    
    u_rel_error = u_error / (np.abs(ref_u.cpu().numpy()) + 1e-8)
    v_rel_error = v_error / (np.abs(ref_v.cpu().numpy()) + 1e-8)
    
    print(f"u 相對誤差: mean={u_rel_error.mean():.6f}, median={np.median(u_rel_error):.6f}")
    print(f"v 相對誤差: mean={v_rel_error.mean():.6f}, median={np.median(v_rel_error):.6f}")
    
    # === 可視化 ===
    print("\n" + "="*60)
    print("生成診斷圖...")
    print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 第一行：預測 vs 參考
    axes[0, 0].hist(pred_u.detach().cpu().numpy(), bins=50, alpha=0.7, label='Predicted', density=True)
    axes[0, 0].hist(ref_u.cpu().numpy(), bins=50, alpha=0.7, label='Reference', density=True)
    axes[0, 0].set_xlabel('u velocity')
    axes[0, 0].set_title('u Velocity Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(pred_v.detach().cpu().numpy(), bins=50, alpha=0.7, label='Predicted', density=True)
    axes[0, 1].hist(ref_v.cpu().numpy(), bins=50, alpha=0.7, label='Reference', density=True)
    axes[0, 1].set_xlabel('v velocity')
    axes[0, 1].set_title('v Velocity Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].hist(div_array, bins=100, edgecolor='black', alpha=0.7)
    axes[0, 2].axvline(0, color='red', linestyle='--', label='Zero divergence')
    axes[0, 2].set_xlabel('Divergence')
    axes[0, 2].set_title(f'Divergence Distribution (RMS={np.sqrt(np.mean(div_array**2)):.3f})')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 第二行：誤差分析
    axes[1, 0].scatter(ref_u.cpu().numpy(), pred_u.detach().cpu().numpy(), 
                       alpha=0.5, s=1)
    axes[1, 0].plot([ref_u.min().item(), ref_u.max().item()], 
                    [ref_u.min().item(), ref_u.max().item()], 
                    'r--', label='Perfect prediction')
    axes[1, 0].set_xlabel('Reference u')
    axes[1, 0].set_ylabel('Predicted u')
    axes[1, 0].set_title('u Velocity: Predicted vs Reference')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].scatter(ref_v.cpu().numpy(), pred_v.detach().cpu().numpy(), 
                       alpha=0.5, s=1)
    axes[1, 1].plot([ref_v.min().item(), ref_v.max().item()], 
                    [ref_v.min().item(), ref_v.max().item()], 
                    'r--', label='Perfect prediction')
    axes[1, 1].set_xlabel('Reference v')
    axes[1, 1].set_ylabel('Predicted v')
    axes[1, 1].set_title('v Velocity: Predicted vs Reference')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 散度空間分布（假設是 2D 場）
    if coords.shape[1] >= 2:
        scatter = axes[1, 2].scatter(coords_np[:, 0], coords_np[:, 1], 
                                     c=abs_div, cmap='hot', s=1, vmax=np.percentile(abs_div, 99))
        axes[1, 2].set_xlabel('x')
        axes[1, 2].set_ylabel('y')
        axes[1, 2].set_title('|Divergence| Spatial Distribution')
        plt.colorbar(scatter, ax=axes[1, 2], label='|div|')
    
    plt.tight_layout()
    
    output_dir = Path('evaluation_results_phase4b_diagnosis')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'conservation_diagnosis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 診斷圖已保存: {output_path}")
    
    # 保存數值報告
    report_path = output_dir / 'diagnosis_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("質量守恆誤差診斷報告\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"模型: {checkpoint_path}\n")
        f.write(f"Epoch: {checkpoint.get('epoch', 'Unknown')}\n")
        f.write(f"數據點數: {coords.shape[0]}\n\n")
        
        f.write("--- 數值範圍 ---\n")
        f.write(f"參考 u: [{ref_u.min():.6f}, {ref_u.max():.6f}]\n")
        f.write(f"預測 u: [{pred_u.min():.6f}, {pred_u.max():.6f}]\n")
        f.write(f"參考 v: [{ref_v.min():.6f}, {ref_v.max():.6f}]\n")
        f.write(f"預測 v: [{pred_v.min():.6f}, {pred_v.max():.6f}]\n\n")
        
        f.write("--- 散度統計 ---\n")
        f.write(f"RMS 散度: {np.sqrt(np.mean(div_array**2)):.6e}\n")
        f.write(f"平均散度: {div_array.mean():.6e}\n")
        f.write(f"散度標準差: {div_array.std():.6e}\n")
        f.write(f"散度範圍: [{div_array.min():.6f}, {div_array.max():.6f}]\n\n")
        
        f.write("--- 誤差統計 ---\n")
        f.write(f"u 平均絕對誤差: {u_error.mean():.6f}\n")
        f.write(f"v 平均絕對誤差: {v_error.mean():.6f}\n")
        f.write(f"u 平均相對誤差: {u_rel_error.mean():.6f}\n")
        f.write(f"v 平均相對誤差: {v_rel_error.mean():.6f}\n")
    
    print(f"✓ 數值報告已保存: {report_path}")
    
    print("\n" + "="*60)
    print("診斷完成！")
    print("="*60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default=None)
    
    args = parser.parse_args()
    
    diagnose_conservation(args.checkpoint, args.config)
