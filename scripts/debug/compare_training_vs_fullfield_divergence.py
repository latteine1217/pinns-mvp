#!/usr/bin/env python3
"""
驗證假設A：Loss計算域不匹配

對比 continuity_loss 在訓練點 (4096) 的計算結果
與全場點 (8192) 的實際守恆誤差

用於解釋為何 loss 下降但實際守恆誤差上升的矛盾現象
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

def compare_divergence_domains(checkpoint_path, config_path):
    """對比訓練域和全場域的散度計算"""
    
    from scripts.train import create_model, get_device
    from pinnx.dataio.channel_flow_loader import ChannelFlowLoader
    from pinnx.evals.metrics import conservation_error
    
    device = get_device('auto')
    print(f"使用設備: {device}")
    print("="*70)
    
    # === 載入模型 ===
    print("\n[1/4] 載入模型...")
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
    epoch = checkpoint.get('epoch', 'Unknown')
    print(f"  ✓ 模型載入完成 (epoch: {epoch})")
    
    # === 載入數據 ===
    print("\n[2/4] 載入數據...")
    loader = ChannelFlowLoader(config_path=config_path)
    
    # 全場數據 (8192 點)
    channel_data = loader.load_full_field_data()
    full_coords = torch.from_numpy(channel_data.sensor_points).float().to(device)
    print(f"  ✓ 全場數據點數: {full_coords.shape[0]}")
    
    # 模擬訓練時的 PDE 採樣點 (4096 點)
    # 按照 train.py 中的相同方式生成
    def generate_pde_points(n_points, domain_bounds):
        """按照訓練腳本的邏輯生成 PDE 點"""
        # 參考 train.py 的實現
        wall_clustering = config.get('wall_clustering', 0.3)
        
        # x 均勻分布
        x = torch.rand(n_points, 1) * (domain_bounds['x'][1] - domain_bounds['x'][0]) + domain_bounds['x'][0]
        
        # y 方向使用 wall clustering
        if wall_clustering > 0:
            # 生成更多靠近壁面的點
            y_uniform = torch.rand(n_points, 1) * 2 - 1  # [-1, 1]
            y_clustered = torch.sign(y_uniform) * (1 - (1 - torch.abs(y_uniform))**wall_clustering)
            y = y_clustered
        else:
            y = torch.rand(n_points, 1) * 2 - 1
        
        return torch.cat([x, y], dim=1)
    
    # 獲取域邊界
    domain_bounds = {
        'x': [0.0, 25.13272],  # 從配置檔讀取
        'y': [-1.0, 1.0]
    }
    
    # 生成訓練 PDE 點
    training_pde_points = generate_pde_points(4096, domain_bounds).to(device)
    print(f"  ✓ 訓練 PDE 點數: {training_pde_points.shape[0]}")
    
    # === 分別計算散度 ===
    print("\n[3/4] 計算散度...")
    
    def compute_divergence(coords, model, name):
        """計算給定點集的散度"""
        coords_eval = coords.clone().detach().requires_grad_(True)
        
        # 模型預測
        pred = model(coords_eval)
        pred_u = pred[:, 0]
        pred_v = pred[:, 1]
        
        # 計算梯度
        u_x = torch.autograd.grad(pred_u, coords_eval,
                                  grad_outputs=torch.ones_like(pred_u),
                                  create_graph=True, retain_graph=True)[0][:, 0]
        
        v_y = torch.autograd.grad(pred_v, coords_eval,
                                  grad_outputs=torch.ones_like(pred_v),
                                  create_graph=True, retain_graph=True)[0][:, 1]
        
        divergence = u_x + v_y
        div_array = divergence.detach().cpu().numpy()
        
        # 計算統計量
        rms_div = np.sqrt(np.mean(div_array**2))
        mean_div = np.mean(div_array)
        std_div = np.std(div_array)
        max_abs_div = np.max(np.abs(div_array))
        
        print(f"\n  === {name} 散度統計 ===")
        print(f"  點數: {coords.shape[0]}")
        print(f"  RMS 散度: {rms_div:.6e}")
        print(f"  平均散度: {mean_div:.6e}")
        print(f"  散度標準差: {std_div:.6e}")
        print(f"  最大|散度|: {max_abs_div:.6e}")
        
        # 使用官方函數驗證
        official_error = conservation_error(pred_u, pred_v, coords_eval)
        print(f"  官方 conservation_error: {official_error:.6e}")
        
        return {
            'coords': coords_eval.detach().cpu().numpy(),
            'divergence': div_array,
            'rms': rms_div,
            'mean': mean_div,
            'std': std_div,
            'max_abs': max_abs_div,
            'official_error': official_error.item() if hasattr(official_error, 'item') else float(official_error),
            'pred_u': pred_u.detach().cpu().numpy(),
            'pred_v': pred_v.detach().cpu().numpy()
        }
    
    # 計算兩個域的散度
    training_results = compute_divergence(training_pde_points, model, "訓練域 (4096點)")
    fullfield_results = compute_divergence(full_coords, model, "全場域 (8192點)")
    
    # === 對比分析 ===
    print("\n[4/4] 對比分析...")
    print("="*70)
    print("訓練域 vs 全場域散度對比:")
    print("="*70)
    
    ratio_rms = fullfield_results['rms'] / training_results['rms']
    ratio_mean = abs(fullfield_results['mean']) / abs(training_results['mean']) if training_results['mean'] != 0 else float('inf')
    ratio_max = fullfield_results['max_abs'] / training_results['max_abs']
    ratio_official = fullfield_results['official_error'] / training_results['official_error']
    
    print(f"RMS 散度比值 (全場/訓練): {ratio_rms:.3f}")
    print(f"平均散度比值 (全場/訓練): {ratio_mean:.3f}")
    print(f"最大散度比值 (全場/訓練): {ratio_max:.3f}")
    print(f"官方誤差比值 (全場/訓練): {ratio_official:.3f}")
    
    # 分析原因
    print("\n" + "="*70)
    print("分析結果:")
    print("="*70)
    
    if ratio_rms > 1.2:
        print("⚠️  全場域散度明顯大於訓練域！")
        print("   → 這解釋了為何 continuity_loss 下降但實際守恆誤差上升")
        print("   → 訓練僅優化了 4096 個特定點，未覆蓋全部 8192 點")
    elif ratio_rms < 0.8:
        print("✅ 全場域散度小於訓練域")
        print("   → Loss 計算域不匹配不是主因")
    else:
        print("🟡 兩域散度相近")
        print("   → Loss 計算域可能不是主要問題")
    
    # 檢查採樣偏差
    print("\n--- 採樣偏差分析 ---")
    
    # 檢查 y 方向分布
    training_y = training_pde_points[:, 1].cpu().numpy()
    fullfield_y = full_coords[:, 1].cpu().numpy()
    
    # 統計靠近壁面的點
    wall_threshold = 0.8
    training_near_wall = np.sum(np.abs(training_y) > wall_threshold) / len(training_y)
    fullfield_near_wall = np.sum(np.abs(fullfield_y) > wall_threshold) / len(fullfield_y)
    
    print(f"訓練域靠近壁面的點 (|y| > {wall_threshold}): {training_near_wall:.1%}")
    print(f"全場域靠近壁面的點 (|y| > {wall_threshold}): {fullfield_near_wall:.1%}")
    
    if abs(training_near_wall - fullfield_near_wall) > 0.1:
        print("⚠️  訓練域和全場域的 y 分布存在明顯差異")
        print("   → wall_clustering 可能導致採樣偏差")
    
    # === 可視化 ===
    print("\n生成對比圖...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 第一行：散度分布對比
    axes[0, 0].hist(training_results['divergence'], bins=50, alpha=0.7, 
                    label=f'Training (RMS={training_results["rms"]:.3e})', density=True)
    axes[0, 0].hist(fullfield_results['divergence'], bins=50, alpha=0.7, 
                    label=f'Full Field (RMS={fullfield_results["rms"]:.3e})', density=True)
    axes[0, 0].set_xlabel('Divergence')
    axes[0, 0].set_title('Divergence Distribution Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.5)
    
    # 採樣點分布對比 (y 方向)
    axes[0, 1].hist(training_y, bins=50, alpha=0.7, label='Training Points', density=True)
    axes[0, 1].hist(fullfield_y, bins=50, alpha=0.7, label='Full Field Points', density=True)
    axes[0, 1].set_xlabel('y coordinate')
    axes[0, 1].set_title('Y-direction Sampling Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 散度空間分布
    scatter1 = axes[0, 2].scatter(training_results['coords'][:, 0], 
                                  training_results['coords'][:, 1],
                                  c=np.abs(training_results['divergence']), 
                                  cmap='hot', s=2, alpha=0.7,
                                  vmax=np.percentile(np.abs(fullfield_results['divergence']), 99))
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    axes[0, 2].set_title('Training Points |Divergence|')
    plt.colorbar(scatter1, ax=axes[0, 2], label='|div|')
    
    # 第二行：詳細對比
    scatter2 = axes[1, 0].scatter(fullfield_results['coords'][:, 0], 
                                  fullfield_results['coords'][:, 1],
                                  c=np.abs(fullfield_results['divergence']), 
                                  cmap='hot', s=2, alpha=0.7,
                                  vmax=np.percentile(np.abs(fullfield_results['divergence']), 99))
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].set_title('Full Field |Divergence|')
    plt.colorbar(scatter2, ax=axes[1, 0], label='|div|')
    
    # 統計對比圖
    metrics = ['RMS', 'Mean', 'Std', 'Max Abs']
    training_vals = [training_results['rms'], abs(training_results['mean']), 
                     training_results['std'], training_results['max_abs']]
    fullfield_vals = [fullfield_results['rms'], abs(fullfield_results['mean']), 
                      fullfield_results['std'], fullfield_results['max_abs']]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    axes[1, 1].bar(x_pos - width/2, training_vals, width, label='Training', alpha=0.7)
    axes[1, 1].bar(x_pos + width/2, fullfield_vals, width, label='Full Field', alpha=0.7)
    axes[1, 1].set_xlabel('Metrics')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Divergence Metrics Comparison')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].legend()
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 比值圖
    ratios = [ratio_rms, ratio_mean, 1.0, ratio_max]  # ratio_std 設為 1.0 作為參考
    axes[1, 2].bar(metrics, ratios, alpha=0.7, color=['red' if r > 1.2 else 'green' if r < 0.8 else 'orange' for r in ratios])
    axes[1, 2].axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Equal')
    axes[1, 2].axhline(y=1.2, color='red', linestyle=':', alpha=0.5, label='20% threshold')
    axes[1, 2].axhline(y=0.8, color='red', linestyle=':', alpha=0.5)
    axes[1, 2].set_xlabel('Metrics')
    axes[1, 2].set_ylabel('Full Field / Training Ratio')
    axes[1, 2].set_title('Domain Ratio Analysis')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存結果
    output_dir = Path('evaluation_results_phase4b_diagnosis')
    output_dir.mkdir(exist_ok=True)
    
    plot_path = output_dir / f'training_vs_fullfield_divergence_epoch{epoch}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ 對比圖已保存: {plot_path}")
    
    # 保存數值報告
    report_path = output_dir / f'domain_comparison_epoch{epoch}.txt'
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("訓練域 vs 全場域散度對比報告\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"模型: {checkpoint_path}\n")
        f.write(f"Epoch: {epoch}\n")
        f.write(f"配置: {config_path}\n\n")
        
        f.write("--- 散度統計對比 ---\n")
        f.write(f"訓練域 (4096點) RMS 散度: {training_results['rms']:.6e}\n")
        f.write(f"全場域 (8192點) RMS 散度: {fullfield_results['rms']:.6e}\n")
        f.write(f"比值 (全場/訓練): {ratio_rms:.3f}\n\n")
        
        f.write(f"訓練域平均散度: {training_results['mean']:.6e}\n")
        f.write(f"全場域平均散度: {fullfield_results['mean']:.6e}\n")
        f.write(f"比值 (全場/訓練): {ratio_mean:.3f}\n\n")
        
        f.write(f"訓練域最大|散度|: {training_results['max_abs']:.6e}\n")
        f.write(f"全場域最大|散度|: {fullfield_results['max_abs']:.6e}\n")
        f.write(f"比值 (全場/訓練): {ratio_max:.3f}\n\n")
        
        f.write(f"訓練域官方誤差: {training_results['official_error']:.6e}\n")
        f.write(f"全場域官方誤差: {fullfield_results['official_error']:.6e}\n")
        f.write(f"比值 (全場/訓練): {ratio_official:.3f}\n\n")
        
        f.write("--- 採樣分析 ---\n")
        f.write(f"訓練域靠近壁面點比例: {training_near_wall:.1%}\n")
        f.write(f"全場域靠近壁面點比例: {fullfield_near_wall:.1%}\n")
        f.write(f"差異: {abs(training_near_wall - fullfield_near_wall):.1%}\n\n")
        
        f.write("--- 結論 ---\n")
        if ratio_rms > 1.2:
            f.write("⚠️  假設 A 證實：Loss 計算域不匹配是主要問題\n")
            f.write("   訓練僅優化了部分點，導致 loss 下降但全場守恆惡化\n")
        elif ratio_rms < 0.8:
            f.write("✅ 假設 A 否定：Loss 計算域不是主要問題\n")
        else:
            f.write("🟡 假設 A 部分支持：可能存在輕微的計算域問題\n")
    
    print(f"✓ 數值報告已保存: {report_path}")
    
    print("\n" + "="*70)
    print("假設 A 驗證完成！")
    print("="*70)
    
    return {
        'ratio_rms': ratio_rms,
        'ratio_official': ratio_official,
        'hypothesis_A_confirmed': ratio_rms > 1.2,
        'training_results': training_results,
        'fullfield_results': fullfield_results
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Checkpoint 文件路徑')
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路徑')
    
    args = parser.parse_args()
    
    result = compare_divergence_domains(args.checkpoint, args.config)
    
    print(f"\n最終結論：假設 A {'已證實' if result['hypothesis_A_confirmed'] else '未證實'}")
    print(f"RMS 散度比值: {result['ratio_rms']:.3f}")
    print(f"官方誤差比值: {result['ratio_official']:.3f}")