#!/usr/bin/env python3
"""
驗證剩餘假設D和E：
- 假設D：RANS Prior 拖累 (prior_weight=0.7)
- 假設E：PDE 採樣偏差 (wall_clustering=0.3)

分析這兩個因素如何影響模型預測範圍和質量守恆
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

def verify_remaining_hypotheses(checkpoint_path, config_path):
    """驗證假設D和E"""
    
    from scripts.train import create_model, get_device
    from pinnx.dataio.channel_flow_loader import ChannelFlowLoader
    
    device = get_device('auto')
    print(f"使用設備: {device}")
    print("="*70)
    
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
    epoch = checkpoint.get('epoch', 'Unknown')
    print(f"  ✓ 模型載入完成 (epoch: {epoch})")
    
    # === 載入數據 ===
    print("\n[2/5] 載入數據...")
    loader = ChannelFlowLoader(config_path=config_path)
    
    # 全場數據
    field_dataset = loader.load_full_field_data()
    full_coords_np, full_fields = field_dataset.to_points(order=('x', 'y', 'z'))
    full_coords = torch.from_numpy(full_coords_np).float().to(device)
    full_u = torch.from_numpy(full_fields['u']).float().to(device)
    full_v = torch.from_numpy(full_fields['v']).float().to(device)
    
    print(f"  ✓ 全場數據點數: {full_coords.shape[0]}")
    
    # === 假設D：分析RANS Prior影響 ===
    print("\n[3/5] 假設D驗證：RANS Prior拖累分析...")
    print("="*70)
    
    # 檢查配置中的prior設置
    prior_weight = config.get('losses', {}).get('prior_weight', 0.0)
    prior_type = config.get('data', {}).get('lowfi', {}).get('type', 'none')
    
    print(f"Prior 權重: {prior_weight}")
    print(f"Prior 類型: {prior_type}")
    
    # 載入帶有prior的數據
    sensor_data = loader.load_sensor_data(strategy='qr_pivot', K=50)
    sensor_data_with_prior = loader.add_lowfi_prior(sensor_data)
    
    hypothesis_d_indicators = []
    
    # 初始化變量以避免未定義錯誤
    u_suppression = 0.0
    v_suppression = 0.0
    high_speed_threshold = 12.0
    
    if sensor_data_with_prior.has_lowfi_prior():
        prior_samples = sensor_data_with_prior.lowfi_prior
        prior_fields = prior_samples.values if prior_samples is not None else {}
        if prior_fields is not None and isinstance(prior_fields, dict):
            print(f"  ✓ 載入了先驗場: {list(prior_fields.keys())}")
            
            if 'u' in prior_fields and 'v' in prior_fields:
                prior_u = prior_fields['u']
                prior_v = prior_fields['v']
                
                # 分析prior的統計特性
                print(f"\nPrior場統計:")
                print(f"  u: [{np.min(prior_u):.3f}, {np.max(prior_u):.3f}], mean={np.mean(prior_u):.3f}")
                print(f"  v: [{np.min(prior_v):.3f}, {np.max(prior_v):.3f}], mean={np.mean(prior_v):.3f}")
                
                # 與真實場對比
                sensor_coords = sensor_data.sensor_points
                sensor_u_true = sensor_data.sensor_data['u']
                sensor_v_true = sensor_data.sensor_data['v']
                
                print(f"\n感測點真實場統計:")
                print(f"  u: [{np.min(sensor_u_true):.3f}, {np.max(sensor_u_true):.3f}], mean={np.mean(sensor_u_true):.3f}")
                print(f"  v: [{np.min(sensor_v_true):.3f}, {np.max(sensor_v_true):.3f}], mean={np.mean(sensor_v_true):.3f}")
                
                # 計算prior與真實場的差異
                u_diff = np.abs(prior_u - sensor_u_true)
                v_diff = np.abs(prior_v - sensor_v_true)
                
                print(f"\nPrior誤差:")
                print(f"  u 平均絕對誤差: {np.mean(u_diff):.3f}")
                print(f"  v 平均絕對誤差: {np.mean(v_diff):.3f}")
                
                # 檢查prior是否壓制了變化範圍
                u_range_prior = np.max(prior_u) - np.min(prior_u)
                v_range_prior = np.max(prior_v) - np.min(prior_v)
                u_range_true = np.max(sensor_u_true) - np.min(sensor_u_true)
                v_range_true = np.max(sensor_v_true) - np.min(sensor_v_true)
                
                u_suppression = u_range_prior / u_range_true if u_range_true > 0 else 0
                v_suppression = v_range_prior / v_range_true if v_range_true > 0 else 0
                
                print(f"\n範圍壓制比率:")
                print(f"  u: {u_suppression:.3f} ({'壓制' if u_suppression < 0.8 else '正常'})")
                print(f"  v: {v_suppression:.3f} ({'壓制' if v_suppression < 0.8 else '正常'})")
                
                # 檢查是否存在拖累跡象
                if prior_weight > 0.5:
                    hypothesis_d_indicators.append(f"Prior權重偏高 ({prior_weight})")
                
                if np.mean(u_diff) > 2.0:  # u的平均誤差>2
                    hypothesis_d_indicators.append("Prior u場誤差較大")
                
                if v_suppression < 0.5:  # v範圍被嚴重壓制
                    hypothesis_d_indicators.append("Prior嚴重壓制v場變化範圍")
                
                if u_suppression < 0.7:  # u範圍被明顯壓制
                    hypothesis_d_indicators.append("Prior明顯壓制u場變化範圍")
            else:
                print("  ⚠️ Prior 場中缺少 u 或 v 數據")
                hypothesis_d_indicators.append("Prior 場數據不完整")
        else:
            print("  ⚠️ Prior 場為空或格式錯誤")
            hypothesis_d_indicators.append("Prior 場數據無效")
                
    else:
        print("  ⚠️ 未找到先驗場數據")
        hypothesis_d_indicators.append("無先驗場數據，無法驗證")
    
    # === 假設E：PDE採樣偏差分析 ===
    print("\n[4/5] 假設E驗證：PDE採樣偏差分析...")
    print("="*70)
    
    # 檢查採樣配置
    sampling_config = config.get('training', {}).get('sampling', {})
    pde_points = sampling_config.get('pde_points', 4096)
    wall_clustering = sampling_config.get('wall_clustering', 0.0)
    
    print(f"PDE 採樣點數: {pde_points}")
    print(f"Wall clustering: {wall_clustering}")
    
    # 模擬PDE採樣點分佈（基於配置重現採樣邏輯）
    print("\n模擬PDE採樣分佈...")
    
    # 從域配置獲取邊界
    domain_config = config.get('domain', {})
    x_range = domain_config.get('x_range', [0.0, 25.13])
    y_range = domain_config.get('y_range', [-1.0, 1.0])
    
    # 模擬採樣（簡化版本）
    np.random.seed(42)  # 固定隨機種子以便重現
    
    if wall_clustering > 0:
        # 有wall clustering的採樣
        # 在y方向使用bias採樣，更多點靠近壁面
        n_points = pde_points
        
        # x方向均勻採樣
        x_samples = np.random.uniform(x_range[0], x_range[1], n_points)
        
        # y方向帶wall clustering的採樣
        # 使用beta分佈在[-1, 1]區間內偏向邊界
        alpha = 1.0 - wall_clustering + 0.1  # 避免alpha=0
        beta = alpha
        
        # Beta分佈產生[0,1]，轉換到[-1,1]，然後映射到實際y範圍
        y_beta = np.random.beta(alpha, beta, n_points)
        y_normalized = 2 * y_beta - 1  # 轉換到[-1, 1]
        
        # 進一步增強wall clustering（讓更多點靠近壁面）
        y_normalized = np.sign(y_normalized) * np.power(np.abs(y_normalized), 1.0 - wall_clustering)
        
        # 映射到實際y範圍
        y_samples = (y_normalized + 1.0) / 2.0 * (y_range[1] - y_range[0]) + y_range[0]
    else:
        # 均勻採樣
        x_samples = np.random.uniform(x_range[0], x_range[1], pde_points)
        y_samples = np.random.uniform(y_range[0], y_range[1], pde_points)
    
    pde_coords = np.column_stack([x_samples, y_samples])
    
    # 分析PDE採樣點在y方向的分佈
    y_bins = [-1.0, -0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 1.0]
    y_labels = ['壁面(-1~-0.8)', '近壁(-0.8~-0.5)', '中下(-0.5~-0.2)', 
                '內層(-0.2~0)', '內層(0~0.2)', '中上(0.2~0.5)', 
                '近壁(0.5~0.8)', '壁面(0.8~1)']
    
    pde_y_hist, _ = np.histogram(y_samples, bins=y_bins)
    pde_y_frac = pde_y_hist / len(y_samples)
    
    # 全場y分佈作對比
    full_y = full_coords[:, 1].cpu().numpy()
    full_y_hist, _ = np.histogram(full_y, bins=y_bins)
    full_y_frac = full_y_hist / len(full_y)
    
    print("\ny方向採樣分佈對比:")
    print("區域             全場分布    PDE採樣     偏差比率")
    hypothesis_e_indicators = []
    
    for i, label in enumerate(y_labels):
        if full_y_frac[i] > 0:
            bias_ratio = pde_y_frac[i] / full_y_frac[i]
            status = "過表示" if bias_ratio > 1.5 else "欠表示" if bias_ratio < 0.5 else "正常"
            print(f"{label:>12}: {full_y_frac[i]:>8.1%}    {pde_y_frac[i]:>8.1%}    {bias_ratio:>6.2f}x ({status})")
            
            # 記錄顯著偏差
            if bias_ratio > 2.0 or bias_ratio < 0.3:
                hypothesis_e_indicators.append(f"{label} {status} ({bias_ratio:.2f}x)")
        else:
            print(f"{label:>12}: {full_y_frac[i]:>8.1%}    {pde_y_frac[i]:>8.1%}    N/A")
    
    # 分析在不同y位置的速度分佈覆蓋
    print("\n不同y區域的速度分佈覆蓋:")
    
    # 將全場數據按y位置分組
    full_coords_np = full_coords.cpu().numpy()
    full_u_np = full_u.cpu().numpy()
    
    # 中心區域 (|y| < 0.3) vs 壁面區域 (|y| > 0.7)
    center_mask = np.abs(full_coords_np[:, 1]) < 0.3
    wall_mask = np.abs(full_coords_np[:, 1]) > 0.7
    
    center_u = full_u_np[center_mask]
    wall_u = full_u_np[wall_mask]
    
    pde_center_mask = np.abs(y_samples) < 0.3
    pde_wall_mask = np.abs(y_samples) > 0.7
    
    print(f"中心區域點數 - 全場: {np.sum(center_mask)}, PDE採樣: {np.sum(pde_center_mask)}")
    print(f"壁面區域點數 - 全場: {np.sum(wall_mask)}, PDE採樣: {np.sum(pde_wall_mask)}")
    
    if len(center_u) > 0 and len(wall_u) > 0:
        print(f"中心區域 u 範圍: [{np.min(center_u):.1f}, {np.max(center_u):.1f}]")
        print(f"壁面區域 u 範圍: [{np.min(wall_u):.1f}, {np.max(wall_u):.1f}]")
        
        # 檢查高速區域覆蓋
        high_speed_threshold = 12.0
        center_high_speed = np.sum(center_u > high_speed_threshold) / len(center_u)
        wall_high_speed = np.sum(wall_u > high_speed_threshold) / len(wall_u)
        
        print(f"高速區域(u>{high_speed_threshold})比例 - 中心: {center_high_speed:.1%}, 壁面: {wall_high_speed:.1%}")
        
        # 檢查PDE採樣是否充分覆蓋高速區域
        pde_center_ratio = np.sum(pde_center_mask) / len(y_samples)
        full_center_ratio = np.sum(center_mask) / len(full_coords_np)
        
        if pde_center_ratio < 0.5 * full_center_ratio:
            hypothesis_e_indicators.append("PDE採樣嚴重欠缺中心高速區域")
        elif pde_center_ratio < 0.8 * full_center_ratio:
            hypothesis_e_indicators.append("PDE採樣輕微欠缺中心高速區域")
    
    # 檢查wall_clustering參數是否過高
    if wall_clustering > 0.5:
        hypothesis_e_indicators.append(f"wall_clustering參數過高 ({wall_clustering})")
    
    # === 生成可視化 ===
    print("\n[5/5] 生成分析圖...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 第一行：假設D - Prior分析
    if (sensor_data_with_prior.has_lowfi_prior() and 
        sensor_data_with_prior.lowfi_prior is not None and 
        isinstance(sensor_data_with_prior.lowfi_prior, dict) and
        'u' in sensor_data_with_prior.lowfi_prior):
        
        prior_u = sensor_data_with_prior.lowfi_prior['u']
        prior_v = sensor_data_with_prior.lowfi_prior['v']
        sensor_u_true = sensor_data.sensor_data['u']
        sensor_v_true = sensor_data.sensor_data['v']
        
        # Prior vs 真實場對比 (u)
        axes[0, 0].scatter(sensor_u_true, prior_u, alpha=0.7, s=30)
        axes[0, 0].plot([sensor_u_true.min(), sensor_u_true.max()], 
                        [sensor_u_true.min(), sensor_u_true.max()], 'r--', label='Perfect')
        axes[0, 0].set_xlabel('True u')
        axes[0, 0].set_ylabel('Prior u')
        axes[0, 0].set_title('Prior vs True u Field')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Prior vs 真實場對比 (v)
        axes[0, 1].scatter(sensor_v_true, prior_v, alpha=0.7, s=30)
        axes[0, 1].plot([sensor_v_true.min(), sensor_v_true.max()], 
                        [sensor_v_true.min(), sensor_v_true.max()], 'r--', label='Perfect')
        axes[0, 1].set_xlabel('True v')
        axes[0, 1].set_ylabel('Prior v')
        axes[0, 1].set_title('Prior vs True v Field')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Prior影響分析
        fields = ['u', 'v']
        suppressions = [u_suppression, v_suppression]
        colors = ['red' if s < 0.5 else 'orange' if s < 0.8 else 'green' for s in suppressions]
        
        axes[0, 2].bar(fields, suppressions, color=colors, alpha=0.7)
        axes[0, 2].axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='No suppression')
        axes[0, 2].axhline(y=0.8, color='orange', linestyle=':', alpha=0.5, label='Mild suppression')
        axes[0, 2].axhline(y=0.5, color='red', linestyle=':', alpha=0.5, label='Strong suppression')
        axes[0, 2].set_ylabel('Range Ratio (Prior/True)')
        axes[0, 2].set_title('Prior Range Suppression')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    else:
        for i in range(3):
            axes[0, i].text(0.5, 0.5, 'No Prior Data Available', 
                           ha='center', va='center', transform=axes[0, i].transAxes)
            axes[0, i].set_title(f'Prior Analysis {i+1}')
    
    # 第二行：假設E - PDE採樣分析
    # y方向分佈對比
    x_pos = np.arange(len(y_labels))
    width = 0.35
    axes[1, 0].bar(x_pos - width/2, full_y_frac, width, label='Full Field', alpha=0.7)
    axes[1, 0].bar(x_pos + width/2, pde_y_frac, width, label='PDE Sampling', alpha=0.7)
    axes[1, 0].set_xlabel('y Regions')
    axes[1, 0].set_ylabel('Fraction')
    axes[1, 0].set_title('y-Direction Sampling Distribution')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(y_labels, rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # PDE採樣點空間分布
    axes[1, 1].scatter(full_coords_np[:, 0], full_coords_np[:, 1], 
                       c=full_u_np, cmap='viridis', s=1, alpha=0.7, label='Full Field')
    axes[1, 1].scatter(pde_coords[:500, 0], pde_coords[:500, 1], 
                       c='red', s=2, alpha=0.8, label='PDE Sample (subset)')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].set_title('PDE Sampling Distribution')
    axes[1, 1].legend()
    
    # 不同區域的速度分佈
    if 'center_u' in locals() and 'wall_u' in locals() and len(center_u) > 0 and len(wall_u) > 0:
        axes[1, 2].hist(center_u, bins=30, alpha=0.7, label='Center (|y|<0.3)', density=True)
        axes[1, 2].hist(wall_u, bins=30, alpha=0.7, label='Wall (|y|>0.7)', density=True)
        axes[1, 2].axvline(high_speed_threshold, color='red', linestyle='--', 
                          label=f'High-speed threshold ({high_speed_threshold})')
        axes[1, 2].set_xlabel('u velocity')
        axes[1, 2].set_ylabel('Density')
        axes[1, 2].set_title('Velocity Distribution by Region')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].text(0.5, 0.5, 'Insufficient data for regional analysis', 
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Regional Velocity Analysis')
    
    plt.tight_layout()
    
    # 保存結果
    output_dir = Path('evaluation_results_phase4b_diagnosis')
    output_dir.mkdir(exist_ok=True)
    
    plot_path = output_dir / f'hypotheses_de_analysis_epoch{epoch}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ 假設D&E分析圖已保存: {plot_path}")
    
    # === 診斷結論 ===
    print("\n" + "="*70)
    print("假設D&E診斷結論:")
    print("="*70)
    
    # 假設D結論
    print("\n--- 假設D：RANS Prior拖累 ---")
    if hypothesis_d_indicators:
        print("⚠️  檢測到Prior拖累指標：")
        for indicator in hypothesis_d_indicators:
            print(f"   • {indicator}")
        hypothesis_d_confirmed = True
        print("🔴 假設D確認：RANS Prior是影響因素之一")
    else:
        hypothesis_d_confirmed = False
        print("✅ 假設D否定：未檢測到明顯的Prior拖累")
    
    # 假設E結論
    print("\n--- 假設E：PDE採樣偏差 ---")
    if hypothesis_e_indicators:
        print("⚠️  檢測到採樣偏差指標：")
        for indicator in hypothesis_e_indicators:
            print(f"   • {indicator}")
        hypothesis_e_confirmed = True
        print("🔴 假設E確認：PDE採樣偏差是影響因素之一")
    else:
        hypothesis_e_confirmed = False
        print("✅ 假設E否定：未檢測到明顯的採樣偏差")
    
    # 保存數值報告
    report_path = output_dir / f'hypotheses_de_epoch{epoch}.txt'
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("假設D&E驗證報告\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"模型: {checkpoint_path}\n")
        f.write(f"Epoch: {epoch}\n")
        f.write(f"配置: {config_path}\n\n")
        
        f.write("--- 假設D：RANS Prior拖累 ---\n")
        f.write(f"Prior權重: {prior_weight}\n")
        f.write(f"Prior類型: {prior_type}\n")
        if hypothesis_d_indicators:
            f.write("檢測到的問題:\n")
            for indicator in hypothesis_d_indicators:
                f.write(f"  • {indicator}\n")
        f.write(f"結論: {'確認' if hypothesis_d_confirmed else '否定'}\n\n")
        
        f.write("--- 假設E：PDE採樣偏差 ---\n")
        f.write(f"PDE採樣點數: {pde_points}\n")
        f.write(f"Wall clustering: {wall_clustering}\n")
        if hypothesis_e_indicators:
            f.write("檢測到的問題:\n")
            for indicator in hypothesis_e_indicators:
                f.write(f"  • {indicator}\n")
        f.write(f"結論: {'確認' if hypothesis_e_confirmed else '否定'}\n")
    
    print(f"✓ 數值報告已保存: {report_path}")
    
    return {
        'hypothesis_d_confirmed': hypothesis_d_confirmed,
        'hypothesis_e_confirmed': hypothesis_e_confirmed,
        'hypothesis_d_indicators': hypothesis_d_indicators,
        'hypothesis_e_indicators': hypothesis_e_indicators,
        'prior_weight': prior_weight,
        'wall_clustering': wall_clustering
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Checkpoint 文件路徑')
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路徑')
    
    args = parser.parse_args()
    
    result = verify_remaining_hypotheses(args.checkpoint, args.config)
    
    print(f"\n最終結論：")
    print(f"假設D（RANS Prior拖累）: {'確認' if result['hypothesis_d_confirmed'] else '否定'}")
    print(f"假設E（PDE採樣偏差）: {'確認' if result['hypothesis_e_confirmed'] else '否定'}")
