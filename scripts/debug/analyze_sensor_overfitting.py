#!/usr/bin/env python3
"""
驗證假設C：K=50感測點過擬合分析

分析K=50感測點的統計分布，檢查是否過度集中在低速區域
導致模型預測範圍塌縮
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

def analyze_sensor_overfitting(checkpoint_path, config_path):
    """分析K=50感測點過擬合問題"""
    
    from scripts.train import create_model, get_device
    from pinnx.dataio.channel_flow_loader import ChannelFlowLoader
    
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
    
    # 全場數據
    channel_data = loader.load_full_field_data()
    full_coords = torch.from_numpy(channel_data.sensor_points).float().to(device)
    full_u = torch.from_numpy(channel_data.sensor_data['u']).float().to(device)
    full_v = torch.from_numpy(channel_data.sensor_data['v']).float().to(device)
    full_p = torch.from_numpy(channel_data.sensor_data['p']).float().to(device)
    
    # K=50 感測點數據
    sensor_data = loader.load_sensor_data(strategy='qr_pivot', K=50)
    sensor_coords = torch.from_numpy(sensor_data.sensor_points).float().to(device)
    sensor_u = torch.from_numpy(sensor_data.sensor_data['u']).float().to(device)
    sensor_v = torch.from_numpy(sensor_data.sensor_data['v']).float().to(device)
    sensor_p = torch.from_numpy(sensor_data.sensor_data['p']).float().to(device)
    
    print(f"  ✓ 全場數據點數: {full_coords.shape[0]}")
    print(f"  ✓ K=50 感測點數: {sensor_coords.shape[0]}")
    
    # === 統計分析 ===
    print("\n[3/4] 感測點統計分析...")
    
    # 轉為numpy以便分析
    full_coords_np = full_coords.cpu().numpy()
    full_u_np = full_u.cpu().numpy()
    full_v_np = full_v.cpu().numpy()
    full_p_np = full_p.cpu().numpy()
    
    sensor_coords_np = sensor_coords.cpu().numpy()
    sensor_u_np = sensor_u.cpu().numpy()
    sensor_v_np = sensor_v.cpu().numpy()
    sensor_p_np = sensor_p.cpu().numpy()
    
    # 計算統計量
    def compute_stats(data, name):
        return {
            'min': np.min(data),
            'max': np.max(data),
            'mean': np.mean(data),
            'std': np.std(data),
            'median': np.median(data),
            'p25': np.percentile(data, 25),
            'p75': np.percentile(data, 75),
            'range': np.max(data) - np.min(data)
        }
    
    # 各場的統計
    full_u_stats = compute_stats(full_u_np, "全場 u")
    sensor_u_stats = compute_stats(sensor_u_np, "感測 u")
    
    full_v_stats = compute_stats(full_v_np, "全場 v")
    sensor_v_stats = compute_stats(sensor_v_np, "感測 v")
    
    full_p_stats = compute_stats(full_p_np, "全場 p")
    sensor_p_stats = compute_stats(sensor_p_np, "感測 p")
    
    print("="*70)
    print("統計對比分析:")
    print("="*70)
    
    def print_comparison(field, full_stats, sensor_stats):
        print(f"\n--- {field} velocity ---")
        print(f"                  全場              感測點           覆蓋率")
        print(f"範圍:    [{full_stats['min']:.3f}, {full_stats['max']:.3f}]  [{sensor_stats['min']:.3f}, {sensor_stats['max']:.3f}]  {100*sensor_stats['range']/full_stats['range']:.1f}%")
        print(f"均值:    {full_stats['mean']:.3f}              {sensor_stats['mean']:.3f}             {100*abs(sensor_stats['mean']-full_stats['mean'])/abs(full_stats['mean']):.1f}% diff")
        print(f"標準差:  {full_stats['std']:.3f}              {sensor_stats['std']:.3f}             {100*sensor_stats['std']/full_stats['std']:.1f}%")
        print(f"中位數:  {full_stats['median']:.3f}              {sensor_stats['median']:.3f}")
    
    print_comparison("u", full_u_stats, sensor_u_stats)
    print_comparison("v", full_v_stats, sensor_v_stats)
    print_comparison("p", full_p_stats, sensor_p_stats)
    
    # 分析速度區間覆蓋
    print("\n--- 速度區間覆蓋分析 ---")
    
    # 定義速度區間
    u_bins = [0, 2, 5, 8, 12, 16.5]
    bin_labels = ['0-2', '2-5', '5-8', '8-12', '12-16.5']
    
    full_u_hist, _ = np.histogram(full_u_np, bins=u_bins)
    sensor_u_hist, _ = np.histogram(sensor_u_np, bins=u_bins)
    
    full_u_frac = full_u_hist / len(full_u_np)
    sensor_u_frac = sensor_u_hist / len(sensor_u_np)
    
    print("速度區間     全場分布    感測點分布   過/欠表示")
    for i, label in enumerate(bin_labels):
        if full_u_frac[i] > 0:
            ratio = sensor_u_frac[i] / full_u_frac[i]
            status = "過表示" if ratio > 1.5 else "欠表示" if ratio < 0.5 else "正常"
            print(f"u ∈ [{label:>8}]: {full_u_frac[i]:>8.1%}    {sensor_u_frac[i]:>10.1%}    {ratio:>5.2f}x ({status})")
        else:
            print(f"u ∈ [{label:>8}]: {full_u_frac[i]:>8.1%}    {sensor_u_frac[i]:>10.1%}    N/A")
    
    # 空間分布分析
    print("\n--- 空間分布分析 ---")
    
    # y方向分布
    y_bins = [-1.0, -0.5, 0.0, 0.5, 1.0]
    y_labels = ['下層(-1~-0.5)', '中下(-0.5~0)', '中上(0~0.5)', '上層(0.5~1)']
    
    full_y_hist, _ = np.histogram(full_coords_np[:, 1], bins=y_bins)
    sensor_y_hist, _ = np.histogram(sensor_coords_np[:, 1], bins=y_bins)
    
    full_y_frac = full_y_hist / len(full_coords_np)
    sensor_y_frac = sensor_y_hist / len(sensor_coords_np)
    
    print("y 位置區間      全場分布    感測點分布   過/欠表示")
    for i, label in enumerate(y_labels):
        if full_y_frac[i] > 0:
            ratio = sensor_y_frac[i] / full_y_frac[i]
            status = "過表示" if ratio > 1.5 else "欠表示" if ratio < 0.5 else "正常"
            print(f"{label:>12}: {full_y_frac[i]:>8.1%}    {sensor_y_frac[i]:>10.1%}    {ratio:>5.2f}x ({status})")
        else:
            print(f"{label:>12}: {full_y_frac[i]:>8.1%}    {sensor_y_frac[i]:>10.1%}    N/A")
    
    # === 模型性能分析 ===
    print("\n[4/4] 模型在感測點的性能分析...")
    
    # 在感測點處預測
    with torch.no_grad():
        sensor_pred = model(sensor_coords)
    
    sensor_pred_u = sensor_pred[:, 0].cpu().numpy()
    sensor_pred_v = sensor_pred[:, 1].cpu().numpy()
    sensor_pred_p = sensor_pred[:, 2].cpu().numpy()
    
    # 計算感測點誤差
    sensor_u_error = np.abs(sensor_pred_u - sensor_u_np)
    sensor_v_error = np.abs(sensor_pred_v - sensor_v_np)
    sensor_p_error = np.abs(sensor_pred_p - sensor_p_np)
    
    sensor_u_rel_error = sensor_u_error / (np.abs(sensor_u_np) + 1e-8)
    sensor_v_rel_error = sensor_v_error / (np.abs(sensor_v_np) + 1e-8)
    
    print(f"\n感測點預測誤差:")
    print(f"u 絕對誤差: mean={np.mean(sensor_u_error):.6f}, max={np.max(sensor_u_error):.6f}")
    print(f"v 絕對誤差: mean={np.mean(sensor_v_error):.6f}, max={np.max(sensor_v_error):.6f}")
    print(f"p 絕對誤差: mean={np.mean(sensor_p_error):.6f}, max={np.max(sensor_p_error):.6f}")
    
    print(f"u 相對誤差: mean={np.mean(sensor_u_rel_error):.3%}, median={np.median(sensor_u_rel_error):.3%}")
    print(f"v 相對誤差: mean={np.mean(sensor_v_rel_error):.3%}, median={np.median(sensor_v_rel_error):.3%}")
    
    # 全場預測（用於對比）
    with torch.no_grad():
        full_pred = model(full_coords)
    
    full_pred_u = full_pred[:, 0].cpu().numpy()
    full_pred_v = full_pred[:, 1].cpu().numpy()
    
    full_pred_u_stats = compute_stats(full_pred_u, "預測 u")
    full_pred_v_stats = compute_stats(full_pred_v, "預測 v")
    
    print(f"\n模型預測範圍:")
    print(f"u: [{full_pred_u_stats['min']:.3f}, {full_pred_u_stats['max']:.3f}] (參考: [{full_u_stats['min']:.3f}, {full_u_stats['max']:.3f}])")
    print(f"v: [{full_pred_v_stats['min']:.3f}, {full_pred_v_stats['max']:.3f}] (參考: [{full_v_stats['min']:.3f}, {full_v_stats['max']:.3f}])")
    
    range_coverage_u = full_pred_u_stats['range'] / full_u_stats['range']
    range_coverage_v = full_pred_v_stats['range'] / full_v_stats['range']
    
    print(f"預測範圍覆蓋率: u={range_coverage_u:.1%}, v={range_coverage_v:.1%}")
    
    # === 過擬合分析結論 ===
    print("\n" + "="*70)
    print("過擬合診斷結論:")
    print("="*70)
    
    # 檢查是否存在過擬合指標
    overfitting_indicators = []
    
    # 1. 感測點範圍覆蓋不足
    u_coverage = sensor_u_stats['range'] / full_u_stats['range']
    v_coverage = sensor_v_stats['range'] / full_v_stats['range']
    
    if u_coverage < 0.7:
        overfitting_indicators.append(f"u 感測點範圍覆蓋不足 ({u_coverage:.1%})")
    if v_coverage < 0.7:
        overfitting_indicators.append(f"v 感測點範圍覆蓋不足 ({v_coverage:.1%})")
    
    # 2. 感測點偏向低速區
    high_speed_threshold = 10.0
    full_high_speed_frac = np.sum(full_u_np > high_speed_threshold) / len(full_u_np)
    sensor_high_speed_frac = np.sum(sensor_u_np > high_speed_threshold) / len(sensor_u_np)
    
    if full_high_speed_frac > 0 and sensor_high_speed_frac / full_high_speed_frac < 0.5:
        overfitting_indicators.append(f"高速區域 (u>{high_speed_threshold}) 欠表示")
    
    # 3. 感測點誤差過小
    if np.mean(sensor_u_rel_error) < 0.05:  # 5%
        overfitting_indicators.append("感測點誤差異常小，可能過擬合")
    
    # 4. 預測範圍嚴重塌縮
    if range_coverage_u < 0.6 or range_coverage_v < 0.3:
        overfitting_indicators.append("預測範圍嚴重塌縮")
    
    if overfitting_indicators:
        print("⚠️  檢測到過擬合指標：")
        for indicator in overfitting_indicators:
            print(f"   • {indicator}")
        
        hypothesis_confirmed = True
        print("\n🔴 假設 C 確認：K=50 感測點過擬合是主要問題")
    else:
        hypothesis_confirmed = False
        print("✅ 假設 C 否定：未檢測到明顯的感測點過擬合")
    
    # === 可視化 ===
    print("\n生成過擬合分析圖...")
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # 第一行：分布對比
    axes[0, 0].hist(full_u_np, bins=50, alpha=0.7, label='Full Field', density=True)
    axes[0, 0].hist(sensor_u_np, bins=20, alpha=0.7, label='K=50 Sensors', density=True)
    axes[0, 0].set_xlabel('u velocity')
    axes[0, 0].set_title('u Velocity Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(full_v_np, bins=50, alpha=0.7, label='Full Field', density=True)
    axes[0, 1].hist(sensor_v_np, bins=20, alpha=0.7, label='K=50 Sensors', density=True)
    axes[0, 1].set_xlabel('v velocity')
    axes[0, 1].set_title('v Velocity Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].hist(full_coords_np[:, 1], bins=50, alpha=0.7, label='Full Field', density=True)
    axes[0, 2].hist(sensor_coords_np[:, 1], bins=20, alpha=0.7, label='K=50 Sensors', density=True)
    axes[0, 2].set_xlabel('y coordinate')
    axes[0, 2].set_title('Spatial Distribution (y)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 第二行：感測點位置和值
    scatter1 = axes[1, 0].scatter(full_coords_np[:, 0], full_coords_np[:, 1], 
                                  c=full_u_np, cmap='viridis', s=2, alpha=0.7)
    axes[1, 0].scatter(sensor_coords_np[:, 0], sensor_coords_np[:, 1], 
                       c='red', s=20, marker='x', alpha=1.0, label='K=50 Sensors')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].set_title('Sensor Locations on u Field')
    axes[1, 0].legend()
    plt.colorbar(scatter1, ax=axes[1, 0], label='u')
    
    # 感測點誤差分布
    axes[1, 1].scatter(sensor_u_np, sensor_pred_u, alpha=0.7)
    axes[1, 1].plot([sensor_u_np.min(), sensor_u_np.max()], 
                    [sensor_u_np.min(), sensor_u_np.max()], 'r--', label='Perfect')
    axes[1, 1].set_xlabel('Reference u')
    axes[1, 1].set_ylabel('Predicted u')
    axes[1, 1].set_title('Sensor Point Accuracy (u)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 速度區間統計
    x_pos = np.arange(len(bin_labels))
    width = 0.35
    axes[1, 2].bar(x_pos - width/2, full_u_frac, width, label='Full Field', alpha=0.7)
    axes[1, 2].bar(x_pos + width/2, sensor_u_frac, width, label='K=50 Sensors', alpha=0.7)
    axes[1, 2].set_xlabel('Velocity Bins')
    axes[1, 2].set_ylabel('Fraction')
    axes[1, 2].set_title('Velocity Range Coverage')
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(bin_labels)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # 第三行：預測範圍對比
    axes[2, 0].hist(full_u_np, bins=50, alpha=0.7, label='Reference', density=True)
    axes[2, 0].hist(full_pred_u, bins=50, alpha=0.7, label='Predicted', density=True)
    axes[2, 0].set_xlabel('u velocity')
    axes[2, 0].set_title('Prediction Range vs Reference')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].hist(full_v_np, bins=50, alpha=0.7, label='Reference', density=True)
    axes[2, 1].hist(full_pred_v, bins=50, alpha=0.7, label='Predicted', density=True)
    axes[2, 1].set_xlabel('v velocity')
    axes[2, 1].set_title('Prediction Range vs Reference')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    # 範圍覆蓋率
    fields = ['u', 'v']
    coverages = [range_coverage_u, range_coverage_v]
    colors = ['red' if c < 0.6 else 'orange' if c < 0.8 else 'green' for c in coverages]
    
    axes[2, 2].bar(fields, coverages, color=colors, alpha=0.7)
    axes[2, 2].axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Full coverage')
    axes[2, 2].axhline(y=0.8, color='orange', linestyle=':', alpha=0.5, label='80% threshold')
    axes[2, 2].axhline(y=0.6, color='red', linestyle=':', alpha=0.5, label='60% threshold')
    axes[2, 2].set_ylabel('Range Coverage Ratio')
    axes[2, 2].set_title('Prediction Range Coverage')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存結果
    output_dir = Path('evaluation_results_phase4b_diagnosis')
    output_dir.mkdir(exist_ok=True)
    
    plot_path = output_dir / f'sensor_overfitting_analysis_epoch{epoch}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ 過擬合分析圖已保存: {plot_path}")
    
    # 保存數值報告
    report_path = output_dir / f'sensor_overfitting_epoch{epoch}.txt'
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("K=50 感測點過擬合分析報告\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"模型: {checkpoint_path}\n")
        f.write(f"Epoch: {epoch}\n")
        f.write(f"配置: {config_path}\n\n")
        
        f.write("--- 統計對比 ---\n")
        f.write(f"u 範圍覆蓋: {u_coverage:.1%}\n")
        f.write(f"v 範圍覆蓋: {v_coverage:.1%}\n")
        f.write(f"預測 u 範圍覆蓋: {range_coverage_u:.1%}\n")
        f.write(f"預測 v 範圍覆蓋: {range_coverage_v:.1%}\n\n")
        
        f.write("--- 感測點誤差 ---\n")
        f.write(f"u 平均相對誤差: {np.mean(sensor_u_rel_error):.3%}\n")
        f.write(f"v 平均相對誤差: {np.mean(sensor_v_rel_error):.3%}\n\n")
        
        f.write("--- 過擬合指標 ---\n")
        if overfitting_indicators:
            for indicator in overfitting_indicators:
                f.write(f"⚠️  {indicator}\n")
        else:
            f.write("✅ 未檢測到明顯過擬合\n")
        
        f.write(f"\n--- 結論 ---\n")
        if hypothesis_confirmed:
            f.write("🔴 假設 C 確認：K=50 感測點過擬合是主要問題\n")
        else:
            f.write("✅ 假設 C 否定：未檢測到明顯的感測點過擬合\n")
    
    print(f"✓ 數值報告已保存: {report_path}")
    
    print("\n" + "="*70)
    print("假設 C 驗證完成！")
    print("="*70)
    
    return {
        'hypothesis_confirmed': hypothesis_confirmed,
        'overfitting_indicators': overfitting_indicators,
        'u_coverage': u_coverage,
        'v_coverage': v_coverage,
        'range_coverage_u': range_coverage_u,
        'range_coverage_v': range_coverage_v,
        'sensor_u_error': np.mean(sensor_u_rel_error),
        'sensor_v_error': np.mean(sensor_v_rel_error)
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Checkpoint 文件路徑')
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路徑')
    
    args = parser.parse_args()
    
    result = analyze_sensor_overfitting(args.checkpoint, args.config)
    
    print(f"\n最終結論：假設 C {'已確認' if result['hypothesis_confirmed'] else '未確認'}")
    print(f"過擬合指標數量: {len(result['overfitting_indicators'])}")