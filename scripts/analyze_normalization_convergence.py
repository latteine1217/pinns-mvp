#!/usr/bin/env python3
"""
進階分析：標準化對訓練收斂特性的影響

分析內容：
1. 收斂速度對比（達到特定損失閾值所需 epoch）
2. 損失曲線平滑度（震盪強度）
3. 收斂率（損失下降斜率）
4. 生成額外視覺化圖表

使用方式：
    python scripts/analyze_normalization_convergence.py
    
輸出：
    - results/normalization_analysis/convergence_speed.png
    - results/normalization_analysis/smoothness_comparison.png
    - results/normalization_analysis/convergence_rate.png
    - results/normalization_analysis/detailed_analysis_report.json
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Tuple
import sys

# 設置中文字體（macOS）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'Songti SC']
plt.rcParams['axes.unicode_minus'] = False


def load_checkpoint_history(checkpoint_path: Path) -> Dict:
    """載入檢查點中的訓練歷史"""
    ckpt = torch.load(str(checkpoint_path), map_location='cpu')
    return ckpt['history']


def compute_convergence_epoch(loss_history: List[float], threshold: float) -> int:
    """
    計算達到特定損失閾值所需的 epoch
    
    Args:
        loss_history: 損失歷史列表
        threshold: 損失閾值
        
    Returns:
        達到閾值的 epoch（若未達到則返回 -1）
    """
    for epoch, loss in enumerate(loss_history):
        if loss <= threshold:
            return epoch
    return -1


def compute_smoothness_score(loss_history: List[float], window: int = 10) -> float:
    """
    計算損失曲線平滑度（震盪強度）
    
    使用滑動窗口內的標準差作為震盪指標
    
    Args:
        loss_history: 損失歷史列表
        window: 滑動窗口大小
        
    Returns:
        平均震盪強度（越小越平滑）
    """
    smoothness_scores = []
    for i in range(window, len(loss_history)):
        window_data = loss_history[i-window:i]
        std = np.std(window_data)
        smoothness_scores.append(std)
    
    return float(np.mean(smoothness_scores))


def compute_convergence_rate(loss_history: List[float], start: int = 0, end: int = -1) -> float:
    """
    計算收斂率（損失下降斜率）
    
    使用線性擬合計算對數空間的斜率
    
    Args:
        loss_history: 損失歷史列表
        start: 起始 epoch
        end: 結束 epoch（-1 = 使用全部）
        
    Returns:
        收斂率（負值表示下降）
    """
    if end == -1:
        end = len(loss_history)
    
    epochs = np.arange(start, end)
    losses = np.array(loss_history[start:end])
    
    # 避免 log(0) 錯誤
    losses = np.maximum(losses, 1e-10)
    log_losses = np.log(losses)
    
    # 線性擬合
    coeffs = np.polyfit(epochs, log_losses, 1)
    return float(coeffs[0])  # 斜率


def analyze_convergence_phases(loss_history: List[float]) -> Dict:
    """
    分析訓練的不同階段
    
    Returns:
        Dict with keys: 'early', 'mid', 'late' (各階段的統計資訊)
    """
    total_epochs = len(loss_history)
    early_end = total_epochs // 3
    mid_end = 2 * total_epochs // 3
    
    phases = {
        'early': loss_history[:early_end],
        'mid': loss_history[early_end:mid_end],
        'late': loss_history[mid_end:]
    }
    
    stats = {}
    for phase_name, phase_losses in phases.items():
        stats[phase_name] = {
            'mean': float(np.mean(phase_losses)),
            'std': float(np.std(phase_losses)),
            'min': float(np.min(phase_losses)),
            'max': float(np.max(phase_losses)),
            'rate': float(compute_convergence_rate(phase_losses))
        }
    
    return stats


def plot_convergence_speed(baseline_history: List[float], 
                          normalized_history: List[float],
                          output_path: Path):
    """繪製收斂速度對比圖（達到不同閾值所需 epoch）"""
    
    # 定義多個損失閾值
    thresholds = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
    
    baseline_epochs = []
    normalized_epochs = []
    
    for threshold in thresholds:
        baseline_ep = compute_convergence_epoch(baseline_history, threshold)
        normalized_ep = compute_convergence_epoch(normalized_history, threshold)
        baseline_epochs.append(baseline_ep if baseline_ep != -1 else 200)  # 未達到則用最大值
        normalized_epochs.append(normalized_ep if normalized_ep != -1 else 200)
    
    # 繪圖
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(thresholds))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_epochs, width, label='Baseline (無標準化)', color='#e74c3c')
    bars2 = ax.bar(x + width/2, normalized_epochs, width, label='Normalized (Z-Score)', color='#3498db')
    
    ax.set_xlabel('Loss Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Epochs to Converge', fontsize=12, fontweight='bold')
    ax.set_title('Convergence Speed Comparison\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t:.4f}' for t in thresholds], rotation=45)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加數值標籤
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height < 200:  # 只標記達到閾值的情況
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9)
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       'N/A',
                       ha='center', va='bottom', fontsize=9, color='red')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def plot_smoothness_comparison(baseline_history: List[float], 
                               normalized_history: List[float],
                               output_path: Path):
    """繪製損失曲線平滑度對比"""
    
    window_sizes = [5, 10, 20, 50]
    baseline_smoothness = []
    normalized_smoothness = []
    
    for window in window_sizes:
        baseline_smoothness.append(compute_smoothness_score(baseline_history, window))
        normalized_smoothness.append(compute_smoothness_score(normalized_history, window))
    
    # 繪圖
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(window_sizes))
    width = 0.35
    
    ax.bar(x - width/2, baseline_smoothness, width, label='Baseline (無標準化)', color='#e74c3c')
    ax.bar(x + width/2, normalized_smoothness, width, label='Normalized (Z-Score)', color='#3498db')
    
    ax.set_xlabel('Smoothing Window Size (epochs)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Oscillation Intensity (Std Dev)', fontsize=12, fontweight='bold')
    ax.set_title('Training Stability Comparison\n(Lower is More Stable)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(window_sizes)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_yscale('log')  # 對數尺度更清楚
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def plot_convergence_rate(baseline_history: List[float], 
                          normalized_history: List[float],
                          output_path: Path):
    """繪製不同階段的收斂率對比"""
    
    baseline_phases = analyze_convergence_phases(baseline_history)
    normalized_phases = analyze_convergence_phases(normalized_history)
    
    phases = ['early', 'mid', 'late']
    baseline_rates = [baseline_phases[p]['rate'] for p in phases]
    normalized_rates = [normalized_phases[p]['rate'] for p in phases]
    
    # 繪圖
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(phases))
    width = 0.35
    
    ax.bar(x - width/2, baseline_rates, width, label='Baseline (無標準化)', color='#e74c3c')
    ax.bar(x + width/2, normalized_rates, width, label='Normalized (Z-Score)', color='#3498db')
    
    ax.set_xlabel('Training Phase', fontsize=12, fontweight='bold')
    ax.set_ylabel('Convergence Rate (log loss / epoch)', fontsize=12, fontweight='bold')
    ax.set_title('Phase-wise Convergence Rate Comparison\n(More Negative = Faster Convergence)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Early (0-33%)', 'Mid (33-66%)', 'Late (66-100%)'])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def generate_detailed_report(baseline_history: List[float], 
                            normalized_history: List[float],
                            output_path: Path):
    """生成詳細分析報告（JSON 格式）"""
    
    report = {
        "analysis_date": "2025-10-17",
        "baseline": {
            "total_epochs": len(baseline_history),
            "final_loss": float(baseline_history[-1]),
            "best_loss": float(min(baseline_history)),
            "smoothness_score_10ep": float(compute_smoothness_score(baseline_history, 10)),
            "convergence_rate_overall": float(compute_convergence_rate(baseline_history)),
            "phases": analyze_convergence_phases(baseline_history),
            "epochs_to_threshold": {
                "0.01": compute_convergence_epoch(baseline_history, 0.01),
                "0.005": compute_convergence_epoch(baseline_history, 0.005),
                "0.001": compute_convergence_epoch(baseline_history, 0.001)
            }
        },
        "normalized": {
            "total_epochs": len(normalized_history),
            "final_loss": float(normalized_history[-1]),
            "best_loss": float(min(normalized_history)),
            "smoothness_score_10ep": float(compute_smoothness_score(normalized_history, 10)),
            "convergence_rate_overall": float(compute_convergence_rate(normalized_history)),
            "phases": analyze_convergence_phases(normalized_history),
            "epochs_to_threshold": {
                "0.01": compute_convergence_epoch(normalized_history, 0.01),
                "0.005": compute_convergence_epoch(normalized_history, 0.005),
                "0.001": compute_convergence_epoch(normalized_history, 0.001)
            }
        },
        "comparison": {
            "final_loss_improvement": f"{(1 - normalized_history[-1] / baseline_history[-1]) * 100:.1f}%",
            "smoothness_improvement": f"{(1 - compute_smoothness_score(normalized_history, 10) / compute_smoothness_score(baseline_history, 10)) * 100:.1f}%",
            "convergence_rate_ratio": float(compute_convergence_rate(normalized_history) / compute_convergence_rate(baseline_history))
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved: {output_path}")
    return report


def main():
    """主執行函數"""
    
    # 配置路徑
    baseline_ckpt = Path("checkpoints/quick_val_baseline/best_model.pth")
    normalized_ckpt = Path("checkpoints/quick_val_normalized/best_model.pth")
    output_dir = Path("results/normalization_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 載入訓練歷史
    print("📂 Loading training histories...")
    baseline_history = load_checkpoint_history(baseline_ckpt)['total_loss']
    normalized_history = load_checkpoint_history(normalized_ckpt)['total_loss']
    
    print(f"   - Baseline: {len(baseline_history)} epochs")
    print(f"   - Normalized: {len(normalized_history)} epochs")
    
    # 生成視覺化
    print("\n📊 Generating visualizations...")
    
    plot_convergence_speed(
        baseline_history, 
        normalized_history,
        output_dir / "convergence_speed.png"
    )
    
    plot_smoothness_comparison(
        baseline_history,
        normalized_history,
        output_dir / "smoothness_comparison.png"
    )
    
    plot_convergence_rate(
        baseline_history,
        normalized_history,
        output_dir / "convergence_rate.png"
    )
    
    # 生成詳細報告
    print("\n📝 Generating detailed report...")
    report = generate_detailed_report(
        baseline_history,
        normalized_history,
        output_dir / "detailed_analysis_report.json"
    )
    
    # 打印關鍵結論
    print("\n" + "="*60)
    print("🎯 Key Findings:")
    print("="*60)
    print(f"✅ Final Loss Improvement: {report['comparison']['final_loss_improvement']}")
    print(f"✅ Smoothness Improvement: {report['comparison']['smoothness_improvement']}")
    print(f"✅ Convergence Rate Ratio: {report['comparison']['convergence_rate_ratio']:.2f}x")
    print(f"\n✅ Epochs to reach 0.001 loss:")
    print(f"   - Baseline: {report['baseline']['epochs_to_threshold']['0.001']} epochs")
    print(f"   - Normalized: {report['normalized']['epochs_to_threshold']['0.001']} epochs")
    print("\n📁 All outputs saved to: results/normalization_analysis/")
    print("="*60)


if __name__ == "__main__":
    main()
