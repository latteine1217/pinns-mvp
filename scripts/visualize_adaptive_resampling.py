#!/usr/bin/env python3
"""
視覺化自適應重採樣歷史

功能：
1. 顯示碰撞點隨時間的移動軌跡
2. 繪製殘差熱圖演化
3. 分析槓桿分數分佈
4. 計算重採樣統計量

使用方式：
    python scripts/visualize_adaptive_resampling.py --checkpoint checkpoints/exp/best_model.pth

作者：AI Assistant
日期：2025-12-07
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional

# 設定中文字體（避免方框）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_resampling_history(checkpoint_path: Path) -> Optional[Dict]:
    """
    從檢查點載入重採樣歷史
    
    Args:
        checkpoint_path: 檢查點檔案路徑
    
    Returns:
        重採樣歷史字典，若不存在則返回 None
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'resampling_history' not in checkpoint:
        print("⚠️ 檢查點中未找到重採樣歷史（可能未啟用 track_history）")
        return None
    
    return checkpoint['resampling_history']


def plot_resampling_timeline(history: Dict, output_dir: Path):
    """
    繪製重採樣時間軸（觸發 epochs 與移除/添加點數）
    
    Args:
        history: 重採樣歷史字典
        output_dir: 輸出目錄
    """
    epochs = [entry['epoch'] for entry in history['events']]
    n_removed = [entry['n_removed'] for entry in history['events']]
    n_added = [entry['n_added'] for entry in history['events']]
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # 繪製柱狀圖
    x = np.arange(len(epochs))
    width = 0.35
    
    ax.bar(x - width/2, n_removed, width, label='Removed Points', color='coral', alpha=0.8)
    ax.bar(x + width/2, n_added, width, label='Added Points', color='skyblue', alpha=0.8)
    
    # 設定標籤
    ax.set_xlabel('Resampling Event')
    ax.set_ylabel('Number of Points')
    ax.set_title('Adaptive Resampling Timeline')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Epoch {e}" for e in epochs], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'resampling_timeline.png', dpi=300)
    print(f"✅ Saved: {output_dir / 'resampling_timeline.png'}")
    plt.close()


def plot_leverage_score_distribution(history: Dict, output_dir: Path):
    """
    繪製槓桿分數分佈隨時間的變化
    
    Args:
        history: 重採樣歷史字典
        output_dir: 輸出目錄
    """
    if 'leverage_scores' not in history['events'][0]:
        print("⚠️ 歷史中未記錄槓桿分數，跳過此圖")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左圖：箱線圖（每次重採樣的分數分佈）
    epochs = [entry['epoch'] for entry in history['events']]
    scores = [entry['leverage_scores'] for entry in history['events']]
    
    axes[0].boxplot(scores, labels=[f"{e}" for e in epochs])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Leverage Score')
    axes[0].set_title('Leverage Score Distribution Over Time')
    axes[0].grid(axis='y', alpha=0.3)
    
    # 右圖：直方圖（最後一次重採樣）
    final_scores = scores[-1]
    axes[1].hist(final_scores, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1].axvline(np.mean(final_scores), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(final_scores):.4f}')
    axes[1].axvline(np.median(final_scores), color='orange', linestyle='--', 
                    label=f'Median: {np.median(final_scores):.4f}')
    axes[1].set_xlabel('Leverage Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Final Leverage Scores (Epoch {epochs[-1]})')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'leverage_score_distribution.png', dpi=300)
    print(f"✅ Saved: {output_dir / 'leverage_score_distribution.png'}")
    plt.close()


def plot_point_movement_2d(history: Dict, output_dir: Path):
    """
    繪製 2D 平面上碰撞點的移動軌跡（僅適用於 2D 問題）
    
    Args:
        history: 重採樣歷史字典
        output_dir: 輸出目錄
    """
    if 'removed_points' not in history['events'][0]:
        print("⚠️ 歷史中未記錄移除點座標，跳過此圖")
        return
    
    # 檢查是否為 2D 問題
    first_removed = history['events'][0]['removed_points']
    if first_removed.shape[1] != 2:
        print(f"⚠️ 此視覺化僅支援 2D 問題（當前維度: {first_removed.shape[1]}）")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 繪製所有移除點（紅色 x）
    for event in history['events']:
        removed = event['removed_points']
        ax.scatter(removed[:, 0], removed[:, 1], c='red', marker='x', 
                  alpha=0.3, s=50, label='Removed' if event == history['events'][0] else '')
    
    # 繪製所有添加點（綠色 o）
    for event in history['events']:
        added = event['added_points']
        ax.scatter(added[:, 0], added[:, 1], c='green', marker='o', 
                  alpha=0.3, s=50, label='Added' if event == history['events'][0] else '')
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Collocation Point Movement (2D)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'point_movement_2d.png', dpi=300)
    print(f"✅ Saved: {output_dir / 'point_movement_2d.png'}")
    plt.close()


def print_resampling_statistics(history: Dict):
    """
    打印重採樣統計摘要
    
    Args:
        history: 重採樣歷史字典
    """
    n_events = len(history['events'])
    epochs = [entry['epoch'] for entry in history['events']]
    n_removed_total = sum(entry['n_removed'] for entry in history['events'])
    n_added_total = sum(entry['n_added'] for entry in history['events'])
    
    print("\n" + "="*60)
    print("📊 Adaptive Resampling Statistics")
    print("="*60)
    print(f"Total resampling events:        {n_events}")
    print(f"Epochs with resampling:         {epochs}")
    print(f"Total points removed:           {n_removed_total}")
    print(f"Total points added:             {n_added_total}")
    print(f"Average removal per event:      {n_removed_total / n_events:.1f}")
    print(f"Average addition per event:     {n_added_total / n_events:.1f}")
    
    if n_events > 1:
        epoch_intervals = [epochs[i+1] - epochs[i] for i in range(len(epochs)-1)]
        print(f"Average epoch interval:         {np.mean(epoch_intervals):.1f}")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize adaptive resampling history')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file with resampling history')
    parser.add_argument('--output', type=str, default='results/adaptive_resampling',
                       help='Output directory for visualizations')
    args = parser.parse_args()
    
    # 創建輸出目錄
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 載入歷史
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    history = load_resampling_history(checkpoint_path)
    if history is None:
        return
    
    # 打印統計
    print_resampling_statistics(history)
    
    # 生成視覺化
    print("📊 Generating visualizations...")
    plot_resampling_timeline(history, output_dir)
    plot_leverage_score_distribution(history, output_dir)
    plot_point_movement_2d(history, output_dir)
    
    # 保存 JSON 摘要
    summary = {
        'n_events': len(history['events']),
        'epochs': [entry['epoch'] for entry in history['events']],
        'total_removed': sum(entry['n_removed'] for entry in history['events']),
        'total_added': sum(entry['n_added'] for entry in history['events']),
    }
    
    with open(output_dir / 'resampling_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ All visualizations saved to: {output_dir}")
    print(f"✅ Summary saved to: {output_dir / 'resampling_summary.json'}")


if __name__ == '__main__':
    main()
