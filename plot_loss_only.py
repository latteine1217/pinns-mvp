#!/usr/bin/env python
"""只繪製 Loss Curve"""

import matplotlib.pyplot as plt
from pathlib import Path

def plot_loss_curve(log_file, output_dir):
    """從日誌文件繪製 loss curve"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📊 繪製 Loss Curve from {log_file}...")

    epochs = []
    total_losses = []
    data_losses = []
    pde_losses = []
    div_losses = []
    prior_losses = []

    with open(log_file, 'r') as f:
        for line in f:
            # 解析訓練日誌（包含 Stage 標記的 kolmogorov 訓練日誌）
            if 'Epoch' in line and 'total_loss' in line and 'Stage:' in line:
                parts = line.split('|')

                epoch_val = None
                total_loss = None
                data_loss = None
                pde_loss = None
                div_loss = None
                prior_loss = None

                for part in parts:
                    part = part.strip()
                    if 'Epoch' in part and '/' in part:
                        # 提取 "Epoch X/Y" 中的 X
                        epoch_str = part.split('Epoch')[1].strip().split('/')[0].strip()
                        epoch_val = int(epoch_str)
                    elif 'total_loss:' in part and epoch_val is not None:
                        total_loss = float(part.split('total_loss:')[1].strip())
                    elif 'data_loss:' in part and not 'weighted_data_loss' in part:
                        data_loss = float(part.split('data_loss:')[1].strip())
                    elif 'pde_loss:' in part and not 'weighted_pde_loss' in part:
                        pde_loss = float(part.split('pde_loss:')[1].strip())
                    elif 'div_loss:' in part and not 'weighted_div_loss' in part:
                        div_loss = float(part.split('div_loss:')[1].strip())
                    elif 'prior_consistency_loss:' in part:
                        prior_loss = float(part.split('prior_consistency_loss:')[1].strip())

                # 添加到列表（確保所有值都已解析）
                if epoch_val is not None and total_loss is not None:
                    epochs.append(epoch_val)
                    total_losses.append(total_loss)
                    if data_loss is not None:
                        data_losses.append(data_loss)
                    if pde_loss is not None:
                        pde_losses.append(pde_loss)
                    if div_loss is not None:
                        div_losses.append(div_loss)
                    if prior_loss is not None:
                        prior_losses.append(prior_loss)

    if not epochs:
        print("⚠️  日誌中找不到 loss 數據")
        return

    # 繪製 loss curves
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Loss Curves - Kolmogorov Re=50 K=100 (RANS Prior)', fontsize=16, fontweight='bold')

    # Total Loss
    axes[0, 0].plot(epochs, total_losses, linewidth=1.5, color='purple', alpha=0.8)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Total Loss', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    axes[0, 0].set_yscale('log')

    # Data Loss
    axes[0, 1].plot(epochs, data_losses, linewidth=1.5, color='orange', alpha=0.8)
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Loss', fontsize=11)
    axes[0, 1].set_title('Data Loss (Sensor Matching)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    axes[0, 1].set_yscale('log')

    # PDE Loss
    axes[0, 2].plot(epochs, pde_losses, linewidth=1.5, color='green', alpha=0.8)
    axes[0, 2].set_xlabel('Epoch', fontsize=11)
    axes[0, 2].set_ylabel('Loss', fontsize=11)
    axes[0, 2].set_title('PDE Loss (Momentum)', fontsize=12, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3, linestyle='--')
    axes[0, 2].set_yscale('log')

    # Divergence Loss
    axes[1, 0].plot(epochs, div_losses, linewidth=1.5, color='red', alpha=0.8)
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('Loss', fontsize=11)
    axes[1, 0].set_title('Continuity Loss (Divergence)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    axes[1, 0].set_yscale('log')

    # Prior Loss
    if prior_losses:
        axes[1, 1].plot(epochs, prior_losses, linewidth=1.5, color='blue', alpha=0.8)
        axes[1, 1].set_xlabel('Epoch', fontsize=11)
        axes[1, 1].set_ylabel('Loss', fontsize=11)
        axes[1, 1].set_title('RANS Prior Consistency Loss', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, linestyle='--')
        axes[1, 1].set_yscale('log')
    else:
        axes[1, 1].text(0.5, 0.5, 'No Prior Loss Data', ha='center', va='center', fontsize=14)
        axes[1, 1].axis('off')

    # Loss 比例分析
    if len(data_losses) > 0 and len(pde_losses) > 0:
        # 計算 weighted loss 比例
        total_weighted = [d*10 + p for d, p in zip(data_losses, pde_losses)]  # 假設 data_weight=10
        data_ratio = [(d*10 / t * 100) if t > 0 else 0 for d, t in zip(data_losses, total_weighted)]
        pde_ratio = [(p / t * 100) if t > 0 else 0 for p, t in zip(pde_losses, total_weighted)]

        axes[1, 2].plot(epochs, data_ratio, label='Data Loss %', linewidth=1.5, color='orange', alpha=0.8)
        axes[1, 2].plot(epochs, pde_ratio, label='PDE Loss %', linewidth=1.5, color='green', alpha=0.8)
        axes[1, 2].set_xlabel('Epoch', fontsize=11)
        axes[1, 2].set_ylabel('Percentage (%)', fontsize=11)
        axes[1, 2].set_title('Loss Contribution Ratio', fontsize=12, fontweight='bold')
        axes[1, 2].legend(fontsize=10)
        axes[1, 2].grid(True, alpha=0.3, linestyle='--')
    else:
        axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png', dpi=200, bbox_inches='tight')
    plt.close()

    print(f"✅ Loss curves 已保存至: {output_dir / 'loss_curves.png'}")

    # 統計資訊
    print(f"\n📊 訓練統計:")
    print(f"   總 Epochs: {len(epochs)}")
    print(f"   初始 Total Loss: {total_losses[0]:.4f}")
    print(f"   最終 Total Loss: {total_losses[-1]:.4f}")
    print(f"   Data Loss: {data_losses[0]:.4f} → {data_losses[-1]:.4f}")
    print(f"   PDE Loss: {pde_losses[0]:.4f} → {pde_losses[-1]:.4f}")
    print(f"   Div Loss: {div_losses[0]:.4f} → {div_losses[-1]:.4f}")
    if prior_losses:
        print(f"   Prior Loss: {prior_losses[0]:.4f} → {prior_losses[-1]:.4f}")

    # 檢查收斂性
    print(f"\n⚠️  訓練診斷:")
    final_100_std = sum([(total_losses[i] - sum(total_losses[-100:])/100)**2 for i in range(-100, 0)])**0.5 / 100
    print(f"   最後 100 epochs std: {final_100_std:.4f}")
    if final_100_std > 0.5:
        print(f"   ❌ Loss 震盪嚴重，訓練不穩定")
    else:
        print(f"   ✅ Loss 相對穩定")


if __name__ == '__main__':
    log_file = "pinnx.log"
    output_dir = "results/loss_analysis"

    print("="*70)
    print("  Loss Curve 分析")
    print("="*70)

    plot_loss_curve(log_file, output_dir)

    print("\n" + "="*70)
    print("✅ 分析完成！")
    print("="*70)
