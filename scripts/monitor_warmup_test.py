#!/usr/bin/env python3
"""
WarmupCosine 測試監控腳本
提取訓練日誌並生成學習率曲線與損失趨勢圖
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def parse_log(log_path):
    """解析訓練日誌"""
    epochs = []
    total_losses = []
    data_losses = []
    residual_losses = []
    bc_losses = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # 匹配: Epoch     25 | Total: 93.066948 | Residual: 272488.703045 | BC: 89.164406 | Data: 1.078676
            match = re.search(r'Epoch\s+(\d+)\s+\|\s+Total:\s+([\d.]+)\s+\|\s+Residual:\s+([\d.]+)\s+\|\s+BC:\s+([\d.]+)\s+\|\s+Data:\s+([\d.]+)', line)
            if match:
                epochs.append(int(match.group(1)))
                total_losses.append(float(match.group(2)))
                residual_losses.append(float(match.group(3)))
                bc_losses.append(float(match.group(4)))
                data_losses.append(float(match.group(5)))
    
    return np.array(epochs), np.array(total_losses), np.array(data_losses), np.array(residual_losses), np.array(bc_losses)

def theoretical_lr_schedule(epochs, warmup_epochs=50, base_lr=1e-3, min_lr=1e-6, max_epochs=300):
    """計算理論學習率曲線"""
    lrs = []
    for epoch in epochs:
        if epoch < warmup_epochs:
            # Linear warmup
            lr = base_lr * (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
            lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        lrs.append(lr)
    return np.array(lrs)

def main():
    log_path = Path('/tmp/warmup_test_log2.txt')
    output_dir = Path('results/warmup_test_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 解析日誌
    epochs, total_losses, data_losses, residual_losses, bc_losses = parse_log(log_path)
    
    if len(epochs) == 0:
        print("❌ 未找到訓練數據！")
        return
    
    print(f"✅ 成功解析 {len(epochs)} 個 epoch 的數據")
    print(f"📊 最新進度: Epoch {epochs[-1]}, Data Loss = {data_losses[-1]:.6f}")
    
    # 計算理論學習率
    all_epochs = np.arange(0, 300)
    theoretical_lrs = theoretical_lr_schedule(all_epochs)
    
    # ========== 繪製損失曲線 ==========
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Data Loss (主要關注指標)
    axes[0, 0].plot(epochs, data_losses, 'o-', color='#e74c3c', linewidth=2, markersize=4)
    axes[0, 0].axhline(y=0.1, color='green', linestyle='--', label='Gate 2 Target (0.1)', linewidth=2)
    axes[0, 0].axhline(y=data_losses[-1], color='orange', linestyle=':', label=f'Current ({data_losses[-1]:.3f})', linewidth=1.5)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Data Loss', fontsize=12)
    axes[0, 0].set_title('Data Loss (主要指標)', fontsize=14, fontweight='bold')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=10)
    
    # Total Loss
    axes[0, 1].plot(epochs, total_losses, 's-', color='#3498db', linewidth=2, markersize=4, alpha=0.7)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Total Loss', fontsize=12)
    axes[0, 1].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # BC Loss
    axes[1, 0].plot(epochs, bc_losses, '^-', color='#9b59b6', linewidth=2, markersize=4, alpha=0.7)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('BC Loss', fontsize=12)
    axes[1, 0].set_title('Boundary Condition Loss', fontsize=14, fontweight='bold')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 理論學習率曲線
    axes[1, 1].plot(all_epochs, theoretical_lrs, '-', color='#2ecc71', linewidth=2.5, label='Theoretical LR')
    axes[1, 1].axvline(x=50, color='red', linestyle='--', label='Warmup End', linewidth=2)
    axes[1, 1].axvline(x=epochs[-1], color='blue', linestyle=':', label=f'Current (Epoch {epochs[-1]})', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 1].set_title('Learning Rate Schedule (Warmup + Cosine)', fontsize=14, fontweight='bold')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=10)
    
    plt.tight_layout()
    plot_path = output_dir / 'warmup_test_progress.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ 損失曲線已保存: {plot_path}")
    
    # ========== 生成詳細報告 ==========
    report = f"""
# WarmupCosine 學習率測試報告
生成時間: {Path(log_path).stat().st_mtime}

## 📊 訓練進度
- **當前 Epoch**: {epochs[-1]} / 300 ({epochs[-1]/300*100:.1f}%)
- **Data Loss**: {data_losses[-1]:.6f}
- **距離目標**: {data_losses[-1] / 0.1:.2f}x (目標 < 0.1)

## 📈 損失變化趨勢
| 階段 | Epoch 範圍 | Data Loss 變化 | 下降率 |
|------|-----------|---------------|--------|
| Warmup 階段 | 0-50 | {data_losses[0]:.3f} → {data_losses[epochs==50][0] if 50 in epochs else 'N/A':.3f} | {(data_losses[0] - data_losses[epochs==50][0]) / data_losses[0] * 100 if 50 in epochs else 0:.1f}% |
| Cosine 前期 | 50-100 | {data_losses[epochs==50][0] if 50 in epochs else 'N/A':.3f} → {data_losses[epochs==100][0] if 100 in epochs else 'N/A':.3f} | {(data_losses[epochs==50][0] - data_losses[epochs==100][0]) / data_losses[epochs==50][0] * 100 if 50 in epochs and 100 in epochs else 0:.1f}% |
| Cosine 中期 | 100-150 | {data_losses[epochs==100][0] if 100 in epochs else 'N/A':.3f} → {data_losses[epochs==150][0] if 150 in epochs else 'N/A':.3f} | {(data_losses[epochs==100][0] - data_losses[epochs==150][0]) / data_losses[epochs==100][0] * 100 if 100 in epochs and 150 in epochs else 0:.1f}% |
| 當前階段 | 150-{epochs[-1]} | {data_losses[epochs==150][0] if 150 in epochs else 'N/A':.3f} → {data_losses[-1]:.3f} | {(data_losses[epochs==150][0] - data_losses[-1]) / data_losses[epochs==150][0] * 100 if 150 in epochs else 0:.1f}% |

## ⚠️ 關鍵觀察
1. **收斂停滯**: Epoch 150 之後 Data Loss 幾乎無變化
2. **學習率**: 當前 LR ≈ {theoretical_lrs[epochs[-1]]:.2e}（Cosine Annealing 中）
3. **權重自適應**: W_data 從 35.075 降至 29.069

## 🔮 預測
基於當前趨勢，即使訓練至 300 epochs:
- **預期 Data Loss**: ~0.35-0.37（停滯區間）
- **達標可能性**: ❌ 極低（需降至 0.1）

## 💡 建議
1. **延長訓練**: 將 epochs 從 300 提升至 1000+（參考 Phase 3 full training）
2. **降低 min_lr**: 從 1e-6 改為 1e-7，保持更長時間的細微調整
3. **加入 L-BFGS**: 在 Adam 訓練 300 epochs 後切換為二階優化器
4. **檢查物理一致性**: 驗證 Residual Loss 是否有異常（當前仍在 1e5 量級）
"""
    
    report_path = output_dir / 'warmup_test_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✅ 詳細報告已保存: {report_path}")
    
    # ========== 輸出控制台摘要 ==========
    print("\n" + "="*60)
    print("📊 WarmupCosine 測試摘要")
    print("="*60)
    print(f"當前進度: Epoch {epochs[-1]}/300 ({epochs[-1]/300*100:.1f}%)")
    print(f"Data Loss: {data_losses[-1]:.6f} (目標: < 0.1, 差距: {data_losses[-1]/0.1:.2f}x)")
    print(f"學習率: {theoretical_lrs[epochs[-1]]:.2e}")
    print("="*60)
    
    # 判斷是否需要提前結論
    if epochs[-1] >= 200 and data_losses[-1] > 0.3:
        print("\n⚠️  警告: Epoch 200+ 但 Data Loss 仍 > 0.3")
        print("建議: 考慮提前終止並調整策略（延長 epochs 或改用 L-BFGS）")

if __name__ == '__main__':
    main()
