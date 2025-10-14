#!/usr/bin/env python3
"""
從訓練日誌中提取損失歷史並繪圖
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def extract_from_log(log_path):
    """從日誌文件提取訓練數據"""
    epochs = []
    data_loss = []
    pde_loss = []
    total_loss = []
    u_loss = []
    v_loss = []
    pressure_loss = []
    lr = []
    
    pattern = re.compile(
        r'Epoch (\d+)/\d+ \| '
        r'total_loss: ([\d.]+) \| '
        r'data_loss: ([\d.]+) \| '
        r'pde_loss: ([\d.]+) \|.*?'
        r'u_loss: ([\d.]+) \| '
        r'v_loss: ([\d.]+) \| '
        r'pressure_loss: ([\d.]+) \| '
        r'lr: ([\d.e+-]+)'
    )
    
    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                total_loss.append(float(match.group(2)))
                data_loss.append(float(match.group(3)))
                pde_loss.append(float(match.group(4)))
                u_loss.append(float(match.group(5)))
                v_loss.append(float(match.group(6)))
                pressure_loss.append(float(match.group(7)))
                lr.append(float(match.group(8)))
    
    return {
        'epochs': np.array(epochs),
        'total_loss': np.array(total_loss),
        'data_loss': np.array(data_loss),
        'pde_loss': np.array(pde_loss),
        'u_loss': np.array(u_loss),
        'v_loss': np.array(v_loss),
        'pressure_loss': np.array(pressure_loss),
        'lr': np.array(lr)
    }

def main():
    log_path = Path("log/fourier_k500_6000ep_20251013_172811.log")
    
    print("=" * 80)
    print("提取訓練歷史數據")
    print("=" * 80)
    
    data = extract_from_log(log_path)
    
    print(f"\n提取到 {len(data['epochs'])} 個訓練記錄")
    print(f"Epoch 範圍: {data['epochs'][0]} - {data['epochs'][-1]}")
    
    # 統計分析
    print("\n" + "=" * 80)
    print("統計分析")
    print("=" * 80)
    
    # 檢查異常值
    anomaly_threshold = 10.0
    anomalies = data['total_loss'] > anomaly_threshold
    if np.any(anomalies):
        print(f"\n⚠️  發現 {np.sum(anomalies)} 個異常 epochs (total_loss > {anomaly_threshold}):")
        for idx in np.where(anomalies)[0]:
            print(f"  Epoch {data['epochs'][idx]}: total_loss={data['total_loss'][idx]:.2f}, "
                  f"pde_loss={data['pde_loss'][idx]:.2f}")
    
    # 正常範圍統計
    normal_mask = ~anomalies
    print(f"\n正常範圍統計 ({np.sum(normal_mask)} epochs):")
    print(f"  total_loss: {data['total_loss'][normal_mask].mean():.4f} ± {data['total_loss'][normal_mask].std():.4f}")
    print(f"  data_loss: {data['data_loss'][normal_mask].mean():.4f} ± {data['data_loss'][normal_mask].std():.4f}")
    print(f"  pde_loss: {data['pde_loss'][normal_mask].mean():.4f} ± {data['pde_loss'][normal_mask].std():.4f}")
    
    print(f"\n分場損失統計:")
    print(f"  u_loss: {data['u_loss'][normal_mask].mean():.4f} ± {data['u_loss'][normal_mask].std():.4f}")
    print(f"  v_loss: {data['v_loss'][normal_mask].mean():.4f} ± {data['v_loss'][normal_mask].std():.4f}")
    print(f"  pressure_loss: {data['pressure_loss'][normal_mask].mean():.4f} ± {data['pressure_loss'][normal_mask].std():.4f}")
    
    # 計算場損失比例
    total_field_loss = data['u_loss'] + data['v_loss'] + data['pressure_loss']
    u_ratio = data['u_loss'] / total_field_loss
    v_ratio = data['v_loss'] / total_field_loss
    p_ratio = data['pressure_loss'] / total_field_loss
    
    print(f"\n分場損失貢獻比例 (最後 100 epochs):")
    print(f"  u_loss: {u_ratio[-100:].mean():.1%}")
    print(f"  v_loss: {v_ratio[-100:].mean():.1%}")
    print(f"  pressure_loss: {p_ratio[-100:].mean():.1%}")
    
    # 繪圖
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # [0,0] 總損失（對數尺度）
    ax = axes[0, 0]
    ax.semilogy(data['epochs'], data['total_loss'], linewidth=1.5, alpha=0.8)
    if np.any(anomalies):
        ax.scatter(data['epochs'][anomalies], data['total_loss'][anomalies], 
                  color='red', s=100, marker='x', label='Anomaly', zorder=5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss (log scale)')
    ax.set_title('Total Loss History')
    ax.grid(True, alpha=0.3)
    if np.any(anomalies):
        ax.legend()
    
    # [0,1] 數據損失 vs PDE損失
    ax = axes[0, 1]
    ax.semilogy(data['epochs'], data['data_loss'], label='data_loss', linewidth=1.5, alpha=0.8)
    ax.semilogy(data['epochs'], data['pde_loss'], label='pde_loss', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Data Loss vs PDE Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # [0,2] 分場數據損失
    ax = axes[0, 2]
    ax.semilogy(data['epochs'], data['u_loss'], label='u_loss', linewidth=1.5, alpha=0.7)
    ax.semilogy(data['epochs'], data['v_loss'], label='v_loss', linewidth=1.5, alpha=0.7)
    ax.semilogy(data['epochs'], data['pressure_loss'], label='pressure_loss', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Field-wise Data Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # [1,0] 損失比例
    ax = axes[1, 0]
    loss_ratio = data['data_loss'] / data['pde_loss']
    ax.plot(data['epochs'], loss_ratio, linewidth=1.5, color='purple', alpha=0.8)
    ax.axhline(y=1.0, color='red', linestyle='--', label='Equal contribution', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('data_loss / pde_loss')
    ax.set_title('Loss Ratio (Data/PDE)')
    ax.set_ylim([0, 20])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # [1,1] 場損失貢獻比例
    ax = axes[1, 1]
    ax.plot(data['epochs'], u_ratio, label='u', linewidth=1.5, alpha=0.7)
    ax.plot(data['epochs'], v_ratio, label='v', linewidth=1.5, alpha=0.7)
    ax.plot(data['epochs'], p_ratio, label='p', linewidth=1.5, alpha=0.7)
    ax.axhline(y=1/3, color='gray', linestyle='--', alpha=0.3, label='Equal (33%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Contribution Ratio')
    ax.set_title('Field-wise Loss Contribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # [1,2] 學習率
    ax = axes[1, 2]
    ax.plot(data['epochs'], data['lr'], linewidth=1.5, color='green', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    output_path = Path("results/task008_training_history_analysis.png")
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n圖表已保存至：{output_path}")
    
    # 保存數據
    np.savez(
        "results/task008_training_history.npz",
        **data
    )
    print(f"數據已保存至：results/task008_training_history.npz")

if __name__ == "__main__":
    main()
