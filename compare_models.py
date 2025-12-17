#!/usr/bin/env python
"""比較 RANS prior 和 vanilla 版本的結果"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import re

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

def parse_training_log(log_path):
    """解析訓練 log 文件，提取 loss 曲線"""
    epochs = []
    total_losses = []
    data_losses = []
    pde_losses = []
    div_losses = []

    with open(log_path, 'r') as f:
        for line in f:
            # 匹配 epoch 行
            match = re.search(r'Epoch (\d+)/\d+ \| total_loss: ([\d.]+) \| data_loss: ([\d.]+) \| pde_loss: ([\d.]+) \| div_loss: ([\d.]+)', line)
            if match:
                epoch = int(match.group(1))
                total_loss = float(match.group(2))
                data_loss = float(match.group(3))
                pde_loss = float(match.group(4))
                div_loss = float(match.group(5))

                epochs.append(epoch)
                total_losses.append(total_loss)
                data_losses.append(data_loss)
                pde_losses.append(pde_loss)
                div_losses.append(div_loss)

    return {
        'epochs': np.array(epochs),
        'total_loss': np.array(total_losses),
        'data_loss': np.array(data_losses),
        'pde_loss': np.array(pde_losses),
        'div_loss': np.array(div_losses)
    }

def plot_loss_curves():
    """繪製 loss curve 比較圖"""
    # 載入兩個版本的 log
    rans_log = parse_training_log('log/kolmogorov_re50_kf4_K100_full_1k/training.log')
    vanilla_log = parse_training_log('log/kolmogorov_re50_kf4_K100_vanilla_1k/training_stdout_correct.log')

    # 創建 2x2 子圖
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Loss Comparison: RANS Prior vs Vanilla', fontsize=16, fontweight='bold')

    # Total Loss
    ax = axes[0, 0]
    ax.plot(rans_log['epochs'], rans_log['total_loss'], label='RANS Prior', linewidth=2, alpha=0.8)
    ax.plot(vanilla_log['epochs'], vanilla_log['total_loss'], label='Vanilla', linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Data Loss
    ax = axes[0, 1]
    ax.plot(rans_log['epochs'], rans_log['data_loss'], label='RANS Prior', linewidth=2, alpha=0.8)
    ax.plot(vanilla_log['epochs'], vanilla_log['data_loss'], label='Vanilla', linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Data Loss')
    ax.set_title('Data Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # PDE Loss
    ax = axes[1, 0]
    ax.plot(rans_log['epochs'], rans_log['pde_loss'], label='RANS Prior', linewidth=2, alpha=0.8)
    ax.plot(vanilla_log['epochs'], vanilla_log['pde_loss'], label='Vanilla', linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PDE Loss')
    ax.set_title('PDE Loss (Momentum)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Divergence Loss
    ax = axes[1, 1]
    ax.plot(rans_log['epochs'], rans_log['div_loss'], label='RANS Prior', linewidth=2, alpha=0.8)
    ax.plot(vanilla_log['epochs'], vanilla_log['div_loss'], label='Vanilla', linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Divergence Loss')
    ax.set_title('Divergence Loss (Continuity)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig('results/loss_curves_comparison.png', dpi=300, bbox_inches='tight')
    print('✅ Loss curves saved to: results/loss_curves_comparison.png')

def plot_metrics_comparison():
    """繪製指標對比圖"""
    # 載入兩個版本的指標
    with open('results/evaluation_rans_prior_epoch1000/metrics.json', 'r') as f:
        rans_metrics = json.load(f)

    with open('results/evaluation_vanilla_epoch1000/metrics.json', 'r') as f:
        vanilla_metrics = json.load(f)

    # 創建對比圖
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Evaluation Metrics Comparison: RANS Prior vs Vanilla (Epoch 1000, t=25.0)',
                 fontsize=14, fontweight='bold')

    # 場重建誤差
    ax = axes[0]
    metrics_names = ['Overall', 'u', 'v', 'p']
    rans_l2 = [
        rans_metrics['relative_l2_overall'] * 100,
        rans_metrics['relative_l2_u'] * 100,
        rans_metrics['relative_l2_v'] * 100,
        rans_metrics['relative_l2_p'] * 100
    ]
    vanilla_l2 = [
        vanilla_metrics['relative_l2_overall'] * 100,
        vanilla_metrics['relative_l2_u'] * 100,
        vanilla_metrics['relative_l2_v'] * 100,
        vanilla_metrics['relative_l2_p'] * 100
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, rans_l2, width, label='RANS Prior', alpha=0.8)
    bars2 = ax.bar(x + width/2, vanilla_l2, width, label='Vanilla', alpha=0.8)

    ax.set_ylabel('Relative L2 Error (%)')
    ax.set_title('Field Reconstruction Error')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=15, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Target (15%)')

    # 添加數值標籤
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=8)

    # 物理守恆誤差
    ax = axes[1]
    div_names = ['Div (mean)', 'Div (max)']
    rans_div = [
        rans_metrics['divergence_error_mean'],
        rans_metrics['divergence_error_max']
    ]
    vanilla_div = [
        vanilla_metrics['divergence_error_mean'],
        vanilla_metrics['divergence_error_max']
    ]

    x = np.arange(len(div_names))
    bars1 = ax.bar(x - width/2, rans_div, width, label='RANS Prior', alpha=0.8)
    bars2 = ax.bar(x + width/2, vanilla_div, width, label='Vanilla', alpha=0.8)

    ax.set_ylabel('Divergence Magnitude')
    ax.set_title('Mass Conservation Error')
    ax.set_xticks(x)
    ax.set_xticklabels(div_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

    # 添加數值標籤
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2e}',
                   ha='center', va='bottom', fontsize=8, rotation=45)

    # 壓力梯度誤差
    ax = axes[2]
    grad_names = ['∂p/∂x', '∂p/∂y']
    rans_grad = [
        rans_metrics['pressure_gradient_dpdx_l2'] * 100,
        rans_metrics['pressure_gradient_dpdy_l2'] * 100
    ]
    vanilla_grad = [
        vanilla_metrics['pressure_gradient_dpdx_l2'] * 100,
        vanilla_metrics['pressure_gradient_dpdy_l2'] * 100
    ]

    x = np.arange(len(grad_names))
    bars1 = ax.bar(x - width/2, rans_grad, width, label='RANS Prior', alpha=0.8)
    bars2 = ax.bar(x + width/2, vanilla_grad, width, label='Vanilla', alpha=0.8)

    ax.set_ylabel('Relative L2 Error (%)')
    ax.set_title('Pressure Gradient Error')
    ax.set_xticks(x)
    ax.set_xticklabels(grad_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 添加數值標籤
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}%',
                   ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('results/metrics_comparison.png', dpi=300, bbox_inches='tight')
    print('✅ Metrics comparison saved to: results/metrics_comparison.png')

def create_summary_table():
    """創建總結表格"""
    # 載入兩個版本的指標
    with open('results/evaluation_rans_prior_epoch1000/metrics.json', 'r') as f:
        rans_metrics = json.load(f)

    with open('results/evaluation_vanilla_epoch1000/metrics.json', 'r') as f:
        vanilla_metrics = json.load(f)

    # 載入訓練時間信息
    rans_time = 37269.3  # seconds
    vanilla_time = 1353.5  # seconds

    print('\n' + '='*80)
    print('📊 Model Comparison Summary')
    print('='*80)
    print(f'\n{"Metric":<30} {"RANS Prior":<20} {"Vanilla":<20} {"Improvement"}')
    print('-'*80)

    # 訓練時間
    print(f'{"Training Time (hours)":<30} {rans_time/3600:<20.2f} {vanilla_time/3600:<20.2f} {"-"}')

    # 場重建誤差
    print(f'\n{"Field Reconstruction Error:":<30}')
    print(f'{"  Overall L2 (%)":<30} {rans_metrics["relative_l2_overall"]*100:<20.2f} '
          f'{vanilla_metrics["relative_l2_overall"]*100:<20.2f} '
          f'{(vanilla_metrics["relative_l2_overall"] - rans_metrics["relative_l2_overall"])*100:+.2f}%')
    print(f'{"  u L2 (%)":<30} {rans_metrics["relative_l2_u"]*100:<20.2f} '
          f'{vanilla_metrics["relative_l2_u"]*100:<20.2f} '
          f'{(vanilla_metrics["relative_l2_u"] - rans_metrics["relative_l2_u"])*100:+.2f}%')
    print(f'{"  v L2 (%)":<30} {rans_metrics["relative_l2_v"]*100:<20.2f} '
          f'{vanilla_metrics["relative_l2_v"]*100:<20.2f} '
          f'{(vanilla_metrics["relative_l2_v"] - rans_metrics["relative_l2_v"])*100:+.2f}%')
    print(f'{"  p L2 (%)":<30} {rans_metrics["relative_l2_p"]*100:<20.2f} '
          f'{vanilla_metrics["relative_l2_p"]*100:<20.2f} '
          f'{(vanilla_metrics["relative_l2_p"] - rans_metrics["relative_l2_p"])*100:+.2f}%')

    # 物理守恆
    print(f'\n{"Mass Conservation:":<30}')
    print(f'{"  Div (mean)":<30} {rans_metrics["divergence_error_mean"]:<20.2e} '
          f'{vanilla_metrics["divergence_error_mean"]:<20.2e} '
          f'{((vanilla_metrics["divergence_error_mean"] - rans_metrics["divergence_error_mean"])/rans_metrics["divergence_error_mean"]*100):+.1f}%')
    print(f'{"  Div (max)":<30} {rans_metrics["divergence_error_max"]:<20.2e} '
          f'{vanilla_metrics["divergence_error_max"]:<20.2e} '
          f'{((vanilla_metrics["divergence_error_max"] - rans_metrics["divergence_error_max"])/rans_metrics["divergence_error_max"]*100):+.1f}%')

    # 壓力梯度
    print(f'\n{"Pressure Gradient Error:":<30}')
    print(f'{"  ∂p/∂x L2 (%)":<30} {rans_metrics["pressure_gradient_dpdx_l2"]*100:<20.2f} '
          f'{vanilla_metrics["pressure_gradient_dpdx_l2"]*100:<20.2f} '
          f'{(vanilla_metrics["pressure_gradient_dpdx_l2"] - rans_metrics["pressure_gradient_dpdx_l2"])*100:+.2f}%')
    print(f'{"  ∂p/∂y L2 (%)":<30} {rans_metrics["pressure_gradient_dpdy_l2"]*100:<20.2f} '
          f'{vanilla_metrics["pressure_gradient_dpdy_l2"]*100:<20.2f} '
          f'{(vanilla_metrics["pressure_gradient_dpdy_l2"] - rans_metrics["pressure_gradient_dpdy_l2"])*100:+.2f}%')

    print('\n' + '='*80)
    print('📌 Key Findings:')
    print('  1. RANS prior 版本在整體誤差上略優於 vanilla（147.85% vs 206.38%）')
    print('  2. 但兩個版本的誤差都遠超目標指標（10-15%），訓練失敗')
    print('  3. 物理守恆（散度）違反嚴重，需要重新檢查訓練配置')
    print('  4. RANS prior 版本訓練時間是 vanilla 的 27.5 倍（10.4h vs 0.38h）')
    print('='*80 + '\n')

if __name__ == '__main__':
    # 創建結果目錄
    Path('results').mkdir(exist_ok=True)

    # 繪製 loss curves
    print('📊 繪製 loss curves...')
    plot_loss_curves()

    # 繪製指標對比
    print('📊 繪製指標對比...')
    plot_metrics_comparison()

    # 創建總結表格
    create_summary_table()

    print('✅ 所有對比圖表已生成！')
