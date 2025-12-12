#!/usr/bin/env python3
"""
論文用 RANS vs DNS 圖表生成腳本
================================

根據 RESULTS_NEXT_STEPS.md 的要求，整理並生成論文需要的 RANS vs DNS 對比圖表。

輸出：
1. 統計對比圖（選 Re100）
2. 能譜對比圖（選 Re100）
3. 速度場對比圖（選 Re100）
4. 綜合對比表格（Re50/100/500）

使用範例：
--------
python scripts/generate_paper_figures_rans.py \
    --output results/paper_figures/

作者：PINNs-MVP 團隊
日期：2025-12-12
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import h5py
from pathlib import Path
import argparse
import json
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# 設定論文風格
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


def load_comparison_results(base_dir: Path) -> Dict:
    """載入所有 RANS vs DNS 比較結果"""
    results = {}
    
    cases = ['re50', 're100', 're500']
    for case in cases:
        case_dir = base_dir / f'rans_vs_dns_{case}'
        if not case_dir.exists():
            logging.warning(f"找不到 {case} 的結果目錄: {case_dir}")
            continue
        
        # 讀取現有圖片和數據
        results[case] = {
            'dir': case_dir,
            'images': {
                'field_u': case_dir / 'field_comparison_u.png',
                'field_v': case_dir / 'field_comparison_v.png',
                'spectrum': case_dir / 'spectrum_comparison.png',
                'statistics': case_dir / 'statistics_comparison.png',
            }
        }
    
    return results


def create_summary_table(results: Dict, output_dir: Path):
    """創建 RANS vs DNS 誤差總結表"""
    logging.info("創建誤差總結表...")
    
    # 從已有的比較圖中提取數據（這裡我們用之前計算的數值）
    summary_data = {
        'Re50': {
            'DNS_Re': 50.0,
            'RANS_Re': 33.4,
            'L2_u': 67.1,
            'L2_v': 100.0,
            'L2_total': 72.9,
            'RMSE_u': 0.431,
            'RMSE_v': 0.268,
        },
        'Re100': {
            'DNS_Re': 100.0,
            'RANS_Re': 66.7,
            'L2_u': 98.3,
            'L2_v': 100.0,
            'L2_total': 99.1,
            'RMSE_u': 0.736,
            'RMSE_v': 0.665,
        },
        'Re500': {
            'DNS_Re': 500.0,
            'RANS_Re': 333.7,
            'L2_u': 100.5,
            'L2_v': 100.0,
            'L2_total': 100.3,
            'RMSE_u': 1.948,
            'RMSE_v': 1.847,
        }
    }
    
    # 創建表格圖
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # 表格數據
    table_data = [
        ['Case', 'DNS Re', 'RANS Re', 'L2(u) %', 'L2(v) %', 'L2(total) %', 'RMSE(u)', 'RMSE(v)'],
    ]
    
    for case_name, data in summary_data.items():
        row = [
            case_name,
            f"{data['DNS_Re']:.0f}",
            f"{data['RANS_Re']:.1f}",
            f"{data['L2_u']:.1f}",
            f"{data['L2_v']:.1f}",
            f"{data['L2_total']:.1f}",
            f"{data['RMSE_u']:.3f}",
            f"{data['RMSE_v']:.3f}",
        ]
        table_data.append(row)
    
    # 創建表格
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.12, 0.12, 0.12, 0.12, 0.12, 0.14, 0.13, 0.13])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # 標題行樣式
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # 數據行樣式（交替顏色）
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#E7E6E6')
    
    plt.title('RANS vs DNS Error Summary\n(Low-Fidelity Baseline Characterization)',
              fontsize=11, fontweight='bold', pad=10)
    
    fig.savefig(output_dir / 'table_rans_dns_summary.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 保存為 JSON
    with open(output_dir / 'rans_dns_summary.json', 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    logging.info(f"  ✅ 總結表已保存")


def create_combined_figure(results: Dict, output_dir: Path):
    """創建組合圖（統計 + 能譜）用於論文"""
    logging.info("創建組合對比圖...")
    
    # 選擇 Re100 作為代表案例
    re100_dir = results['re100']['dir']
    
    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)
    
    # 左：統計對比（從已有圖片讀取）
    ax1 = fig.add_subplot(gs[0, 0])
    img_stats = plt.imread(results['re100']['images']['statistics'])
    ax1.imshow(img_stats)
    ax1.axis('off')
    ax1.set_title('(a) Statistical Comparison', fontsize=11, fontweight='bold')
    
    # 右：能譜對比
    ax2 = fig.add_subplot(gs[0, 1])
    img_spectrum = plt.imread(results['re100']['images']['spectrum'])
    ax2.imshow(img_spectrum)
    ax2.axis('off')
    ax2.set_title('(b) Energy Spectrum', fontsize=11, fontweight='bold')
    
    plt.suptitle('RANS vs DNS Comparison (Re=100)\nDemonstrating Low-Fidelity Prior Bias',
                 fontsize=12, fontweight='bold', y=1.02)
    
    fig.savefig(output_dir / 'fig_rans_dns_combined_re100.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logging.info(f"  ✅ 組合圖已保存")


def create_error_scaling_plot(output_dir: Path):
    """創建誤差隨 Re 變化的趨勢圖"""
    logging.info("創建誤差-雷諾數趨勢圖...")
    
    # 數據
    Re_vals = np.array([50, 100, 500])
    L2_u = np.array([67.1, 98.3, 100.5])
    L2_v = np.array([100.0, 100.0, 100.0])
    L2_total = np.array([72.9, 99.1, 100.3])
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    ax.plot(Re_vals, L2_u, 'o-', label='L2(u)', linewidth=2, markersize=8)
    ax.plot(Re_vals, L2_v, 's-', label='L2(v)', linewidth=2, markersize=8)
    ax.plot(Re_vals, L2_total, '^-', label='L2(total)', linewidth=2, markersize=8)
    
    ax.set_xlabel('DNS Reynolds Number', fontsize=10)
    ax.set_ylabel('Relative L2 Error (%)', fontsize=10)
    ax.set_title('RANS Error Scaling with Reynolds Number', fontsize=11, fontweight='bold')
    ax.set_xscale('log')
    ax.set_ylim([60, 105])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right')
    
    # 添加註解
    ax.axhline(100, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(300, 101.5, '100% error (complete mismatch)', fontsize=8, color='red', alpha=0.7)
    
    fig.savefig(output_dir / 'fig_rans_error_scaling.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logging.info(f"  ✅ 誤差趨勢圖已保存")


def generate_latex_table(output_dir: Path):
    """生成 LaTeX 表格代碼"""
    logging.info("生成 LaTeX 表格...")
    
    latex_code = r"""
\begin{table}[htbp]
\centering
\caption{RANS vs DNS Error Summary: Low-Fidelity Baseline Characterization}
\label{tab:rans_dns_error}
\begin{tabular}{lcccccccc}
\toprule
\textbf{Case} & \textbf{DNS Re} & \textbf{RANS Re} & \textbf{L2(u) \%} & \textbf{L2(v) \%} & \textbf{L2(total) \%} & \textbf{RMSE(u)} & \textbf{RMSE(v)} \\
\midrule
Re50  & 50.0  & 33.4  & 67.1  & 100.0 & 72.9  & 0.431 & 0.268 \\
Re100 & 100.0 & 66.7  & 98.3  & 100.0 & 99.1  & 0.736 & 0.665 \\
Re500 & 500.0 & 333.7 & 100.5 & 100.0 & 100.3 & 1.948 & 1.847 \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: RANS uses standard k-$\epsilon$ model with spatially-averaged eddy viscosity. 
The high error (70-100\%) demonstrates the systematic bias of the low-fidelity model, 
which PINNs must correct using sparse measurements.
\end{tablenotes}
\end{table}
"""
    
    with open(output_dir / 'table_rans_dns_latex.tex', 'w') as f:
        f.write(latex_code)
    
    logging.info(f"  ✅ LaTeX 表格已保存")


def main():
    parser = argparse.ArgumentParser(description='生成論文用 RANS vs DNS 圖表')
    parser.add_argument('--results', type=str, default='results/',
                       help='結果目錄（包含 rans_vs_dns_reXX/）')
    parser.add_argument('--output', type=str, default='results/paper_figures/',
                       help='輸出目錄')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info("=" * 70)
    logging.info("論文圖表生成：RANS vs DNS 對比")
    logging.info("=" * 70)
    
    # 載入結果
    results = load_comparison_results(results_dir)
    
    if not results:
        logging.error("找不到任何 RANS vs DNS 結果！")
        return
    
    logging.info(f"找到 {len(results)} 個案例：{list(results.keys())}")
    
    # 生成圖表
    create_summary_table(results, output_dir)
    create_combined_figure(results, output_dir)
    create_error_scaling_plot(output_dir)
    generate_latex_table(output_dir)
    
    logging.info("\n" + "=" * 70)
    logging.info("✅ 所有論文圖表已生成！")
    logging.info(f"   輸出目錄: {output_dir}")
    logging.info(f"   生成檔案:")
    logging.info(f"     - table_rans_dns_summary.png")
    logging.info(f"     - fig_rans_dns_combined_re100.png")
    logging.info(f"     - fig_rans_error_scaling.png")
    logging.info(f"     - table_rans_dns_latex.tex")
    logging.info(f"     - rans_dns_summary.json")
    logging.info("=" * 70)


if __name__ == '__main__':
    main()
