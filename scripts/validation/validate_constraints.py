"""
物理約束驗證腳本
檢驗RANS模型中湍動能k和耗散率ε是否滿足物理約束條件
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
import sys
import os

# 添加項目根目錄到路徑
sys.path.append('/Users/latteine/Documents/coding/pinns-mvp')

from pinnx.physics.turbulence import apply_physical_constraints, physical_constraint_penalty

def generate_test_turbulence_values() -> tuple:
    """生成測試用的湍流量數值"""
    # 生成包含正負值的測試數據
    torch.manual_seed(42)
    batch_size = 1000
    
    # k: 模擬一些負值情況 (約20%負值)
    k_positive = torch.abs(torch.randn(int(batch_size * 0.8))) * 0.1
    k_negative = -torch.abs(torch.randn(int(batch_size * 0.2))) * 0.05
    k = torch.cat([k_positive, k_negative]).unsqueeze(1)
    k = k[torch.randperm(batch_size)]  # 隨機打亂
    
    # ε: 模擬一些負值情況 (約15%負值)
    eps_positive = torch.abs(torch.randn(int(batch_size * 0.85))) * 1.0
    eps_negative = -torch.abs(torch.randn(int(batch_size * 0.15))) * 0.5
    epsilon = torch.cat([eps_positive, eps_negative]).unsqueeze(1)
    epsilon = epsilon[torch.randperm(batch_size)]  # 隨機打亂
    
    return k, epsilon

def validate_physical_constraints(k: torch.Tensor, 
                                epsilon: torch.Tensor,
                                constraint_types: list = ["none", "relu", "softplus", "clip"]) -> Dict[str, Any]:
    """驗證物理約束效果"""
    
    results = {}
    
    for constraint_type in constraint_types:
        print(f"\n=== 測試約束類型: {constraint_type} ===")
        
        # 應用約束
        k_constrained, epsilon_constrained = apply_physical_constraints(
            k.clone(), epsilon.clone(), constraint_type
        )
        
        # 計算約束懲罰項
        penalty = physical_constraint_penalty(k, epsilon, penalty_weight=1.0)
        
        # 統計分析
        stats = {
            'constraint_type': constraint_type,
            'k_raw': {
                'min': k.min().item(),
                'max': k.max().item(),
                'mean': k.mean().item(),
                'negative_count': (k < 0).sum().item(),
                'negative_ratio': (k < 0).float().mean().item()
            },
            'epsilon_raw': {
                'min': epsilon.min().item(),
                'max': epsilon.max().item(),
                'mean': epsilon.mean().item(),
                'negative_count': (epsilon < 0).sum().item(),
                'negative_ratio': (epsilon < 0).float().mean().item()
            },
            'k_constrained': {
                'min': k_constrained.min().item(),
                'max': k_constrained.max().item(),
                'mean': k_constrained.mean().item(),
                'negative_count': (k_constrained < 0).sum().item(),
                'negative_ratio': (k_constrained < 0).float().mean().item()
            },
            'epsilon_constrained': {
                'min': epsilon_constrained.min().item(),
                'max': epsilon_constrained.max().item(),
                'mean': epsilon_constrained.mean().item(),
                'negative_count': (epsilon_constrained < 0).sum().item(),
                'negative_ratio': (epsilon_constrained < 0).float().mean().item()
            },
            'penalty': penalty.item()
        }
        
        results[constraint_type] = stats
        
        # 輸出統計結果
        print(f"k (原始): 範圍 [{stats['k_raw']['min']:.6f}, {stats['k_raw']['max']:.6f}], "
              f"負值比例: {stats['k_raw']['negative_ratio']:.2%}")
        print(f"ε (原始): 範圍 [{stats['epsilon_raw']['min']:.6f}, {stats['epsilon_raw']['max']:.6f}], "
              f"負值比例: {stats['epsilon_raw']['negative_ratio']:.2%}")
        print(f"k (約束後): 範圍 [{stats['k_constrained']['min']:.6f}, {stats['k_constrained']['max']:.6f}], "
              f"負值比例: {stats['k_constrained']['negative_ratio']:.2%}")
        print(f"ε (約束後): 範圍 [{stats['epsilon_constrained']['min']:.6f}, {stats['epsilon_constrained']['max']:.6f}], "
              f"負值比例: {stats['epsilon_constrained']['negative_ratio']:.2%}")
        print(f"約束懲罰項: {stats['penalty']:.6f}")
    
    return results

def create_constraint_visualization(results: Dict[str, Any], save_path: str = 'tasks/task-006/constraint_validation.png'):
    """創建約束效果可視化"""
    
    # 確保目錄存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    constraint_types = list(results.keys())
    n_types = len(constraint_types)
    
    # 創建圖表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Physical Constraints Validation - RANS Turbulence Variables', fontsize=16)
    
    # k 原始值分布
    ax = axes[0, 0]
    k_mins = [results[ct]['k_raw']['min'] for ct in constraint_types]
    k_maxs = [results[ct]['k_raw']['max'] for ct in constraint_types]
    
    x = np.arange(n_types)
    width = 0.35
    
    ax.bar(x - width/2, k_mins, width, label='Min k', alpha=0.7, color='lightcoral')
    ax.bar(x + width/2, k_maxs, width, label='Max k', alpha=0.7, color='lightblue')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Physical Boundary (k≥0)')
    ax.set_xlabel('Constraint Type')
    ax.set_ylabel('k Value')
    ax.set_title('Turbulent Kinetic Energy (k) - Raw Values')
    ax.set_xticks(x)
    ax.set_xticklabels(constraint_types)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ε 原始值分布
    ax = axes[0, 1]
    eps_mins = [results[ct]['epsilon_raw']['min'] for ct in constraint_types]
    eps_maxs = [results[ct]['epsilon_raw']['max'] for ct in constraint_types]
    
    ax.bar(x - width/2, eps_mins, width, label='Min ε', alpha=0.7, color='lightcoral')
    ax.bar(x + width/2, eps_maxs, width, label='Max ε', alpha=0.7, color='lightblue')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Physical Boundary (ε≥0)')
    ax.set_xlabel('Constraint Type')
    ax.set_ylabel('ε Value')
    ax.set_title('Dissipation Rate (ε) - Raw Values')
    ax.set_xticks(x)
    ax.set_xticklabels(constraint_types)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 負值比例對比 (原始)
    ax = axes[1, 0]
    k_neg_ratios = [results[ct]['k_raw']['negative_ratio'] * 100 for ct in constraint_types]
    eps_neg_ratios = [results[ct]['epsilon_raw']['negative_ratio'] * 100 for ct in constraint_types]
    
    ax.bar(x - width/2, k_neg_ratios, width, label='k negative %', alpha=0.7, color='orange')
    ax.bar(x + width/2, eps_neg_ratios, width, label='ε negative %', alpha=0.7, color='purple')
    ax.set_xlabel('Constraint Type')
    ax.set_ylabel('Negative Values (%)')
    ax.set_title('Percentage of Non-Physical Values (Before Constraints)')
    ax.set_xticks(x)
    ax.set_xticklabels(constraint_types)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 約束後負值比例
    ax = axes[1, 1]
    k_const_neg = [results[ct]['k_constrained']['negative_ratio'] * 100 for ct in constraint_types]
    eps_const_neg = [results[ct]['epsilon_constrained']['negative_ratio'] * 100 for ct in constraint_types]
    
    bars1 = ax.bar(x - width/2, k_const_neg, width, label='k negative %', alpha=0.7, color='green')
    bars2 = ax.bar(x + width/2, eps_const_neg, width, label='ε negative %', alpha=0.7, color='cyan')
    
    # 在零值上標記 "100% Compliant"
    for i, (k_val, eps_val) in enumerate(zip(k_const_neg, eps_const_neg)):
        if k_val == 0:
            ax.text(i - width/2, 0.5, '✓', ha='center', va='bottom', fontsize=16, color='green', weight='bold')
        if eps_val == 0:
            ax.text(i + width/2, 0.5, '✓', ha='center', va='bottom', fontsize=16, color='green', weight='bold')
    
    ax.set_xlabel('Constraint Type')
    ax.set_ylabel('Negative Values (%)')
    ax.set_title('Percentage of Non-Physical Values (After Constraints)')
    ax.set_xticks(x)
    ax.set_xticklabels(constraint_types)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, max(max(k_const_neg), max(eps_const_neg)) + 2)
    
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n約束驗證圖表已保存: {save_path}")
    
    return fig

def analyze_constraint_effectiveness(results: Dict[str, Any]) -> str:
    """分析約束機制有效性"""
    
    analysis = []
    analysis.append("="*60)
    analysis.append("物理約束機制有效性分析")
    analysis.append("="*60)
    
    for constraint_type, stats in results.items():
        analysis.append(f"\n約束類型: {constraint_type.upper()}")
        analysis.append("-" * 40)
        
        # 檢查是否100%滿足約束
        k_violation = stats['k_constrained']['negative_ratio'] > 0
        eps_violation = stats['epsilon_constrained']['negative_ratio'] > 0
        
        if constraint_type == "none":
            analysis.append(f"原始負值比例: k={stats['k_raw']['negative_ratio']:.1%}, ε={stats['epsilon_raw']['negative_ratio']:.1%}")
            analysis.append(f"約束懲罰項: {stats['penalty']:.6f}")
        else:
            if not k_violation and not eps_violation:
                analysis.append("✅ 100% 滿足物理約束 (k≥0, ε≥0)")
                analysis.append(f"約束修正效果: k負值 {stats['k_raw']['negative_ratio']:.1%}→0%, ε負值 {stats['epsilon_raw']['negative_ratio']:.1%}→0%")
            else:
                analysis.append(f"❌ 約束違反: k負值{stats['k_constrained']['negative_ratio']:.2%}, "
                              f"ε負值{stats['epsilon_constrained']['negative_ratio']:.2%}")
                
            # 數值範圍分析
            k_shift = stats['k_constrained']['min'] - stats['k_raw']['min']
            eps_shift = stats['epsilon_constrained']['min'] - stats['epsilon_raw']['min']
            if k_shift > 0 or eps_shift > 0:
                analysis.append(f"數值修正: k最小值提升{k_shift:.6f}, ε最小值提升{eps_shift:.6f}")
    
    # 推薦最佳約束方法
    analysis.append("\n" + "="*60)
    analysis.append("推薦配置")
    analysis.append("="*60)
    
    # 尋找完全合規的方法
    compliant_methods = []
    for ct, stats in results.items():
        if ct != "none" and stats['k_constrained']['negative_ratio'] == 0 and stats['epsilon_constrained']['negative_ratio'] == 0:
            compliant_methods.append(ct)
    
    if compliant_methods:
        analysis.append(f"✅ 推薦使用: {', '.join(compliant_methods)}")
        analysis.append("這些方法能100%確保物理約束的滿足")
        
        # softplus 的優勢分析
        if "softplus" in compliant_methods:
            softplus_stats = results["softplus"]
            analysis.append(f"\n💡 Softplus優勢:")
            analysis.append(f"  - 可微分約束，梯度友好")
            analysis.append(f"  - k範圍: [{softplus_stats['k_constrained']['min']:.6f}, {softplus_stats['k_constrained']['max']:.6f}]")
            analysis.append(f"  - ε範圍: [{softplus_stats['epsilon_constrained']['min']:.6f}, {softplus_stats['epsilon_constrained']['max']:.6f}]")
    else:
        analysis.append("⚠️  沒有找到完全合規的約束方法，建議檢查實現")
    
    return "\n".join(analysis)

def main():
    """主驗證程序"""
    print("=== RANS物理約束驗證程序 ===")
    print("正在生成測試數據...")
    
    # 生成測試數據
    k, epsilon = generate_test_turbulence_values()
    print(f"測試數據大小: {k.shape[0]} 個樣本")
    print(f"原始 k 範圍: [{k.min():.6f}, {k.max():.6f}], 負值比例: {(k<0).float().mean():.1%}")
    print(f"原始 ε 範圍: [{epsilon.min():.6f}, {epsilon.max():.6f}], 負值比例: {(epsilon<0).float().mean():.1%}")
    
    # 驗證約束
    results = validate_physical_constraints(k, epsilon)
    
    # 創建可視化
    fig = create_constraint_visualization(results)
    
    # 生成分析報告
    analysis_report = analyze_constraint_effectiveness(results)
    print(analysis_report)
    
    # 保存分析報告
    report_path = 'tasks/task-006/constraint_validation_report.txt'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(analysis_report)
    print(f"\n分析報告已保存: {report_path}")
    
    print("\n約束驗證完成！")
    print("- 圖表: tasks/task-006/constraint_validation.png") 
    print("- 報告: tasks/task-006/constraint_validation_report.txt")

if __name__ == "__main__":
    main()