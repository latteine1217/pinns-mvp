"""
Task-006: 500 Epochs穩定性測試分析工具
分析RANS湍流系統的長期穩定性表現
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def parse_training_log(log_file):
    """解析訓練日誌"""
    data = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if "Epoch" in line and "Total:" in line and "INFO" in line:
                try:
                    # 提取INFO消息部分
                    info_part = line.split("INFO - ")[-1]
                    
                    # 解析epoch信息
                    parts = info_part.split(' | ')
                    epoch_info = {}
                    
                    for part in parts:
                        part = part.strip()
                        if part.startswith('Epoch'):
                            # 處理 "Epoch 123" 格式
                            epoch_match = re.search(r'Epoch\s+(\d+)', part)
                            if epoch_match:
                                epoch_info['epoch'] = int(epoch_match.group(1))
                        elif part.startswith('Stage:'):
                            stage_match = re.search(r'Stage:\s+(\w+)', part)
                            if stage_match:
                                epoch_info['stage'] = stage_match.group(1)
                        elif part.startswith('Total:'):
                            epoch_info['total_loss'] = float(part.split(':')[1].strip())
                        elif part.startswith('Mom_x:'):
                            epoch_info['momentum_x'] = float(part.split(':')[1].strip())
                        elif part.startswith('Mom_y:'):
                            epoch_info['momentum_y'] = float(part.split(':')[1].strip())
                        elif part.startswith('Cont:'):
                            epoch_info['continuity'] = float(part.split(':')[1].strip())
                        elif part.startswith('k:'):
                            epoch_info['k_equation'] = float(part.split(':')[1].strip())
                        elif part.startswith('ε:'):
                            epoch_info['epsilon_equation'] = float(part.split(':')[1].strip())
                        elif part.startswith('BC:'):
                            epoch_info['boundary'] = float(part.split(':')[1].strip())
                        elif part.startswith('Data:'):
                            epoch_info['data'] = float(part.split(':')[1].strip())
                        elif part.startswith('Time:'):
                            time_str = part.split(':')[1].replace('s', '').strip()
                            epoch_info['time'] = float(time_str)
                    
                    if 'epoch' in epoch_info and 'total_loss' in epoch_info:
                        # 設置默認stage
                        if 'stage' not in epoch_info:
                            epoch_info['stage'] = 'unknown'
                        data.append(epoch_info)
                        
                except Exception as e:
                    print(f"解析錯誤在行: {line.strip()}")
                    print(f"錯誤: {e}")
                    continue
    
    return data

def generate_stability_report(data, output_dir):
    """生成穩定性分析報告"""
    print(f"=== Task-006: 500 Epochs 穩定性分析報告 ===")
    print(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"數據點數: {len(data)}")
    
    if len(data) < 10:
        print("❌ 數據不足，無法進行分析")
        return
    
    # 轉換為numpy arrays
    epochs = np.array([d['epoch'] for d in data])
    total_loss = np.array([d['total_loss'] for d in data])
    momentum_x = np.array([d.get('momentum_x', 0) for d in data])
    momentum_y = np.array([d.get('momentum_y', 0) for d in data])
    continuity = np.array([d.get('continuity', 0) for d in data])
    k_eq = np.array([d.get('k_equation', 0) for d in data])
    epsilon_eq = np.array([d.get('epsilon_equation', 0) for d in data])
    
    # 1. 總體穩定性分析
    print("\n=== 總體穩定性分析 ===")
    initial_loss = total_loss[0] if len(total_loss) > 0 else np.nan
    final_loss = total_loss[-1] if len(total_loss) > 0 else np.nan
    improvement = (initial_loss - final_loss) / initial_loss * 100 if initial_loss > 0 else 0
    
    print(f"初始損失: {initial_loss:.6f}")
    print(f"最終損失: {final_loss:.6f}")
    print(f"總體改善: {improvement:.2f}%")
    
    # 檢查數值穩定性
    has_nan = np.any(np.isnan(total_loss))
    has_inf = np.any(np.isinf(total_loss))
    max_loss = np.max(total_loss)
    
    print(f"數值穩定性: {'❌ 有NaN' if has_nan else '✅ 無NaN'}")
    print(f"數值穩定性: {'❌ 有Inf' if has_inf else '✅ 無Inf'}")
    print(f"最大損失值: {max_loss:.6f}")
    
    # 2. 量級平衡分析
    print("\n=== 損失量級平衡分析 ===")
    final_data = data[-1]
    components = ['momentum_x', 'momentum_y', 'continuity', 'k_equation', 'epsilon_equation']
    
    for comp in components:
        if comp in final_data:
            value = final_data[comp]
            magnitude = np.floor(np.log10(abs(value))) if value > 0 else -np.inf
            print(f"{comp:15s}: {value:.6f} (10^{magnitude:.0f})")
    
    # 檢查量級平衡 (目標: 10^0 量級)
    balance_scores = []
    for comp in components:
        if comp in final_data and final_data[comp] > 0:
            magnitude = abs(np.log10(final_data[comp]))
            balance_scores.append(magnitude)
    
    avg_magnitude = np.mean(balance_scores) if balance_scores else np.inf
    print(f"平均量級偏差: {avg_magnitude:.2f} (目標: ≤1.0)")
    balance_status = "✅ 良好平衡" if avg_magnitude <= 1.0 else "⚠️ 需要調優"
    print(f"量級平衡狀態: {balance_status}")
    
    # 3. 收斂性分析
    print("\n=== 收斂性分析 ===")
    slope = 0  # 初始化
    stability_ratio = 0  # 初始化
    
    if len(total_loss) > 50:
        # 計算最後50個epoch的趨勢
        recent_epochs = epochs[-50:]
        recent_loss = total_loss[-50:]
        
        # 線性回歸計算斜率
        coeffs = np.polyfit(recent_epochs, recent_loss, 1)
        slope = coeffs[0]
        
        print(f"最近50 epochs斜率: {slope:.8f}")
        convergence_status = "✅ 持續收斂" if slope < 0 else "⚠️ 收斂停滯"
        print(f"收斂狀態: {convergence_status}")
        
        # 計算標準差(穩定性指標)
        recent_std = np.std(recent_loss)
        stability_ratio = recent_std / np.mean(recent_loss) * 100
        print(f"最近穩定性: {stability_ratio:.2f}% (CV)")
        stability_status = "✅ 高度穩定" if stability_ratio < 5 else "⚠️ 有波動"
        print(f"穩定性狀態: {stability_status}")
    
    # 4. 生成可視化圖表
    print(f"\n=== 生成可視化圖表 ===")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1) 總損失演化
    ax1.semilogy(epochs, total_loss, 'b-', linewidth=2, label='Total Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss (log scale)')
    ax1.set_title('Loss Evolution')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2) 各分量演化
    ax2.plot(epochs, momentum_x, 'r-', label='Momentum X', alpha=0.8)
    ax2.plot(epochs, momentum_y, 'g-', label='Momentum Y', alpha=0.8)
    ax2.plot(epochs, continuity, 'b-', label='Continuity', alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss Components')
    ax2.set_title('Momentum & Continuity Terms')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3) 湍流分量演化
    ax3.plot(epochs, k_eq, 'orange', label='k-equation', alpha=0.8)
    ax3.plot(epochs, epsilon_eq, 'purple', label='ε-equation', alpha=0.8)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Turbulence Loss')
    ax3.set_title('Turbulence Terms (k-ε)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4) 訓練階段標記
    stages = [d['stage'] for d in data]
    unique_stages = list(set(stages))
    colors = ['red', 'blue', 'green']
    
    for i, stage in enumerate(unique_stages):
        stage_epochs = [d['epoch'] for d in data if d['stage'] == stage]
        stage_loss = [d['total_loss'] for d in data if d['stage'] == stage]
        if stage_epochs:
            ax4.semilogy(stage_epochs, stage_loss, 'o-', 
                        color=colors[i % len(colors)], 
                        label=f'Stage: {stage}', alpha=0.7)
    
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Total Loss (log scale)')
    ax4.set_title('Training Stages')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/500epochs_stability_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 可視化圖表已保存: {output_dir}/500epochs_stability_analysis.png")
    
    # 5. 生成文字報告
    report_path = f'{output_dir}/500epochs_stability_report.md'
    with open(report_path, 'w') as f:
        f.write(f"# Task-006: 500 Epochs 長期穩定性測試報告\n\n")
        f.write(f"**生成時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**數據範圍**: Epoch 1 - {epochs[-1]}  \n")
        f.write(f"**總訓練時間**: {data[-1]['time']:.1f}秒 ({data[-1]['time']/60:.1f}分鐘)  \n\n")
        
        f.write(f"## 📊 關鍵指標達成\n\n")
        f.write(f"| 指標 | 結果 | 狀態 |\n")
        f.write(f"|------|------|------|\n")
        f.write(f"| 數值穩定性 | {'無NaN/Inf' if not (has_nan or has_inf) else '有數值問題'} | {'✅' if not (has_nan or has_inf) else '❌'} |\n")
        f.write(f"| 量級平衡 | 平均偏差{avg_magnitude:.2f} | {'✅' if avg_magnitude <= 1.0 else '⚠️'} |\n")
        f.write(f"| 損失改善 | {improvement:.2f}% | {'✅' if improvement > 0 else '❌'} |\n")
        
        converging = True
        stable = True
        if len(total_loss) > 50:
            converging = slope < 0
            stable = stability_ratio < 5
            f.write(f"| 收斂趨勢 | 斜率{slope:.2e} | {'✅' if converging else '⚠️'} |\n")
            f.write(f"| 近期穩定性 | CV={stability_ratio:.2f}% | {'✅' if stable else '⚠️'} |\n")
        
        f.write(f"\n## 🎯 Task-006 驗收結論\n\n")
        
        all_passed = (not (has_nan or has_inf) and 
                     avg_magnitude <= 1.0 and 
                     improvement > 0)
        
        if len(total_loss) > 50:
            all_passed = all_passed and converging and stable
        
        if all_passed:
            f.write(f"✅ **Task-006 全部驗收標準達成**\n\n")
            f.write(f"- 湍流損失權重調優: 量級平衡完美實現\n")
            f.write(f"- 物理約束機制: 穩定運行500 epochs無異常\n") 
            f.write(f"- 長期穩定性: 持續收斂且數值穩定\n")
            f.write(f"- 系統穩健性: 達到工程可用級別\n")
        else:
            f.write(f"⚠️ **需要進一步優化**\n\n")
            if has_nan or has_inf:
                f.write(f"- 數值穩定性問題需解決\n")
            if avg_magnitude > 1.0:
                f.write(f"- 量級平衡需進一步調優\n")
            if improvement <= 0:
                f.write(f"- 收斂性能需改善\n")
    
    print(f"✅ 穩定性報告已保存: {report_path}")
    
    return {
        'final_loss': final_loss,
        'improvement': improvement,
        'numerical_stable': not (has_nan or has_inf),
        'magnitude_balanced': avg_magnitude <= 1.0,
        'converging': slope < 0 if len(total_loss) > 50 else True,
        'all_passed': all_passed
    }

if __name__ == "__main__":
    log_file = 'rans_stability_500epochs_v3.log'
    output_dir = 'tasks/task-006'
    
    print("開始分析500 epochs穩定性測試結果...")
    
    # 解析訓練日誌
    data = parse_training_log(log_file)
    
    # 生成穩定性報告
    results = generate_stability_report(data, output_dir)
    
    print(f"\n=== 分析完成 ===")
    print(f"所有分析結果已保存到: {output_dir}/")
    print(f"Task-006 穩定性測試: {'✅ 完全成功' if results['all_passed'] else '⚠️ 需要優化'}")