#!/usr/bin/env python3
"""
分析訓練悖論：為何 continuity loss 下降但 mass error 上升？
"""
import re
import numpy as np
import matplotlib.pyplot as plt

def parse_training_log(filename):
    """提取完整訓練指標"""
    epochs = []
    continuity_losses = []
    data_losses = []
    total_losses = []
    mass_errors = []
    
    with open(filename, 'r') as f:
        for line in f:
            # Epoch 資訊
            match_epoch = re.search(r'Epoch (\d+)/', line)
            if match_epoch:
                epoch = int(match_epoch.group(1))
                
                # 提取損失
                match_cont = re.search(r'continuity_loss:\s+([\d.e+-]+)', line)
                match_data = re.search(r'weighted_data_loss:\s+([\d.e+-]+)', line)
                match_total = re.search(r'total_loss:\s+([\d.e+-]+)', line)
                
                if match_cont and match_data and match_total:
                    epochs.append(epoch)
                    continuity_losses.append(float(match_cont.group(1)))
                    data_losses.append(float(match_data.group(1)))
                    total_losses.append(float(match_total.group(1)))
            
            # 質量守恆誤差
            match_mass = re.search(r'質量守恆誤差:\s+([\d.e+-]+)', line)
            if match_mass:
                mass_errors.append(float(match_mass.group(1)))
    
    return {
        'epochs': np.array(epochs),
        'continuity_loss': np.array(continuity_losses),
        'data_loss': np.array(data_losses),
        'total_loss': np.array(total_losses),
        'mass_error': np.array(mass_errors)
    }

# 解析最新的訓練日誌
log_file = "log/kolm_bugfix.log"
data = parse_training_log(log_file)

print("=" * 60)
print("訓練悖論分析")
print("=" * 60)
print(f"總 epochs: {len(data['epochs'])}")
print()

# 關鍵發現
print("🔍 關鍵指標變化:")
print(f"  Continuity Loss: {data['continuity_loss'][0]:.3f} → {data['continuity_loss'][-1]:.3f}")
print(f"    變化率: {100 * (data['continuity_loss'][-1] - data['continuity_loss'][0]) / data['continuity_loss'][0]:.1f}%")
print()
print(f"  Data Loss: {data['data_loss'][0]:.1f} → {data['data_loss'][-1]:.1f}")
print(f"    變化率: {100 * (data['data_loss'][-1] - data['data_loss'][0]) / data['data_loss'][0]:.1f}%")
print()
print(f"  Mass Error (L∞): {data['mass_error'][0]:.2f} → {data['mass_error'][-1]:.2f}")
print(f"    變化率: {100 * (data['mass_error'][-1] - data['mass_error'][0]) / data['mass_error'][0]:.1f}%")
print()

# 找最小質量守恆誤差的 epoch
min_mass_idx = np.argmin(data['mass_error'])
min_mass_epoch = min_mass_idx  # mass_error 每個 epoch 都記錄
min_mass_value = data['mass_error'][min_mass_idx]

print(f"📊 最小質量守恆誤差:")
print(f"  Epoch: {min_mass_epoch}")
print(f"  Mass Error: {min_mass_value:.3f}")
print(f"  之後上升了: {data['mass_error'][-1] - min_mass_value:.3f}")
print()

# 分析損失比例
loss_ratio = data['data_loss'] / (data['continuity_loss'] + 1e-8)
print(f"💡 Data Loss / Continuity Loss 比例:")
print(f"  初始: {loss_ratio[0]:.1f}")
print(f"  最終: {loss_ratio[-1]:.1f}")
print(f"  變化: {loss_ratio[-1] / loss_ratio[0]:.2f}x")
print()

# 檢查權重配置
print("⚙️ 配置的損失權重:")
import yaml
with open('configs/kolmogorov_2d_baseline.yml', 'r') as f:
    config = yaml.safe_load(f)
print(f"  data_weight: {config['losses']['data_weight']}")
print(f"  continuity_weight: {config['losses']['continuity_weight']}")
print(f"  normalize_losses: {config['losses']['normalize_losses']}")
print(f"  adaptive_weighting: {config['losses']['adaptive_weighting']['enabled']}")
print()

# 視覺化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Continuity Loss vs Mass Error (雙軸)
ax1 = axes[0, 0]
ax1_twin = ax1.twinx()
ax1.plot(data['epochs'], data['continuity_loss'], 'b-', linewidth=2, label='Continuity Loss (MSE)')
ax1_twin.plot(data['mass_error'], 'r--', linewidth=2, label='Mass Error (L∞)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Continuity Loss (MSE)', color='b')
ax1_twin.set_ylabel('Mass Error (L∞)', color='r')
ax1.set_title('核心悖論：Continuity Loss ↓ but Mass Error ↑')
ax1.grid(True, alpha=0.3)
ax1.axvline(min_mass_epoch, color='g', linestyle=':', label=f'Min Mass Error (epoch {min_mass_epoch})')
ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')

# 2. Total Loss 分解
ax2 = axes[0, 1]
ax2.plot(data['epochs'], data['total_loss'], 'k-', linewidth=2, label='Total Loss')
ax2.plot(data['epochs'], data['data_loss'], 'orange', linewidth=1.5, label='Data Loss (weighted)')
ax2.plot(data['epochs'], data['continuity_loss'] * config['losses']['continuity_weight'], 
         'b--', linewidth=1.5, label='Continuity Loss (weighted)')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Loss 組成分析')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

# 3. Loss Ratio (Data / Continuity)
ax3 = axes[1, 0]
ax3.plot(data['epochs'], loss_ratio, 'purple', linewidth=2)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Data Loss / Continuity Loss')
ax3.set_title('損失比例趨勢（資料主導 vs 物理約束）')
ax3.grid(True, alpha=0.3)
ax3.axhline(1.0, color='r', linestyle='--', label='平衡點')
ax3.legend()

# 4. Mass Error 變化率
ax4 = axes[1, 1]
mass_error_change = np.diff(data['mass_error'], prepend=data['mass_error'][0])
ax4.plot(mass_error_change, 'darkred', linewidth=1.5)
ax4.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax4.set_xlabel('Epoch')
ax4.set_ylabel('ΔMass Error')
ax4.set_title('質量守恆誤差變化率')
ax4.grid(True, alpha=0.3)
ax4.fill_between(range(len(mass_error_change)), 0, mass_error_change, 
                  where=(mass_error_change > 0), color='red', alpha=0.3, label='Increasing')
ax4.fill_between(range(len(mass_error_change)), 0, mass_error_change, 
                  where=(mass_error_change <= 0), color='green', alpha=0.3, label='Decreasing')
ax4.legend()

plt.tight_layout()
plt.savefig('results/training_paradox_analysis.png', dpi=150)
print("📊 視覺化已保存: results/training_paradox_analysis.png")

# 診斷結論
print("\n" + "=" * 60)
print("🔬 診斷結論")
print("=" * 60)

if loss_ratio[-1] > 100:
    print("❌ 問題：資料損失主導訓練（Data Loss >> Continuity Loss）")
    print("   → 模型過度擬合感測點，忽略物理約束")
    print("   → 建議：增加 continuity_weight 或減少 data_weight")
elif data['continuity_loss'][-1] < 0.01:
    print("⚠️  Continuity Loss 已經很小，但 Mass Error 仍高")
    print("   → 可能原因：")
    print("      1. MSE 無法約束局部高散度（平均小但最大值大）")
    print("      2. 配置點分佈不足（2000 點可能覆蓋不夠）")
    print("      3. 需要 hard constraint（如 divergence-free projection）")
else:
    print("✅ Continuity Loss 仍有下降空間")
    print("   → 建議：繼續訓練或調整權重")
