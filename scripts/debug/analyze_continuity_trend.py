#!/usr/bin/env python3
import re
import matplotlib.pyplot as plt

def parse_losses(filename):
    """提取訓練損失"""
    continuity_losses = []
    mass_errors = []
    data_losses = []
    
    with open(filename, 'r') as f:
        for line in f:
            # 提取 continuity_loss
            match_cont = re.search(r'continuity_loss:\s+([\d.e+-]+)', line)
            if match_cont:
                continuity_losses.append(float(match_cont.group(1)))
            
            # 提取 data_loss
            match_data = re.search(r'weighted_data_loss:\s+([\d.e+-]+)', line)
            if match_data:
                data_losses.append(float(match_data.group(1)))
            
            # 提取質量守恆誤差
            match_mass = re.search(r'質量守恆誤差:\s+([\d.e+-]+)', line)
            if match_mass:
                mass_errors.append(float(match_mass.group(1)))
    
    return continuity_losses, data_losses, mass_errors

cont1, data1, mass1 = parse_losses("log/kolm_weights_check.log")
cont2, data2, mass2 = parse_losses("log/kolm_no_scaling.log")

print("Run 1 (with normalization):")
print(f"  Continuity loss: {cont1[0]:.3f} → {cont1[-1]:.3f} (下降 {100*(cont1[0]-cont1[-1])/cont1[0]:.1f}%)")
print(f"  Data loss: {data1[0]:.3f} → {data1[-1]:.3f}")
print(f"  Mass error: {mass1[0]:.2f} → {mass1[-1]:.2f}")
print()
print("Run 2 (no normalization):")
print(f"  Continuity loss: {cont2[0]:.3f} → {cont2[-1]:.3f} (下降 {100*(cont2[0]-cont2[-1])/cont2[0]:.1f}%)")
print(f"  Data loss: {data2[0]:.3f} → {data2[-1]:.3f}")
print(f"  Mass error: {mass2[0]:.2f} → {mass2[-1]:.2f}")

# 繪製雙軸圖：continuity loss vs mass error
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Run 1
ax1.plot(cont1, 'b-', label='Continuity Loss', linewidth=2)
ax1_twin = ax1.twinx()
ax1_twin.plot(mass1, 'r--', label='Mass Error', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Continuity Loss', color='b')
ax1_twin.set_ylabel('Mass Conservation Error', color='r')
ax1.set_title('Run 1: With Normalization + GradNorm')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')

# Run 2
ax2.plot(cont2, 'b-', label='Continuity Loss', linewidth=2)
ax2_twin = ax2.twinx()
ax2_twin.plot(mass2, 'r--', label='Mass Error', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Continuity Loss', color='b')
ax2_twin.set_ylabel('Mass Conservation Error', color='r')
ax2.set_title('Run 2: No Normalization/GradNorm')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left')
ax2_twin.legend(loc='upper right')

plt.tight_layout()
plt.savefig('results/continuity_vs_mass_error.png', dpi=150)
print("\nPlot saved: results/continuity_vs_mass_error.png")
