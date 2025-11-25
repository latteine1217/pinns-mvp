#!/usr/bin/env python3
"""
比較不同訓練運行的質量守恆誤差趨勢
"""
import re
import matplotlib.pyplot as plt

def parse_log(filename):
    """從日誌中提取質量守恆誤差"""
    errors = []
    with open(filename, 'r') as f:
        for line in f:
            match = re.search(r'質量守恆誤差:\s+([\d.e+-]+)', line)
            if match:
                errors.append(float(match.group(1)))
    return errors

# 比較兩個訓練運行
log1 = "log/kolm_weights_check.log"  # 有 normalization + GradNorm
log2 = "log/kolm_no_scaling.log"     # 無 normalization/GradNorm

errors1 = parse_log(log1)
errors2 = parse_log(log2)

print(f"Run 1 (with normalization): {len(errors1)} epochs")
print(f"  Initial: {errors1[0]:.2f}, Final: {errors1[-1]:.2f}")
print(f"  Min: {min(errors1):.2f}, Max: {max(errors1):.2f}")
print()
print(f"Run 2 (no normalization): {len(errors2)} epochs")
print(f"  Initial: {errors2[0]:.2f}, Final: {errors2[-1]:.2f}")
print(f"  Min: {min(errors2):.2f}, Max: {max(errors2):.2f}")

# 繪圖比較
plt.figure(figsize=(10, 6))
plt.plot(errors1[:50], 'o-', label='With Normalization + GradNorm', alpha=0.7)
plt.plot(errors2[:50], 's-', label='No Normalization/GradNorm', alpha=0.7)
plt.axhline(y=0.01, color='r', linestyle='--', label='Target (0.01)')
plt.xlabel('Epoch')
plt.ylabel('Mass Conservation Error')
plt.title('Comparison: Mass Conservation Error Trends')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig('results/mass_error_comparison.png', dpi=150)
print("\nPlot saved: results/mass_error_comparison.png")
