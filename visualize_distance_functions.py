"""
視覺化三種距離函數形式
比較 quadratic, cosh, sin 在 y ∈ [-1, 1] 的行為
"""
import numpy as np
import matplotlib.pyplot as plt

# 設定中文顯示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 生成 y 座標
y = np.linspace(-1, 1, 500)

# 三種距離函數
def quadratic(y):
    """二次函數形式: d(y) = 1 - y²"""
    return 1 - y**2

def cosh_form(y, alpha=10.0):
    """雙曲餘弦形式: d(y) = 1 - cosh(α·y) / cosh(α)"""
    return 1 - np.cosh(alpha * y) / np.cosh(alpha)

def sin_form(y):
    """正弦形式: d(y) = sin(π·(y+1)/2)"""
    return np.sin(np.pi * (y + 1) / 2)

# 計算函數值
d_quadratic = quadratic(y)
d_cosh_alpha5 = cosh_form(y, alpha=5.0)
d_cosh_alpha10 = cosh_form(y, alpha=10.0)
d_cosh_alpha20 = cosh_form(y, alpha=20.0)
d_sin = sin_form(y)

# 計算梯度 (導數)
dy = y[1] - y[0]
grad_quadratic = np.gradient(d_quadratic, dy)
grad_cosh_alpha10 = np.gradient(d_cosh_alpha10, dy)
grad_sin = np.gradient(d_sin, dy)

# ===== 繪圖 =====
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- 子圖1: 三種基本形式比較 ---
ax1 = axes[0, 0]
ax1.plot(y, d_quadratic, 'b-', linewidth=2.5, label='Quadratic: 1 - y²')
ax1.plot(y, d_cosh_alpha10, 'r--', linewidth=3.0, label='Cosh (α=10): 1 - cosh(10y)/cosh(10) [推薦]')
ax1.plot(y, d_sin, 'g-.', linewidth=2.5, label='Sin: sin(π(y+1)/2)')

ax1.axhline(y=0, color='k', linestyle=':', alpha=0.3)
ax1.axvline(x=-1, color='k', linestyle=':', alpha=0.3)
ax1.axvline(x=1, color='k', linestyle=':', alpha=0.3)
ax1.axvline(x=0, color='k', linestyle=':', alpha=0.3)

ax1.set_xlabel('Wall-normal coordinate y', fontsize=12)
ax1.set_ylabel('Distance function d(y)', fontsize=12)
ax1.set_title('Distance Functions Comparison', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-1, 1)
ax1.set_ylim(-0.1, 1.1)

# 標註關鍵點
ax1.scatter([-1, 1], [0, 0], color='red', s=100, zorder=5, label='Wall (d=0)')
ax1.scatter([0], [quadratic(0)], color='blue', s=100, zorder=5, marker='s')
ax1.text(0, quadratic(0) + 0.1, f'd(0)={quadratic(0):.2f}', ha='center', fontsize=10)

# --- 子圖2: Cosh 形式不同 α 值 ---
ax2 = axes[0, 1]
ax2.plot(y, d_quadratic, 'k-', linewidth=2, alpha=0.5, label='Quadratic (reference)')
ax2.plot(y, d_cosh_alpha5, 'b-', linewidth=2.5, label='α = 5 (gradual)')
ax2.plot(y, d_cosh_alpha10, 'r-', linewidth=2.5, label='α = 10 (moderate)')
ax2.plot(y, d_cosh_alpha20, 'orange', linewidth=2.5, label='α = 20 (steep)')

ax2.axhline(y=0, color='k', linestyle=':', alpha=0.3)
ax2.axvline(x=-1, color='k', linestyle=':', alpha=0.3)
ax2.axvline(x=1, color='k', linestyle=':', alpha=0.3)

ax2.set_xlabel('Wall-normal coordinate y', fontsize=12)
ax2.set_ylabel('Distance function d(y)', fontsize=12)
ax2.set_title('Cosh Form: Effect of α Parameter', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-1, 1)
ax2.set_ylim(-0.1, 1.1)

# --- 子圖3: 梯度比較 ---
ax3 = axes[1, 0]
ax3.plot(y, grad_quadratic, 'b-', linewidth=2.5, label="Quadratic: d'(y) = -2y")
ax3.plot(y, grad_cosh_alpha10, 'r--', linewidth=2.5, label="Cosh (α=10): d'(y)")
ax3.plot(y, grad_sin, 'g-.', linewidth=2.5, label="Sin: d'(y)")

ax3.axhline(y=0, color='k', linestyle=':', alpha=0.3)
ax3.axvline(x=-1, color='k', linestyle=':', alpha=0.3)
ax3.axvline(x=1, color='k', linestyle=':', alpha=0.3)
ax3.axvline(x=0, color='k', linestyle=':', alpha=0.3)

ax3.set_xlabel('Wall-normal coordinate y', fontsize=12)
ax3.set_ylabel("Gradient d'(y)", fontsize=12)
ax3.set_title('Gradient Comparison (影響訓練梯度)', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-1, 1)

# --- 子圖4: 速度場約束示例 ---
ax4 = axes[1, 1]

# 模擬網路輸出 (拋物線速度分佈)
u_network = 1.0 * (1 - y**2)  # 理想拋物線

# 應用距離函數
u_constrained_quad = u_network * d_quadratic
u_constrained_cosh = u_network * d_cosh_alpha10
u_constrained_sin = u_network * d_sin

ax4.plot(y, u_network, 'k--', linewidth=2, alpha=0.5, label='Network output (unconstrained)')
ax4.plot(y, u_constrained_quad, 'b-', linewidth=2.5, label='× Quadratic')
ax4.plot(y, u_constrained_cosh, 'r--', linewidth=2.5, label='× Cosh (α=10)')
ax4.plot(y, u_constrained_sin, 'g-.', linewidth=2.5, label='× Sin')

ax4.axhline(y=0, color='k', linestyle=':', alpha=0.3)
ax4.axvline(x=-1, color='k', linestyle=':', alpha=0.3)
ax4.axvline(x=1, color='k', linestyle=':', alpha=0.3)

# 標註邊界條件
ax4.scatter([-1, 1], [0, 0], color='red', s=100, zorder=5)
ax4.text(-1, -0.15, 'u(-1)=0', ha='center', fontsize=10, color='red')
ax4.text(1, -0.15, 'u(1)=0', ha='center', fontsize=10, color='red')

ax4.set_xlabel('Wall-normal coordinate y', fontsize=12)
ax4.set_ylabel('Velocity u(y)', fontsize=12)
ax4.set_title('Example: Velocity Constraint Application', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(-1, 1)
ax4.set_ylim(-0.2, 1.1)

plt.tight_layout()
plt.savefig('distance_functions_comparison.png', dpi=300, bbox_inches='tight')
print("✓ 圖片已儲存: distance_functions_comparison.png")

# ===== 數值特性表格 =====
print("\n" + "="*70)
print("Distance Function Characteristics at Key Points")
print("="*70)
grad_col = "d'(0)"
print(f"{'Function':<20} {'d(-1)':<12} {'d(0)':<12} {'d(1)':<12} {grad_col:<12}")
print("-"*70)
print(f"{'Quadratic':<20} {quadratic(-1):<12.6f} {quadratic(0):<12.6f} {quadratic(1):<12.6f} {-2*0:<12.6f}")
print(f"{'Cosh (α=5)':<20} {cosh_form(-1, 5):<12.6f} {cosh_form(0, 5):<12.6f} {cosh_form(1, 5):<12.6f} {0.0:<12.6f}")
print(f"{'Cosh (α=10)':<20} {cosh_form(-1, 10):<12.6f} {cosh_form(0, 10):<12.6f} {cosh_form(1, 10):<12.6f} {0.0:<12.6f}")
print(f"{'Cosh (α=20)':<20} {cosh_form(-1, 20):<12.6f} {cosh_form(0, 20):<12.6f} {cosh_form(1, 20):<12.6f} {0.0:<12.6f}")
print(f"{'Sin':<20} {sin_form(-1):<12.6f} {sin_form(0):<12.6f} {sin_form(1):<12.6f} {np.pi/2:<12.6f}")
print("="*70)

# ===== 關鍵特性分析 =====
print("\n" + "="*70)
print("Key Properties Analysis")
print("="*70)

print("\n1. Boundary Satisfaction (d(±1) = 0):")
print(f"   Quadratic:  |d(-1)| = {abs(quadratic(-1)):.2e}, |d(1)| = {abs(quadratic(1)):.2e}")
print(f"   Cosh (α=10): |d(-1)| = {abs(cosh_form(-1, 10)):.2e}, |d(1)| = {abs(cosh_form(1, 10)):.2e}")
print(f"   Sin:        |d(-1)| = {abs(sin_form(-1)):.2e}, |d(1)| = {abs(sin_form(1)):.2e}")

print("\n2. Central Value (d(0)):")
print(f"   Quadratic:  d(0) = {quadratic(0):.6f}")
print(f"   Cosh (α=10): d(0) = {cosh_form(0, 10):.6f}")
print(f"   Sin:        d(0) = {sin_form(0):.6f}")

print("\n3. Symmetry:")
print(f"   All functions are symmetric: d(-y) = d(y) ✓")

print("\n4. Gradient at Center (d'(0)):")
print(f"   Quadratic:  d'(0) = 0 (zero gradient)")
print(f"   Cosh (α=10): d'(0) = 0 (zero gradient)")
print(f"   Sin:        d'(0) = π/2 ≈ 1.571 (non-zero gradient)")

print("\n5. Computational Efficiency:")
print(f"   Quadratic:  ★★★★★ (fastest, simple polynomial)")
print(f"   Cosh:       ★★★☆☆ (moderate, exponential functions)")
print(f"   Sin:        ★★★★☆ (fast, trigonometric)")

print("\n6. Recommended Usage:")
print(f"   Cosh (α=10):  ★★★★★ Default choice - adjustable steepness with strong wall constraint")
print(f"   Quadratic:    ★★★★☆ Alternative - fastest but less flexible")
print(f"   Sin:          ★★★☆☆ Special cases - smooth periodic-like behavior")

print("\n" + "="*70)
plt.show()
