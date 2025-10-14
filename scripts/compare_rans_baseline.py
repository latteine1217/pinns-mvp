"""
TASK-008 Phase 4 Step 7: RANS vs. 無 RANS 對比分析
對比兩個訓練結果，評估 RANS 整合的效果
"""

import re
import matplotlib.pyplot as plt
import numpy as np

def parse_training_log(log_path):
    """解析訓練日誌，提取關鍵指標"""
    epochs = []
    total_loss = []
    data_loss = []
    pde_loss = []
    wall_loss = []
    
    # RANS 相關損失（僅 RANS 啟用時有值）
    rans_loss = []
    k_equation_loss = []
    epsilon_equation_loss = []
    turbulent_viscosity_loss = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # 匹配 Epoch 日誌行
            match = re.search(r'Epoch (\d+)/\d+ \| total_loss: ([\d.]+)', line)
            if match:
                epoch = int(match.group(1))
                total = float(match.group(2))
                
                # 提取其他損失項
                data = re.search(r'data_loss: ([\d.]+)', line)
                pde = re.search(r'pde_loss: ([\d.]+)', line)
                wall = re.search(r'wall_loss: ([\d.]+)', line)
                
                epochs.append(epoch)
                total_loss.append(total)
                data_loss.append(float(data.group(1)) if data else 0.0)
                pde_loss.append(float(pde.group(1)) if pde else 0.0)
                wall_loss.append(float(wall.group(1)) if wall else 0.0)
                
                # 嘗試提取 RANS 損失
                rans = re.search(r'rans_loss: ([\d.]+)', line)
                k_eq = re.search(r'k_equation_loss: ([\d.]+)', line)
                eps_eq = re.search(r'epsilon_equation_loss: ([\d.]+)', line)
                turb_visc = re.search(r'turbulent_viscosity_loss: ([\d.]+)', line)
                
                rans_loss.append(float(rans.group(1)) if rans else 0.0)
                k_equation_loss.append(float(k_eq.group(1)) if k_eq else 0.0)
                epsilon_equation_loss.append(float(eps_eq.group(1)) if eps_eq else 0.0)
                turbulent_viscosity_loss.append(float(turb_visc.group(1)) if turb_visc else 0.0)
    
    return {
        'epochs': np.array(epochs),
        'total_loss': np.array(total_loss),
        'data_loss': np.array(data_loss),
        'pde_loss': np.array(pde_loss),
        'wall_loss': np.array(wall_loss),
        'rans_loss': np.array(rans_loss),
        'k_equation_loss': np.array(k_equation_loss),
        'epsilon_equation_loss': np.array(epsilon_equation_loss),
        'turbulent_viscosity_loss': np.array(turbulent_viscosity_loss),
    }

# 解析兩個日誌
rans_enabled = parse_training_log('log/test_rans_phase4_validation.log')
rans_disabled = parse_training_log('log/test_rans_disabled_baseline.log')

print("=" * 80)
print("TASK-008 Phase 4 Step 7: RANS vs. 無 RANS 對比分析報告")
print("=" * 80)
print()

# 1. 最終損失對比
print("1️⃣ 最終損失對比 (Epoch 100)")
print("-" * 80)
print(f"{'指標':<30} {'RANS 啟用':>15} {'RANS 禁用':>15} {'差異':>15}")
print("-" * 80)

rans_final = rans_enabled['total_loss'][-1]
baseline_final = rans_disabled['total_loss'][-1]
diff_pct = (rans_final - baseline_final) / baseline_final * 100

print(f"{'總損失 (total_loss)':<30} {rans_final:>15.4f} {baseline_final:>15.4f} {diff_pct:>+14.1f}%")
print(f"{'資料損失 (data_loss)':<30} {rans_enabled['data_loss'][-1]:>15.4f} {rans_disabled['data_loss'][-1]:>15.4f} {(rans_enabled['data_loss'][-1] - rans_disabled['data_loss'][-1]) / rans_disabled['data_loss'][-1] * 100:>+14.1f}%")
print(f"{'PDE 損失 (pde_loss)':<30} {rans_enabled['pde_loss'][-1]:>15.4f} {rans_disabled['pde_loss'][-1]:>15.4f} {(rans_enabled['pde_loss'][-1] - rans_disabled['pde_loss'][-1]) / rans_disabled['pde_loss'][-1] * 100:>+14.1f}%")
print(f"{'壁面損失 (wall_loss)':<30} {rans_enabled['wall_loss'][-1]:>15.4f} {rans_disabled['wall_loss'][-1]:>15.4f} {(rans_enabled['wall_loss'][-1] - rans_disabled['wall_loss'][-1]) / rans_disabled['wall_loss'][-1] * 100:>+14.1f}%")

if rans_enabled['rans_loss'][-1] > 0:
    print(f"{'RANS 損失 (rans_loss)':<30} {rans_enabled['rans_loss'][-1]:>15.4f} {'N/A':>15} {'N/A':>15}")

print()

# 2. 收斂速度對比
print("2️⃣ 收斂速度對比")
print("-" * 80)

# 定義收斂閾值
threshold_1 = 5.0  # 快速收斂閾值
threshold_2 = 1.0  # 深度收斂閾值

def find_convergence_epoch(loss_history, threshold):
    """找到損失首次低於閾值的 epoch"""
    idx = np.where(loss_history < threshold)[0]
    return idx[0] if len(idx) > 0 else None

conv_1_rans = find_convergence_epoch(rans_enabled['total_loss'], threshold_1)
conv_1_baseline = find_convergence_epoch(rans_disabled['total_loss'], threshold_1)

conv_2_rans = find_convergence_epoch(rans_enabled['total_loss'], threshold_2)
conv_2_baseline = find_convergence_epoch(rans_disabled['total_loss'], threshold_2)

print(f"收斂至 total_loss < {threshold_1:.1f}:")
print(f"  - RANS 啟用: {conv_1_rans if conv_1_rans else '未達到'}")
print(f"  - RANS 禁用: {conv_1_baseline if conv_1_baseline else '未達到'}")

print(f"\n收斂至 total_loss < {threshold_2:.1f}:")
print(f"  - RANS 啟用: {conv_2_rans if conv_2_rans else '未達到'}")
print(f"  - RANS 禁用: {conv_2_baseline if conv_2_baseline else '未達到'}")

print()

# 3. RANS 損失穩定性分析
print("3️⃣ RANS 損失穩定性分析")
print("-" * 80)

if np.any(rans_enabled['rans_loss'] > 0):
    rans_loss_mean = np.mean(rans_enabled['rans_loss'][1:])  # 跳過 epoch 0
    rans_loss_std = np.std(rans_enabled['rans_loss'][1:])
    rans_loss_min = np.min(rans_enabled['rans_loss'][1:])
    rans_loss_max = np.max(rans_enabled['rans_loss'][1:])
    
    print(f"RANS 總損失統計 (epoch 1-100):")
    print(f"  - 平均值: {rans_loss_mean:.4f}")
    print(f"  - 標準差: {rans_loss_std:.4f}")
    print(f"  - 最小值: {rans_loss_min:.4f}")
    print(f"  - 最大值: {rans_loss_max:.4f}")
    print(f"  - 變異係數 (CV): {rans_loss_std / rans_loss_mean * 100:.2f}%")
    
    # 湍流黏度損失分析
    turb_visc_mean = np.mean(rans_enabled['turbulent_viscosity_loss'][1:])
    turb_visc_max = np.max(rans_enabled['turbulent_viscosity_loss'])
    
    print(f"\n湍流黏度損失統計:")
    print(f"  - 平均值: {turb_visc_mean:.4f}")
    print(f"  - 最大值: {turb_visc_max:.4f}")
    print(f"  - Epoch 0 值: {rans_enabled['turbulent_viscosity_loss'][0]:.4f}")
    print(f"  - Epoch 100 值: {rans_enabled['turbulent_viscosity_loss'][-1]:.4f}")
    
    if turb_visc_mean > 10.0:
        print(f"\n  ⚠️  警告: 湍流黏度損失平均值 ({turb_visc_mean:.2f}) 仍大於目標閾值 10.0")
        print(f"      建議: 進一步降低 turbulent_viscosity_weight 或調整 k/ε 初始化")
    else:
        print(f"\n  ✅ 湍流黏度損失在可接受範圍內 (< 10.0)")

print()

# 4. 關鍵發現總結
print("4️⃣ 關鍵發現與建議")
print("-" * 80)

print(f"\n📊 總損失對比:")
if rans_final > baseline_final:
    increase = (rans_final - baseline_final) / baseline_final * 100
    print(f"  ⚠️  RANS 啟用後總損失增加 {increase:.1f}%")
    print(f"      - RANS 項貢獻: {rans_enabled['rans_loss'][-1]:.4f}")
    print(f"      - 可能原因: RANS 約束過強或初始化不當")
else:
    decrease = (baseline_final - rans_final) / baseline_final * 100
    print(f"  ✅ RANS 啟用後總損失降低 {decrease:.1f}%")

print(f"\n🏃 收斂速度:")
if conv_1_baseline and conv_1_rans:
    if conv_1_rans < conv_1_baseline:
        print(f"  ✅ RANS 加速收斂: 提前 {conv_1_baseline - conv_1_rans} epochs 達到 total_loss < {threshold_1}")
    else:
        print(f"  ⚠️  RANS 減慢收斂: 延遲 {conv_1_rans - conv_1_baseline} epochs 達到 total_loss < {threshold_1}")
else:
    print(f"  ℹ️  兩者收斂速度相近（在 100 epochs 內未達到閾值 {threshold_1}）")

print(f"\n🎯 下一步建議:")
if rans_final > baseline_final * 1.5:
    print(f"  1. RANS 損失過大，建議:")
    print(f"     - 降低 turbulent_viscosity_weight 至 0.001（當前 0.01）")
    print(f"     - 或進一步降低閾值至 50（當前 100）")
    print(f"     - 檢查 k/ε 初始化是否合理")
elif rans_final > baseline_final:
    print(f"  1. RANS 損失適中，建議:")
    print(f"     - 保持當前權重設定")
    print(f"     - 延長訓練至 500-1000 epochs 觀察長期效果")
    print(f"     - 對比最終預測場的物理一致性（質量/動量守恆）")
else:
    print(f"  1. RANS 整合成功，建議:")
    print(f"     - 進行完整物理驗證（質量守恆、能譜、壁面剪應力）")
    print(f"     - 評估湍流統計量（k, ε, ν_t）的合理性")
    print(f"     - 與 DNS 基準對比預測誤差")

print("\n" + "=" * 80)
print("分析完成！詳細曲線圖將保存至 results/comparison_rans_vs_baseline.png")
print("=" * 80)

# 5. 生成對比圖
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('TASK-008 Phase 4: RANS vs. Baseline Comparison', fontsize=16, fontweight='bold')

# (1) 總損失對比
ax = axes[0, 0]
ax.semilogy(rans_enabled['epochs'], rans_enabled['total_loss'], 'b-', linewidth=2, label='RANS Enabled')
ax.semilogy(rans_disabled['epochs'], rans_disabled['total_loss'], 'r--', linewidth=2, label='RANS Disabled (Baseline)')
ax.axhline(y=threshold_1, color='gray', linestyle=':', label=f'Threshold = {threshold_1}')
ax.set_xlabel('Epoch')
ax.set_ylabel('Total Loss')
ax.set_title('(a) Total Loss Convergence')
ax.legend()
ax.grid(True, alpha=0.3)

# (2) 資料損失對比
ax = axes[0, 1]
ax.plot(rans_enabled['epochs'], rans_enabled['data_loss'], 'b-', linewidth=2, label='RANS Enabled')
ax.plot(rans_disabled['epochs'], rans_disabled['data_loss'], 'r--', linewidth=2, label='RANS Disabled')
ax.set_xlabel('Epoch')
ax.set_ylabel('Data Loss')
ax.set_title('(b) Data Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# (3) PDE 損失對比
ax = axes[0, 2]
ax.plot(rans_enabled['epochs'], rans_enabled['pde_loss'], 'b-', linewidth=2, label='RANS Enabled')
ax.plot(rans_disabled['epochs'], rans_disabled['pde_loss'], 'r--', linewidth=2, label='RANS Disabled')
ax.set_xlabel('Epoch')
ax.set_ylabel('PDE Loss')
ax.set_title('(c) PDE Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# (4) RANS 損失項拆解
ax = axes[1, 0]
if np.any(rans_enabled['rans_loss'] > 0):
    ax.semilogy(rans_enabled['epochs'], rans_enabled['k_equation_loss'], 'g-', linewidth=2, label='k equation')
    ax.semilogy(rans_enabled['epochs'], rans_enabled['epsilon_equation_loss'], 'm-', linewidth=2, label='ε equation')
    ax.semilogy(rans_enabled['epochs'], rans_enabled['turbulent_viscosity_loss'], 'c-', linewidth=2, label='ν_t penalty')
    ax.axhline(y=10.0, color='red', linestyle='--', linewidth=1, label='Target threshold (10.0)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RANS Loss Components')
    ax.set_title('(d) RANS Loss Breakdown')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, 'No RANS Loss\n(Disabled)', ha='center', va='center', fontsize=14, color='gray')
    ax.set_title('(d) RANS Loss Breakdown')

# (5) 壁面損失對比
ax = axes[1, 1]
ax.semilogy(rans_enabled['epochs'], rans_enabled['wall_loss'], 'b-', linewidth=2, label='RANS Enabled')
ax.semilogy(rans_disabled['epochs'], rans_disabled['wall_loss'], 'r--', linewidth=2, label='RANS Disabled')
ax.set_xlabel('Epoch')
ax.set_ylabel('Wall Loss')
ax.set_title('(e) Wall Boundary Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# (6) 損失差異（絕對值）
ax = axes[1, 2]
loss_diff = rans_enabled['total_loss'] - rans_disabled['total_loss']
ax.plot(rans_enabled['epochs'], loss_diff, 'k-', linewidth=2)
ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax.fill_between(rans_enabled['epochs'], 0, loss_diff, where=(loss_diff > 0), alpha=0.3, color='red', label='RANS worse')
ax.fill_between(rans_enabled['epochs'], 0, loss_diff, where=(loss_diff <= 0), alpha=0.3, color='green', label='RANS better')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss Difference (RANS - Baseline)')
ax.set_title('(f) Loss Difference')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/comparison_rans_vs_baseline.png', dpi=150, bbox_inches='tight')
print(f"\n✅ 圖表已保存: results/comparison_rans_vs_baseline.png")

