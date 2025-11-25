# Re=200 DNS 完成後檢查清單

## ⏱️ 預估完成時間
- **當前進度**: ~20% (3,900/20,000)
- **預估完成**: 2025-11-21 約 23:30 (4.5小時後)

---

## ✅ 驗證流程 (按順序執行)

### 第 1 步：基本檢查 (1 分鐘)
```bash
# 1.1 確認檔案存在
ls -lh data/kolmogorov_dns_re200_512x512_v2.h5

# 1.2 檢查檔案大小（應為 1-2 GB）
du -h data/kolmogorov_dns_re200_512x512_v2.h5

# 1.3 查看最後 10 行日誌
tail -10 log/dns_generation_re200_v2.log
```

**預期輸出**:
- 檔案大小: 1-2 GB
- 最後一行包含 "DNS 求解完成"
- 無 NaN 或 Inf 錯誤

---

### 第 2 步：快速物理驗證 (2 分鐘)
```bash
python scripts/validate_dns_v2_quick.py \
    --dns-file data/kolmogorov_dns_re200_512x512_v2.h5
```

**成功標準**:
- ✅ v_max(t=10) > 1.0 (必須通過)
- ✅ v_max(t=10) > 5.0 (理想目標)
- ✅ Divergence < 1e-8
- ✅ 無 NaN/Inf 值

---

### 第 3 步：完整驗證與視覺化 (5 分鐘)
```bash
python scripts/validate_dns_final.py \
    --dns-file data/kolmogorov_dns_re200_512x512_v2.h5 \
    --output results/dns_validation_re200_v2/
```

**輸出檔案**:
- `results/dns_validation_re200_v2/final_validation_report.png` (6面板綜合圖)
- `results/dns_validation_re200_v2/velocity_magnitude_t10_HD.png` (高清速度場)

**檢查要點**:
- v-velocity 場有清晰的渦旋結構
- 動能隨時間增長並趨於穩定
- 能譜峰值在 k ≈ 3-4
- 散度誤差 < 1e-8

---

### 第 4 步：生成 QR 感測器 (3 分鐘)
```bash
python scripts/generate_sensors_k500.py \
    --input data/kolmogorov_dns_re200_512x512_v2.h5 \
    --K 100 \
    --output data/kolmogorov_qr_sensors_re200_K100_v2.npz
```

**成功標準**:
- ✅ 條件數 < 300 (可接受)
- ✅ POD 能量比 > 0.99
- ✅ 感測點空間分佈均勻

---

### 第 5 步：Re=100 vs Re=200 對比 (3 分鐘)
```bash
python3 << 'SCRIPT'
import h5py
import numpy as np
import matplotlib.pyplot as plt

# 讀取兩個數據集
with h5py.File('data/kolmogorov_dns_re100_512x512_v2.h5', 'r') as f100:
    u100 = f100['u'][100]  # t=10
    v100 = f100['v'][100]
    t100 = f100['time'][:]
    
with h5py.File('data/kolmogorov_dns_re200_512x512_v2.h5', 'r') as f200:
    u200 = f200['u'][100]  # t=10
    v200 = f200['v'][100]
    t200 = f200['time'][:]

# 計算統計
mag100 = np.sqrt(u100**2 + v100**2)
mag200 = np.sqrt(u200**2 + v200**2)

vmax100 = np.abs(v100).max()
vmax200 = np.abs(v200).max()

# 創建對比圖
fig = plt.figure(figsize=(16, 10))

# Re=100
ax1 = plt.subplot(2, 3, 1)
im1 = ax1.imshow(mag100.T, cmap='hot', origin='lower')
ax1.set_title(f'Re=100 Velocity Magnitude\n|v|_max = {vmax100:.2f}', 
              fontsize=14, fontweight='bold')
plt.colorbar(im1, ax=ax1)

ax2 = plt.subplot(2, 3, 2)
ax2.imshow(v100.T, cmap='RdBu_r', origin='lower', vmin=-vmax100, vmax=vmax100)
ax2.set_title(f'Re=100 v-velocity', fontsize=14, fontweight='bold')

ax3 = plt.subplot(2, 3, 3)
ax3.imshow(u100.T, cmap='RdBu_r', origin='lower')
ax3.set_title(f'Re=100 u-velocity', fontsize=14, fontweight='bold')

# Re=200
ax4 = plt.subplot(2, 3, 4)
im4 = ax4.imshow(mag200.T, cmap='hot', origin='lower')
ax4.set_title(f'Re=200 Velocity Magnitude\n|v|_max = {vmax200:.2f}', 
              fontsize=14, fontweight='bold')
plt.colorbar(im4, ax=ax4)

ax5 = plt.subplot(2, 3, 5)
ax5.imshow(v200.T, cmap='RdBu_r', origin='lower', vmin=-vmax200, vmax=vmax200)
ax5.set_title(f'Re=200 v-velocity', fontsize=14, fontweight='bold')

ax6 = plt.subplot(2, 3, 6)
ax6.imshow(u200.T, cmap='RdBu_r', origin='lower')
ax6.set_title(f'Re=200 u-velocity', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('results/re100_vs_re200_comparison.png', dpi=150, bbox_inches='tight')

# 打印比較統計
print(f"╔═══════════════════════════════════════════════════════╗")
print(f"║      Re=100 vs Re=200 Comparison at t=10             ║")
print(f"╚═══════════════════════════════════════════════════════╝")
print(f"\n📊 Velocity Statistics:")
print(f"  Re=100:")
print(f"    v_max   = {vmax100:.3f}")
print(f"    u_mean  = {u100.mean():.3f}")
print(f"    |v|_avg = {mag100.mean():.3f}")
print(f"\n  Re=200:")
print(f"    v_max   = {vmax200:.3f}")
print(f"    u_mean  = {u200.mean():.3f}")
print(f"    |v|_avg = {mag200.mean():.3f}")
print(f"\n🔍 Improvement:")
improvement = (vmax200 / vmax100 - 1) * 100
print(f"    v_max improvement: {improvement:+.1f}%")
print(f"\n✅ 對比圖已保存: results/re100_vs_re200_comparison.png")
SCRIPT
```

---

### 第 6 步：動能演化對比 (2 分鐘)
```bash
python3 << 'SCRIPT'
import h5py
import numpy as np
import matplotlib.pyplot as plt

# 讀取時間序列數據
with h5py.File('data/kolmogorov_dns_re100_512x512_v2.h5', 'r') as f100:
    diag100 = f100['diagnostics']
    t100 = diag100['time'][:]
    ke100 = diag100['kinetic_energy'][:]
    ens100 = diag100['enstrophy'][:]
    
with h5py.File('data/kolmogorov_dns_re200_512x512_v2.h5', 'r') as f200:
    diag200 = f200['diagnostics']
    t200 = diag200['time'][:]
    ke200 = diag200['kinetic_energy'][:]
    ens200 = diag200['enstrophy'][:]

# 創建演化圖
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# 動能演化
ax1 = axes[0]
ax1.plot(t100, ke100, 'b-', linewidth=2, label='Re=100', alpha=0.8)
ax1.plot(t200, ke200, 'r-', linewidth=2, label='Re=200', alpha=0.8)
ax1.axvline(x=10, color='gray', linestyle='--', alpha=0.5, label='t=10 (evaluation)')
ax1.set_xlabel('Time (t)', fontsize=12)
ax1.set_ylabel('Kinetic Energy', fontsize=12)
ax1.set_title('Kinetic Energy Evolution: Re=100 vs Re=200', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# 渦量演化
ax2 = axes[1]
ax2.plot(t100, ens100, 'b-', linewidth=2, label='Re=100', alpha=0.8)
ax2.plot(t200, ens200, 'r-', linewidth=2, label='Re=200', alpha=0.8)
ax2.axvline(x=10, color='gray', linestyle='--', alpha=0.5, label='t=10 (evaluation)')
ax2.set_xlabel('Time (t)', fontsize=12)
ax2.set_ylabel('Enstrophy', fontsize=12)
ax2.set_title('Enstrophy Evolution: Re=100 vs Re=200', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/re100_vs_re200_evolution.png', dpi=150, bbox_inches='tight')

print("✅ 演化對比圖已保存: results/re100_vs_re200_evolution.png")
SCRIPT
```

---

## 📊 預期結果總結

### Re=200 vs Re=100 預期對比

| 指標 | Re=100 (實測) | Re=200 (預期) | 改善 |
|------|--------------|--------------|------|
| **v_max(t=10)** | 3.43 | 6.0 | +75% |
| **轉捩時間** | ~8 | ~3-4 | -50% |
| **渦量峰值** | ~80 | ~150 | +87% |
| **動能峰值** | ~4 | ~18 | +350% |

### 成功判定標準

#### 🟢 完全成功
- v_max(t=10) > 5.0
- v_max > Re=100 的 1.5 倍
- 散度誤差 < 1e-8

#### 🟡 部分成功
- v_max(t=10) > 3.5 (優於 Re=100)
- v_max > Re=100 的 1.0 倍
- 散度誤差 < 1e-6

#### 🔴 失敗
- v_max(t=10) < 3.0
- 出現 NaN/Inf
- 散度誤差 > 1e-5

---

## 🚀 下一步規劃

### 如果 Re=200 成功

1. **更新文檔**
   - 更新 `DNS_RE200_SIMULATION_REPORT.md`
   - 更新 `DNS_REYNOLDS_COMPARISON.md`
   - 更新 `SESSION_SUMMARY_20251121.md`

2. **準備 PINNs 訓練**
   - 使用 Re=200 資料集訓練
   - 配置檔案: `configs/kolmogorov_experiments/kolmogorov_2d_re200.yml`
   - K=100 QR 感測器

3. **論文素材**
   - 所有對比圖
   - 驗證報告
   - 統計表格

### 如果 Re=200 未達預期

1. **診斷分析**
   - 檢查 v(t) 演化曲線
   - 分析轉捩延遲原因
   - 能譜分析

2. **可能調整**
   - 提高擾動振幅 (amplitude = 15?)
   - 延長模擬時間 (T_end = 30?)
   - 嘗試 Re=150 作為中間值

---

## 💾 備份建議

```bash
# 完成後備份關鍵檔案
mkdir -p backup/re200_v2/
cp data/kolmogorov_dns_re200_512x512_v2.h5 backup/re200_v2/
cp data/kolmogorov_qr_sensors_re200_K100_v2.npz backup/re200_v2/
cp -r results/dns_validation_re200_v2/ backup/re200_v2/
cp log/dns_generation_re200_v2.log backup/re200_v2/
tar -czf backup/re200_v2_$(date +%Y%m%d).tar.gz backup/re200_v2/
```

---

**最後更新**: 2025-11-21 19:00  
**預計完成**: 2025-11-21 23:30
