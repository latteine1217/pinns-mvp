# 資料類型說明文檔

## 📌 重要澄清：本專案使用「瞬時流場快照」

**最後更新時間**：2025-12-15

---

## 🎯 資料類型定義

### 瞬時流場快照 (Instantaneous Snapshot)

本專案從 JHTDB (Johns Hopkins Turbulence Database) 擷取的 Channel Flow Re_tau=1000 資料為：

- **資料類型**：瞬時流場快照 (Instantaneous Snapshot)
- **時間點**：`timestep=0` 對應某一時刻的完整 3D 湍流場
- **物理意義**：捕捉某一瞬間的湍流結構（渦旋、剪切層、脈動等）
- **應用場景**：
  - ✅ 瞬時流場重建
  - ✅ 湍流結構識別
  - ✅ 稀疏感測器逆問題
  - ✅ 瞬時速度/壓力場預測

### ⚠️ 非時間平均場

本專案 **不使用** 時間平均流場 (Time-Averaged Field)，因此：

- ❌ 不適合直接比較「統計穩態」量（如平均速度剖面 $\langle u \rangle_t$）
- ❌ 不提供 Reynolds 平均流場（需手動後處理多個時間步）
- ❌ 不包含湍流統計量（如 Reynolds 應力 $\langle u'u' \rangle$）

---

## 📂 相關修改記錄

### 修改文件列表

| 文件路徑 | 修改內容 | 修改日期 |
|---------|---------|----------|
| `scripts/tools/fetch_channel_flow.py` | 移除誤導性時間平均註解 | 2025-12-15 |
| `scripts/tools/fetch_channel_flow.py` | 添加資料類型警告說明 | 2025-12-15 |
| `scripts/tools/fetch_channel_flow.py` | 標記 `time_average_window` 為 DEPRECATED | 2025-12-15 |

### 具體修改點

#### 1. 文件開頭說明（第 8-12 行）

**修改前：**
```python
2. 2D 切片提取與時間平均處理
```

**修改後：**
```python
⚠️  重要說明：資料類型為「瞬時流場快照」(Instantaneous Snapshot)
    - 本腳本擷取的是某一時刻的完整 3D 湍流場（timestep=0）
    - 並非統計平均場或時間平均場
    - 適合用於瞬時流場重建、湍流結構識別等研究
    - 如需時間平均場，需手動對多個 timesteps 進行後處理

2. 2D 切片提取與預處理 (瞬時流場快照)
```

#### 2. ChannelFlowConfig 類別（第 72 行）

**修改前：**
```python
time_average_window: Optional[List[float]] = None  # [20.0, 26.0]
```

**修改後：**
```python
time_average_window: Optional[List[float]] = None  # DEPRECATED: 未實作時間平均 [20.0, 26.0]
```

#### 3. fetch_cutout_data 方法（第 302 行）

**修改前：**
```python
timestep=0  # 使用瞬時資料，後續進行時間平均
```

**修改後：**
```python
timestep=0  # 擷取瞬時流場快照 (instantaneous snapshot at t=0)
```

---

## 🔬 物理意義與評估指標

### 瞬時場 vs 平均場的差異

| 特性 | 瞬時場 (本專案) | 時間平均場 |
|------|----------------|-----------|
| **物理量** | $u(x, y, z, t=0)$ | $\langle u(x, y, z) \rangle_t$ |
| **脈動特性** | 包含全部湍流脈動 | 脈動已被平均抹除 |
| **誤差範圍** | 通常較大（10-20%） | 通常較小（1-5%） |
| **評估指標** | 相對 L2、點對點誤差 | 統計量比較、剖面誤差 |
| **應用場景** | 瞬時預測、結構識別 | 工程設計、平均特性 |

### 合理的誤差閾值

對於瞬時流場重建，以下為合理的驗收標準：

- ✅ **速度場相對 L2 誤差** ≤ 15-20%（瞬時場含湍流脈動）
- ✅ **壓力場相對 L2 誤差** ≤ 20-25%（壓力脈動更劇烈）
- ✅ **散度誤差** ≤ 1e-2（守恆性檢查）
- ✅ **感測點 RMSE** ≤ 0.1 * std(真實場)

⚠️ **注意**：瞬時場誤差通常是時間平均場的 2-3 倍，這是正常的物理現象！

---

## 📚 技術參考

### JHTDB Channel Flow 資料集

- **資料集名稱**：Channel Flow (Re_tau=1000)
- **時間範圍**：t ∈ [0, 26.0]
- **時間步長**：dt = 0.0065
- **資料類型**：每個 timestep 為獨立的瞬時快照
- **官方文檔**：https://turbulence.pha.jhu.edu/

### 相關文獻

1. **Lee & Moser (2015)**：Direct numerical simulation of turbulent channel flow up to Re_τ ≈ 5200
   - 說明 DNS 資料集的瞬時特性
   - 定義時間平均統計量的計算方法

2. **Brunton et al. (2016)**：Compressed sensing and dynamic mode decomposition
   - 稀疏感測器重建瞬時流場
   - 討論瞬時場 vs 統計場的重建差異

---

## 🚀 未來擴展（如需時間平均功能）

如果未來需要實作時間平均流場處理，建議的實施步驟：

### 1. 多時間步資料擷取

```python
def fetch_time_averaged_cutout(self, t_start: float, t_end: float, n_samples: int):
    """擷取並平均多個時間步的流場"""
    timesteps = np.linspace(t_start, t_end, n_samples)
    fields_accumulated = None
    
    for t in timesteps:
        cutout = self.jhtdb_manager.fetch_cutout(..., timestep=t)
        if fields_accumulated is None:
            fields_accumulated = cutout['data']
        else:
            for var in fields_accumulated:
                fields_accumulated[var] += cutout['data'][var]
    
    # 時間平均
    for var in fields_accumulated:
        fields_accumulated[var] /= n_samples
    
    return fields_accumulated
```

### 2. 統計量計算

```python
def compute_turbulence_statistics(self, snapshots: List[Dict]):
    """從多個瞬時快照計算 Reynolds 應力等統計量"""
    # 平均場
    mean_u = np.mean([s['u'] for s in snapshots], axis=0)
    
    # Reynolds 應力
    u_prime = [s['u'] - mean_u for s in snapshots]
    reynolds_stress = np.mean([up * up for up in u_prime], axis=0)
    
    return {'mean': mean_u, 'reynolds_stress': reynolds_stress}
```

### 3. 配置文件擴展

```yaml
data:
  type: "time_averaged"  # 或 "instantaneous"
  time_averaging:
    enabled: true
    window: [20.0, 26.0]  # 統計穩態區間
    n_samples: 100
```

---

## ✅ 驗證清單

使用者或開發者可以用以下清單確認資料類型的正確性：

- [ ] 確認 `fetch_channel_flow.py` 的文件說明包含瞬時場警告
- [ ] 確認 `timestep=0` 的註解說明為「瞬時快照」
- [ ] 確認 `time_average_window` 標記為 DEPRECATED
- [ ] 確認評估指標的閾值適合瞬時場（而非統計場）
- [ ] 確認相關文檔（README、技術報告）中明確說明資料類型

---

## 📞 聯絡資訊

如有任何關於資料類型、物理意義或評估標準的問題，請參考：

- 技術文檔：`docs/TECHNICAL_DOCUMENTATION.md`
- 配置參考：`docs/CONFIG_REFERENCE.md`
- 疑難排解：`docs/TROUBLESHOOTING.md`

---

**文檔版本**：1.0  
**建立日期**：2025-12-15  
**維護者**：PINNs-MVP 開發團隊
