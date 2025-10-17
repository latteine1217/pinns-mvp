# JHTDB API 問題診斷報告

## 📅 日期：2025-10-16

## 🎯 目標
從 JHTDB 獲取真實的多時間步數據（50 個時間步），用於 QR-Pivot 感測點選擇，以降低條件數。

## ❌ 發現的問題

### 1. **HTTP API 完全失效**
**症狀**：所有 SOAP 請求返回 "Base64 數據為空"

**測試代碼**：
```bash
python scripts/fetch_temporal_snapshots.py \
  --mode 2d --n_time 50 --resolution 128 64 \
  --time_start 0.0 --time_end 10.0 \
  --output data/jhtdb/channel_flow_re1000/temporal/xy_n50_real.npz
```

**結果**：
```
ERROR - Base64 數據為空
ERROR - 數據提取失敗: 響應中的數據為空
WARNING - Token 驗證失敗，啟用 Mock fallback 機制
```

**影響**：所有緩存數據（`data/jhtdb/cache/channel_*.h5`）**都是 Mock 數據**，而非真實 JHTDB 數據。

---

### 2. **物理座標轉換邏輯錯誤**
**位置**：`pinnx/dataio/jhtdb_client.py` 第 986 行

**錯誤代碼**：
```python
start_int = [max(1, int(s) + 1) for s in start]
```

**問題**：
- 直接將物理座標（如 `0.0, -1.0, 4.558`）轉為整數，未考慮數據集的物理範圍
- Channel Flow 的範圍：`L_x × L_y × L_z = 8π × 2 × 3π`
- 網格：`2048 × 512 × 1536`

**正確轉換應為**：
```python
# 物理座標 → 網格索引
L = [8*np.pi, 2, 3*np.pi]  # 物理範圍
N = [2048, 512, 1536]      # 網格解析度
start_int = [max(1, int((s / L[i]) * N[i]) + 1) for i, s in enumerate(start)]
```

---

### 3. **pyJHTDB 不兼容**
**症狀**：`pip install pyJHTDB` 失敗

**錯誤**：
```python
AttributeError: module 'numpy' has no attribute 'int'.
`np.int` was a deprecated alias for the builtin `int`.
```

**原因**：pyJHTDB 使用了已棄用的 `np.int`（NumPy 1.20 後移除）

**可能解決方案**：
1. 從源碼安裝修改後的 pyJHTDB（替換 `np.int` → `int`）
2. 降級 NumPy 到 < 1.20（不推薦，會破壞其他依賴）

---

## 🔍 診斷證據

### 緩存數據完全相同
```bash
# 檢查前 5 個緩存文件
python3 -c "
import h5py, glob, numpy as np
files = sorted(glob.glob('data/jhtdb/cache/channel_*.h5'))[:5]
for f in files:
    with h5py.File(f, 'r') as h:
        u = h['u'][:]
        print(f'{f.split(\"/\")[-1][:30]}: mean={u.mean():.4f}')
"
```

**輸出**：
```
channel_03c1e76d2a7afc6d7367c4: mean=9.8408
channel_0b14aae916449784a3c243: mean=9.8408
channel_0d9cefe09ff14340802bcd: mean=9.8408
channel_0f650bf764473f2b7b1401: mean=9.8408
channel_113a121688123c807634bb: mean=9.8408
```

**結論**：所有文件統計量**完全相同** → Mock fallback 生成了重複數據

---

### 時間步元數據正確，但數據相同
```bash
# 檢查元數據中的 timestep
python3 -c "
import h5py, json, glob
files = sorted(glob.glob('data/jhtdb/cache/channel_*.h5'))[:5]
for f in files:
    with h5py.File(f, 'r') as h:
        params = json.loads(h.attrs['query_params'])
        print(f'timestep={params[\"timestep\"]:4d}')
"
```

**輸出**：
```
timestep= 125
timestep=1318
timestep= 533
timestep=1193
timestep=1004
```

**結論**：時間步參數**不同**，但數據**相同** → API 請求失敗後，fallback 生成了**相同的**模擬數據

---

## ✅ 已完成的修復

### 修復 1：數據提取邏輯
**文件**：`scripts/fetch_temporal_snapshots.py` 第 155-167 行

**原始代碼**（錯誤）：
```python
data = manager.fetch_cutout(...)
u_snapshots[i] = data['u'].squeeze()  # ❌ KeyError: 'u'
```

**修正後**：
```python
result = manager.fetch_cutout(...)
data = result['data']  # ✅ 正確訪問數據
u_snapshots[i] = data['u'].squeeze()
logger.info(f"來源: {'緩存' if result.get('from_cache') else 'JHTDB API'}")
```

**狀態**：✅ 已修復並驗證

---

## 🚧 待修復問題

### 優先級 1：修復 HTTP API 座標轉換
**任務**：修改 `pinnx/dataio/jhtdb_client.py` 第 979-1027 行

**需要的數據**：
- Channel Flow 物理範圍：`[8π, 2, 3π]`
- 網格解析度：`[2048, 512, 1536]`

**修復計畫**：
1. 創建數據集配置映射（物理範圍 + 網格解析度）
2. 修改 `_call_get_any_cutout_web()` 的座標轉換邏輯
3. 添加單元測試驗證轉換正確性

**預估時間**：1-2 小時

---

### 優先級 2：修復 pyJHTDB 兼容性
**任務**：從源碼安裝修改後的 pyJHTDB

**步驟**：
```bash
# 克隆 pyJHTDB
git clone https://github.com/idies/pyJHTDB.git external/pyJHTDB
cd external/pyJHTDB

# 替換 np.int → int
find . -name "*.py" -exec sed -i '' 's/np\.int,/int,/g' {} \;
find . -name "*.py" -exec sed -i '' 's/np\.int)/int)/g' {} \;

# 安裝
pip install -e .
```

**預估時間**：30 分鐘

---

## 🔄 備選方案

### 方案 A：使用改進的 Mock 數據（推薦短期）
**優點**：
- ✅ 立即可用，無需等待 API 修復
- ✅ 可驗證 QR-Pivot 流程的正確性
- ✅ 數據包含物理真實性（湍流模式）

**缺點**：
- ❌ 不是真實 JHTDB 數據
- ❌ 無法用於論文級結果

**使用方式**：
```bash
# 已生成的 Mock 數據（需重新生成時間演化版本）
python scripts/fetch_temporal_snapshots.py \
  --mode 2d --n_time 50 --resolution 128 64 \
  --use-mock \
  --output data/jhtdb/channel_flow_re1000/temporal/xy_n50_mock_improved.npz
```

---

### 方案 B：使用單時間步真實數據
**描述**：從 JHTDB 獲取單個時間步的大範圍 cutout，使用空間多樣性代替時間演化

**優點**：
- ✅ 避免時間演化的 API 問題
- ✅ 空間多樣性可能足以降低條件數

**缺點**：
- ❌ 需要更大的空間域（增加數據量）
- ❌ 無法捕捉時間動力學特徵

**實施方式**：
```bash
# 獲取更大的空間 cutout（如 256×128 而非 128×64）
python scripts/fetch_channel_flow.py \
  --mode cutout \
  --resolution 256 128 --z_pos 4.558 \
  --output data/jhtdb/channel_flow_re1000/slices/xy_pos4.558_256x128.npz
```

---

### 方案 C：完整修復 JHTDB API（推薦長期）
**任務清單**：
1. ✅ 修復數據提取邏輯（已完成）
2. 🚧 修復座標轉換邏輯（優先級 1）
3. 🚧 修復 pyJHTDB 兼容性（優先級 2）
4. 🔲 添加完整的集成測試
5. 🔲 更新文檔

**預估總時間**：3-4 小時

---

## 📊 條件數比較

| 數據來源 | 條件數 | 狀態 | 備註 |
|---------|--------|------|------|
| **Mock 數據（當前）** | `1.9e+16` | ❌ 失敗 | 所有時間步相同 → 秩=1 |
| **Mock 數據（理論）** | `~1e+4` | 🔲 未測試 | 需重新生成時間演化版本 |
| **真實 JHTDB（目標）** | `<100` | 🔲 未測試 | 需修復 API |

---

## 🎯 建議下一步

### 短期（今日）：
1. **生成改進的 Mock 數據**（包含真實時間演化）
2. **驗證 QR-Pivot 流程**能正確降低條件數
3. **記錄當前進度**到 `context/decisions_log.md`

### 中期（本周）：
1. **修復 HTTP API 座標轉換**（優先級 1）
2. **安裝修改後的 pyJHTDB**（優先級 2）
3. **重新測試真實數據獲取**

### 長期（論文）：
1. **完整集成測試**（真實數據 + QR-Pivot）
2. **性能基準測試**（條件數 vs K）
3. **文檔與可重現性**

---

## 📝 參考資料

- **JHTDB 官方文檔**：http://turbulence.pha.jhu.edu/
- **GetAnyCutoutWeb API**：http://turbulence.pha.jhu.edu/webquery/query.aspx
- **pyJHTDB GitHub**：https://github.com/idies/pyJHTDB
- **Channel Flow 參數**：`AGENTS.md` 第 X 行

---

## ✍️ 記錄者
- **日期**：2025-10-16
- **會話**：QR-Pivot 條件數改進
- **相關任務**：TASK-QR-IMPROVEMENT-001
