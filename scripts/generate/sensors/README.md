# Sensor Generation Scripts

此目錄包含用於生成 Kolmogorov Flow 感測器的腳本。

## 腳本說明

### 1. `generate_kolmogorov_temporal_qr.py`

使用 **QR-Pivot** 方法生成時間序列感測器，基於時空特徵矩陣的訊息量選取最具代表性的空間點。

**特點：**
- 考慮 u, v, p 及其梯度、渦度等 10 個物理量
- 選取具有最大訊息量的空間點
- 提供條件數、覆蓋率、能量比等品質指標

**使用範例：**
```bash
python scripts/generate/sensors/generate_kolmogorov_temporal_qr.py \
  --input data/kolmogorov_dns/kolmogorov_dns_100.npy \
  --output data/kolmogorov_sensors/re100 \
  --K 50 100 200 400 \
  --time-range 0.0 20.0 \
  --time-stride 1 \
  --include-dns-values \
  --dns-values data/kolmogorov_dns/kolmogorov_dns_100.npy
```

### 2. `generate_kolmogorov_temporal_random.py`

使用 **隨機取樣** 方法生成時間序列感測器，作為 QR-Pivot 方法的對照基準。

**特點：**
- 完全隨機選取空間點（不考慮訊息量）
- 使用固定隨機種子確保可重現性
- 保持與 QR-Pivot 相同的檔案格式以便對比

**使用範例：**
```bash
python scripts/generate/sensors/generate_kolmogorov_temporal_random.py \
  --input data/kolmogorov_dns/kolmogorov_dns_100.npy \
  --output data/kolmogorov_sensors/re100 \
  --K 50 100 200 400 \
  --time-range 0.0 20.0 \
  --time-stride 1 \
  --include-dns-values \
  --dns-values data/kolmogorov_dns/kolmogorov_dns_100.npy \
  --seed 42
```

## 參數說明

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--input` | 輸入 DNS/LES NPY 檔案路徑 | 必填 |
| `--output` | 輸出資料夾路徑 | `./data/sensors/kolmogorov` |
| `--K` | 感測器數量列表 | `[50, 100, 200, 400]` |
| `--time-range` | 時間範圍 [t_start, t_end] (秒) | `[0.0, 20.0]` |
| `--time-stride` | 時間採樣間隔（每幾個時間步取一個快照） | `1` |
| `--include-dns-values` | 是否附加 DNS time series values | `False` |
| `--dns-values` | DNS NPY 檔案路徑（用於 values） | `None` |
| `--seed` | 隨機種子（僅 random 腳本） | `42` |

## 輸出格式

### JSON 檔案

每個感測器配置會生成一個 JSON 檔案，包含：
- `indices`: 選定的空間點索引（1D flattened）
- `K`: 感測器數量
- `resolution`: 網格解析度
- `method`: 選點方法（"QR-Pivot" 或 "Random Sampling"）
- `time_range`: 時間範圍
- `time_steps`: 時間步數
- `selected_coordinates`: 選定點的空間座標 (x, y)
- QR-Pivot 獨有：`condition_number`, `subspace_coverage`, `energy_ratio`
- Random 獨有：`seed`

### NPZ 檔案（選用）

如果使用 `--include-dns-values`，會額外生成包含 DNS 真值的 NPZ 檔案：
- `time`: 時間陣列
- `u`, `v`, `p`, `omega`: 各感測點的物理量時間序列

## 檔案命名規則

- QR-Pivot: `sensors_temporal_K{K}_N{N}_t{t_start}-{t_end}.json`
- Random: `sensors_temporal_random_K{K}_re{Re}_N{N}_t{t_start}-{t_end}.json`
- DNS Values: 在檔名後加上 `_dns_values.npz`

## 範例：Re100 完整生成流程

```bash
# 1. 生成 QR-Pivot 感測器
python scripts/generate/sensors/generate_kolmogorov_temporal_qr.py \
  --input data/kolmogorov_dns/kolmogorov_dns_100.npy \
  --output data/kolmogorov_sensors/re100 \
  --K 50 100 200 400 \
  --time-range 0.0 20.0 \
  --time-stride 1 \
  --include-dns-values \
  --dns-values data/kolmogorov_dns/kolmogorov_dns_100.npy

# 2. 生成 Random 感測器（對照組）
python scripts/generate/sensors/generate_kolmogorov_temporal_random.py \
  --input data/kolmogorov_dns/kolmogorov_dns_100.npy \
  --output data/kolmogorov_sensors/re100 \
  --K 50 100 200 400 \
  --time-range 0.0 20.0 \
  --time-stride 1 \
  --include-dns-values \
  --dns-values data/kolmogorov_dns/kolmogorov_dns_100.npy \
  --seed 42
```

生成的檔案結構：
```
data/kolmogorov_sensors/re100/
├── sensors_temporal_K50_N256_t0-20.json
├── sensors_temporal_K50_N256_t0-20_dns_values.npz
├── sensors_temporal_random_K50_re62_N256_t0-20.json
├── sensors_temporal_random_K50_re62_N256_t0-20_dns_values.npz
└── ... (其他 K 值)
```

## 注意事項

1. **時間範圍選擇**：建議避開初始瞬態（如前 15 秒），選擇統計穩定的時間區間
2. **時間採樣間隔**：較大的 `time_stride` 可減少計算量，但會降低時間解析度
3. **記憶體需求**：生成感測器時會載入完整時間序列並計算梯度場，大規模數據可能需要較大記憶體
4. **隨機種子**：建議使用固定隨機種子（如 42）以確保實驗可重現性
