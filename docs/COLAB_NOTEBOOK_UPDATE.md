# Colab Notebook 更新說明

## 📅 更新日期：2025-10-16

## 🎯 更新目的
因應訓練架構重構（`pinnx/train/trainer.py`），調整 Colab Notebook 以保持相容性。

---

## ✅ 主要變更

### 1. **配置文件名稱修正**
| 項目 | 舊版本 | 新版本 |
|------|--------|--------|
| 配置文件 | `colab_piratenet_1k.yml` (不存在) | `colab_piratenet_2d_slice.yml` ✅ |
| 實驗名稱 | `colab_piratenet_1k` | `colab_piratenet_2d_slice` |
| 檢查點路徑 | `checkpoints/colab_piratenet_1k/` | `checkpoints/colab_piratenet_2d_slice/` |
| 結果路徑 | `results/colab_piratenet_1k/` | `results/colab_piratenet_2d_slice/` |

### 2. **訓練命令（保持不變）**
```bash
# ✅ 新舊架構都使用相同命令
python scripts/train.py --cfg configs/colab_piratenet_2d_slice.yml
```

**關鍵點**：
- ✅ `scripts/train.py` 已重構，但**命令行接口完全向後相容**
- ✅ 使用新的 `Trainer` 類別（`pinnx/train/trainer.py`）
- ✅ 所有配置文件格式保持不變

### 3. **Notebook 說明更新**
新增以下內容：
- ✅ 標註使用「2D 切片版本」（記憶體優化）
- ✅ 說明新訓練架構（Trainer 類別）
- ✅ 調整預期誤差目標（20% for 2D slice）
- ✅ 調整訓練時間估計（30-60 分鐘）

---

## 🔍 架構變更說明

### 舊架構（已棄用）
```
scripts/train.py (1200+ 行)
  ├─ 資料載入
  ├─ 模型初始化
  ├─ 訓練循環（內嵌在腳本中）
  ├─ 驗證與檢查點
  └─ 結果保存
```

### 新架構（當前）
```
scripts/train.py (精簡協調器)
  ├─ 參數解析與配置載入
  ├─ 資料載入與預處理
  ├─ 模型/物理/損失初始化
  └─ 調用 Trainer 類別
      │
      └─ pinnx/train/trainer.py (核心訓練器)
          ├─ step(): 單步訓練
          ├─ validate(): 驗證循環
          ├─ train(): 完整訓練循環
          ├─ 動態權重調度
          └─ 檢查點管理
```

**優勢**：
- ✅ 訓練邏輯可重用（單元測試友好）
- ✅ 腳本邏輯清晰（協調 vs. 執行分離）
- ✅ 向後相容（所有舊配置仍可用）

---

## 📋 Notebook 更新清單

### Cell 5: 驗證配置文件
```python
# ✅ 已更新
with open('configs/colab_piratenet_2d_slice.yml', 'r') as f:
```

### Cell 6: 訓練命令
```bash
# ✅ 已更新
!python scripts/train.py --cfg configs/colab_piratenet_2d_slice.yml
```

### Cell 7: TensorBoard 監控
```python
# ✅ 已更新
%tensorboard --logdir checkpoints/colab_piratenet_2d_slice
```

### Cell 8: 評估命令
```bash
# ✅ 已更新（腳本路徑正確）
!python scripts/evaluate_piratenet_vs_jhtdb.py \
    --checkpoint checkpoints/colab_piratenet_2d_slice/epoch_1000.pth \
    --config configs/colab_piratenet_2d_slice.yml \
    --device cuda
```

### Cell 9-11: 視覺化
```python
# ✅ 已更新所有路徑
results/colab_piratenet_2d_slice/vs_jhtdb_statistics.json
results/colab_piratenet_2d_slice/vs_jhtdb_predictions.npz
```

### Cell 12: Google Drive 保存
```bash
# ✅ 已更新
!cp -r checkpoints/colab_piratenet_2d_slice /content/drive/MyDrive/pinns-mvp/checkpoints/
```

---

## 🚀 使用方式（無變化）

### 快速啟動
```python
# 1. 掛載 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. 克隆專案
!git clone https://github.com/your-repo/pinns-mvp.git
%cd pinns-mvp

# 3. 安裝依賴
!pip install -q pyyaml h5py tensorboard matplotlib seaborn scipy

# 4. 訓練（使用新架構）
!python scripts/train.py --cfg configs/colab_piratenet_2d_slice.yml
```

---

## ⚠️ 注意事項

### 1. **配置文件選擇**
- ✅ **2D 切片版**：`colab_piratenet_2d_slice.yml`（記憶體 ~8GB）
- ⚠️ **3D 完整版**：需要更多記憶體（~40GB），Colab 免費版可能不足

### 2. **預期結果**
| 版本 | 訓練時間 | 目標誤差 | 記憶體 |
|------|---------|---------|--------|
| 2D 切片 | 30-60 min | ≤ 20% | ~8 GB |
| 3D 完整 | 2-8 hrs | ≤ 15% | ~40 GB |

### 3. **檢查點恢復**
```bash
# ✅ 支援從中斷處恢復（新舊架構一致）
!python scripts/train.py \
    --cfg configs/colab_piratenet_2d_slice.yml \
    --resume checkpoints/colab_piratenet_2d_slice/latest.pth
```

---

## 🔧 故障排除

### 問題 1: 找不到配置文件
```
FileNotFoundError: configs/colab_piratenet_1k.yml
```
**解決方案**：使用 `colab_piratenet_2d_slice.yml`

### 問題 2: 記憶體不足（OOM）
```
RuntimeError: CUDA out of memory
```
**解決方案**：
1. 確認使用 2D 切片配置
2. 降低批次大小（2048 → 1024）
3. 降低採樣點數（pde_points: 2048 → 1024）

### 問題 3: 訓練損失為 NaN
**檢查清單**：
- ✅ 壁面邊界：`y_min: -1.0, y_max: 1.0`（而非 0, 2）
- ✅ 學習率：`lr: 1.0e-3`（不要過大）
- ✅ 梯度裁剪：`gradient_clip: 1.0`（已啟用）

---

## 📊 驗證新架構是否正常工作

### 檢查 1: 訓練日誌
```bash
!tail -50 log/colab_piratenet_2d_slice/training.log
```

**預期輸出**：
```
Epoch 10/1000 | Total Loss: 1.234e-03 | LR: 9.8e-04
  ├─ Data Loss: 5.67e-04 (weight: 100.0)
  ├─ PDE Loss: 3.45e-04 (weight: 1.0)
  ├─ Wall Loss: 2.12e-05 (weight: 10.0)  # ✅ 不為零
  └─ Time: 2.3s/epoch
```

### 檢查 2: 權重調度
```bash
# ✅ GradNorm 應該每 100 steps 更新權重
# 查看日誌中是否出現 "GradNorm updated weights"
```

### 檢查 3: 學習率遞減
```bash
# ✅ 學習率應該從 1e-3 逐步衰減到 ~1e-6
# 檢查 TensorBoard 中的 learning_rate 曲線
```

---

## 📚 相關文檔

- [訓練架構說明](../AGENTS.md#訓練架構設計)
- [配置模板指南](../configs/templates/README.md)
- [訓練器 API 文檔](../pinnx/train/README.md)
- [故障排除指南](../docs/TROUBLESHOOTING.md)

---

## ✅ 總結

| 項目 | 狀態 | 說明 |
|------|------|------|
| Notebook 更新 | ✅ 完成 | 所有路徑與配置已修正 |
| 訓練命令 | ✅ 相容 | 無需修改命令行參數 |
| 配置文件 | ✅ 可用 | `colab_piratenet_2d_slice.yml` 存在 |
| 評估腳本 | ✅ 正確 | `evaluate_piratenet_vs_jhtdb.py` 可用 |
| 向後相容性 | ✅ 保持 | 舊配置仍可使用 |

**結論**：Notebook 已完全適配新架構，可直接在 Colab 中使用。
