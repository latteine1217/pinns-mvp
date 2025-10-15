# 📦 配置模板索引

本目錄包含 4 個標準化 YAML 模板，涵蓋從快速測試到生產級訓練的完整流程。

---

## 🚀 快速選擇指南

| 場景 | 模板 | 時間 | 用途 |
|------|------|------|------|
| **快速驗證想法** | `2d_quick_baseline.yml` | 5-10 分鐘 | 調試代碼、初步可行性測試 |
| **特徵消融研究** | `2d_medium_ablation.yml` | 15-30 分鐘 | 量化特徵貢獻、對照實驗 |
| **課程式訓練** | `3d_slab_curriculum.yml` | 30-60 分鐘 | 多階段學習、穩健收斂 |
| **論文級結果** | `3d_full_production.yml` | 2-8 小時 | 高精度重建、完整驗證 |
| **PirateNet 複現** | `piratenet_baseline.yml` | 1-2 小時 | 複現論文架構、SOAP 優化器 |

---

## 📋 模板詳細說明

### 1️⃣ `2d_quick_baseline.yml` - 快速基準測試

**用途**：快速驗證想法、調試代碼、初步可行性測試

**配置摘要**：
- **空間維度**：2D 切片（xy 平面）
- **感測點 K**：50（最少）
- **訓練輪數**：100 epochs
- **網路架構**：128×4（輕量）
- **特徵開關**：
  - ✅ Fourier（簡化，m=16）
  - ✅ VS-PINN
  - ❌ 自適應權重
  - ❌ 課程學習
  - ❌ RANS prior

**預期結果**：
- 訓練時間：5-10 分鐘（GPU）
- 最終 L2：30-50%
- PDE ratio：10-20%

**成功標準**：
- ✅ 無 NaN/Inf（穩定性）
- ✅ Loss 持續下降
- ✅ L2 < 50%
- ✅ PDE ratio > 10%

**使用範例**：
```bash
python scripts/train.py --cfg configs/templates/2d_quick_baseline.yml
```

**適用場景**：
- 調試新功能
- 驗證資料載入
- 快速參數試探
- 初學者入門

---

### 2️⃣ `2d_medium_ablation.yml` - 中期消融實驗

**用途**：特徵消融研究、參數敏感度分析、建立對照基線

**配置摘要**：
- **空間維度**：2D 切片（xy 平面）
- **感測點 K**：100（中等）
- **訓練輪數**：1000 epochs
- **網路架構**：256×6（標準）
- **特徵開關**：
  - ✅ Fourier（完整，m=32）
  - ✅ VS-PINN
  - ✅ 自適應權重（GradNorm）
  - ❌ 課程學習
  - ❌ RANS prior

**預期結果**：
- 訓練時間：15-30 分鐘（GPU）
- 最終 L2：20-30%
- PDE ratio：30-40%

**成功標準**：
- ✅ L2 < 30%
- ✅ PDE ratio > 30%
- ✅ 可量化 ΔL2（vs. baseline ≥ 10%）

**消融實驗設計範例**：
```yaml
# 基線實驗
Experiment A (baseline):      fourier_m=32, adaptive=true

# 消融實驗（每次僅改變一個變數）
Experiment B (no_fourier):    fourier_m=0,  adaptive=true
Experiment C (no_adaptive):   fourier_m=32, adaptive=false
Experiment D (minimal):       fourier_m=0,  adaptive=false
```

**使用範例**：
```bash
# 基線實驗
python scripts/train.py --cfg configs/templates/2d_medium_ablation.yml

# 消融實驗（修改配置後）
python scripts/train.py --cfg configs/2d_ablation_no_fourier.yml
```

**適用場景**：
- 特徵重要性分析
- 超參數敏感度測試
- 建立性能基線
- 對照實驗設計

---

### 3️⃣ `3d_slab_curriculum.yml` - 3D 課程學習

**用途**：課程式訓練、多階段學習、3D 流場重建

**配置摘要**：
- **空間維度**：3D Slab（z: 4.0-5.5，Δz≈1.5）
- **感測點 K**：100
- **訓練輪數**：1000 epochs（2 階段）
- **網路架構**：256×6（標準）
- **特徵開關**：
  - ✅ Fourier（完整，m=32）
  - ✅ VS-PINN
  - ✅ 自適應權重
  - ✅ 課程學習（2 階段）
  - ❌ RANS prior

**課程設計**：
```yaml
Stage1 (0-400 epochs):    PDE 暖啟（weights: data=5, PDE=5）
                          目標: PDE ratio ≥ 25%, τ_w 非零

Stage2 (400-1000 epochs): PDE 主導（weights: data=5, PDE=8）
                          目標: PDE ratio ≥ 35%, L2 < 25%
```

**預期結果**：
- 訓練時間：30-60 分鐘（GPU）
- Stage1 完成：Loss ↓ 50%, PDE ratio ≈ 25%
- Stage2 完成：L2 ≈ 20-25%, PDE ratio ≈ 35-40%

**成功標準**：
- ✅ 階段轉換平滑（Δloss < 20%）
- ✅ Stage1: PDE ratio ≥ 25%
- ✅ Stage2: PDE ratio ≥ 35%, L2 < 25%
- ✅ τ_w ratio ≥ 5.0

**使用範例**：
```bash
python scripts/train.py --cfg configs/templates/3d_slab_curriculum.yml
```

**適用場景**：
- 高雷諾數流動（需穩健訓練）
- 3D 流場重建
- 逆問題求解
- 需要強物理一致性的案例

**注意事項**：
- 監控階段轉換時的 loss 跳變（應 < 20%）
- Stage2 若 PDE ratio 下降 > 5%，考慮調整權重
- 可根據需求擴展至 3-4 階段

---

### 4️⃣ `3d_full_production.yml` - 生產級訓練

**用途**：生產級訓練、完整 3D 重建、論文級結果

**配置摘要**：
- **空間維度**：3D 完整域（z: 0.0-9.42）
- **感測點 K**：500（高密度）
- **訓練輪數**：5000 epochs（3 階段）
- **網路架構**：256×8（加深）
- **特徵開關**：
  - ✅ Fourier（完整，m=64，高頻擴展）
  - ✅ VS-PINN
  - ✅ 自適應權重
  - ✅ 課程學習（3 階段）
  - ⚠️ RANS prior（可選）
  - ⚠️ Ensemble UQ（可選）

**課程設計**：
```yaml
Stage1 (0-1000):      基礎建立（PDE=5.0, LR=1e-3）
                      目標: PDE ratio ≥ 25%, τ_w ≥ 1.0

Stage2 (1000-3000):   PDE 主導（PDE=8.0, LR=5e-4）
                      目標: PDE ratio ≥ 35%, L2 < 20%

Stage3 (3000-5000):   精煉優化（PDE=10.0, LR=1e-4）
                      目標: PDE ratio ≥ 40%, L2 < 15%
```

**預期結果**：
- 訓練時間：2-8 小時（GPU，取決於硬體）
- Stage1 (1000ep)：Loss ↓ 60%, PDE ratio ≈ 25%
- Stage2 (3000ep)：L2 ≈ 18-20%, PDE ratio ≈ 35%
- Stage3 (5000ep)：L2 ≈ 12-15%, PDE ratio ≈ 40-45%

**成功標準（論文級）**：
- ✅ L2 ≤ 15%（u 速度）
- ✅ PDE ratio ≥ 40%（最終階段）
- ✅ τ_w ratio ≥ 10.0
- ✅ Mass conservation < 1e-4
- ✅ RMSE improvement ≥ 30%（vs. RANS）
- ✅ Spectrum RMSE ≤ 25%

**使用範例**：
```bash
# 完整訓練（5000 epochs）
python scripts/train.py --cfg configs/templates/3d_full_production.yml

# 啟用 Ensemble（不確定性量化）
python scripts/train.py --cfg configs/templates/3d_full_production.yml --ensemble

# 從檢查點恢復
python scripts/train.py --cfg configs/templates/3d_full_production.yml \
    --resume checkpoints/3d_full_production_final/stage2_complete.pth
```

**適用場景**：
- 論文級結果產出
- 完整 3D 流場重建
- 需要高精度的應用
- 最終模型交付

**注意事項**：
- 長期訓練建議使用 tmux/screen（防斷線）
- 監控 GPU 記憶體（batch_size 可能需調整）
- 定期備份檢查點（每 500 epochs）
- 建議啟用 W&B 以便遠端監控

**記憶體估算**：
- 模型參數：~5M
- 批次資料：~200MB
- 峰值記憶體：~8-12GB（CUDA）

**效能調優**：
- GPU 充足：`batch_size=4096, width=512`
- GPU 受限：`batch_size=1024, width=256, depth=6`
- 極速模式：關閉 Fourier（`fourier_m=0`）
- 高精度：`depth=10, fourier_m=128`

---

## 🔧 如何使用這些模板

### 步驟 1：選擇模板
根據上述「快速選擇指南」選擇合適的模板。

### 步驟 2：複製並修改
```bash
# 複製模板到 configs/ 目錄
cp configs/templates/2d_quick_baseline.yml configs/my_experiment.yml

# 修改關鍵參數
vim configs/my_experiment.yml
```

**必改參數**：
- `experiment.name`: 改為你的實驗名稱
- `output.checkpoint_dir`: 改為對應路徑
- `output.results_dir`: 改為對應路徑

### 步驟 3：執行訓練
```bash
python scripts/train.py --cfg configs/my_experiment.yml
```

### 步驟 4：監控訓練
```bash
# 即時監控日誌
tail -f log/<exp_name>/training.log

# 使用 TensorBoard
tensorboard --logdir checkpoints/<exp_name>
```

---

## 🚀 混合精度訓練（AMP）

### 啟用條件
- ✅ **CUDA 設備**（Tesla T4, A100, V100 等）
- ❌ **MPS 設備**（Apple Silicon，不支援 GradScaler）
- ❌ **CPU 設備**（無效能提升）

### 預期效果
| 指標 | 效果 |
|------|------|
| **記憶體節省** | 30-50% |
| **速度影響** | < 10%（可忽略） |
| **數值穩定性** | 自動損失縮放保護 |

### 配置範例
```yaml
training:
  amp:
    enabled: true  # CUDA 設備推薦啟用
```

### 記憶體對比
| 模板 | AMP 關閉 | AMP 開啟 | 節省 |
|------|---------|---------|------|
| 2D Medium | 6 GB | 4 GB | 33% |
| 3D Slab | 10 GB | 6 GB | 40% |
| 3D Full | 14 GB | 8 GB | 43% |

### 使用建議
- **Colab 免費版（T4, 15GB）**: ✅ 必須啟用（3D Full 需求 8GB < 15GB）
- **本地調試**: ⚠️ 可關閉以簡化除錯
- **生產訓練**: ✅ 推薦啟用（加速檢查點保存）

### 故障排除
| 問題 | 原因 | 解決 |
|------|------|------|
| 訓練崩潰且日誌顯示 `MPS device` | MPS 不支援 GradScaler | Trainer 會自動禁用並發出警告，無需手動修改 |
| 記憶體溢出（CUDA OOM） | AMP 未啟用 | 在配置中設定 `amp.enabled: true` |

### 技術限制
- **MPS 設備**：自動禁用（友好警告）
- **CPU 設備**：自動禁用（無警告）
- **CUDA 設備**：正常啟用

---

## 📊 對照實驗設計範例

### 範例 1：Fourier 特徵消融

**研究問題**：Fourier 特徵對精度的貢獻有多大？

**實驗設計**：
```bash
# 基線（啟用 Fourier）
cp configs/templates/2d_medium_ablation.yml configs/ablation_fourier_on.yml
python scripts/train.py --cfg configs/ablation_fourier_on.yml

# 對照（禁用 Fourier）
cp configs/templates/2d_medium_ablation.yml configs/ablation_fourier_off.yml
# 修改: fourier_m: 0
python scripts/train.py --cfg configs/ablation_fourier_off.yml
```

**預期結果**：ΔL2 ≈ 10-15%（基線更優）

---

### 範例 2：自適應權重消融

**研究問題**：自適應權重是否改善 PDE 參與度？

**實驗設計**：
```bash
# 基線（啟用自適應）
cp configs/templates/2d_medium_ablation.yml configs/ablation_adaptive_on.yml
python scripts/train.py --cfg configs/ablation_adaptive_on.yml

# 對照（禁用自適應）
cp configs/templates/2d_medium_ablation.yml configs/ablation_adaptive_off.yml
# 修改: adaptive_weighting: false
python scripts/train.py --cfg configs/ablation_adaptive_off.yml
```

**預期結果**：
- 基線：PDE ratio ≈ 35%
- 對照：PDE ratio ≈ 15-20%（下降顯著）

---

### 範例 3：課程學習有效性

**研究問題**：課程學習是否改善收斂速度與穩健性？

**實驗設計**：
```bash
# 基線（2 階段課程）
cp configs/templates/3d_slab_curriculum.yml configs/curriculum_2stage.yml
python scripts/train.py --cfg configs/curriculum_2stage.yml

# 對照（單階段，無課程）
cp configs/templates/3d_slab_curriculum.yml configs/curriculum_single.yml
# 修改: curriculum.enable: false
python scripts/train.py --cfg configs/curriculum_single.yml
```

**預期結果**：
- 基線：收斂平滑，最終 L2 ≈ 22%
- 對照：可能震盪，最終 L2 ≈ 25-28%

---

## ⚠️ 常見錯誤與修正

| 錯誤 | 原因 | 修正 |
|------|------|------|
| `KeyError: 'checkpoint_dir'` | 未修改輸出路徑 | 修改 `output.checkpoint_dir` |
| `CUDA out of memory` | batch_size 過大 | 降低至 512 或 1024 |
| `PDE ratio < 10%` | 權重失衡 | 檢查 `adaptive_weighting` 與初始權重 |
| `NaN after 50 epochs` | 學習率過高 | 降低至 `5e-4` 或啟用 warmup |
| `Stage transition loss jump > 50%` | 權重變化過激 | 限制權重變化至 ≤ 2× |

---

## 📚 延伸閱讀

- **配置標準化指南**：`configs/README.md`
- **訓練腳本使用說明**：`scripts/README.md`（如存在）
- **開發者指引**：`AGENTS.md`
- **技術文檔**：`TECHNICAL_DOCUMENTATION.md`

---

## 🆘 需要幫助？

如遇到問題，請按以下順序檢查：

1. **檢查配置文件**：確認所有必要欄位已填寫
2. **查看日誌**：`tail -f log/<exp_name>/training.log`
3. **檢查 GPU**：`nvidia-smi`（確認記憶體充足）
4. **參考 README**：`configs/README.md` 的故障排除章節
5. **查看過往實驗**：`configs/experiment_tracking.md`（如存在）

---

**最後更新**：2025-10-15  
**維護者**：PINNs-MVP Team  
**版本**：v1.0
