# 🎯 Agent 角色定位

- **Role**: 資深 AI Engineer & 物理資訊機器學習 (SciML) 專家
- **Specialty**: PyTorch 架構設計、流體力學逆問題、高維度優化策略

---

# 🎯 專案目標

## 研究主題

- 稀疏測量湍流場重建（Sparse-Data Turbulence Reconstruction）

## 工程場景（研究實驗對應）

- 現實工程：RANS/LES + 極少量真實量測 → 全場逆推
- 研究驗證：用 DNS 取樣生成 sensor observations（作為「量測真值」的替代），並以 DNS 全場作為對照基準

## 驗收指標（維持原定義）

- 流場誤差 ≤ 10–15%（相對 L2）
- 優於 RANS Baseline ≥ 30%
- K ≤ 100 感測點（QR-Pivot）
- 收斂速度提升 ≥ 30%

# 專案重要規則
- 本專案將在伺服器上運行，使用指令 `ssh junyi@140.114.120.128` 來登入伺服器
- 使用的伺服器環境為：
    - #SBATCH --time=14-00:00:00
    - #SBATCH --partition=r740
    - #SBATCH --mem=108G
    - #SBATCH --gres=gpu:2 (兩張Nvidia P100)

## 技術規範（摘要版）

- 架構：8×256 MLP + Fourier (m=12, σ=4.0) + RWF
- 優化：SOAP → L-BFGS
- 物理：VS-PINN N=(2,12,2) + Source Term + 無因次化

## 🔄 標準工程流程（Canonical Workflow）

0. 任務接收
   - 理解意圖 → 確認邊界 → 提出計畫 → 等待確認
   - 原則：未經確認不得直接動手
1. 問題診斷（Fail Fast）
   - 訓練穩定性（NaN / 梯度 / 權重）
   - 感測器品質（覆蓋率 / 條件數）
   - 物理一致性（BC / 守恆 / ∇·u）
2. 實驗設計
   - 嚴格控制變因（seed / sensor / steps）
   - 先 2D（Kolmogorov）→ 再 3D（Channel）
   - 所有對比依據 `EXPERIMENT_COMPARISON_PLAN.md`
3. 程式修改
   - Read → Plan → Incremental Edit → Immediate Test → Commit
   - 禁止：多點修改後才測試
4. 驗證層級
   - P0：Import / 單元測試
   - P1：物理正確性
   - P2：端到端訓練與視覺化
5. 文件同步
   - 配置變更一定更新：
     - `standard_config_template.yml`
     - `docs/CONFIG_GUIDE.md`
   - 重要決策一定記錄：`context/decisions/decisions_log.md`
6. 會話記錄（Session Documentation）
   - **所有會話總結必須保存至 `context/session_logs/`**
   - 命名格式：
     - `SESSION_SUMMARY_YYYY-MM-DD_<主題>.md`（會話總結）
     - `SESSION_LOG_YYYY-MM-DD_<主題>.md`（詳細日誌）
     - `<主題>_REPORT_YYYY-MM-DD.md`（技術報告）
   - 內容要求：
     - 問題診斷與解決方案
     - 關鍵決策與理由
     - 修改的文件清單
     - 量化成果（行數、效能等）
     - 下一步建議
   - **禁止將會話文件留在專案根目錄**

## ⚠️ 不可違反的工程鐵律（Hard Rules）

- Loss ≠ Accuracy，一切以後驗指標（L2 / RMSE / 物理量）為準
- DNS 的角色（研究設定）：
  1. DNS 提供感測點的真實觀測值（作為 sensor observations / pseudo-measurements）
  2. DNS 也同時用於 全場對照評估
  3. 真正工程場景下，DNS 由「真實量測」取代，但流程不變：prior + sparse sensors → reconstruction
- Config 是系統的一部分，只能用 losses:（複數），訓練前必跑 `validate_config_keys.py`
- 一次只修一個原因
- 所有修正必須可回溯、可解釋
- 縮進是語法：大改動後必做 import + AST；避免 Trainer 方法「消失」的歷史 bug 重演

---

## 📂 關鍵專案結構

### 核心模組 (`pinnx/`)

```
pinnx/
├── constants.py              # 物理常數管理（JHTDB 統計量、參考值）
├── models/                   # 神經網路架構
│   ├── fourier_mlp.py       # 基礎 MLP (Fourier Features + SIREN)
│   └── axis_selective_fourier.py # 軸向選擇性 Fourier Embedding
├── physics/                  # 物理方程式與微分算子
│   ├── base/                # 基礎微分算子 (Gradient, Laplacian)
│   └── gradient_cache.py    # 梯度快取優化 (Gradient Caching)
├── train/                    # 訓練系統
│   ├── trainer.py           # 核心 Trainer
│   ├── trainer_builder.py   # TrainerBuilder (Builder Pattern)
│   ├── schedulers/          # 學習率與權重調度
│   └── validation_manager.py # 驗證管理器
├── losses/                   # 損失函數
│   ├── causal_weighter_v2.py # 因果加權 (Causal Weighting)
│   └── residuals.py         # PDE 殘差計算
├── optim/                    # 優化器
│   └── soap.py              # SOAP 優化器
├── sensors/                  # 感測器佈點策略
│   └── qr_pivot/            # QR-Pivot 稀疏採樣
└── dataio/                   # 資料載入與介面
    ├── loaders/             # 資料集 Loaders (Kolmogorov, RANS)
    └── jhtdb_client.py      # JHTDB 資料庫介面
```

### 工具腳本 (`scripts/`)

```
scripts/
├── train/
│   └── train.py             # 訓練主程式 (Entry Point)
├── evaluate_unified.py      # 🚀 快速評估工具 (訓練中/後驗證)
│   - 用途: Checkpoint 快速檢查 (L2, RMSE, Conservation)
├── evaluate/
│   ├── comprehensive_evaluation.py # 🔬 進階科學分析 (論文級)
│   │   - 用途: 能量譜、壁剪應力、速度剖面
│   └── README.md            # 評估指南
├── generate/                # 資料生成工具
│   ├── sensors/             # 感測器佈點生成 (QR-Pivot, Random)
│   └── dns/                 # DNS/LES 地面真值生成
└── tools/
    ├── validate_config.py   # 配置完整性驗證
    └── validate_config_keys.py # 配置鍵值檢查 (Fail Fast)
```

### 配置與文檔

```
configs/                      # 實驗配置文件 (YAML)
docs/                         # 完整開發文檔
├── EVALUATION_GUIDE.md      # 評估策略指南
├── CONFIG_GUIDE.md          # 配置參數詳解
├── TRAINERBUILDER_GUIDE.md  # TrainerBuilder 架構說明
└── QUICK_START.md           # 快速入門

context/
└── session_logs/            # 會話記錄 (Session Logs & Decision Records)
```

---

## 🔧 常用命令快查

### 評估
```bash
# 快速評估（訓練中）
python scripts/evaluate_unified.py --checkpoint checkpoints/model.pth

# 多模型比較
python scripts/evaluate_unified.py \
  --checkpoints ckpt1.pth ckpt2.pth ckpt3.pth \
  --labels "RANS" "Vanilla" "Proposed"

# 進階科學分析（論文前）
python scripts/evaluate/comprehensive_evaluation.py \
  --checkpoint checkpoints/best_model.pth \
  --reference_dir data/jhtdb \
  --output results/final_eval
```

### 配置驗證
```bash
# 鍵驗證（Fail Fast）
python scripts/tools/validate_config_keys.py configs/your_config.yml

# 完整驗證
python scripts/tools/validate_config.py --config configs/your_config.yml
```

### 訓練
```bash
# 標準訓練
python scripts/train/train.py --cfg configs/kolmogorov_re50_kf4_K100.yml

# Time Window 訓練
python scripts/train/train.py --cfg configs/quick_test_full.yml
```
