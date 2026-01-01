# 🧠 PINNx: 核心框架指南

## 總覽

`pinnx` 是 PINNs-MVP 專案的核心函式庫，它封裝了實現物理資訊神經網路所需的所有基礎建構模塊。從模型定義、物理方程實現，到訓練循環和評估，所有核心邏輯都包含在此模組中。其設計旨在實現模組化、可配置性和可擴展性。

---

## 核心設計理念

- **配置驅動**: 所有實驗行為（模型架構、物理參數、訓練策略）均由 `configs/` 目錄下的 YAML 文件定義，使得實驗可被精確重現。
- **物理與模型分離**: `physics` 模組專注於物理方程的數學表達，而 `models` 模組專注於神經網路架構，兩者解耦。
- **可擴展性**: 清晰的模組劃分使得添加新的物理模型、網路架構或損失函數變得簡單。

---

## 模組結構詳解

以下是 `pinnx` 內部各子模組的職責劃分：

- **`pinnx/models/`**: 神經網路架構
  - 包含專案的核心模型，如 `Fourier-MLP`，它結合了傅立葉特徵（Fourier Features）和正弦激活函數（SIREN-style）以捕捉高頻細節。

- **`pinnx/physics/`**: 物理方程引擎
  - 實現專案所求解的偏微分方程（PDEs）。核心是 `VSPINNChannelFlow`，它實現了針對通道流剛性問題的變數縮放（Variable-Scaling）版 Navier-Stokes 方程。

- **`pinnx/losses/`**: 損失函數
  - 定義了所有自定義的損失函數。這包括計算 PDE 殘差的損失，以及如 `GradNorm` 等用於動態平衡多個損失項的自適應權重策略。

- **`pinnx/train/`**: 訓練框架（**Phase 1-4 重構完成** ✅）
  - 此模組是訓練過程的心臟。`Trainer` 類別負責管理整個訓練循環，經過大規模重構後採用模組化設計：
    - **核心訓練方法**: `step()`, `train()`, `validate()`, `save_checkpoint()` 已完成重構（平均減少 74% 行數）
    - **TrainingLoopManager**: 獨立類別管理 TensorBoard 日誌與自適應更新協調
    - **Helper Methods**: 17 個輔助方法實現單一職責原則，提升可測試性與可維護性
    - **功能完整**: 包含梯度計算、優化器步驟、驗證、檢查點管理、學習率調度器和課程學習邏輯

- **`pinnx/sensors/`**: 感測器佈局算法
  - 實現用於選擇最優感測器位置的算法。核心是 `qr_sampling.py`，它使用 QR 分解來識別流場中最具資訊量的點。

- **`pinnx/dataio/`**: 數據 I/O
  - 負責所有數據的載入和預處理。包含與 JHTDB 資料庫互動的客戶端，以及用於載入不同格式數據（如 `.npz`, `.h5`）的工具。

- **`pinnx/evals/`**: 評估與視覺化
  - 提供計算模型性能指標（如相對 L2 誤差、壁面剪應力）和生成視覺化圖表（如速度剖面、能譜）的工具。

- **`pinnx/optim/`**: 自定義優化器
  - 包含為本專案實現的特定優化器，例如 `SOAP`，它在處理 PINN 的不良條件損失曲面時可能比標準 `Adam` 更有效。

- **`pinnx/utils/`**: 通用工具
  - 包含在整個套件中重複使用的輔助函數，例如標準化工具、日誌設定等。

---

## 工作流程概念

一個典型的訓練流程如下：

1.  **配置載入**: `scripts/train.py` 作為使用者入口，負責解析指定的 YAML 配置文件。
2.  **組件初始化**: 腳本調用 `pinnx/train/factory.py` 中的工廠函數，根據配置實例化 `model`, `physics`, `optimizer` 等核心組件。
3.  **訓練器設置**: 這些組件被傳遞給 `pinnx/train/trainer.py` 中的 `Trainer` 物件，`Trainer` 初始化時會：
    - 設置數據標準化器（`OutputTransform`）
    - 配置物理驗證器（`PhysicsValidator`）
    - 初始化 TensorBoard 日誌管理（`TrainingLoopManager`）
    - 設置檢查點管理與早停機制
4.  **訓練執行**: `Trainer` 物件接管控制權，執行完整的訓練循環：
    - **step()**: 單步訓練（前向傳播 → 損失計算 → 反向傳播 → 優化器更新）
    - **validate()**: 驗證指標計算（MSE、relative L2）
    - **train()**: 完整訓練循環（epoch 迭代 → 自適應更新 → 日誌記錄）
    - **save_checkpoint()**: 檢查點保存（物理驗證 → 狀態打包 → 磁碟寫入）

---

## 重構歷史

### Phase 1-4 重構（2025-12）

`pinnx/train/` 模組經歷了大規模重構，大幅改善代碼品質：

| Phase | Method | Lines Before | Lines After | Reduction |
|-------|--------|--------------|-------------|-----------|
| Phase 1 | `step()` | 371 | 92 | **-75%** |
| Phase 2 | `train()` | 371 | 92 | **-75%** |
| Phase 3 | `validate()` | 71 | 21 | **-70%** |
| Phase 4 | `save_checkpoint()` | 158 | 46 | **-71%** |
| **Total** | 4 methods | **971** | **251** | **-74%** |

**關鍵改進**:
- ✅ 模組化設計：17 個 helper methods + 1 個管理類（`TrainingLoopManager`）
- ✅ 單一職責原則：每個方法專注於單一任務
- ✅ 可測試性提升：每個組件可獨立測試
- ✅ 零回歸：所有功能測試通過

**詳細報告**:
- `REFACTORING_REPORT_PHASE1-3.md`: Phase 1 完整報告
- `REFACTORING_REPORT_PHASE2.md`: Phase 2 完整報告
- `REFACTORING_REPORT_PHASE3.md`: Phase 3 完整報告
- `REFACTORING_REPORT_PHASE4.md`: Phase 4 完整報告
