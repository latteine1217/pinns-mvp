# Project Context

## Purpose
**PINNs-MVP** 是一個基於物理訊息神經網路（Physics-Informed Neural Networks, PINNs）的湍流場重建研究專案。核心目標是從**極少量感測點資料（K ≤ 50-80）**重建完整的 3D 湍流速度與壓力場，並以 Johns Hopkins Turbulence Database (JHTDB) 的通道流數據（Re_τ=1000）作為高保真基準。

**研究題目**: 少量資料 × 物理先驗：基於公開湍流資料庫的 PINNs 逆重建  

## Tech Stack

### 核心技術
- **Python 3.8+**: 主要開發語言
- **PyTorch 2.0+**: 深度學習框架（自動微分、GPU 加速）
- **NumPy/SciPy**: 數值計算與科學運算
- **HDF5/NetCDF**: 高保真數據儲存格式
- **Matplotlib/Seaborn**: 科學視覺化

### 專案特定技術
- **VS-PINN (Variable Scaling PINN)**: 自適應變數標準化
- **GradNorm**: 動態損失權重平衡
- **QR-Pivot Sensor Selection**: 最優感測點選擇
- **Curriculum Learning**: 階段式訓練策略
- **JHTDB API (pyJHTDB)**: 湍流數據庫存取

### 開發工具
- **Git**: 版本控制
- **Conda**: 環境管理
- **pytest**: 單元測試框架
- **mypy**: 靜態類型檢查
- **ripgrep (rg)**: 快速程式碼搜尋
- **fd**: 快速檔案搜尋
- **tree**: 目錄結構視覺化

## Project Conventions

### Code Style
- **語言**: 程式碼與註解使用**中文**，作圖標題/label 使用**英文**
- **類型安全**: 所有函數必須使用 type hints，並通過 mypy 檢查
- **命名規範**:
  - 變數/函數: `snake_case`
  - 類別: `PascalCase`
  - 常數: `UPPER_SNAKE_CASE`
  - 私有成員: `_leading_underscore`
- **文檔字串**: 使用 Google Style Docstrings
- **行長度**: 最大 120 字元
- **註解原則**: 在功能前簡略說明，使用 `===` 或 `---` 分隔輸出區塊

### Architecture Patterns
- **分層架構**:
  - `pinnx/`: 核心框架（物理、模型、損失、訓練）
  - `scripts/`: 可執行腳本（訓練、評估、診斷）
  - `configs/`: YAML 配置檔案（模板化、標準化）
  - `tests/`: 單元測試與整合測試
- **模組化設計**: 各 API 檔案必須獨立，避免循環依賴
- **效能優先**: 避免 >3 層迴圈，優先使用向量化運算
- **簡潔原則**: 優先使用最簡潔優雅的解決方案，盡可能少修改程式碼

### Testing Strategy
- **測試驅動開發 (TDD)**: 新功能必須附帶單元測試
- **物理驗證優先**: 所有變更必須通過物理一致性檢查（質量守恆、動量守恆、邊界條件）
- **測試覆蓋率**: 目標 >90%
- **測試分類**:
  - 單元測試: `tests/test_*.py`
  - 物理驗證: `tests/test_physics*.py`
  - 整合測試: `tests/test_*_integration.py`
- **驗證流程**: Physics Gate → Debug/Performance 規劃 → Reviewer → 實作

### Git Workflow
- **分支策略**: 主分支為 `main`，功能開發使用 feature branches
- **提交規範**:
  - 格式: `<type>: <description>`
  - Type: `feat`, `fix`, `refactor`, `test`, `docs`, `perf`
  - 範例: `feat: add GradNorm dynamic weighting`
- **提交前檢查**:
  1. 執行 `git status` 查看變更
  2. 執行 `git diff` 確認修改內容
  3. 確保測試通過
  4. 更新相關文檔（README.md, TECHNICAL_DOCUMENTATION.md）
- **禁止操作**: 不主動執行 `git push`，不更新 git config

## Domain Context

### 物理背景
- **研究對象**: 湍流通道流（Channel Flow, Re_τ=1000）
- **控制方程**: 3D 不可壓縮 Navier-Stokes 方程
- **邊界條件**: 
  - 壁面: 無滑移條件（u=v=w=0）
  - 週期性: x, z 方向週期邊界
- **物理約束**:
  - 質量守恆: ∇·u = 0
  - 動量守恆: ∂u/∂t + (u·∇)u = -∇p + ν∇²u
  - 能量守恆: 驗證湍流動能平衡

### 數據規格
- **JHTDB Channel Flow Re_τ=1000**:
  - 計算域: 8πh × 2h × 3πh (h=1)
  - 網格: 2048 × 512 × 1536 (wavemodes)
  - 時間步: Δt=0.0013, δt=0.0065
  - 儲存範圍: t ∈ [0, 25.9935]
  - 摩擦速度: u_τ = 4.9968×10⁻²
  - 黏滯係數: ν = 5×10⁻⁵

### 成功門檻
1. **流場誤差**: 速度/壓力相對 L2 error ≤ 10-15%
2. **可識別性**: K ≤ 50 感測點達標（噪聲 σ=1-3%）
3. **效率**: 相較固定權重基線，收斂 epoch 下降 ≥30%
4. **資料重現性**: 所有評測以 JHTDB cutout/散點取樣為標準

## Important Constraints

### 物理約束（不可違反）
- **Physics Gate**: 任何變更必須通過物理驗證（質量/動量/能量守恆）
- **量綱一致性**: 所有物理量必須保持正確量綱
- **邊界條件**: 必須嚴格強制執行（壁面無滑移、週期性）
- **數值穩定性**: 禁止 NaN、Inf 或梯度爆炸

### 開發約束
- **向後相容性**: 保持與現有 30+ 配置檔案的相容性
- **可重現性**: 所有實驗必須固定隨機種子，配置檔案版本化
- **文檔同步**: 程式碼變更必須同步更新技術文檔
- **效能要求**: 訓練時間不得顯著增加（除非有明確效能證據）

### 資料安全
- **JHTDB Token**: 不得硬編碼，必須使用環境變數（`.env`）
- **敏感資訊**: 禁止提交至版本控制（`.gitignore` 保護）
- **資料引用**: 使用 JHTDB 數據必須遵循官方引用規範

## External Dependencies

### 數據來源
- **JHTDB (Johns Hopkins Turbulence Database)**:
  - URL: http://turbulence.pha.jhu.edu/
  - API: pyJHTDB (HTTP/SciServer)
  - 認證: 需申請個人 auth token
  - 引用: 必須在論文中引用 JHTDB

### Python 套件（核心）
- `torch >= 2.0.0`: 深度學習框架
- `numpy >= 1.21.0`: 數值計算
- `scipy >= 1.7.0`: 科學運算
- `h5py >= 3.0.0`: HDF5 檔案處理
- `pyyaml >= 5.4.0`: YAML 配置解析
- `matplotlib >= 3.4.0`: 視覺化
- `pytest >= 6.2.0`: 測試框架

### 開發工具
- **ripgrep**: 快速程式碼搜尋（替代 grep）
- **fd**: 快速檔案搜尋（替代 find）
- **tree**: 目錄結構視覺化

### 計算資源
- **GPU**: 建議 NVIDIA GPU（CUDA 支援）
- **記憶體**: 建議 ≥16GB RAM
- **儲存**: 建議 ≥50GB（JHTDB 數據快取）

## Agent Workflow (主 Agent 協調規則)

### 核心原則
1. **Sub-agent 僅研究/規劃/審查**，嚴禁直接改動程式碼
2. **長內容寫入檔案系統**（Markdown），不塞進對話歷史
3. **主 Agent 是唯一實作者**，決策記錄至 `context/decisions_log.md`
4. **Gate 順序**: Physics ✅ → Debug/Performance 規劃 ✅ → Reviewer ✅ → 實作
5. **物理優先**: 若任一 Gate 未過或發現物理疑慮，強制阻斷實作

### 檔案結構
- **全域上下文**: `context/context_session_*.md`
- **決策日誌**: `context/decisions_log.md`
- **任務目錄**: `tasks/<task-id>/`
  - `task_brief.md`: 任務簡述（主 Agent 撰寫）
  - `physics_review.md`: 物理審查（Physicist）
  - `debug_playbook.md`: 除錯手冊（Debug Engineer）
  - `perf_profile_plan.md`: 效能分析（Performance Engineer）
  - `review_report.md`: 審查報告（Reviewer）
  - `impl_plan.md`: 實作計畫（主 Agent）

### 委派規則
- 每個 sub-agent 指令必須包含：任務目標、相關路徑、輸出檔名、交付時限、驗收格式
- Sub-agent 開始前讀 `context_session_*.md` 與 `task_brief.md`
- Sub-agent 回傳格式: "我已完成，請閱讀 <相對路徑>；重點摘要：<三行內>"

### 風險保護
- **Physics 優先**: 若 `physics_test_spec.md` 未規範或未通過，禁止任何效能優化合併
- **偵錯優先**: 若 `debug_playbook.md` 指出根因在資料/BC/IC，即刻暫停調參
- **雙指標驗證**: Performance 建議需同時驗證「效能↑」與「物理不退化」
- **上下文鎖**: 同一 `<task-id>` 不得同時開兩條互斥分支

## Configuration Management

### 配置檔案規範
- **位置**: `configs/`
- **格式**: YAML
- **命名**: `<category>_<feature>_<variant>.yml`
- **模板**: `configs/templates/` 提供 4 個標準化模板
  - `2d_quick_baseline.yml`: 快速驗證（5-10 min）
  - `2d_medium_ablation.yml`: 特徵消融（15-30 min）
  - `3d_slab_curriculum.yml`: 課程學習（30-60 min）
  - `3d_full_production.yml`: 論文級結果（2-8 hrs）

### 必要欄位
```yaml
experiment:
  name: "<experiment_name>"
  seed: 42
  device: "auto"

output:
  checkpoint_dir: "./checkpoints/<experiment_name>"
  results_dir: "./results/<experiment_name>"

training:
  optimizer: "adam"
  lr: 1.0e-3
  epochs: 1000
```

### 配置驗證
- 使用前必須驗證 YAML 語法
- 確保所有路徑存在或可自動創建
- 檢查參數範圍合理性（學習率、權重、網格大小）

## Troubleshooting Resources

### 診斷工具
1. **訓練失敗診斷**: `scripts/debug/diagnose_piratenet_failure.py`
   - 檢查點完整性分析
   - 損失項趨勢圖
   - 配置參數驗證
   - 文檔: `docs/PIRATENET_TRAINING_FAILURE_DIAGNOSIS.md`

2. **QR-Pivot 感測點分析**: `scripts/visualize_qr_sensors.py`
   - 2D/3D 分佈視覺化
   - 品質指標（條件數、能量比例）
   - 策略比較（QR-Pivot vs POD vs Greedy）
   - 文檔: `docs/QR_SENSOR_VISUALIZATION_GUIDE.md`

3. **標準化效果驗證**: `scripts/quick_validation_normalization.py`
   - 快速驗證標準化對訓練穩定性的影響
   - 文檔: `docs/NORMALIZATION_USER_GUIDE.md`

### 常見問題
- **NaN/梯度爆炸**: 檢查學習率、梯度裁剪、標準化設定
- **收斂緩慢**: 啟用 GradNorm、調整損失權重、使用課程學習
- **物理殘差高**: 檢查 NS 方程實現、邊界條件、網格解析度
- **感測點過擬合**: 增加感測點數量 K、改善 QR-Pivot 品質

### 文檔索引
- **技術文檔**: `TECHNICAL_DOCUMENTATION.md`
- **開發指引**: `AGENTS.md`
- **決策日誌**: `context/decisions_log.md`
- **配置說明**: `configs/README.md`
- **腳本說明**: `scripts/README.md`
