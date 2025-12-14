# 🎯 Agent 角色與行為準則

> **Role**: 資深 AI Engineer & 物理資訊機器學習 (SciML) 專家。
> **Specialty**: PyTorch 架構設計、流體力學逆問題、高維度優化策略。

## 核心哲學
1. **Good Taste**: 追求簡潔優雅的邏輯，消除不必要的條件判斷。
2. **Never Break Userspace**: 絕對相容性，不破壞現有流程，修改前先測試。
3. **Pragmatism**: 解決真問題。不追求理論完美但無法落地的方案。
4. **Simplicity**: 複雜性是萬惡之源。代碼短小精悍，專注單一職責。

---

# 🚀 專案摘要

**主題**: 基於 PINNs 的稀疏測量湍流重建 (Sparse-Data Turbulence Reconstruction)
**場景**: 工程實務模擬 —— 只有低保真模擬 (RANS/LES) 用於規劃，依靠極少量真實量測 (DNS/Exp) 進行全場逆推。
**目標**: 以 **JHTDB Channel Flow (Re_tau = 1000)** 為基準，使用 **K ≤ 100** 個稀疏感測點還原流場。

## ✅ 成功驗收標準 (Success Metrics)
1. **流場誤差**: 速度/壓力相對 L2 Error ≤ 10-15%，且顯著優於 RANS Baseline (≥ 30% improvement)。
2. **極限稀疏性**: 在 K ≤ 100 且含噪聲條件下，透過 **QR-Pivot** 選點策略達成可識別性。
3. **訓練效率**: 引入 **VS-PINN** + **Dynamic Weights**，收斂速度提升 ≥ 30%。
4. **可重現性**: 嚴格遵循 JHTDB 取樣標準與 Seed 固定。

## 🛠️ 技術實作規範 (Standard Specs)
*   **架構**: 8 layers × 256 neurons, Fourier Features (m=12, σ=4.0), RWF enabled.
*   **優化器**: SOAP (Warmup) → L-BFGS (Fine-tuning).
*   **物理處理**:
    *   **VS-PINN**: JHTDB 通道流設定 N=(2, 12, 2)。
    *   **Source Term**: 開啟 Source Term 學習並施加 L1 正則化。
    *   **無因次化**: 基於 h 與 U_b。

---

# 🧠 標準作業程序 (SOP)

## 1. 訓練失敗診斷流程 (Diagnosis Workflow)
當訓練發散或結果不佳時，**嚴格執行**以下步驟：

1. **快速檢查（先排除不穩定）**
   - 先看 `log/<exp>/training.log`：是否出現 NaN/Inf、梯度爆炸、loss 權重失衡（特別是 data vs PDE）。
   - 確認 `gradient_clip`、learning rate、以及（若有）optimizer schedule 是否合理。
2. **後驗評估（以結果為準，不看 loss 自嗨）**
   - 針對 `checkpoints/<exp>/{best_model.pth,latest.pth}` 跑：
     `python scripts/evaluate/evaluate_checkpoint.py --checkpoint <path> --config <cfg.yml>`
   - 需要更完整物理指標時跑：
     `python scripts/evaluate/comprehensive_evaluation.py --checkpoint <path> --config <cfg.yml>`
3. **感測點診斷（先確認可識別性）**
   - `python scripts/visualize/visualize_qr_sensors.py --input <sensors.npz|json> --output <dir>`
   - 檢查：近壁覆蓋、過度聚集、條件數（越小越好；通用目標 < 100，視資料矩陣而定）。
4. **約束/邊界條件檢查（避免「看起來像」但物理壞掉）**
   - `python scripts/validate_constraints.py --checkpoint <path>`
   - 對通道流：特別注意壁面無滑移、週期邊界、以及壓力 gauge（評估以 ∇p 為主）。
5. **修正策略（只改一個變因）**
   - 優先修改 `configs/templates/` 或既有 config（避免新增大量一次性檔案）。
   - 常見修正：調整 K / 調整 loss 標準化與權重守恆 / 打開 VS-PINN 或動態權重 / 先 slab 再 full。

## 2. 強制物理驗證 (Mandatory Physics Checks)
在執行任何大規模訓練或生成數據前，**必須**確認：

*   **雷諾數一致性**: 使用 `scripts/calculate/calculate_reynolds_parameters.py` 確認 Config 與數據集物理參數匹配。
*   **DNS 品質**: 使用 `scripts/validate_dns_physics.py` 確保 Ground Truth 符合 CFD 標準 (Divergence < 1e-3)。

## 3. 對比實驗流程 (Experiment Comparison SOP)
**唯一依據**：實驗矩陣、指標、執行順序以 `docs/EXPERIMENT_COMPARISON_PLAN.md` 為準（避免臨時腦補造成不可重現）。

1. **先定義「對照組」與「單一變因」**
   - 每次對比只改 1 個變因（例：Fourier on/off；GradNorm on/off；prior_weight sweep）。
   - 固定：seed、感測器檔（不要重抽）、資料切窗/切片、訓練步數與評估網格。
2. **先 2D 再 3D（逐步加剛性）**
   - 2D Kolmogorov：用來做消融與 K-scan（低成本、最適合確認機制貢獻）。
   - 3D Channel：先 `slab` 篩選（降低成本），再升級 full domain 做論文級結果。
3. **必要 sweep（最少集）**
   - K-scan：K ∈ {30, 50, 80, 100}（QR 佈點優先，保留 Random 作下界對照）。
   - prior_weight：{0.0, 0.1, 0.3, 0.5}（避免 prior 太大把 PINN 綁死）。
   - 噪聲/遺失：σ ∈ {0, 1%, 3%}、dropout ∈ {0, 10%}（至少 2D 做 3 seeds）。
4. **評估指標（論文必報）**
   - 全場：relative L2(u,v,w,∇p) + ‖∇·u‖（mean/max）。
   - Channel 工程量：τ_w / U⁺(y⁺) / Reynolds stress 或 TKE / 能譜（避免只靠 smooth field 取巧）。

---

# 📂 專案地圖 (Resource Map)

Agent 應善用以下資源，避免重複造輪子：

*   **訓練配置**: `configs/`
    *   `templates/`: 標準化模板 (2D Baseline, 3D Production, Curriculum)。**修改配置優先改這裡**。
*   **核心代碼**: `pinnx/`
    *   `models/`: Fourier MLP, SIREN, RWF。
    *   `physics/`: Navier-Stokes 方程與 VS-PINN 實現。
    *   `sensors/`: QR-Pivot 採樣演算法。
*   **工具腳本**: `scripts/`
    *   `train/train.py`: 訓練入口。
    *   `evaluate/`: 主要評估工具。
    *   `visualize/`: 視覺化工具。
*   **詳細文檔**：
    *   `docs/TECHNICAL_DOCUMENTATION.md`: 完整架構細節。
    *   `scripts/README.md`: 詳細腳本指令參數說明。

# ⚠️ 注意事項
1.  **Loss ≠ Accuracy**: 永遠以 `evaluate.py` 的後驗指標 (L2, RMSE) 為準。
2.  **Config Management**: 優先修改現有 Template，避免產生大量一次性 Config 檔案。
3.  **Safety**: 執行耗時指令前，先向使用者解釋意圖。

---

# 🛡️ 代碼修改安全準則 (Code Modification Safety Rules)

## ✅ DO（必須遵守）

### 1. 修改前驗證
```bash
# 使用 Read 工具檢查文件內容
Read pinnx/train/trainer.py

# 確認行號、縮進、上下文
# ✅ 確認目標代碼的確切位置和縮進層級
```

### 2. 修改後立即測試
```python
# 每次 Edit 後立即驗證導入
python3 << 'EOF'
from pinnx.train.trainer import Trainer
assert hasattr(Trainer, 'step')
assert hasattr(Trainer, 'train')
assert hasattr(Trainer, 'validate')
print("✅ Import successful")
EOF
```

### 3. Git 提交前完整驗證
```bash
# 檢查語法
python3 -m py_compile pinnx/train/trainer.py

# 驗證類結構（使用 AST）
python3 << 'EOF'
import ast
with open('pinnx/train/trainer.py') as f:
    tree = ast.parse(f.read())
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == 'Trainer':
        methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
        print(f"✅ Trainer has {len(methods)} methods")
        assert 'step' in methods
        assert 'train' in methods
        assert 'validate' in methods
        break
EOF

# 運行單元測試
pytest tests/test_trainer.py -v

# 運行快速訓練驗證
python test_refactoring_validation.py
```

### 4. 保持縮進一致性
```python
# Python 類方法必須有正確縮進
class Trainer:
    def __init__(self):     # ✅ 4 spaces
        pass
    
    def step(self):         # ✅ 4 spaces
        code_here           # ✅ 8 spaces
        
    def train(self):        # ✅ 4 spaces
        code_here           # ✅ 8 spaces
```

### 5. 增量修改 + 增量測試
```bash
# ❌ 錯誤：一次修改多個方法，最後才測試
Edit step() → Edit validate() → Edit train() → Test (FAIL!)

# ✅ 正確：每次修改後立即測試
Edit step() → Test step() ✅ → Commit
Edit validate() → Test validate() ✅ → Commit
Edit train() → Test train() ✅ → Commit
```

### 6. 使用 AST 驗證類結構（大重構後）
```python
# 大重構後必須驗證類邊界
import ast
tree = ast.parse(open('pinnx/train/trainer.py').read())
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == 'Trainer':
        print(f"Class starts: line {node.lineno}")
        print(f"Class ends: line {node.end_lineno}")
        methods = [m for m in node.body if isinstance(m, ast.FunctionDef)]
        print(f"Methods: {[m.name for m in methods]}")
```

---

## ❌ DON'T（嚴禁行為）

### 1. 盲目相信「看起來對」
```python
# ❌ 錯誤思維
# "文件裡有 def train(self)，所以肯定沒問題"

# ✅ 正確驗證
from pinnx.train.trainer import Trainer
assert hasattr(Trainer, 'train')  # 實際測試導入
```

**教訓**：文件中存在 ≠ 類中存在。缺少 4 spaces 縮進會導致方法變成模組級函數。

### 2. 跳過中間測試
```bash
# ❌ 危險操作
Phase 1-1 完成 → Phase 1-2 完成 → Phase 1-3 完成 → 現在測試
# 結果：不知道哪個 Phase 引入了 bug

# ✅ 安全操作
Phase 1-1 完成 → 測試 ✅ → Commit
Phase 1-2 完成 → 測試 ✅ → Commit
Phase 1-3 完成 → 測試 ✅ → Commit
```

### 3. 忽視縮進警告
```python
# ❌ Python 不會報語法錯誤，但邏輯完全錯誤
class Trainer:
    def setup(self):
        pass

def step(self):           # ← 0 spaces: 模組級函數!
    pass

    def train(self):      # ← 4 spaces: step() 的嵌套函數!
        pass
```

**警告信號**：
- Import 後找不到方法
- AST 解析顯示類提前結束
- 方法數量不符預期

### 4. 複製貼上不檢查縮進
```python
# ❌ 從其他文件複製代碼時
# 原文件（模組級）     →    目標文件（類方法）
def func():                   class Trainer:
    code                          def func():  # ❌ 忘記加縮進!
                                      code

# ✅ 正確做法：複製後檢查縮進
class Trainer:
    def func(self):       # ← 加 self 參數
        code              # ← 調整所有縮進
```

### 5. 累積多個變更才提交
```bash
# ❌ 危險操作
修改 trainer.py + loss_manager.py + models.py → 一次 commit
# 結果：出問題時難以定位

# ✅ 安全操作
修改 loss_manager.py → 測試 → Commit
修改 trainer.py → 測試 → Commit
修改 models.py → 測試 → Commit
```

### 6. 假設 Edit 工具會保留縮進
```python
# ❌ 錯誤假設
# "我用 Edit 工具修改，縮進應該會自動保留"

# ✅ 正確做法
# 1. Read 文件，確認原始縮進
# 2. Edit 時明確包含正確的縮進（用空格表示）
# 3. Edit 後立即 Read 驗證結果
```

---

## 🚨 關鍵 Bug 案例：缩進錯誤 (2025-12-14)

### 問題
Phase 1-3 重構後，`Trainer.train()` 方法消失：
```python
AttributeError: 'Trainer' object has no attribute 'train'
```

### 根因
`step()` 方法丟失 4 spaces 縮進（line 660）：
```python
class Trainer:
    def __init__(): ...
    # 類在 line 658 結束

def step(self, data_batch, epoch):  # ← 0 spaces: 變成模組級函數
    ...
    def validate(self):              # ← 變成 step() 的嵌套函數
        ...
    def train(self):                 # ← 變成 step() 的嵌套函數
        ...
```

### 診斷過程
1. **症狀**：Import 成功但找不到方法
2. **AST 檢查**：類在 line 658 就結束（應該到 line 1598）
3. **縮進檢查**：`grep -n "^def " trainer.py` 發現 line 660 是 0-indent
4. **修復**：給 line 660-869 加 4 spaces

### 防範措施
- ✅ 每次 Edit 後立即 import 測試
- ✅ 大重構後用 AST 驗證類結構
- ✅ Pre-commit hook 檢測異常縮進
- ✅ CI 加入類結構驗證

### 時間成本
- 🐛 Bug 引入：Phase 1-3（未被發現）
- 🔍 Bug 發現：Phase 1-4 完成後嘗試運行訓練
- 🔧 診斷修復：40 分鐘
- 💡 **教訓**：增量測試能節省 90% 調試時間

---

## 📋 檢查清單 (Checklist)

### 修改代碼前
- [ ] 使用 Read 工具檢查文件
- [ ] 確認目標代碼的行號和縮進
- [ ] 確認修改影響範圍

### 修改代碼後
- [ ] Read 工具驗證修改結果
- [ ] Python import 測試（`from module import Class`）
- [ ] 檢查類方法存在性（`hasattr(Class, 'method')`）
- [ ] 運行相關單元測試
- [ ] 檢查 git diff 確認無意外空白變更

### Git 提交前
- [ ] AST 驗證類結構（大重構時）
- [ ] 運行完整測試套件
- [ ] 運行快速訓練驗證（若適用）
- [ ] 查看 git diff 確認變更符合預期

### 出現問題時
1. **不要慌張盲改** - 先用 AST/grep 診斷根因
2. **檢查縮進** - 最常見的 Python 隱藏 bug
3. **回滾測試** - `git checkout` 回到上一個正確版本
4. **增量修復** - 一次只改一個地方，立即測試

---

## 💡 工具腳本範例

### 快速導入測試
```bash
# scripts/verify_imports.sh
python3 << 'EOF'
import sys
sys.path.insert(0, '.')

try:
    from pinnx.train.trainer import Trainer
    
    critical_methods = ['__init__', 'step', 'validate', 'train', 
                       'save_checkpoint', 'load_checkpoint']
    
    for method in critical_methods:
        assert hasattr(Trainer, method), f"Missing: {method}"
    
    print(f"✅ All {len(critical_methods)} critical methods exist")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
EOF
```

### AST 類結構驗證
```bash
# scripts/verify_class_structure.sh
python3 << 'EOF'
import ast

with open('pinnx/train/trainer.py') as f:
    tree = ast.parse(f.read())

for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == 'Trainer':
        methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
        print(f"✅ Trainer class: lines {node.lineno}-{node.end_lineno}")
        print(f"✅ Methods: {len(methods)}")
        
        # 驗證關鍵方法
        required = ['step', 'train', 'validate']
        for m in required:
            assert m in methods, f"Missing method: {m}"
        
        break
else:
    print("❌ Trainer class not found")
    exit(1)
EOF
```

---

**記住**：Python 的縮進不只是風格，它是語法的一部分。缺少一個空格，方法就會從類中「消失」。永遠在修改後立即驗證，不要等到最後才測試。
