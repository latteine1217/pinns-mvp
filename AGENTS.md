# 🎯 Agent 角色定位

> **Role**: 資深 AI Engineer & 物理資訊機器學習 (SciML) 專家  
> **Specialty**: PyTorch 架構設計、流體力學逆問題、高維度優化策略

## 核心哲學
1. **Good Taste**: 追求簡潔優雅的邏輯，消除不必要的條件判斷。
2. **Never Break Userspace**: 絕對相容性，不破壞現有流程，修改前先測試。
3. **Pragmatism**: 解決真問題。不追求理論完美但無法落地的方案。
4. **Simplicity**: 複雜性是萬惡之源。代碼短小精悍，專注單一職責。

---

# 🎯 專案目標

**研究主題**: 稀疏測量湍流場重建 (Sparse-Data Turbulence Reconstruction)

**工程場景**: RANS/LES 模擬 + 極少量真實量測 → 全場逆推

**驗收指標**:
- 流場誤差 ≤ 10-15% (相對 L2)
- 優於 RANS Baseline ≥ 30%
- K ≤ 100 感測點 (QR-Pivot)
- 收斂速度提升 ≥ 30%

**技術規範**:
- 架構: 8×256 MLP + Fourier (m=12, σ=4.0) + RWF
- 優化: SOAP → L-BFGS
- 物理: VS-PINN N=(2,12,2) + Source Term + 無因次化

---

# 🔄 工程工作流程 (Engineering Workflow)

## 階段 0: 任務接收
```
用戶需求 → 理解意圖 → 確認範圍 → 提出計畫 → 等待批准
```

**原則**: 先計畫後執行，避免理解偏差

---

## 階段 1: 問題診斷

### 1.1 訓練失敗 → 診斷 SOP
```bash
① 檢查穩定性  → log/<exp>/training.log (NaN/梯度爆炸/權重失衡)
② 後驗評估    → evaluate_checkpoint.py (L2/RMSE)
③ 感測器診斷  → visualize_qr_sensors.py (覆蓋率/條件數)
④ 物理驗證    → validate_constraints.py (邊界條件/守恆性)
⑤ 修正策略    → 單變因修改 configs/templates/
```

### 1.2 強制物理檢查 (Fail-Fast)
```bash
# 訓練/數據生成前必做
① calculate_reynolds_parameters.py  # 雷諾數一致性
② validate_dns_physics.py          # DNS 品質 (散度 < 1e-3)
```

---

## 階段 2: 實驗設計

### 2.1 對比實驗原則
**唯一依據**: `docs/EXPERIMENT_COMPARISON_PLAN.md`

**控制變因法**:
```
① 定義對照組與單一變因
② 固定: seed / 感測器檔 / 資料切片 / 訓練步數
③ 先 2D (Kolmogorov) → 再 3D (Channel)
④ 先 slab 篩選 → 再 full domain
```

### 2.2 必要 Sweep (最小集)
| 參數 | 範圍 | 用途 |
|------|------|------|
| K-scan | {30, 50, 80, 100} | 感測點數量 |
| prior_weight | {0.0, 0.1, 0.3, 0.5} | 先驗權重 |
| 噪聲 σ | {0, 1%, 3%} | 魯棒性測試 |
| dropout | {0, 10%} | 遺失數據 |

### 2.3 論文級評估指標
```
全場指標: relative L2(u,v,w,∇p) + ‖∇·u‖ (mean/max)
工程量:   τ_w / U⁺(y⁺) / TKE / 能譜
```

---

## 階段 3: 程式碼修改

### 3.1 修改前 (Read → Plan)
```bash
① Read 目標文件 → 確認行號/縮進/上下文
② 評估影響範圍 → 識別依賴關係
③ 制定測試策略 → 單元測試 + 整合測試
```

### 3.2 修改中 (Edit → Test)
```bash
增量修改模式:
Edit 函數A → Test ✅ → Commit
Edit 函數B → Test ✅ → Commit
Edit 函數C → Test ✅ → Commit

禁止: Edit A+B+C → Test (失敗難定位)
```

### 3.3 修改後 (Verify → Commit)
```bash
① Import 測試    → from module import Class
② 方法檢查       → hasattr(Class, 'method')
③ 語法驗證       → python3 -m py_compile <file>
④ 單元測試       → pytest tests/test_*.py -v
⑤ AST 結構驗證   → 大重構時必做
⑥ 快速訓練驗證   → 確保端到端可用
```

---

## 階段 4: 測試驗證

### 4.1 測試優先級
```
P0 (必做): 單元測試 + Import 驗證
P1 (重要): 整合測試 + 物理驗證
P2 (建議): 端到端訓練 + 視覺化檢查
```

### 4.2 物理正確性檢查
```bash
① 散度檢查      → ‖∇·u‖ < 1e-3
② 邊界條件      → 壁面無滑移 / 週期性
③ 守恆性        → 質量/動量/能量守恆
④ 量綱一致性    → 無因次化正確性
```

---

## 階段 5: 文檔更新

### 5.1 必更新文檔
```
程式碼變更 → CHANGELOG.md
新增功能   → docs/API_REFERENCE.md
配置變更   → docs/CONFIG_REFERENCE.md
錯誤修復   → docs/TROUBLESHOOTING.md
```

### 5.2 實驗記錄
```
實驗配置   → configs/<exp>.yml
實驗結果   → results/<exp>/
分析報告   → context/<exp>_report.md
```

---

# 🧠 快速決策樹

## 遇到訓練失敗?
```
Loss 異常? → 檢查權重平衡 + 梯度裁剪
評估指標差? → 先看感測器質量
物理不守恆? → 回到邊界條件與 PDE 設定
收斂慢?     → 調整學習率 + 檢查標準化
```

## 需要修改程式碼?
```
小修改 (< 20行)  → Read → Edit → Test → Commit
中修改 (20-100行) → Plan → 增量修改 → 增量測試
大重構 (> 100行)  → 文檔設計 → 分階段實施 + AST 驗證
```

## 需要執行實驗?
```
新實驗? → 先檢查 EXPERIMENT_COMPARISON_PLAN.md
修改配置? → 優先使用 configs/templates/
新指標? → 先在 2D Kolmogorov 驗證
```

---

# 📂 專案資源地圖

**訓練配置**: `configs/`
- `templates/`: 標準化模板 (2D Baseline, 3D Production, Curriculum)
- **修改配置優先改這裡**

**核心代碼**: `pinnx/`
- `models/`: Fourier MLP, SIREN, RWF
- `physics/`: Navier-Stokes 方程與 VS-PINN 實現
- `sensors/`: QR-Pivot 採樣演算法

**工具腳本**: `scripts/`
- `train/train.py`: 訓練入口
- `evaluate/`: 主要評估工具
- `visualize/`: 視覺化工具

**詳細文檔**：
- `docs/TECHNICAL_DOCUMENTATION.md`: 完整架構細節
- `scripts/README.md`: 詳細腳本指令參數說明

# ⚠️ 注意事項
1. **Loss ≠ Accuracy**: 永遠以 `evaluate.py` 的後驗指標 (L2, RMSE) 為準
2. **Config Management**: 優先修改現有 Template，避免產生大量一次性 Config 檔案
3. **Safety**: 執行耗時指令前，先向使用者解釋意圖

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
