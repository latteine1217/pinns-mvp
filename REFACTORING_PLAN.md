# PINNx 模組全面簡潔性重構計畫

## 執行日期
2025-12-14

## 版本控制
- ✅ **Pre-refactoring commit**: `e959bb0` (已推送至 GitHub)
- 🔄 **Refactoring branch**: 將創建 `refactor/code-simplification` 分支
- 📌 **Target**: 減少總代碼複雜度 30-50%，保持 100% 測試通過率

---

## 第一階段：程式碼複雜度分析

### 檔案規模統計

| 檔案 | 行數 | 函數數 | 平均函數長度 | 複雜度評級 |
|------|------|--------|--------------|-----------|
| `pinnx/sensors/qr_pivot.py` | 2,385 | 7 | ~340 | 🔴 **極高** |
| `pinnx/train/trainer.py` | 2,113 | 19 | 107 | 🔴 **極高** |
| `pinnx/dataio/jhtdb_client.py` | 1,697 | ? | ? | 🟡 中 (外部依賴) |
| `pinnx/dataio/lowfi_loader.py` | 1,304 | ? | ? | 🟠 高 |
| `pinnx/train/factory.py` | 1,118 | 8 | ~140 | 🟠 高 |
| `pinnx/models/fourier_mlp.py` | 942 | 2 | ~471 | 🔴 **極高** |
| `pinnx/losses/residuals.py` | 1,011 | ? | ? | 🟠 高 |

**總計**: 31,930 行 Python 代碼

### 🚨 關鍵問題識別

#### 1. **trainer.py - `step()` 方法巨型化**
- **當前**: 785 行（佔整個 Trainer 類的 37%）
- **問題**: 
  - 違反單一職責原則
  - 包含數據載入、前向傳播、損失計算、反向傳播、優化、日誌、檢查點等多重職責
  - 難以測試、維護、擴展
- **目標**: 拆分為 5-10 個獨立方法，每個 < 100 行

#### 2. **fourier_mlp.py - 巨型類設計**
- **當前**: 2 個類，平均 ~471 行/類
- **問題**: 單一類包含過多功能（初始化、前向傳播、Fourier 特徵管理、權重初始化等）
- **目標**: 拆分為多個專注的小類

#### 3. **qr_pivot.py - 超長選擇器**
- **當前**: 7 個函數，平均 ~340 行/函數
- **問題**: 單一函數包含過多邏輯分支
- **目標**: 提取共用邏輯，使用策略模式

---

## 第二階段：逐模組重構計畫

### Phase 1: Trainer (`pinnx/train/trainer.py`)

**優先級**: 🔥 **最高**（影響整個訓練流程）

#### 當前結構分析

```
Trainer.step() - 785 行
├── 數據批次加載 (~50 行)
├── 前向傳播 (~80 行)
├── 損失計算 (~150 行)
│   ├── PDE residuals
│   ├── Data loss
│   ├── Prior loss
│   └── Boundary conditions
├── 反向傳播與優化 (~100 行)
├── 動態權重更新 (~80 行)
├── 自適應採樣 (~60 行)
├── Fourier annealing (~50 行)
├── 課程學習 (~80 行)
├── 日誌與監控 (~100 行)
└── 檢查點與驗證 (~55 行)
```

#### 重構策略

**目標**: 將 `step()` 從 785 行減少到 **< 150 行**

**新架構**:
```python
class Trainer:
    # 核心訓練循環（簡化版）
    def step(self, epoch: int, data_batch: Dict) -> Dict:
        """主訓練步驟（目標: < 150 行）"""
        # 1. 數據準備
        inputs = self._prepare_inputs(data_batch)
        
        # 2. 前向傳播
        predictions = self._forward_pass(inputs)
        
        # 3. 損失計算（委派給 LossManager）
        losses = self.loss_manager.compute_losses(
            predictions, inputs, data_batch
        )
        
        # 4. 反向傳播與優化
        self._backward_and_optimize(losses)
        
        # 5. 後處理（動態權重、採樣等）
        self._post_step_updates(epoch, losses)
        
        return losses
    
    # === 新增獨立方法 ===
    def _prepare_inputs(self, data_batch: Dict) -> Dict:
        """數據預處理與正規化（目標: < 80 行）"""
        pass
    
    def _forward_pass(self, inputs: Dict) -> Dict:
        """前向傳播（目標: < 50 行）"""
        pass
    
    def _backward_and_optimize(self, losses: Dict):
        """反向傳播與梯度更新（目標: < 80 行）"""
        pass
    
    def _post_step_updates(self, epoch: int, losses: Dict):
        """動態權重、採樣、Fourier annealing（目標: < 100 行）"""
        pass
```

**具體任務**:
- [ ] 創建 `LossManager` 類（整合 residuals, priors, weighting）
- [ ] 創建 `DataPreprocessor` 類（處理正規化、坐標轉換）
- [ ] 創建 `TrainingMonitor` 類（整合日誌、TensorBoard、檢查點）
- [ ] 創建 `DynamicStrategyManager` 類（動態權重、採樣、annealing）
- [ ] 將 `step()` 拆分為 5-8 個獨立方法

**預期效益**:
- 代碼行數: 785 → **~150** (-81%)
- 可測試性: 🔴 困難 → ✅ 簡單（每個子方法可獨立測試）
- 可維護性: 🔴 低 → ✅ 高（職責清晰）

---

### Phase 2: Models (`pinnx/models/fourier_mlp.py`)

**優先級**: 🔥 **高**

#### 當前問題
- `FourierMLP` 類 ~600 行，包含：
  - 網路初始化
  - Fourier 特徵生成
  - 前向傳播
  - RWF (Random Weight Factorization) 邏輯
  - 權重初始化策略

#### 重構策略

**拆分為 3 個獨立類**:

```python
# 1. Fourier 特徵處理（獨立模組）
class FourierFeatureLayer(nn.Module):
    """Fourier 特徵編碼（目標: < 100 行）"""
    def __init__(self, in_dim, num_frequencies, sigma):
        pass
    
    def forward(self, x):
        pass

# 2. RWF 處理（獨立模組）
class RandomWeightFactorization(nn.Module):
    """隨機權重分解（目標: < 80 行）"""
    pass

# 3. 主網路（簡化版）
class FourierMLP(nn.Module):
    """主網路架構（目標: < 150 行）"""
    def __init__(self, config):
        self.fourier_layer = FourierFeatureLayer(...)
        self.rwf = RandomWeightFactorization(...) if config.get('rwf') else None
        self.mlp = self._build_mlp(config)
    
    def forward(self, x):
        # 簡潔的前向傳播邏輯（< 30 行）
        x = self.fourier_layer(x) if self.fourier_layer else x
        x = self.rwf(x) if self.rwf else x
        return self.mlp(x)
```

**預期效益**:
- 代碼行數: 942 → **~400** (-58%)
- 模組化: 提升（每個特徵獨立）
- 可重用性: 提升（Fourier/RWF 可獨立使用）

---

### Phase 3: Losses (`pinnx/losses/`)

**優先級**: 🔥 **高**

#### 當前問題
- `residuals.py` (1,011 行): 包含所有 PDE residual 計算
- `priors.py` (792 行): Prior loss 計算
- `weighting.py` (589 行): 多種權重策略

#### 重構策略

**統一介面設計**:

```python
# 1. 抽象基類
class LossComponent(ABC):
    """所有 loss 組件的基類"""
    @abstractmethod
    def compute(self, predictions, targets, **kwargs) -> torch.Tensor:
        pass
    
    @abstractmethod
    def get_weight(self) -> float:
        pass

# 2. 具體實作
class PDEResidualLoss(LossComponent):
    """PDE residual loss（簡化版，< 150 行）"""
    pass

class RANSPriorLoss(LossComponent):
    """RANS prior loss（< 80 行）"""
    pass

class DataLoss(LossComponent):
    """Data-driven loss（< 50 行）"""
    pass

# 3. Loss Manager（整合所有 loss）
class LossManager:
    """統一管理所有 loss 組件（< 200 行）"""
    def __init__(self, config):
        self.components = self._build_components(config)
        self.weighter = self._build_weighter(config)
    
    def compute_losses(self, predictions, inputs, data_batch):
        losses = {}
        for name, component in self.components.items():
            losses[name] = component.compute(predictions, inputs)
        
        # 應用動態權重
        weighted_losses = self.weighter.apply(losses)
        return weighted_losses
```

**預期效益**:
- 代碼行數: 2,392 → **~1,000** (-58%)
- 可擴展性: 提升（新增 loss 只需繼承 `LossComponent`）
- 可測試性: 提升（每個 loss 可獨立測試）

---

### Phase 4: Physics (`pinnx/physics/`)

**優先級**: 🟡 **中**

#### 當前問題
- `ns_2d.py` (814 行): NS 方程殘差計算
- `kolmogorov_flow_2d.py` (748 行): Kolmogorov flow 專用
- `vs_pinn_channel_flow.py` (1,025 行): VS-PINN 實作

#### 重構策略

**提取共用邏輯**:

```python
# 1. 基礎 PDE 類
class PDEBase(ABC):
    """PDE 方程基類（< 100 行）"""
    @abstractmethod
    def compute_residuals(self, coords, predictions, **kwargs):
        pass

# 2. NS 方程共用邏輯
class NavierStokesBase(PDEBase):
    """NS 方程共用邏輯（< 200 行）"""
    def compute_continuity(self, u, v, coords):
        # 連續性方程（所有 NS 變體共用）
        pass
    
    def compute_momentum(self, u, v, p, coords, nu):
        # 動量方程（共用邏輯）
        pass

# 3. 具體實作（簡化版）
class NSEquations2D(NavierStokesBase):
    """2D NS 方程（< 150 行）"""
    pass

class KolmogorovFlow2D(NavierStokesBase):
    """Kolmogorov flow（< 120 行）"""
    pass
```

**預期效益**:
- 代碼行數: 2,587 → **~1,200** (-54%)
- 重複代碼: 減少（共用邏輯統一管理）

---

### Phase 5: Data I/O (`pinnx/dataio/`)

**優先級**: 🟡 **中**

#### 當前問題
- `lowfi_loader.py` (1,304 行): 低保真數據載入
- `channel_flow_loader.py` (893 行): Channel flow 專用
- 大量重複的 HDF5/NetCDF 讀取邏輯

#### 重構策略

**統一數據載入介面**:

```python
# 1. 抽象基類
class DataLoader(ABC):
    """數據載入器基類"""
    @abstractmethod
    def load(self, path: str) -> Dict[str, np.ndarray]:
        pass

# 2. 格式專用載入器
class HDF5Loader(DataLoader):
    """HDF5 格式載入（< 150 行）"""
    pass

class NetCDFLoader(DataLoader):
    """NetCDF 格式載入（< 150 行）"""
    pass

# 3. 數據預處理器
class DataPreprocessor:
    """統一預處理邏輯（< 200 行）"""
    def normalize(self, data):
        pass
    
    def interpolate(self, data, target_grid):
        pass
```

**預期效益**:
- 代碼行數: 3,197 → **~1,500** (-53%)
- 可維護性: 提升（格式處理邏輯統一）

---

### Phase 6: Sensors (`pinnx/sensors/qr_pivot.py`)

**優先級**: 🟠 **中高**

#### 當前問題
- 2,385 行，7 個函數，平均 ~340 行/函數
- 包含多種選擇器策略（QR, POD, Greedy, Multi-objective）
- 大量重複邏輯

#### 重構策略

**策略模式重構**:

```python
# 1. 抽象基類
class SensorSelector(ABC):
    """感測器選擇器基類"""
    @abstractmethod
    def select(self, data, K, **kwargs) -> np.ndarray:
        pass

# 2. 具體策略（每個 < 200 行）
class QRPivotSelector(SensorSelector):
    pass

class PODBasedSelector(SensorSelector):
    pass

class GreedySelector(SensorSelector):
    pass

# 3. 工廠模式
class SelectorFactory:
    """選擇器工廠（< 50 行）"""
    @staticmethod
    def create(strategy: str, **kwargs) -> SensorSelector:
        if strategy == 'qr':
            return QRPivotSelector(**kwargs)
        elif strategy == 'pod':
            return PODBasedSelector(**kwargs)
        # ...
```

**預期效益**:
- 代碼行數: 2,385 → **~1,000** (-58%)
- 可擴展性: 提升（新增策略更簡單）

---

## 第三階段：測試與驗證策略

### 回歸測試檢查點

每個 Phase 完成後必須執行：

```bash
# 1. 單元測試（必須 100% 通過）
pytest tests/ -v --tb=short

# 2. 整合測試
pytest tests/test_*_integration.py -v

# 3. 效能基準測試（確保無退化）
python scripts/benchmark/run_benchmarks.py

# 4. 記憶體洩漏檢測
pytest tests/ --memray
```

### 驗收標準

每個 Phase 必須滿足：
- ✅ 所有現有測試通過（100%）
- ✅ 代碼覆蓋率不降低（維持 > 80%）
- ✅ 效能無明顯退化（< 5% 差異）
- ✅ 向後相容（舊配置檔仍可使用）

---

## 第四階段：量化目標

### 總體目標

| 指標 | 當前 | 目標 | 改進 |
|------|------|------|------|
| **總代碼行數** | 31,930 | **~20,000** | **-37%** |
| **平均函數長度** | ~120 行 | **< 80 行** | **-33%** |
| **最長函數** | 785 行 | **< 200 行** | **-75%** |
| **單元測試覆蓋率** | ~75% | **> 85%** | **+10%** |
| **循環複雜度** | 高 | **中** | 顯著降低 |

### 模組級別目標

| 模組 | 當前行數 | 目標行數 | 減少 |
|------|---------|---------|------|
| `train/trainer.py` | 2,113 | **~1,200** | **-43%** |
| `models/fourier_mlp.py` | 942 | **~400** | **-58%** |
| `sensors/qr_pivot.py` | 2,385 | **~1,000** | **-58%** |
| `losses/*.py` | 2,392 | **~1,000** | **-58%** |
| `physics/*.py` | 5,838 | **~3,500** | **-40%** |
| `dataio/*.py` | 5,197 | **~2,800** | **-46%** |

---

## 第五階段：執行時程

### Week 1: Trainer 重構
- Day 1-2: 分析 `step()` 方法，設計新架構
- Day 3-4: 實作 `LossManager`, `DataPreprocessor`
- Day 5: 重構 `step()` 方法
- Day 6-7: 測試與驗證

### Week 2: Models & Losses 重構
- Day 1-3: `fourier_mlp.py` 拆分
- Day 4-6: `losses/` 統一介面
- Day 7: 測試與驗證

### Week 3: Physics & Data I/O 重構
- Day 1-3: `physics/` 提取共用邏輯
- Day 4-6: `dataio/` 統一載入介面
- Day 7: 測試與驗證

### Week 4: Sensors & 整體優化
- Day 1-3: `qr_pivot.py` 策略模式重構
- Day 4-5: 整體代碼審查與優化
- Day 6-7: 完整回歸測試與文檔更新

---

## 第六階段：風險管理

### 已識別風險

| 風險 | 影響 | 緩解措施 |
|------|------|---------|
| **向後不相容** | 🔴 高 | 保留舊介面作為 deprecated wrapper |
| **測試覆蓋不足** | 🟠 中 | 每個 Phase 前補充單元測試 |
| **效能退化** | 🟡 低 | 每次重構後執行基準測試 |
| **功能遺失** | 🟠 中 | 完整的整合測試覆蓋 |

### 回滾策略

- 每個 Phase 開始前創建獨立分支
- 保留完整的 Git 歷史
- 如發現嚴重問題，可快速回滾到 `e959bb0` commit

---

## 第七階段：文檔更新

### 必須更新的文檔

- [ ] `docs/TECHNICAL_DOCUMENTATION.md` - 新架構說明
- [ ] `docs/API_REFERENCE.md` - 新 API 文檔
- [ ] `README.md` - 更新範例代碼
- [ ] `MIGRATION_GUIDE.md` - 舊代碼遷移指南（新文件）
- [ ] 各模組內部 docstring 更新

---

## 附錄：重構原則檢查表

每次修改前確認：

- [ ] **單一職責**: 每個函數/類只做一件事
- [ ] **函數長度**: < 80 行（硬限制：< 150 行）
- [ ] **參數數量**: < 5 個（建議 < 3 個）
- [ ] **嵌套深度**: < 3 層
- [ ] **循環複雜度**: < 10
- [ ] **命名清晰**: 變數/函數名稱自解釋
- [ ] **註解精簡**: 代碼自說明，註解補充「為什麼」而非「做什麼」
- [ ] **DRY**: 無重複邏輯
- [ ] **YAGNI**: 不實作暫時不需要的功能
- [ ] **測試覆蓋**: 新代碼必須有對應單元測試

---

## 最終交付物

1. ✅ 所有代碼通過重構（減少 ~30-40% 代碼量）
2. ✅ 測試覆蓋率 > 85%
3. ✅ 完整文檔更新
4. ✅ 遷移指南（幫助現有使用者升級）
5. ✅ 效能基準報告（證明無退化）
6. ✅ 重構總結報告（詳細記錄所有變更）

---

**預計完成時間**: 4 週  
**預計代碼減少**: ~12,000 行 (-37%)  
**預計可維護性提升**: 顯著（函數平均長度從 120 → 80 行）
