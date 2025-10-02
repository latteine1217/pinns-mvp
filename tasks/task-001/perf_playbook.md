# Performance Optimization Playbook - PINNs 逆重建

**版本**: v1.0  
**建立時間**: 2025-09-30  
**負責人**: Performance Engineer  
**適用範圍**: PINNs 逆重建專案全生命週期

## 📋 目錄
1. [背景與目標](#背景與目標)
2. [三波優化策略](#三波優化策略)
3. [物理正確性前置條件](#物理正確性前置條件)
4. [實施檢查清單](#實施檢查清單)
5. [監控與回滾](#監控與回滾)

---

## 🎯 背景與目標

### 專案特性
- **科學計算敏感性**: 物理正確性 > 計算效率
- **多尺度問題**: 小規模驗證 → 大規模生產
- **稀疏重建**: 從少量感測點（K≤12）重建完整場

### 效能目標
```yaml
targets:
  small_scale:    # 32×32 網格
    training_time: "< 10 min/1000 epochs"
    memory_usage: "< 2GB"
    sensor_selection: "< 5s"
  medium_scale:   # 128×128 網格  
    training_time: "< 2 hours/1000 epochs"
    memory_usage: "< 16GB"
    sensor_selection: "< 60s"
  large_scale:    # 512×512 網格
    training_time: "< 24 hours/5000 epochs"
    memory_usage: "< 64GB"
    sensor_selection: "< 600s"
```

---

## 🚀 三波優化策略

### 第一波：Quick Wins (1-2 天實施)

**目標**: 立即可得的 20-30% 效能提升，風險極低

#### 1.1 批次計算優化 ⚡
```python
# 當前問題：逐點計算 PDE 殘差
def compute_pde_residual_slow(model, points):
    residuals = []
    for point in points:
        r = compute_single_residual(model, point)
        residuals.append(r)
    return torch.stack(residuals)

# 優化：批次計算
def compute_pde_residual_fast(model, points):
    # 一次性計算所有點的殘差
    return compute_batch_residual(model, points)
```

**實施步驟**:
1. 識別所有逐點計算的地方
2. 改為批次張量運算
3. 驗證數值結果一致性

**預期提升**: 2-5x 前向傳播速度

#### 1.2 Tensor 記憶體管理 💾
```python
# 記憶體池避免重複分配
class TensorPool:
    def __init__(self):
        self.pools = {}
    
    def get_tensor(self, shape, dtype=torch.float32):
        key = (shape, dtype)
        if key not in self.pools:
            self.pools[key] = []
        
        if self.pools[key]:
            return self.pools[key].pop().zero_()
        else:
            return torch.zeros(shape, dtype=dtype)
    
    def return_tensor(self, tensor):
        key = (tuple(tensor.shape), tensor.dtype)
        if key in self.pools:
            self.pools[key].append(tensor)
```

**實施步驟**:
1. 分析記憶體分配熱點
2. 實施 tensor 池
3. 測量記憶體使用降低

**預期提升**: 20-40% 記憶體使用降低

#### 1.3 JIT 編譯關鍵函數 🔥
```python
# JIT 編譯數值密集函數
@torch.jit.script
def fourier_features_jit(x: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    z = 2.0 * math.pi * torch.mm(x, B)
    return torch.cat([torch.cos(z), torch.sin(z)], dim=-1)

@torch.jit.script  
def pde_residual_ns_jit(u: torch.Tensor, v: torch.Tensor, p: torch.Tensor,
                        u_x: torch.Tensor, u_y: torch.Tensor,
                        v_x: torch.Tensor, v_y: torch.Tensor) -> torch.Tensor:
    # 編譯的 N-S 方程殘差計算
    pass
```

**實施步驟**:
1. 識別純數值計算函數
2. 加上 @torch.jit.script 裝飾器
3. 處理 JIT 不支援的語法

**預期提升**: 10-30% 計算核心加速

### 第二波：Mid-term Optimizations (3-5 天實施)

**目標**: 結構性改善，需要較多測試

#### 2.1 稀疏計算利用 🕸️
```python
# 利用 QR-pivot 的稀疏結構
class SparseSensorMatrix:
    def __init__(self, sensor_indices, grid_shape):
        self.indices = sensor_indices
        self.sparse_map = self._build_sparse_mapping()
    
    def apply_sparse_operation(self, full_field):
        # 只計算感測點處的值，避免全場計算
        return full_field[self.indices]
```

**核心策略**:
- 感測器處的計算：避免全網格計算
- 稀疏梯度：只在需要的地方計算梯度
- 快取重用：感測器配置結果快取

**預期提升**: 3-10x 大網格計算加速

#### 2.2 GPU 並行優化 🖥️
```python
# CUDA 流並行
class CUDAStreamManager:
    def __init__(self, n_streams=4):
        self.streams = [torch.cuda.Stream() for _ in range(n_streams)]
        self.current = 0
    
    def run_async(self, func, *args, **kwargs):
        stream = self.streams[self.current]
        self.current = (self.current + 1) % len(self.streams)
        
        with torch.cuda.stream(stream):
            return func(*args, **kwargs)
```

**實施重點**:
- 模型前向/反向並行
- 資料載入與計算重疊
- 多 GPU 資料並行

**預期提升**: 2-4x GPU 利用率提升

#### 2.3 自適應計算精度 🎯
```python
# 訓練過程中動態調整計算精度
class AdaptivePrecisionTrainer:
    def __init__(self):
        self.precision = torch.float32
        self.precision_schedule = {
            0: torch.float32,      # 初期：高精度確保穩定
            1000: torch.float16,   # 中期：降低精度加速
            4000: torch.float32    # 後期：恢復高精度收斂
        }
    
    def update_precision(self, epoch):
        if epoch in self.precision_schedule:
            self.precision = self.precision_schedule[epoch]
```

**實施策略**:
- 混合精度訓練
- 動態精度調整
- 數值穩定性監控

**預期提升**: 40-70% 記憶體節省，1.5-2x 速度提升

### 第三波：Advanced Optimizations (1-2 週實施)

**目標**: 深層架構改善，需要大量驗證

#### 3.1 近似算法替代 🧮
```python
# 大規模 QR 分解的近似算法
class ApproximateQRSelector:
    def __init__(self, rank_fraction=0.1):
        self.rank_fraction = rank_fraction
    
    def select_sensors_approximate(self, data_matrix, n_sensors):
        # 隨機投影降維
        reduced_matrix = self._random_projection(data_matrix)
        # 在降維空間做 QR-pivot
        indices = self._qr_pivot_reduced(reduced_matrix, n_sensors)
        return indices
```

**核心算法**:
- 隨機投影 + QR-pivot
- 分層採樣策略
- 多解析度感測器選擇

**預期提升**: 10-50x 大規模感測器選擇

#### 3.2 分散式計算架構 🌐
```python
# 多 GPU/節點並行訓練
class DistributedPINNTrainer:
    def __init__(self, world_size):
        self.world_size = world_size
        self.device_map = self._setup_device_mapping()
    
    def train_distributed(self, model, data):
        # 模型並行：不同 GPU 負責不同網路層
        # 資料並行：不同節點處理不同空間區域
        pass
```

**架構設計**:
- 空間分解並行
- 模型分割並行
- 異步梯度更新

**預期提升**: 線性擴展到多節點

#### 3.3 Memory Mapping 大資料 💽
```python
# Out-of-core 計算支援
class MemoryMappedDataset:
    def __init__(self, data_path, chunk_size=1000):
        self.data_map = np.memmap(data_path, mode='r')
        self.chunk_size = chunk_size
    
    def get_batch_streaming(self, indices):
        # 只載入需要的資料塊
        return self._load_chunks(indices)
```

**實施重點**:
- HDF5/NetCDF 流式讀取
- JHTDB 資料的增量載入
- 快取友善的資料存取模式

**預期提升**: 支援 TB 級資料集

---

## 🧪 物理正確性前置條件

### 不可妥協的物理約束

#### 1. 數值穩定性 (Numerical Stability)
```python
# 每次優化後必須通過的檢查
def verify_numerical_stability(model, test_data):
    checks = {
        "gradient_finite": check_gradient_finiteness(model, test_data),
        "pde_residual": check_pde_residual_bounded(model, test_data), 
        "conservation": check_conservation_laws(model, test_data),
        "boundary_satisfaction": check_boundary_conditions(model, test_data)
    }
    
    for check_name, result in checks.items():
        assert result, f"Physics check failed: {check_name}"
```

#### 2. 守恆律驗證 (Conservation Laws)
- **質量守恆**: ∇·u = 0 (不可壓縮)
- **動量守恆**: N-S 方程殘差 < 1e-3
- **能量單調性**: 能量耗散 ≥ 0

#### 3. 邊界條件一致性
- Dirichlet 邊界：絕對誤差 < 1e-4
- Neumann 邊界：梯度誤差 < 1e-3
- 周期邊界：周期性保持

### 優化安全約束

#### A. 禁止破壞的項目
```yaml
forbidden_changes:
  - 物理方程係數修改
  - 邊界條件放寬
  - 守恆律檢查移除
  - 維度一致性破壞
```

#### B. 必須保持的精度
```python
precision_requirements = {
    "pde_residual_l2": 1e-3,
    "boundary_error_max": 1e-4,
    "conservation_violation": 1e-5,
    "reconstruction_l2": 0.15  # 15% 相對誤差上限
}
```

#### C. 回歸測試協議
1. **每次優化前**: 建立基準結果
2. **優化實施中**: 實時監控關鍵指標
3. **優化完成後**: 全面物理驗證
4. **發現問題時**: 立即回滾 + 根因分析

---

## ✅ 實施檢查清單

### Phase 1: Quick Wins

#### 批次計算優化
- [ ] 分析當前逐點計算位置
- [ ] 實施批次 PDE 殘差計算  
- [ ] 批次邊界條件評估
- [ ] 數值結果驗證 (相對誤差 < 1e-10)
- [ ] 效能基準測試

#### 記憶體管理
- [ ] 實施 Tensor 池
- [ ] 識別記憶體分配熱點
- [ ] 測量記憶體使用改善
- [ ] 長時間運行穩定性測試

#### JIT 編譯
- [ ] 標記純數值函數
- [ ] 實施 @torch.jit.script
- [ ] 處理 JIT 相容性問題
- [ ] 編譯後功能驗證
- [ ] 效能提升測量

### Phase 2: Mid-term

#### 稀疏計算
- [ ] 設計稀疏感測器數據結構
- [ ] 實施稀疏矩陣運算
- [ ] 快取感測器配置結果
- [ ] 大網格效能驗證

#### GPU 並行
- [ ] CUDA 流管理實施
- [ ] 資料載入並行化
- [ ] 多 GPU 支援
- [ ] GPU 記憶體最佳化

#### 自適應精度
- [ ] 混合精度訓練實施
- [ ] 動態精度調整邏輯
- [ ] 數值穩定性監控
- [ ] 收斂性驗證

### Phase 3: Advanced

#### 近似算法
- [ ] 近似 QR 算法研究
- [ ] 實施與驗證
- [ ] 精度影響評估
- [ ] 大規模測試

#### 分散式計算
- [ ] 多節點架構設計
- [ ] 通信協議實施
- [ ] 負載平衡策略
- [ ] 容錯機制

#### Memory Mapping
- [ ] 大資料集存取設計
- [ ] 流式載入實施
- [ ] JHTDB 整合
- [ ] 記憶體使用最佳化

---

## 📊 監控與回滾

### 關鍵監控指標

#### 效能指標
```python
performance_dashboard = {
    "training_speed": "epochs/hour",
    "memory_peak": "GB",
    "gpu_utilization": "%",
    "sensor_selection_time": "seconds",
    "convergence_rate": "loss/epoch"
}
```

#### 物理指標  
```python
physics_dashboard = {
    "pde_residual_l2": "dimensionless",
    "mass_conservation": "kg/s deviation", 
    "energy_conservation": "J/s deviation",
    "boundary_error_max": "dimensionless",
    "reconstruction_rmse": "% relative error"
}
```

### 自動回滾觸發條件

#### 紅線指標 (立即回滾)
- PDE 殘差 > 1e-2
- 邊界誤差 > 1e-3
- 守恆律偏差 > 1e-4
- 重建誤差 > 20%
- 記憶體使用 > 2x 基線

#### 黃線指標 (警告監控)
- 訓練速度 < 0.8x 基線
- 收斂率下降 > 20%
- GPU 利用率 < 50%

### 回滾操作程序

#### 1. 立即響應 (< 5 分鐘)
```bash
# 自動回滾腳本
git checkout HEAD~1  # 回到前一版本
python scripts/verify_physics.py --full  # 驗證基線正確性
python scripts/benchmark.py --quick     # 快速效能確認
```

#### 2. 根因分析 (< 1 小時)
- 比較程式碼差異
- 分析失敗測試日誌
- 檢查數值穩定性
- 記錄問題報告

#### 3. 修正策略 (< 4 小時)  
- 隔離問題模組
- 設計修正方案
- 實施最小修改
- 全面驗證測試

---

## 📚 參考資源與工具

### 效能分析工具
```bash
# 基本分析
python -m cProfile -o profile.stats script.py
python -m snakeviz profile.stats

# PyTorch 專用
python -c "
import torch.profiler as profiler
with profiler.profile() as prof:
    # 你的程式碼
prof.export_chrome_trace('trace.json')
"

# 記憶體分析
python -m memory_profiler script.py
```

### 物理驗證工具
```python
# 守恆律檢查
from pinnx.evals.physics import ConservationChecker
checker = ConservationChecker()
checker.verify_mass_conservation(u_field, v_field)
checker.verify_momentum_conservation(u, v, p, viscosity)

# 數值穩定性
from pinnx.evals.stability import NumericalStabilityChecker  
stability = NumericalStabilityChecker()
stability.check_gradient_explosion(model, test_points)
```

### 基準測試腳本
```bash
# 效能基準測試
python scripts/benchmark.py \
    --scales small,medium,large \
    --profile \
    --output benchmark_report.json

# 物理正確性測試
python scripts/physics_test.py \
    --conservation \
    --boundary \
    --pde-residual \
    --verbose
```

---

**最後更新**: 2025-09-30  
**版本**: v1.0  
**責任人**: Performance Engineer Sub-agent  
**審查狀態**: 待實施