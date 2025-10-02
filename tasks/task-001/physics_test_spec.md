# 物理一致性測試規格

**文件版本**: v1.0  
**建立時間**: 2025-09-30  
**適用範圍**: PINNs 逆重建專案物理模組  

## 測試目標

建立系統性的物理正確性驗證框架，確保 N-S 方程式實作、尺度化處理、守恆律檢查的數學正確性與數值穩定性。

---

## 測試分類

### Level 1: 基礎數學正確性測試

#### 1.1 量綱一致性測試
```python
def test_dimensional_consistency():
    """驗證所有物理量的量綱正確性"""
    
    # 測試案例: 標準單位系統
    coords = torch.tensor([[0.1, 0.2], [0.3, 0.4]])  # [m]
    velocity = torch.tensor([[1.0, 0.5], [0.8, 1.2]])  # [m/s]
    pressure = torch.tensor([[100.0], [120.0]])  # [Pa = kg/(m·s²)]
    nu = 1e-6  # [m²/s]
    
    # 計算動量方程殘差
    pred = torch.cat([velocity, pressure, torch.zeros_like(pressure)], dim=1)
    mom_x, mom_y, cont = ns_residual_2d(coords, pred, nu)
    
    # 驗證: 動量方程殘差應有加速度量綱 [m/s²]
    assert_dimensional_consistency(mom_x, expected_dim="acceleration")
    assert_dimensional_consistency(cont, expected_dim="divergence")  # [1/s]
```

#### 1.2 解析解驗證測試
```python
def test_analytical_solutions():
    """基於已知解析解驗證方程式實作"""
    
    # 案例1: 泊肃葉流 (Poiseuille flow)
    def poiseuille_solution(y, U_max=1.0, H=1.0):
        """平行板間泊肃葉流的解析解"""
        return U_max * (1 - (2*y/H)**2), 0.0  # u, v
    
    # 案例2: 庫埃特流 (Couette flow)  
    def couette_solution(y, U_wall=1.0, H=1.0):
        """庫埃特流的解析解"""
        return U_wall * y / H, 0.0  # u, v
        
    # 測試: 解析解應使殘差趨近於零
    for solution_func in [poiseuille_solution, couette_solution]:
        residual = verify_analytical_solution(solution_func)
        assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-6)
```

#### 1.3 梯度計算正確性測試
```python
def test_gradient_accuracy():
    """驗證自動微分梯度計算精度"""
    
    # 測試函數: f(x,y) = sin(πx)cos(πy)
    # 已知: ∇²f = -2π²sin(πx)cos(πy)
    def test_function(coords):
        x, y = coords[:, 0], coords[:, 1]
        return torch.sin(np.pi * x) * torch.cos(np.pi * y)
    
    coords = torch.linspace(0, 1, 20).repeat(2, 1).T
    coords.requires_grad_(True)
    
    f = test_function(coords).unsqueeze(1)
    computed_laplacian = compute_laplacian(f, coords)
    
    # 解析拉普拉斯算子
    x, y = coords[:, 0], coords[:, 1]
    analytical_laplacian = -2 * np.pi**2 * torch.sin(np.pi * x) * torch.cos(np.pi * y)
    
    assert torch.allclose(computed_laplacian.squeeze(), analytical_laplacian, atol=1e-4)
```

### Level 2: 物理守恆律測試

#### 2.1 質量守恆測試
```python
def test_mass_conservation():
    """驗證不可壓縮流質量守恆"""
    
    # 測試: 對於真正的不可壓縮流場，連續方程殘差應為零
    def solenoidal_field(coords):
        """無散度速度場: u = sin(y), v = -sin(x)"""
        x, y = coords[:, 0], coords[:, 1]
        u = torch.sin(y).unsqueeze(1)
        v = -torch.sin(x).unsqueeze(1)
        return torch.cat([u, v], dim=1)
    
    coords = torch.randn(50, 2, requires_grad=True)
    velocity = solenoidal_field(coords)
    
    # 檢查散度
    divergence = compute_divergence(velocity, coords)
    assert torch.allclose(divergence, torch.zeros_like(divergence), atol=1e-6)
```

#### 2.2 動量守恆測試
```python  
def test_momentum_conservation():
    """驗證動量守恆在強制零源項時的表現"""
    
    # 測試: 無外力情況下的動量方程
    coords = torch.randn(30, 2, requires_grad=True)
    
    # 建立滿足連續方程的速度場
    velocity = create_divergence_free_field(coords)
    pressure = create_compatible_pressure_field(coords, velocity)
    
    pred = torch.cat([velocity, pressure, torch.zeros(coords.shape[0], 1)], dim=1)
    
    # 檢查無外力時的動量殘差
    mom_x, mom_y, _ = ns_residual_2d(coords, pred, nu=1e-3)
    
    # 在數值誤差範圍內應接近動量平衡
    assert check_momentum_balance(mom_x, mom_y, tolerance=1e-4)
```

#### 2.3 能量耗散測試
```python
def test_energy_dissipation():
    """驗證黏性耗散的正確性"""
    
    # 測試: 黏性流動中能量耗散率應為正
    coords = create_channel_geometry(nx=32, ny=16)
    velocity = create_shear_flow(coords)  # 具有剪切的流場
    
    dissipation_rate = compute_viscous_dissipation(velocity, coords, nu=1e-3)
    
    # 黏性耗散率必須非負
    assert torch.all(dissipation_rate >= 0)
    
    # 對於剪切流，耗散率應顯著大於零
    assert torch.mean(dissipation_rate) > 1e-6
```

### Level 3: 數值穩定性測試

#### 3.1 尺度變化魯棒性測試
```python
def test_scaling_robustness():
    """測試不同尺度下的數值穩定性"""
    
    scale_factors = [1e-3, 1e-1, 1.0, 1e1, 1e3]
    base_coords = torch.randn(20, 2, requires_grad=True)
    
    for scale in scale_factors:
        # 縮放座標
        coords = base_coords * scale
        
        # 建立測試場
        velocity = create_test_velocity_field(coords)
        pressure = create_test_pressure_field(coords)
        pred = torch.cat([velocity, pressure, torch.zeros_like(pressure)], dim=1)
        
        # 計算殘差
        residuals = ns_residual_2d(coords, pred, nu=1e-6 * scale**2)
        
        # 驗證: 殘差不應包含 NaN 或 Inf
        for residual in residuals:
            assert torch.isfinite(residual).all()
            assert not torch.isnan(residual).any()
```

#### 3.2 梯度消失/爆炸檢測
```python
def test_gradient_pathology():
    """檢測梯度消失或爆炸問題"""
    
    # 極端情況測試
    extreme_coords = torch.tensor([
        [1e-8, 1e-8],   # 極小值
        [1e8, 1e8],     # 極大值  
        [1e-8, 1e8],    # 混合尺度
    ], requires_grad=True, dtype=torch.float64)  # 使用雙精度
    
    # 建立高階測試函數
    def high_order_function(coords):
        return torch.sum(coords**10, dim=1, keepdim=True)
    
    f = high_order_function(extreme_coords)
    
    # 計算高階導數
    first_deriv = compute_derivatives(f, extreme_coords, order=1)
    second_deriv = compute_derivatives(f, extreme_coords, order=2)
    
    # 檢查數值健全性
    assert torch.isfinite(first_deriv).all()
    assert torch.isfinite(second_deriv).all()
    
    # 梯度範數不應過小或過大
    grad_norm = torch.norm(first_deriv, dim=1)
    assert torch.all(grad_norm > 1e-12)  # 避免梯度消失
    assert torch.all(grad_norm < 1e12)   # 避免梯度爆炸
```

### Level 4: 尺度化一致性測試  

#### 4.1 VS-PINN 尺度變換測試
```python
def test_vs_scaling_invariance():
    """驗證 VS-PINN 尺度變換的物理一致性"""
    
    # 原始物理量
    input_data = torch.randn(50, 2) * 100  # 大尺度輸入
    output_data = torch.randn(50, 3) * 0.01  # 小尺度輸出
    
    # 建立尺度器
    scaler = VSScaler(input_dim=2, output_dim=3, learnable=False)
    scaler.fit(input_data, output_data)
    
    # 尺度變換
    scaled_input = scaler.transform_input(input_data)
    scaled_output = scaler.transform_output(output_data)
    
    # 逆變換
    recovered_input = scaler.inverse_transform_input(scaled_input)
    recovered_output = scaler.inverse_transform_output(scaled_output)
    
    # 驗證: 逆變換應恢復原始值
    assert torch.allclose(input_data, recovered_input, atol=1e-5)
    assert torch.allclose(output_data, recovered_output, atol=1e-5)
    
    # 驗證: 標準化後的資料應具有標準分佈特性
    assert torch.allclose(scaled_input.mean(dim=0), torch.zeros(2), atol=1e-3)
    assert torch.allclose(scaled_input.std(dim=0), torch.ones(2), atol=1e-3)
```

#### 4.2 梯度尺度變換測試
```python
def test_gradient_scaling_consistency():
    """驗證梯度在尺度變換下的正確性"""
    
    # 建立測試函數與尺度器
    coords = torch.randn(20, 2, requires_grad=True)
    scaler = StandardScaler()
    scaler.fit(coords)
    
    # 原始空間梯度
    f_original = torch.sum(coords**3, dim=1, keepdim=True)
    grad_original = compute_derivatives(f_original, coords)
    
    # 尺度空間梯度
    coords_scaled = scaler.transform(coords)
    f_scaled = torch.sum(coords_scaled**3, dim=1, keepdim=True)  
    grad_scaled = compute_derivatives(f_scaled, coords_scaled)
    
    # 梯度反尺度化
    grad_descaled = denormalize_gradients(grad_scaled, scaler, coords_scaled, f_scaled)
    
    # 驗證: 反尺度化後的梯度應與原始梯度一致 (在尺度因子內)
    scale_factor = scaler.std.squeeze()
    expected_grad = grad_original / scale_factor  # 簡化的尺度關係
    
    assert torch.allclose(grad_descaled, expected_grad, rtol=1e-2)
```

---

## 測試執行框架

### 自動化測試管道
```bash
# 完整物理測試套件
pytest tests/physics/ -v --tb=short

# 按級別執行
pytest tests/physics/test_level1_basic.py    # 基礎數學
pytest tests/physics/test_level2_conservation.py  # 守恆律
pytest tests/physics/test_level3_stability.py     # 數值穩定性  
pytest tests/physics/test_level4_scaling.py       # 尺度一致性
```

### 持續驗證指標
```python
# 在 conftest.py 中定義的全域指標
PHYSICS_TOLERANCE = {
    'mass_conservation': 1e-6,
    'momentum_conservation': 1e-5,
    'energy_conservation': 1e-4,
    'gradient_accuracy': 1e-6,
    'scaling_consistency': 1e-5
}

# 自動報告生成
@pytest.fixture(scope="session", autouse=True)
def physics_test_report():
    """生成物理測試總結報告"""
    yield
    generate_physics_test_summary()
```

---

## 失敗排除指南

### 常見失敗模式
1. **量綱不匹配**: 檢查單位系統一致性
2. **梯度計算錯誤**: 確認 `requires_grad=True` 設定
3. **數值溢出**: 檢查輸入資料範圍與尺度化
4. **守恆律違反**: 驗證邊界條件與初始條件設定

### 除錯工具
```python
def debug_physics_computation(coords, pred, nu):
    """物理計算除錯工具"""
    
    print("=== 物理計算除錯資訊 ===")
    print(f"座標範圍: [{coords.min():.3e}, {coords.max():.3e}]")
    print(f"預測值範圍: [{pred.min():.3e}, {pred.max():.3e}]")  
    print(f"黏性係數: {nu:.3e}")
    
    # 逐步計算與檢查
    try:
        residuals = ns_residual_2d(coords, pred, nu)
        print(f"殘差計算成功: {[r.shape for r in residuals]}")
        
        for i, r in enumerate(residuals):
            print(f"殘差 {i}: 均值={r.mean():.3e}, 標準差={r.std():.3e}")
            
    except Exception as e:
        print(f"計算失敗: {e}")
        import traceback
        traceback.print_exc()
```

---

## 驗收標準

### 通過條件
- ✅ 所有 Level 1-4 測試通過率 ≥ 95%
- ✅ 無梯度病態問題 (消失/爆炸)
- ✅ 守恆律誤差在指定容差內
- ✅ 尺度變換數值穩定

### Physics Gate 通過標準
```python
PHYSICS_GATE_CRITERIA = {
    'analytical_solution_error': 1e-6,
    'conservation_error': 1e-5, 
    'gradient_consistency': 1e-4,
    'scaling_robustness': True,
    'dimensional_consistency': True
}
```

---

**下次更新**: 根據測試執行結果調整容差與測試案例  
**維護責任**: Physicist Sub-Agent  
**相關文檔**: `physics_review.md`, `physics_risks_register.md`