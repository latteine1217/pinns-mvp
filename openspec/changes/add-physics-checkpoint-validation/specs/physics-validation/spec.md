# Physics Validation Specification

## ADDED Requirements

### Requirement: Automatic Physics Validation on Checkpoint Save
訓練器 SHALL 在保存檢查點前自動執行物理一致性驗證，確保模型符合質量守恆、動量守恆與邊界條件約束。

#### Scenario: Validation passes and checkpoint is saved
- **GIVEN** 訓練器準備保存檢查點
- **WHEN** 物理驗證執行且所有指標低於閾值
- **THEN** 檢查點成功保存，且元數據包含物理指標

#### Scenario: Validation fails and checkpoint is rejected
- **GIVEN** 訓練器準備保存檢查點
- **WHEN** 物理驗證執行且任一指標超過閾值
- **THEN** 檢查點保存被拒絕，且記錄警告日誌包含失敗原因與實際誤差值

#### Scenario: Validation is disabled via configuration
- **GIVEN** 配置中設定 `physics_validation.enabled: false`
- **WHEN** 訓練器準備保存檢查點
- **THEN** 跳過物理驗證，直接保存檢查點

---

### Requirement: Mass Conservation Validation
驗證器 SHALL 檢查速度場的散度（divergence）是否接近零，確保質量守恆。

#### Scenario: Mass conservation is satisfied
- **GIVEN** 速度場 `u, v, w` 從模型預測
- **WHEN** 計算 `div_u = ∂u/∂x + ∂v/∂y + ∂w/∂z`
- **THEN** `max(|div_u|) < mass_conservation_threshold` (預設 1e-2)

#### Scenario: Mass conservation is violated
- **GIVEN** 速度場 `u, v, w` 從模型預測
- **WHEN** 計算 `div_u = ∂u/∂x + ∂v/∂y + ∂w/∂z`
- **THEN** `max(|div_u|) >= mass_conservation_threshold`，驗證失敗並記錄實際誤差值

---

### Requirement: Momentum Conservation Validation
驗證器 SHALL 檢查 Navier-Stokes 方程殘差是否在可接受範圍內。

#### Scenario: Momentum conservation is satisfied
- **GIVEN** 速度場 `u, v, w` 與壓力場 `p` 從模型預測
- **WHEN** 計算 NS 殘差 `R = ∂u/∂t + (u·∇)u + ∇p - ν∇²u`
- **THEN** `mean(|R|) < momentum_conservation_threshold` (預設 1e-1)

#### Scenario: Momentum conservation is violated
- **GIVEN** 速度場 `u, v, w` 與壓力場 `p` 從模型預測
- **WHEN** 計算 NS 殘差 `R = ∂u/∂t + (u·∇)u + ∇p - ν∇²u`
- **THEN** `mean(|R|) >= momentum_conservation_threshold`，驗證失敗並記錄實際誤差值

---

### Requirement: Boundary Condition Validation
驗證器 SHALL 檢查壁面邊界條件（無滑移條件）是否被正確強制執行。

#### Scenario: Wall boundary condition is satisfied
- **GIVEN** 壁面位置 `y = 0` 與 `y = 2h`
- **WHEN** 計算壁面速度 `u_wall, v_wall, w_wall`
- **THEN** `max(|u_wall|, |v_wall|, |w_wall|) < boundary_condition_threshold` (預設 1e-3)

#### Scenario: Wall boundary condition is violated
- **GIVEN** 壁面位置 `y = 0` 與 `y = 2h`
- **WHEN** 計算壁面速度 `u_wall, v_wall, w_wall`
- **THEN** `max(|u_wall|, |v_wall|, |w_wall|) >= boundary_condition_threshold`，驗證失敗並記錄實際誤差值

---

### Requirement: Physics Metrics Recording
檢查點元數據 SHALL 包含物理驗證指標，用於後續分析與除錯。

#### Scenario: Metrics are recorded in checkpoint metadata
- **GIVEN** 物理驗證執行完成
- **WHEN** 檢查點保存
- **THEN** 檢查點元數據包含以下欄位：
  - `physics_metrics.mass_conservation_error`: 質量守恆最大誤差
  - `physics_metrics.momentum_conservation_error`: 動量守恆平均誤差
  - `physics_metrics.boundary_condition_error`: 邊界條件最大誤差
  - `physics_metrics.validation_passed`: 布林值，表示驗證是否通過

#### Scenario: Metrics are accessible for analysis
- **GIVEN** 檢查點已保存
- **WHEN** 載入檢查點
- **THEN** 可透過 `checkpoint['physics_metrics']` 存取物理指標

---

### Requirement: Configurable Validation Thresholds
使用者 SHALL 能夠透過配置檔案調整物理驗證閾值，以適應不同精度需求。

#### Scenario: User adjusts thresholds via configuration
- **GIVEN** 配置檔案包含 `physics_validation.thresholds` 區塊
- **WHEN** 訓練器載入配置
- **THEN** 驗證器使用配置中的閾值（而非預設值）

#### Scenario: User uses default thresholds
- **GIVEN** 配置檔案未指定 `physics_validation.thresholds`
- **WHEN** 訓練器載入配置
- **THEN** 驗證器使用預設閾值：
  - `mass_conservation: 1.0e-2`
  - `momentum_conservation: 1.0e-1`
  - `boundary_condition: 1.0e-3`

---

### Requirement: Validation Failure Logging
驗證失敗時，系統 SHALL 記錄詳細的警告日誌，包含失敗原因與實際誤差值。

#### Scenario: Detailed failure log is generated
- **GIVEN** 物理驗證失敗
- **WHEN** 檢查點保存被拒絕
- **THEN** 日誌包含以下資訊：
  - 失敗的驗證項目（質量守恆/動量守恆/邊界條件）
  - 實際誤差值
  - 配置的閾值
  - 建議的除錯步驟（如調整學習率、檢查網格解析度）

#### Scenario: Log is accessible for debugging
- **GIVEN** 驗證失敗日誌已記錄
- **WHEN** 使用者檢查訓練日誌
- **THEN** 可在日誌檔案中找到完整的驗證失敗資訊
