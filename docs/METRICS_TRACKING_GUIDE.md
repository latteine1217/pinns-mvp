# 指標追蹤完整指南（Metrics Tracking Guide）

**目的**：系統性追蹤計算成本與物理約束滿足度，支撐論文投稿。

---

## 📊 1. 計算成本指標（Efficiency Metrics）

### 1.1 訓練時間（Training Time）

#### 必須追蹤：
```python
import time
import wandb

# 訓練開始
start_time = time.time()

for epoch in range(max_epochs):
    epoch_start = time.time()

    # 訓練一個 epoch
    train_one_epoch()

    epoch_time = time.time() - epoch_start

    # 記錄每個 epoch 時間
    wandb.log({
        'timing/epoch_time': epoch_time,
        'timing/cumulative_time': time.time() - start_time,
    }, step=epoch)

    # 達標時間
    if l2_error < target_threshold:
        convergence_time = time.time() - start_time
        wandb.log({'timing/time_to_convergence': convergence_time})

# 訓練結束
total_wall_time = time.time() - start_time
gpu_hours = total_wall_time / 3600

wandb.log({
    'timing/total_wall_time': total_wall_time,
    'timing/gpu_hours': gpu_hours,
})
```

#### 關鍵指標：
- `total_wall_time`（秒）：總訓練時間
- `time_per_epoch`（秒）：每個 epoch 平均時間
- `time_to_convergence`（秒）：達到目標精度的時間
- `gpu_hours`（小時）：GPU 使用時數

---

### 1.2 記憶體使用（Memory Usage）

#### 必須追蹤：
```python
import torch
import wandb

# GPU 記憶體追蹤
if torch.cuda.is_available():
    # 訓練前
    torch.cuda.reset_peak_memory_stats()

    # 訓練中（定期記錄）
    current_memory = torch.cuda.memory_allocated() / 1024**2  # MB
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB

    wandb.log({
        'memory/current_gpu_memory_mb': current_memory,
        'memory/peak_gpu_memory_mb': peak_memory,
    })

# 模型參數量
model_parameters = sum(p.numel() for p in model.parameters())
wandb.log({'memory/model_parameters': model_parameters})

# 激活值記憶體（估算）
activation_memory = estimate_activation_memory(model, batch_size)
wandb.log({'memory/activation_memory_mb': activation_memory})
```

#### 關鍵指標：
- `peak_gpu_memory_mb`（MB）：峰值 GPU 記憶體
- `model_parameters`（個）：模型參數量
- `activation_memory_mb`（MB）：激活值記憶體

---

### 1.3 推理速度（Inference Speed）

#### 必須追蹤：
```python
import torch
import time
import wandb

# 推理速度測試
model.eval()
with torch.no_grad():
    # 預熱
    for _ in range(10):
        _ = model(test_input)

    # 計時
    num_samples = 1000
    inference_start = time.time()

    for _ in range(num_samples):
        _ = model(test_input)

    inference_time = (time.time() - inference_start) / num_samples * 1000  # ms
    throughput = num_samples / (time.time() - inference_start)  # samples/sec

    wandb.log({
        'inference/time_per_sample_ms': inference_time,
        'inference/throughput': throughput,
    })
```

#### 關鍵指標：
- `inference_time_per_sample_ms`（ms）：單樣本推理時間
- `throughput`（samples/sec）：吞吐量

---

### 1.4 效率綜合指標（Efficiency Score）

#### 必須追蹤：
```python
# 效率分數：精度 / 成本
efficiency_score = (1.0 - l2_error) / total_wall_time  # accuracy per second
memory_efficiency = (1.0 - l2_error) / peak_memory  # accuracy per MB
inference_efficiency = (1.0 - l2_error) / inference_time  # accuracy per ms

wandb.log({
    'efficiency/score': efficiency_score,
    'efficiency/memory_efficiency': memory_efficiency,
    'efficiency/inference_efficiency': inference_efficiency,
})
```

#### 關鍵指標：
- `efficiency_score`：精度/訓練時間
- `memory_efficiency`：精度/記憶體
- `inference_efficiency`：精度/推理時間

---

## ⚛️ 2. 物理約束指標（Physics Metrics）

### 2.1 質量守恆（Mass Conservation）

#### 必須追蹤：
```python
import torch
import wandb

def compute_divergence(u, v, w, x, y, z):
    """計算散度 ∇·u"""
    # 使用自動微分計算梯度
    dudx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]
    dvdy = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v),
                                create_graph=True)[0]

    if w is not None:  # 3D
        dwdz = torch.autograd.grad(w, z, grad_outputs=torch.ones_like(w),
                                    create_graph=True)[0]
        divergence = dudx + dvdy + dwdz
    else:  # 2D
        divergence = dudx + dvdy

    return divergence

# 計算散度指標
div = compute_divergence(u_pred, v_pred, w_pred, x, y, z)

div_mean = torch.mean(torch.abs(div)).item()
div_max = torch.max(torch.abs(div)).item()
div_99th = torch.quantile(torch.abs(div), 0.99).item()
div_l2 = torch.norm(div).item() / torch.sqrt(torch.tensor(div.numel()))
div_relative = div_l2 / torch.norm(torch.cat([u_pred, v_pred, w_pred])).item()

wandb.log({
    'physics/mass_conservation/divergence_mean': div_mean,
    'physics/mass_conservation/divergence_max': div_max,
    'physics/mass_conservation/divergence_99th': div_99th,
    'physics/mass_conservation/divergence_l2': div_l2,
    'physics/mass_conservation/divergence_relative': div_relative,
})
```

#### 關鍵指標：
- `divergence_mean`：散度平均值（目標 < 1e-3）
- `divergence_max`：散度最大值
- `divergence_l2`：散度 L2 norm
- `divergence_relative`：相對散度

---

### 2.2 動量守恆（Momentum Conservation）

#### 必須追蹤：
```python
def compute_momentum_residual(u, v, w, p, x, y, z, nu, rho):
    """計算 Navier-Stokes 動量方程殘差"""
    # x-momentum: ∂u/∂t + u·∇u = -1/ρ ∂p/∂x + ν∇²u + f_x
    # 穩態假設：∂u/∂t = 0

    # 對流項
    dudx = grad(u, x)
    dudy = grad(u, y)
    convection_x = u * dudx + v * dudy

    # 壓力項
    dpdx = grad(p, x)
    pressure_x = -1.0 / rho * dpdx

    # 擴散項
    d2udx2 = grad(dudx, x)
    d2udy2 = grad(dudy, y)
    diffusion_x = nu * (d2udx2 + d2udy2)

    # 殘差（假設無外力）
    residual_x = convection_x - pressure_x - diffusion_x

    # 同樣計算 y, z 方向
    residual_y = ...
    residual_z = ...  # 3D only

    return residual_x, residual_y, residual_z

res_x, res_y, res_z = compute_momentum_residual(u, v, w, p, x, y, z, nu, rho)

wandb.log({
    'physics/momentum_conservation/residual_x_mean': torch.mean(torch.abs(res_x)).item(),
    'physics/momentum_conservation/residual_y_mean': torch.mean(torch.abs(res_y)).item(),
    'physics/momentum_conservation/residual_max': torch.max(torch.abs(res_x)).item(),
})
```

#### 關鍵指標：
- `momentum_residual_mean`：動量殘差平均值（目標 < 1e-2）
- `momentum_residual_max`：動量殘差最大值

---

### 2.3 邊界條件（Boundary Conditions）

#### 週期邊界（Kolmogorov）：
```python
# 檢查週期性
u_left = u_pred[:, 0, :]   # x=0
u_right = u_pred[:, -1, :]  # x=L_x

periodicity_x_error = torch.mean(torch.abs(u_left - u_right)).item()

wandb.log({
    'physics/boundary_conditions/periodicity_x_error': periodicity_x_error,
    'physics/boundary_conditions/periodicity_y_error': periodicity_y_error,
})
```

#### 無滑移邊界（Channel flow）：
```python
# 檢查壁面無滑移條件
u_wall = u_pred[:, wall_indices, :]  # 壁面位置
wall_velocity_violation = torch.mean(torch.abs(u_wall)).item()  # 應 = 0

wandb.log({
    'physics/boundary_conditions/wall_velocity_violation': wall_velocity_violation,
})
```

#### 關鍵指標：
- `periodicity_error`：週期性誤差（目標 < 1e-4）
- `wall_velocity_violation`：壁面速度違反（目標 < 1e-5）

---

### 2.4 物理合理性（Physical Plausibility）

#### 能量譜（Energy Spectrum）：
```python
import numpy as np
from scipy.fft import fft2

def compute_energy_spectrum(u, v):
    """計算能量譜 E(k)"""
    # 2D FFT
    u_fft = fft2(u.cpu().numpy())
    v_fft = fft2(v.cpu().numpy())

    # 能量密度
    energy = 0.5 * (np.abs(u_fft)**2 + np.abs(v_fft)**2)

    # 徑向平均得到 E(k)
    E_k = radial_average(energy)

    return E_k

E_k_pred = compute_energy_spectrum(u_pred, v_pred)
E_k_dns = compute_energy_spectrum(u_true, v_true)

# 能譜誤差
spectrum_error = np.mean(np.abs(E_k_pred - E_k_dns) / (E_k_dns + 1e-10))

wandb.log({
    'physics/plausibility/energy_spectrum_error': spectrum_error,
})

# 繪製能譜對比
wandb.log({
    "physics/energy_spectrum": wandb.plot.line_series(
        xs=k_values,
        ys=[E_k_dns, E_k_pred],
        keys=["DNS", "PINN"],
        title="Energy Spectrum E(k)",
        xname="Wavenumber k"
    )
})
```

#### 湍流統計（3D Channel）：
```python
# 壁面剪應力
tau_w_pred = compute_wall_shear_stress(u_pred, y)
tau_w_dns = compute_wall_shear_stress(u_true, y)
tau_w_error = torch.abs(tau_w_pred - tau_w_dns) / tau_w_dns

wandb.log({
    'physics/turbulence/wall_shear_stress_error': tau_w_error.item(),
})

# Reynolds stress
reynolds_stress_pred = compute_reynolds_stress(u_pred, v_pred, w_pred)
reynolds_stress_dns = compute_reynolds_stress(u_true, v_true, w_true)
rs_error = torch.mean(torch.abs(reynolds_stress_pred - reynolds_stress_dns)).item()

wandb.log({
    'physics/turbulence/reynolds_stress_error': rs_error,
})
```

#### 關鍵指標：
- `energy_spectrum_error`：能譜誤差
- `wall_shear_stress_error`：壁面剪應力誤差
- `reynolds_stress_error`：Reynolds stress 誤差

---

### 2.5 綜合物理違反分數（Composite Score）

```python
# 定義權重（根據重要性）
w_div = 10.0      # 質量守恆最重要
w_mom = 1.0       # 動量守恆
w_bc = 5.0        # 邊界條件
w_energy = 0.1    # 能量（相對不重要）

physics_violation_score = (
    w_div * div_l2 +
    w_mom * momentum_residual_mean +
    w_bc * bc_max_violation +
    w_energy * energy_error
)

wandb.log({
    'metrics/physics_violation_score': physics_violation_score,
})
```

---

## 📝 3. 實作建議

### 3.1 在訓練腳本中追蹤（`train.py`）

```python
# 在 train.py 中加入 Timer 和 Memory Tracker
import time
import torch
from pinnx.utils.timer import Timer
from pinnx.utils.memory_tracker import MemoryTracker

timer = Timer()
memory_tracker = MemoryTracker()

timer.start()
for epoch in range(max_epochs):
    # 訓練
    loss = train_one_epoch()

    # 記錄時間
    timer.log_epoch(epoch)

    # 記錄記憶體
    memory_tracker.log(epoch)

    # 每 N epochs 評估物理指標
    if epoch % physics_eval_freq == 0:
        physics_metrics = evaluate_physics_constraints(model)
        wandb.log(physics_metrics, step=epoch)

timer.stop()
timer.log_summary()
```

### 3.2 在評估腳本中追蹤（`evaluate_checkpoint.py`）

```python
# 在 evaluate_checkpoint.py 中加入詳細物理驗證
from pinnx.evaluation.physics_validator import PhysicsValidator

validator = PhysicsValidator(config)

# 評估
results = validator.evaluate(model, test_data)

# 記錄到 WandB
wandb.log({
    **results['accuracy_metrics'],
    **results['physics_metrics'],
    **results['efficiency_metrics'],
})

# 生成報告
validator.generate_report(output_dir)
```

---

## 📊 4. WandB 視覺化建議

### 4.1 效率分析
```python
# Pareto frontier: 精度 vs 訓練時間
wandb.log({
    "efficiency/pareto_frontier": wandb.plot.scatter(
        table=wandb.Table(data=[[time, error, model] for ...]),
        x="training_time",
        y="L2_error",
        title="Efficiency Pareto Frontier"
    )
})
```

### 4.2 物理約束視覺化
```python
# Radar chart: 物理指標對比
wandb.log({
    "physics/radar_chart": wandb.plot.line_series(
        xs=[categories],
        ys=[vanilla_scores, full_scores],
        keys=["Vanilla", "Full"],
        title="Physics Constraint Satisfaction"
    )
})

# Heatmap: 散度空間分佈
wandb.log({
    "physics/divergence_heatmap": wandb.Image(divergence_plot)
})
```

---

## ✅ 5. 檢查清單（論文投稿前）

### 計算成本指標：
- [ ] 記錄總訓練時間（wall-time）
- [ ] 記錄達標時間（time to convergence）
- [ ] 記錄峰值 GPU 記憶體
- [ ] 記錄模型參數量
- [ ] 記錄推理速度
- [ ] 計算效率分數（精度/時間）

### 物理約束指標：
- [ ] 記錄質量守恆（散度）
- [ ] 記錄動量守恆（PDE 殘差）
- [ ] 記錄邊界條件滿足度
- [ ] 記錄能量譜誤差
- [ ] 記錄湍流統計（3D）
- [ ] 計算綜合物理違反分數

### 視覺化：
- [ ] Pareto frontier（精度 vs 時間）
- [ ] Radar chart（物理指標）
- [ ] Heatmap（散度分佈）
- [ ] Energy spectrum（E(k) 對比）

---

**最後更新**：2026-01-02

---

**文檔維護**: PINNs-MVP 團隊  
**版本**: 2.0.0  
**最後更新**: 2026-01-03
