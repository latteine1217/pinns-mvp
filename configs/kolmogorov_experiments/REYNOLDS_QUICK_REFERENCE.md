# Kolmogorov Flow 雷諾數快速參考

## 🔬 標準公式

```
Re = F / (ν² k³)
ν = √(F / (Re × k³))
```

## 📊 常用參數 (k=4)

| 目標 Re | 所需 ν (F=1.0) | 所需 F (ν=0.02) | 物理狀態 |
|---------|----------------|-----------------|---------|
| 30 | 0.022822 | 0.768 | 時空混沌 |
| 50 | 0.017678 | 0.640 | 時空混沌 |
| 60 | 0.016137 | 1.536 | 完全發展湍流 |
| 100 | 0.012500 | 2.560 | 完全發展湍流 |

## ⚡ 快速計算工具

```bash
# 計算所需黏滯度
python scripts/validation/validate_kolmogorov_reynolds.py --compute-nu --Re 30 --F 1.0 --k 4

# 驗證配置文件
python scripts/validation/validate_kolmogorov_reynolds.py --validate

# 生成參數表
python scripts/validation/validate_kolmogorov_reynolds.py --table
```

## ✅ 在 Python 中使用

```python
from pinnx.physics.kolmogorov_flow_2d import KolmogorovFlow2D

# 建立物理模組
physics = KolmogorovFlow2D(
    forcing_params={'amplitude': 0.768, 'wavenumber': 4},
    physics_params={'nu': 0.02}
)

# 計算雷諾數
Re = physics.compute_reynolds_number()  # 30.00

# 獲取完整物理信息
info = physics.get_physics_info()
print(info['physics_parameters']['Reynolds_number'])
```
