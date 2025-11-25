# Re=100 训练启动前检查清单

## ✅ 训练前必检项目

### 1. DNS 数据准备
- [ ] DNS 数据文件存在: `data/kolmogorov_dns_re100_512x512.h5`
- [ ] 文件大小合理（> 50 MB）
- [ ] 包含关键快照 t=10.0（统计稳态）
- [ ] 数据无 NaN/Inf 值
- [ ] 散度误差 < 1e-3

**检查命令**:
```bash
# 文件大小
ls -lh data/kolmogorov_dns_re100_512x512.h5

# 数据完整性
python scripts/check_dns_re100_v2.py \
    --input data/kolmogorov_dns_re100_512x512.h5 \
    --verbose
```

---

### 2. 配置文件验证
- [ ] 配置文件存在: `configs/kolmogorov_experiments/kolmogorov_2d_re100_highres.yml`
- [ ] 雷诺数设置正确: `nu = 0.0125` → Re=100
- [ ] 强迫参数正确: `F=1.0, k=4`
- [ ] 网格匹配 DNS: `N=512`
- [ ] 感测点数合理: `K=100`

**检查命令**:
```bash
# 验证雷诺数
python scripts/validation/validate_kolmogorov_reynolds.py \
    --validate --config-dir configs/kolmogorov_experiments/ \
    | grep "re100_highres"

# 应显示:
# kolmogorov_2d_re100_highres.yml  1.0  4  0.012500  100.00  100  ✅
```

---

### 3. QR-Pivot 感测点
- [ ] 感测点文件存在: `data/kolmogorov_qr_sensors_re100_K100.npz`
- [ ] 包含 100 个感测点
- [ ] 空间分布合理（覆盖全域）
- [ ] 条件数 < 100（可选检查）

**生成命令**（如果不存在）:
```bash
python scripts/generate_2d_slice_qr_sensors_fixed_v2.py \
    --dns-file data/kolmogorov_dns_re100_512x512.h5 \
    --K 100 \
    --output data/kolmogorov_qr_sensors_re100_K100.npz \
    --snapshot-key "t_10.0" \
    --method "qr_pivot"
```

**检查命令**:
```bash
python -c "
import numpy as np
data = np.load('data/kolmogorov_qr_sensors_re100_K100.npz')
print(f'感测点数: {len(data[\"sensor_x\"])}')
print(f'X 范围: [{data[\"sensor_x\"].min():.2f}, {data[\"sensor_x\"].max():.2f}]')
print(f'Y 范围: [{data[\"sensor_y\"].min():.2f}, {data[\"sensor_y\"].max():.2f}]')
print(f'数据字段: {list(data.keys())}')
"
```

---

### 4. 输出目录准备
- [ ] 检查点目录: `checkpoints/kolmogorov_2d_re100_highres/`
- [ ] 结果目录: `results/kolmogorov_2d_re100_highres/`
- [ ] 日志目录: `log/kolmogorov_2d_re100_highres/`

**建立命令**:
```bash
mkdir -p checkpoints/kolmogorov_2d_re100_highres
mkdir -p results/kolmogorov_2d_re100_highres
mkdir -p log/kolmogorov_2d_re100_highres
```

---

### 5. 计算资源确认
- [ ] GPU 可用（推荐）
- [ ] 可用内存 > 8 GB
- [ ] 可用磁盘空间 > 5 GB
- [ ] Python 环境正确（PyTorch 2.x）

**检查命令**:
```bash
# GPU 状态
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# 内存状态
python -c "import psutil; mem = psutil.virtual_memory(); print(f'可用内存: {mem.available / 1024**3:.1f} GB / {mem.total / 1024**3:.1f} GB')"

# 磁盘空间
df -h . | tail -1 | awk '{print "可用磁盘: " $4}'
```

---

### 6. 配置参数检查

#### 模型架构
```yaml
model:
  type: "FourierMLP"
  width: 512          # ✅ 匹配 DNS 分辨率
  depth: 8            # ✅ 足够深度
  fourier_features:
    n_features: 256   # ✅ 高频敏感度
    sigma: 10.0       # ✅ 合理尺度
```

#### 训练参数
```yaml
training:
  n_epochs: 500       # ✅ 足够训练轮数
  optimizer: "SOAP"   # ✅ 前期快速收敛
  lr: 0.0001          # ✅ 稳定学习率
  batch_size: 1024    # ✅ 合理批次大小
```

#### 损失权重
```yaml
losses:
  data_weight: 1.0    # 数据一致性
  pde_weight: 0.1     # 物理残差（自适应调整）
  bc_weight: 1.0      # 边界条件
```

---

## 🚀 启动训练

### 方法 1: 自动化脚本（推荐）
```bash
./scripts/train_kolmogorov_re100_highres.sh
```

此脚本会自动执行：
1. 检查 DNS 数据
2. 生成 QR-Pivot 感测点（如果不存在）
3. 验证配置文件雷诺数
4. 建立输出目录
5. 启动训练（前台执行）

### 方法 2: 直接调用训练脚本
```bash
# 前台执行（可见即时输出）
python scripts/train.py \
    --cfg configs/kolmogorov_experiments/kolmogorov_2d_re100_highres.yml

# 背景执行（适合长时间训练）
nohup python scripts/train.py \
    --cfg configs/kolmogorov_experiments/kolmogorov_2d_re100_highres.yml \
    > log/kolmogorov_2d_re100_highres/training.log 2>&1 &

# 记录进程 ID
echo $! > log/kolmogorov_2d_re100_highres/train.pid
```

---

## 📊 训练监控

### 实时监控命令
```bash
# 监控训练日志
tail -f log/kolmogorov_2d_re100_highres/training.log

# 监控损失变化（每 10 秒刷新）
watch -n 10 'tail -20 log/kolmogorov_2d_re100_highres/training.log | grep "Loss"'

# 监控 GPU 使用率
watch -n 5 nvidia-smi

# 监控检查点生成
watch -n 30 'ls -lht checkpoints/kolmogorov_2d_re100_highres/ | head -10'
```

### 训练进度检查
```bash
# 查看最新 epoch
tail -1 log/kolmogorov_2d_re100_highres/training.log | grep -oP 'Epoch \K\d+'

# 查看平均 loss
tail -50 log/kolmogorov_2d_re100_highres/training.log | grep "Total Loss" | awk '{sum+=$NF; count++} END {print "平均 Loss:", sum/count}'

# 检查是否有 NaN
grep -i "nan\|inf" log/kolmogorov_2d_re100_highres/training.log || echo "✅ 无异常值"
```

---

## ⚠️ 常见启动问题

### 问题 1: CUDA out of memory
**解决方案**:
```yaml
# 降低批次大小
training:
  batch_size: 512    # 从 1024 降至 512
```

### 问题 2: DNS 数据未找到
**解决方案**:
```bash
# 等待 DNS 生成完成
tail -f log/dns_generation_re100.log

# 检查文件是否存在
ls -lh data/kolmogorov_dns_re100_512x512.h5
```

### 问题 3: 雷诺数验证失败
**解决方案**:
```bash
# 检查配置文件中的 nu 值
grep "nu:" configs/kolmogorov_experiments/kolmogorov_2d_re100_highres.yml

# 应显示: nu: 0.0125
```

### 问题 4: 感测点生成失败
**解决方案**:
```bash
# 检查 DNS 数据是否包含 t=10.0 快照
python -c "
import h5py
with h5py.File('data/kolmogorov_dns_re100_512x512.h5', 'r') as f:
    print('快照列表:', list(f['snapshots'].keys()))
"

# 如果没有 t_10.0，使用其他快照
python scripts/generate_2d_slice_qr_sensors_fixed_v2.py \
    --dns-file data/kolmogorov_dns_re100_512x512.h5 \
    --snapshot-key "t_15.0"  # 使用其他稳态快照
```

---

## 📝 训练参数建议

### 快速测试（验证流程）
```yaml
training:
  n_epochs: 100      # 快速验证
  save_interval: 20  # 频繁保存
```

### 标准训练（论文级结果）
```yaml
training:
  n_epochs: 500      # 充分收敛
  save_interval: 50  # 定期保存
  early_stopping:
    patience: 100    # 防止过拟合
```

### 长时间训练（极致性能）
```yaml
training:
  n_epochs: 2000     # 完全收敛
  optimizer: "SOAP"  # 前 1000 epochs
  switch_to_lbfgs:   # 后 1000 epochs
    epoch: 1000
```

---

## ✅ 最终确认

**训练前最后检查**:
```bash
# 一键检查所有依赖
python -c "
import os
from pathlib import Path

checks = {
    'DNS 数据': Path('data/kolmogorov_dns_re100_512x512.h5').exists(),
    '配置文件': Path('configs/kolmogorov_experiments/kolmogorov_2d_re100_highres.yml').exists(),
    '感测点数据': Path('data/kolmogorov_qr_sensors_re100_K100.npz').exists(),
    '输出目录': Path('checkpoints/kolmogorov_2d_re100_highres').exists(),
}

print('='*50)
print('训练前检查')
print('='*50)
for item, status in checks.items():
    icon = '✅' if status else '❌'
    print(f'{icon} {item}')

all_ok = all(checks.values())
print('='*50)
if all_ok:
    print('✅ 所有检查通过，可以启动训练！')
else:
    print('❌ 部分检查失败，请先修正问题')
"
```

**准备好后，执行**:
```bash
./scripts/train_kolmogorov_re100_highres.sh
```

🎯 祝训练顺利！
