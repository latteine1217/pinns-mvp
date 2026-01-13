#!/bin/bash
#SBATCH --job-name=profiler
#SBATCH --partition=r740
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/junyi/pinns-sparse-flow/logs/profiler_%j.out

# 環境設定
cd /home/junyi/pinns-sparse-flow
export PYTHONPATH=/home/junyi/pinns-sparse-flow:$PYTHONPATH

# 顯示環境信息
echo "=========================================="
echo "Profiler Job Started"
echo "=========================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# 執行 profiler（使用簡化版本）
python3 << 'EOF'
import torch
import sys
from pathlib import Path
from torch.profiler import profile, record_function, ProfilerActivity
import time

# 添加路徑
sys.path.insert(0, '/home/junyi/pinns-sparse-flow')

print(f"\n{'='*80}")
print("GPU 檢測")
print(f"{'='*80}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"{'='*80}\n")

# 設置設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 載入配置和數據
from pinnx.train.config_loader import load_config
from pinnx.train.model_physics_factory import create_model
from pinnx.data.kolmogorov_data_loader import load_kolmogorov_data
from pinnx.train.optimizer_factory import create_optimizer
from pinnx.physics.kolmogorov_flow_2d import KolmogorovFlow2D

config_path = "/home/junyi/pinns-sparse-flow/configs/experiments/S2_k_scan/s2_qr_K50_2d_re50.yml"
config = load_config(config_path)
config['training']['epochs'] = 40

print("載入數據...")
data_dict = load_kolmogorov_data(config, device)
print(f"  Sensor data: {data_dict['sensor_data'].shape}")
print(f"  PDE points: {data_dict['pde_points'].shape}")

print("\n創建模型...")
model = create_model(config, device, statistics=None)
model.train()
print(f"  參數量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

print("\n創建物理約束...")
physics = KolmogorovFlow2D(
    Re=config['physics']['Re'],
    kf=config['physics']['kf'],
    input_dim=config['model']['in_dim']
)

print("\n創建優化器...")
optimizer = create_optimizer(model, config)

# 簡化的訓練步驟
def train_step():
    optimizer.zero_grad()
    
    # 數據損失
    x_data = data_dict['sensor_data'][:, :2]
    y_data = data_dict['sensor_data'][:, 2:]
    pred_data = model(x_data)
    data_loss = torch.nn.functional.mse_loss(pred_data, y_data)
    
    # PDE 損失
    x_pde = data_dict['pde_points']
    x_pde.requires_grad_(True)
    pred_pde = model(x_pde)
    
    residuals = physics.compute_residuals(x_pde, pred_pde)
    pde_loss = sum(r.pow(2).mean() for r in residuals.values())
    
    # 總損失
    total_loss = 10.0 * data_loss + pde_loss
    
    # 反向傳播
    total_loss.backward()
    optimizer.step()
    
    return {
        'total': total_loss.item(),
        'data': data_loss.item(),
        'pde': pde_loss.item()
    }

# 預熱
print(f"\n{'='*80}")
print("階段 1: 預熱 20 epochs")
print(f"{'='*80}")

warmup_start = time.time()
for epoch in range(20):
    losses = train_step()
    if epoch % 5 == 0:
        print(f"  Epoch {epoch:2d}: loss = {losses['total']:.6f}")
warmup_time = time.time() - warmup_start

print(f"\n✅ 預熱完成 (用時: {warmup_time:.1f}秒, 平均: {warmup_time/20:.2f}秒/epoch)")

# Profiling
print(f"\n{'='*80}")
print("階段 2: Profiling 10 epochs")
print(f"{'='*80}\n")

activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(ProfilerActivity.CUDA)

with profile(
    activities=activities,
    record_shapes=True,
    profile_memory=True,
    with_stack=False,  # 關閉 stack trace 以加速
) as prof:
    for epoch in range(10):
        with record_function("train_step"):
            losses = train_step()
            if epoch % 2 == 0:
                print(f"  Epoch {epoch:2d}: loss = {losses['total']:.6f}")
        prof.step()

print(f"\n✅ Profiling 完成")

# 分析結果
print(f"\n{'='*80}")
print("Profiling 結果分析")
print(f"{'='*80}\n")

if torch.cuda.is_available():
    print("Top 15 CUDA Operations:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    print()

print("Top 15 CPU Operations:")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

# 統計關鍵操作
events = prof.key_averages()
item_time = sum(evt.cuda_time_total if torch.cuda.is_available() else evt.cpu_time_total 
                for evt in events if 'item' in evt.key.lower() or 'tolist' in evt.key.lower())
grad_time = sum(evt.cuda_time_total if torch.cuda.is_available() else evt.cpu_time_total 
                for evt in events if 'backward' in evt.key.lower() or 'grad' in evt.key.lower())
total_time = sum(evt.cuda_time_total if torch.cuda.is_available() else evt.cpu_time_total for evt in events)

print(f"\n{'='*80}")
print("關鍵指標")
print(f"{'='*80}")
print(f".item() 時間: {item_time/1000:.2f} ms ({item_time/total_time*100:.1f}%)")
print(f"梯度計算:     {grad_time/1000:.2f} ms ({grad_time/total_time*100:.1f}%)")
print(f"總時間:       {total_time/1000:.2f} ms")
print(f"平均/epoch:   {total_time/10/1000:.2f} ms")
print(f"{'='*80}")

# 保存結果
from pathlib import Path
output_dir = Path("/home/junyi/pinns-sparse-flow/profiler_results")
output_dir.mkdir(exist_ok=True)

report_file = output_dir / "profiler_report.txt"
with open(report_file, 'w') as f:
    f.write(f"Device: {device}\n")
    f.write(f"Warmup time: {warmup_time:.2f}s ({warmup_time/20:.3f}s/epoch)\n\n")
    if torch.cuda.is_available():
        f.write("CUDA Operations:\n")
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
        f.write("\n\n")
    f.write("CPU Operations:\n")
    f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

print(f"\n✅ 報告已保存: {report_file}")
print("\n✅ Profiler 完成！")
EOF

echo "=========================================="
echo "Profiler Job Completed"
echo "=========================================="
