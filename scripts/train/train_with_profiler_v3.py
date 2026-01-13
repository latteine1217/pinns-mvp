"""
簡化的 Profiler 腳本 - 直接使用 train.py 的核心邏輯

關鍵改進：
1. 預熱 20 個 epochs 後再開始統計
2. 統計 10 epochs 獲得準確平均值
3. 使用與生產環境相同的配置
"""

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import sys
from pathlib import Path
import logging

# 添加專案路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 導入與 train.py 相同的模組
from pinnx.train.config_loader import load_config
from pinnx.train.model_physics_factory import create_model, create_physics_constraints
from pinnx.data.kolmogorov_data_loader import load_kolmogorov_data
from pinnx.train.optimizer_factory import create_optimizer

def main():
    # 設置日誌
    logging.basicConfig(level=logging.INFO)
    
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"使用設備: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'='*80}\n")
    
    # 載入配置（使用與生產環境相同的配置）
    config_path = project_root / "configs/experiments/S2_k_scan/s2_qr_K50_2d_re50.yml"
    config = load_config(str(config_path))
    
    # 修改配置以加快測試
    config['training']['epochs'] = 40  # 20 warmup + 10 profile + 10 buffer
    config['training']['save_interval'] = 1000  # 禁用保存以加速
    
    print("正在載入數據...")
    data_dict = load_kolmogorov_data(config, device)
    
    print("正在創建模型...")
    model = create_model(config, device, statistics=None)
    model.train()
    
    print(f"模型參數量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    print("正在創建物理約束...")
    physics_constraints = create_physics_constraints(config, device)
    
    print("正在創建優化器...")
    optimizer = create_optimizer(model, config)
    
    # 定義簡單的訓練步驟（不含所有複雜邏輯，只測量核心計算）
    def train_step(model, data_dict, physics_constraints, optimizer):
        optimizer.zero_grad()
        
        # 前向傳播 - 數據點
        x_data = data_dict['sensor_data'][:, :2]  # (x, y)
        outputs_data = model(x_data)
        
        # 前向傳播 - PDE 點
        x_pde = data_dict['pde_points']
        x_pde.requires_grad_(True)
        outputs_pde = model(x_pde)
        
        # 計算 PDE 殘差（包含梯度計算）
        residuals = physics_constraints.compute_residuals(x_pde, outputs_pde)
        
        # 計算損失
        data_loss = torch.nn.functional.mse_loss(outputs_data, data_dict['sensor_data'][:, 2:])
        pde_loss = sum(r.pow(2).mean() for r in residuals.values())
        total_loss = 10.0 * data_loss + pde_loss
        
        # 反向傳播
        total_loss.backward()
        
        # 優化器步驟
        optimizer.step()
        
        # 返回損失值（.item() 調用）
        return {
            'total_loss': total_loss.item(),
            'data_loss': data_loss.item(),
            'pde_loss': pde_loss.item()
        }
    
    # ========================================
    # 階段 1: 預熱（不統計）
    # ========================================
    warmup_epochs = 20
    print(f"\n{'='*80}")
    print(f"階段 1: 預熱 {warmup_epochs} epochs（JIT 編譯 + GPU 預熱）")
    print(f"{'='*80}\n")
    
    for epoch in range(warmup_epochs):
        loss_dict = train_step(model, data_dict, physics_constraints, optimizer)
        if epoch % 5 == 0:
            print(f"  Warmup Epoch {epoch:2d}: loss = {loss_dict['total_loss']:.6f}")
    
    print("\n✅ 預熱完成")
    
    # ========================================
    # 階段 2: Profiling（統計 10 epochs）
    # ========================================
    profile_epochs = 10
    print(f"\n{'='*80}")
    print(f"階段 2: Profiling {profile_epochs} epochs")
    print(f"{'='*80}\n")
    
    # 使用 PyTorch Profiler
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=None,
    ) as prof:
        
        for epoch in range(profile_epochs):
            with record_function("train_step"):
                loss_dict = train_step(model, data_dict, physics_constraints, optimizer)
                
                if epoch % 2 == 0:
                    print(f"  Profile Epoch {epoch:2d}: loss = {loss_dict['total_loss']:.6f}")
            
            prof.step()
    
    print("\n✅ Profiling 完成")
    
    # ========================================
    # 階段 3: 分析結果
    # ========================================
    print(f"\n{'='*80}")
    print("階段 3: 分析 Profiler 結果")
    print(f"{'='*80}\n")
    
    events = prof.key_averages()
    
    # 1. 顯示 Top 操作
    if torch.cuda.is_available():
        print("="*80)
        print("Top 20 CUDA Operations")
        print("="*80)
        print(prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=20
        ))
    
    print("\n" + "="*80)
    print("Top 20 CPU Operations")
    print("="*80)
    print(prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=20
    ))
    
    # 2. 分析關鍵操作
    print("\n" + "="*80)
    print("關鍵操作時間佔比分析")
    print("="*80)
    
    item_time = 0
    grad_time = 0
    mm_time = 0
    
    for evt in events:
        key_lower = evt.key.lower()
        time_val = evt.cuda_time_total if torch.cuda.is_available() else evt.cpu_time_total
        
        if 'item' in key_lower or 'tolist' in key_lower:
            item_time += time_val
        if 'backward' in key_lower or 'grad' in key_lower:
            grad_time += time_val
        if 'mm' in key_lower or 'matmul' in key_lower or 'addmm' in key_lower:
            mm_time += time_val
    
    total_time = sum(evt.cuda_time_total if torch.cuda.is_available() else evt.cpu_time_total for evt in events)
    
    print(f"\n.item() 相關操作: {item_time/1000:.2f} ms ({item_time/total_time*100:.1f}%)")
    print(f"梯度計算:         {grad_time/1000:.2f} ms ({grad_time/total_time*100:.1f}%)")
    print(f"矩陣運算:         {mm_time/1000:.2f} ms ({mm_time/total_time*100:.1f}%)")
    print(f"─────────────────────────────────────")
    print(f"總時間:           {total_time/1000:.2f} ms")
    print(f"平均每 epoch:     {total_time/profile_epochs/1000:.2f} ms")
    
    # 3. 保存詳細報告
    output_dir = project_root / "profiler_results"
    output_dir.mkdir(exist_ok=True)
    
    report_file = output_dir / "report_v3.txt"
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"Profiler Report V3 (預熱後統計)\n")
        f.write(f"Device: {device}\n")
        f.write(f"Warmup epochs: {warmup_epochs}\n")
        f.write(f"Profile epochs: {profile_epochs}\n")
        f.write("="*80 + "\n\n")
        
        if torch.cuda.is_available():
            f.write("Top CUDA Operations:\n")
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
            f.write("\n\n")
        
        f.write("Top CPU Operations:\n")
        f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
        
        f.write("\n\n")
        f.write("="*80 + "\n")
        f.write("Summary Statistics:\n")
        f.write("="*80 + "\n")
        f.write(f".item() calls: {item_time/1000:.2f} ms ({item_time/total_time*100:.1f}%)\n")
        f.write(f"Gradient computation: {grad_time/1000:.2f} ms ({grad_time/total_time*100:.1f}%)\n")
        f.write(f"Matrix operations: {mm_time/1000:.2f} ms ({mm_time/total_time*100:.1f}%)\n")
        f.write(f"Total time: {total_time/1000:.2f} ms\n")
        f.write(f"Average per epoch: {total_time/profile_epochs/1000:.2f} ms\n")
    
    print(f"\n✅ 詳細報告已保存: {report_file}")
    
    # Chrome trace
    trace_file = output_dir / "trace_v3.json"
    prof.export_chrome_trace(str(trace_file))
    print(f"✅ Chrome trace 已保存: {trace_file}")
    
    print("\n" + "="*80)
    print("✅ Profiling 完成！")
    print("="*80)


if __name__ == "__main__":
    main()
