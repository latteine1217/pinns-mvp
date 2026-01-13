"""
改進的 Profiler 腳本 - 包含預熱階段

關鍵改進：
1. 預熱 10-20 個 epochs 後再開始統計
2. 統計更多 epochs（10-20 個）以獲得更準確的平均值
3. 使用接近實際的 batch size
4. 分別統計不同階段的性能
"""

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import sys
from pathlib import Path

# 添加專案路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pinnx.train.trainer_builder import TrainerBuilder
from pinnx.train.config_loader import load_config
import yaml

def main():
    # 設置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 根據可用記憶體調整 batch size
    if torch.cuda.is_available():
        # GPU: 嘗試使用接近實際的 batch size
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU 記憶體: {total_memory:.1f} GB")
        
        if total_memory >= 14:  # P100 16GB
            batch_size = 7000
        elif total_memory >= 6:  # GTX 1060 6GB
            batch_size = 4000
        else:
            batch_size = 2000
    else:
        batch_size = 2000
    
    print(f"Batch size: {batch_size}")
    
    # 載入配置
    config_path = project_root / "configs/experiments/S2_k_scan/s2_qr_K50_2d_re50.yml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 修改 batch size
    config['training']['N_pde'] = batch_size
    config['training']['epochs'] = 50  # 只訓練 50 epochs
    
    # 創建訓練器
    print("\n正在初始化訓練器...")
    builder = TrainerBuilder(config, device)
    trainer = builder.build_all()
    
    print(f"\n模型參數量: {sum(p.numel() for p in trainer.model.parameters()) / 1e6:.2f}M")
    
    # ========================================
    # 階段 1: 預熱（不統計）
    # ========================================
    warmup_epochs = 20
    print(f"\n{'='*80}")
    print(f"階段 1: 預熱 {warmup_epochs} epochs（不統計）")
    print(f"{'='*80}")
    
    for epoch in range(warmup_epochs):
        loss_dict = trainer.train_step(epoch)
        if epoch % 5 == 0:
            print(f"  Warmup Epoch {epoch}: loss = {loss_dict.get('total_loss', 0):.6f}")
    
    print("✅ 預熱完成")
    
    # ========================================
    # 階段 2: Profiling（統計 10 epochs）
    # ========================================
    profile_epochs = 10
    print(f"\n{'='*80}")
    print(f"階段 2: Profiling {profile_epochs} epochs")
    print(f"{'='*80}")
    
    # 使用 PyTorch Profiler
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=None,  # 不自動保存，手動處理
    ) as prof:
        
        for epoch in range(warmup_epochs, warmup_epochs + profile_epochs):
            with record_function("epoch"):
                loss_dict = trainer.train_step(epoch)
                
                if epoch % 2 == 0:
                    print(f"  Profile Epoch {epoch}: loss = {loss_dict.get('total_loss', 0):.6f}")
            
            # 每個 epoch 後記錄一次
            prof.step()
    
    print("\n✅ Profiling 完成")
    
    # ========================================
    # 階段 3: 分析結果
    # ========================================
    print(f"\n{'='*80}")
    print("階段 3: 分析 Profiler 結果")
    print(f"{'='*80}\n")
    
    # 1. 按 CUDA 時間排序（如果有 GPU）
    if torch.cuda.is_available():
        print("=" * 80)
        print("Top 20 CUDA Operations (by total time)")
        print("=" * 80)
        print(prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=20
        ))
        print()
    
    # 2. 按 CPU 時間排序
    print("=" * 80)
    print("Top 20 CPU Operations (by total time)")
    print("=" * 80)
    print(prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=20
    ))
    print()
    
    # 3. 分析關鍵操作
    print("=" * 80)
    print("關鍵操作分析")
    print("=" * 80)
    
    events = prof.key_averages()
    
    # 統計 .item() 相關操作
    item_time = 0
    item_count = 0
    for evt in events:
        if 'item' in evt.key.lower() or 'tolist' in evt.key.lower():
            item_time += evt.cuda_time_total if torch.cuda.is_available() else evt.cpu_time_total
            item_count += evt.count
    
    # 統計梯度計算
    grad_time = 0
    grad_count = 0
    for evt in events:
        if 'backward' in evt.key.lower() or 'grad' in evt.key.lower():
            grad_time += evt.cuda_time_total if torch.cuda.is_available() else evt.cpu_time_total
            grad_count += evt.count
    
    # 統計矩陣運算
    mm_time = 0
    mm_count = 0
    for evt in events:
        if 'mm' in evt.key.lower() or 'matmul' in evt.key.lower():
            mm_time += evt.cuda_time_total if torch.cuda.is_available() else evt.cpu_time_total
            mm_count += evt.count
    
    total_time = sum(evt.cuda_time_total if torch.cuda.is_available() else evt.cpu_time_total for evt in events)
    
    print(f"\n.item() 相關操作:")
    print(f"  總時間: {item_time / 1000:.3f} ms ({item_time / total_time * 100:.1f}%)")
    print(f"  調用次數: {item_count}")
    print(f"  平均時間: {item_time / max(item_count, 1) / 1000:.3f} ms/call")
    
    print(f"\n梯度計算:")
    print(f"  總時間: {grad_time / 1000:.3f} ms ({grad_time / total_time * 100:.1f}%)")
    print(f"  調用次數: {grad_count}")
    
    print(f"\n矩陣運算:")
    print(f"  總時間: {mm_time / 1000:.3f} ms ({mm_time / total_time * 100:.1f}%)")
    print(f"  調用次數: {mm_count}")
    
    print(f"\n總時間: {total_time / 1000:.3f} ms")
    print(f"平均每 epoch: {total_time / profile_epochs / 1000:.3f} ms")
    
    # 4. 保存詳細報告
    output_dir = project_root / "profiler_results"
    output_dir.mkdir(exist_ok=True)
    
    trace_file = output_dir / "trace_v2.json"
    prof.export_chrome_trace(str(trace_file))
    print(f"\n✅ Chrome trace 已保存: {trace_file}")
    print(f"   使用 chrome://tracing 查看")
    
    # 保存文本報告
    report_file = output_dir / "report_v2.txt"
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"Profiler Report (預熱後統計)\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Warmup epochs: {warmup_epochs}\n")
        f.write(f"Profile epochs: {profile_epochs}\n")
        f.write(f"Device: {device}\n")
        f.write("="*80 + "\n\n")
        
        if torch.cuda.is_available():
            f.write("CUDA Operations:\n")
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
            f.write("\n\n")
        
        f.write("CPU Operations:\n")
        f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
        
    print(f"✅ 詳細報告已保存: {report_file}")
    
    print("\n" + "="*80)
    print("✅ Profiling 完成！")
    print("="*80)


if __name__ == "__main__":
    main()
