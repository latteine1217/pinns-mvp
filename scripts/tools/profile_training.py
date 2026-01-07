"""
訓練效能分析工具

用途: 測量訓練循環各階段的實際耗時,識別真實瓶頸
使用: python scripts/tools/profile_training.py --cfg CONFIG_PATH --epochs 100
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
from collections import defaultdict
import json

import torch
import numpy as np


class TrainingProfiler:
    """訓練效能分析器
    
    測量各階段耗時:
    1. Data Loading (從 DataLoader 取批次)
    2. CPU→GPU Transfer (資料傳輸)
    3. Forward Pass (前向傳播)
    4. Loss Computation (損失計算)
    5. Backward Pass (反向傳播)
    6. Optimizer Step (優化器更新)
    7. GradNorm Update (權重更新,每 N 步)
    """
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.step_start_time = None
        self.phase_start_time = None
        self.current_phase = None
        
        # GPU 同步控制 (精確測量 CUDA 操作)
        self.use_cuda = torch.cuda.is_available()
        
    def start_step(self):
        """開始一個訓練步驟"""
        if self.use_cuda:
            torch.cuda.synchronize()
        self.step_start_time = time.perf_counter()
        
    def start_phase(self, phase_name: str):
        """開始測量某個階段"""
        if self.use_cuda:
            torch.cuda.synchronize()
        self.current_phase = phase_name
        self.phase_start_time = time.perf_counter()
        
    def end_phase(self):
        """結束當前階段測量"""
        if self.current_phase is None:
            return
        
        if self.use_cuda:
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - self.phase_start_time
        self.timings[self.current_phase].append(elapsed)
        self.current_phase = None
        
    def end_step(self):
        """結束一個訓練步驟"""
        if self.use_cuda:
            torch.cuda.synchronize()
        
        total_time = time.perf_counter() - self.step_start_time
        self.timings['total_step'].append(total_time)
        
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """獲取統計摘要
        
        Returns:
            字典包含每個階段的 mean/std/min/max/total 時間
        """
        summary = {}
        
        for phase, times in self.timings.items():
            if not times:
                continue
                
            times_arr = np.array(times)
            summary[phase] = {
                'mean': float(np.mean(times_arr)),
                'std': float(np.std(times_arr)),
                'min': float(np.min(times_arr)),
                'max': float(np.max(times_arr)),
                'total': float(np.sum(times_arr)),
                'count': len(times),
                'percentage': 0.0  # 稍後計算
            }
        
        # 計算各階段佔比
        if 'total_step' in summary:
            total_time = summary['total_step']['total']
            for phase in summary:
                if phase != 'total_step':
                    summary[phase]['percentage'] = (summary[phase]['total'] / total_time) * 100
        
        return summary
    
    def print_summary(self):
        """打印格式化的摘要報告"""
        summary = self.get_summary()
        
        print("\n" + "=" * 80)
        print("📊 訓練效能分析報告")
        print("=" * 80)
        
        # 按總時間排序
        sorted_phases = sorted(
            [(k, v) for k, v in summary.items() if k != 'total_step'],
            key=lambda x: x[1]['total'],
            reverse=True
        )
        
        print(f"\n{'階段':<25} {'平均 (ms)':<12} {'佔比 (%)':<10} {'調用次數':<10}")
        print("-" * 80)
        
        for phase, stats in sorted_phases:
            mean_ms = stats['mean'] * 1000
            percentage = stats['percentage']
            count = stats['count']
            print(f"{phase:<25} {mean_ms:>10.2f}   {percentage:>8.1f}   {count:>8}")
        
        if 'total_step' in summary:
            total_stats = summary['total_step']
            print("-" * 80)
            print(f"{'Total Step':<25} {total_stats['mean']*1000:>10.2f}   {100.0:>8.1f}   {total_stats['count']:>8}")
            print("=" * 80)
            
            # 額外統計
            print(f"\n⏱️  總訓練時間: {total_stats['total']:.2f} 秒")
            print(f"⚡ 平均步驟耗時: {total_stats['mean']*1000:.2f} ms")
            print(f"🚀 吞吐量: {1.0/total_stats['mean']:.2f} steps/sec")
            
            # GPU 記憶體使用
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated() / 1e9
                mem_reserved = torch.cuda.memory_reserved() / 1e9
                print(f"\n💾 GPU 記憶體使用:")
                print(f"   已分配: {mem_allocated:.2f} GB")
                print(f"   已保留: {mem_reserved:.2f} GB")
        
        print("=" * 80 + "\n")
        
    def save_to_json(self, output_path: Path):
        """保存結果為 JSON"""
        summary = self.get_summary()
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Profiling 結果已保存至: {output_path}")


def inject_profiler_into_trainer(trainer, profiler: TrainingProfiler):
    """將 profiler 注入到 Trainer 的 step 方法中
    
    策略: Monkey patch Trainer.step() 方法,在關鍵位置添加計時
    """
    original_step = trainer.step
    
    def profiled_step(data_batch: Dict[str, torch.Tensor], epoch: int) -> Dict[str, Any]:
        """帶 profiling 的 step 方法"""
        profiler.start_step()
        
        # === Phase 1: Forward Pass ===
        profiler.start_phase('1_forward_pass')
        predictions = trainer._forward_pass_all_points(data_batch)
        profiler.end_phase()
        
        # === Phase 2: Loss Computation ===
        profiler.start_phase('2_loss_computation')
        losses = trainer._compute_all_losses(predictions, data_batch, epoch)
        profiler.end_phase()
        
        # === Phase 3: Combine & Weight Losses ===
        profiler.start_phase('3_combine_weight_losses')
        total_loss, result = trainer._combine_and_weight_losses(
            losses, predictions['is_vs_pinn'], epoch
        )
        profiler.end_phase()
        
        # === Phase 4: Backward Pass ===
        profiler.start_phase('4_backward_pass')
        trainer.optimizer.zero_grad()
        total_loss.backward()
        profiler.end_phase()
        
        # === Phase 5: Gradient Clipping ===
        if trainer.train_cfg.get('gradient_clip', 0) > 0:
            profiler.start_phase('5_gradient_clipping')
            torch.nn.utils.clip_grad_norm_(
                trainer.model.parameters(),
                trainer.train_cfg['gradient_clip']
            )
            profiler.end_phase()
        
        # === Phase 6: Optimizer Step ===
        profiler.start_phase('6_optimizer_step')
        trainer.optimizer.step()
        profiler.end_phase()
        
        profiler.end_step()
        
        return result
    
    # 替換原方法
    trainer.step = profiled_step
    
    return trainer


def main():
    parser = argparse.ArgumentParser(description='訓練效能分析工具')
    parser.add_argument('--cfg', type=str, required=True, help='配置文件路徑')
    parser.add_argument('--epochs', type=int, default=100, help='Profiling 的 epoch 數量')
    parser.add_argument('--output', type=str, default='./profiling_results.json', help='結果輸出路徑')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup epochs (不計入統計)')
    
    args = parser.parse_args()
    
    # 設置日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "=" * 80)
    print("🔬 開始訓練效能分析")
    print("=" * 80)
    print(f"配置文件: {args.cfg}")
    print(f"Profiling epochs: {args.epochs} (warmup: {args.warmup})")
    print(f"輸出路徑: {args.output}")
    print("=" * 80 + "\n")
    
    # === 載入配置並構建 Trainer ===
    from pinnx.train.trainer_builder import TrainerBuilder
    from pinnx.utils.config import load_config
    
    config = load_config(args.cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 修改配置: 僅訓練指定 epochs
    config['training']['epochs'] = args.epochs + args.warmup
    config['logging']['wandb'] = False  # 禁用 WandB
    config['training']['checkpoint_freq'] = 0  # 禁用檢查點
    config['training']['validation_freq'] = 0  # 禁用驗證
    
    logging.info("🔧 構建 Trainer...")
    builder = TrainerBuilder(config, device)
    trainer = builder.build()
    
    # === 注入 Profiler ===
    profiler = TrainingProfiler()
    trainer = inject_profiler_into_trainer(trainer, profiler)
    
    logging.info(f"⏩ 執行 {args.warmup} epochs warmup...")
    
    # === Warmup (不計入統計) ===
    for epoch in range(args.warmup):
        trainer.step(trainer.training_data, epoch)
    
    # 清空 warmup 期間的計時數據
    profiler.timings.clear()
    
    logging.info(f"📊 開始 profiling ({args.epochs} epochs)...")
    
    # === Profiling ===
    overall_start = time.time()
    
    for epoch in range(args.warmup, args.epochs + args.warmup):
        trainer.step(trainer.training_data, epoch)
        
        # 每 10 epochs 打印進度
        if (epoch - args.warmup + 1) % 10 == 0:
            progress = (epoch - args.warmup + 1) / args.epochs * 100
            logging.info(f"進度: {progress:.0f}% ({epoch - args.warmup + 1}/{args.epochs} epochs)")
    
    overall_time = time.time() - overall_start
    
    logging.info(f"✅ Profiling 完成! 總時間: {overall_time:.2f} 秒")
    
    # === 生成報告 ===
    profiler.print_summary()
    
    # 保存 JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.save_to_json(output_path)
    
    # === 瓶頸識別 ===
    summary = profiler.get_summary()
    sorted_phases = sorted(
        [(k, v) for k, v in summary.items() if k != 'total_step'],
        key=lambda x: x[1]['percentage'],
        reverse=True
    )
    
    print("\n" + "🎯 瓶頸識別 (Top 3)")
    print("-" * 80)
    for i, (phase, stats) in enumerate(sorted_phases[:3], 1):
        print(f"{i}. {phase}: {stats['percentage']:.1f}% (平均 {stats['mean']*1000:.2f} ms)")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
