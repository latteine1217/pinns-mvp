"""
簡化版訓練效能分析工具

直接在訓練循環中添加詳細計時,無需 monkey patching
使用方法: python scripts/tools/simple_profiler.py --cfg CONFIG_PATH --epochs 100
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from collections import defaultdict
import json

import torch
import numpy as np

# 添加專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class SimpleTimer:
    """簡單計時器"""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.start_times = {}
        self.use_cuda = torch.cuda.is_available()
    
    def start(self, phase: str):
        if self.use_cuda:
            torch.cuda.synchronize()
        self.start_times[phase] = time.perf_counter()
    
    def end(self, phase: str):
        if self.use_cuda:
            torch.cuda.synchronize()
        
        if phase in self.start_times:
            elapsed = time.perf_counter() - self.start_times[phase]
            self.timings[phase].append(elapsed)
            del self.start_times[phase]
    
    def get_stats(self, phase: str) -> dict:
        """獲取某個階段的統計資訊"""
        if phase not in self.timings or not self.timings[phase]:
            return {}
        
        times = np.array(self.timings[phase])
        return {
            'mean': float(np.mean(times)),
            'std': float(np.std(times)),
            'min': float(np.min(times)),
            'max': float(np.max(times)),
            'total': float(np.sum(times)),
            'count': len(times)
        }
    
    def print_report(self):
        """打印報告"""
        print("\n" + "=" * 100)
        print("📊 訓練效能分析報告")
        print("=" * 100)
        
        # 收集所有階段
        all_phases = sorted(self.timings.keys())
        
        # 計算總時間
        total_time = 0.0
        phase_times = {}
        for phase in all_phases:
            stats = self.get_stats(phase)
            if stats:
                phase_times[phase] = stats['total']
                total_time += stats['total']
        
        # 按時間排序
        sorted_phases = sorted(phase_times.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n{'階段':<30} {'總時間 (s)':<12} {'平均 (ms)':<12} {'佔比 (%)':<10} {'調用次數':<10}")
        print("-" * 100)
        
        for phase, phase_total in sorted_phases:
            stats = self.get_stats(phase)
            percentage = (phase_total / total_time * 100) if total_time > 0 else 0
            mean_ms = stats['mean'] * 1000
            
            print(f"{phase:<30} {phase_total:>10.2f}   {mean_ms:>10.2f}   {percentage:>8.1f}   {stats['count']:>8}")
        
        print("-" * 100)
        print(f"{'總計':<30} {total_time:>10.2f}   {'':>10}   {100.0:>8.1f}   {'':>8}")
        print("=" * 100)
        
        # GPU 資訊
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            print(f"\n💾 GPU 記憶體:")
            print(f"   已分配: {mem_allocated:.2f} GB")
            print(f"   已保留: {mem_reserved:.2f} GB")
        
        print("\n" + "=" * 100 + "\n")
        
        # 瓶頸識別
        print("🎯 瓶頸識別 (Top 5)")
        print("-" * 100)
        for i, (phase, phase_total) in enumerate(sorted_phases[:5], 1):
            stats = self.get_stats(phase)
            percentage = (phase_total / total_time * 100) if total_time > 0 else 0
            print(f"{i}. {phase}: {percentage:.1f}% (平均 {stats['mean']*1000:.2f} ms, 調用 {stats['count']} 次)")
        print("=" * 100 + "\n")
    
    def save_json(self, filepath: str):
        """保存為 JSON"""
        data = {}
        for phase in self.timings.keys():
            data[phase] = self.get_stats(phase)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ 結果已保存至: {filepath}")


def profile_training(config_path: str, num_epochs: int, warmup_epochs: int, output_path: str):
    """執行 profiling 訓練"""
    
    from pinnx.train.trainer_builder import TrainerBuilder
    from pinnx.train.config_loader import load_config
    from pinnx.train.model_physics_factory import create_model, create_physics
    from pinnx.train.loss_factory import create_loss_functions
    from pinnx.dataio.loaders.kolmogorov import prepare_kolmogorov_training_data
    
    # 載入配置
    config = load_config(config_path)
    
    # 從配置中獲取設備設定
    device_str = config.get('experiment', {}).get('device', 'auto')
    if device_str == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_str)
    
    print("\n" + "=" * 100)
    print("🔬 開始訓練效能分析")
    print("=" * 100)
    print(f"配置: {config_path}")
    print(f"設備: {device}")
    print(f"Profiling epochs: {num_epochs} (warmup: {warmup_epochs})")
    print(f"輸出: {output_path}")
    print("=" * 100 + "\n")
    
    # 1. 創建模型
    logging.info("🔧 創建模型...")
    model = create_model(config, device)
    
    # 2. 創建物理模組
    logging.info("🔧 創建物理模組...")
    physics = create_physics(config, device)
    
    # 3. 創建損失函數
    logging.info("🔧 創建損失函數...")
    losses = create_loss_functions(config, device)
    
    # 4. 準備訓練數據
    logging.info("🔧 準備訓練數據...")
    training_data = prepare_kolmogorov_training_data(config, device)
    
    # 5. 構建 Trainer
    logging.info("🔧 構建 Trainer...")
    builder = TrainerBuilder(config, device)
    builder.with_model(model)
    builder.with_physics(physics)
    builder.with_losses(losses)
    builder.with_training_data(training_data)
    trainer = builder.build()
    
    # 創建計時器
    timer = SimpleTimer()
    
    # === Warmup ===
    if warmup_epochs > 0:
        logging.info(f"⏩ 執行 {warmup_epochs} epochs warmup...")
        for epoch in range(warmup_epochs):
            trainer.step(trainer.training_data, epoch)
            if (epoch + 1) % 10 == 0:
                logging.info(f"Warmup 進度: {epoch + 1}/{warmup_epochs}")
    
    # === Profiling 訓練循環 ===
    logging.info(f"\n📊 開始 profiling ({num_epochs} epochs)...")
    
    for epoch in range(warmup_epochs, warmup_epochs + num_epochs):
        # === 整個 step 計時 ===
        timer.start('0_total_step')
        
        data_batch = trainer.training_data
        
        # === 1. Forward Pass ===
        timer.start('1_forward_pass')
        predictions = trainer._forward_pass_all_points(data_batch)
        timer.end('1_forward_pass')
        
        # === 2. Loss Computation ===
        timer.start('2_loss_computation')
        losses = trainer._compute_all_losses(predictions, data_batch, epoch)
        timer.end('2_loss_computation')
        
        # === 3. Combine & Weight Losses ===
        timer.start('3_combine_weight')
        total_loss, result = trainer._combine_and_weight_losses(
            losses, predictions['is_vs_pinn'], epoch
        )
        timer.end('3_combine_weight')
        
        # === 4. Backward Pass ===
        timer.start('4_backward')
        trainer.optimizer.zero_grad()
        total_loss.backward()
        timer.end('4_backward')
        
        # === 5. Gradient Clipping ===
        if trainer.train_cfg.get('gradient_clip', 0) > 0:
            timer.start('5_grad_clip')
            torch.nn.utils.clip_grad_norm_(
                trainer.model.parameters(),
                trainer.train_cfg['gradient_clip']
            )
            timer.end('5_grad_clip')
        
        # === 6. Optimizer Step ===
        timer.start('6_optimizer_step')
        trainer.optimizer.step()
        timer.end('6_optimizer_step')
        
        timer.end('0_total_step')
        
        # 進度打印
        if (epoch - warmup_epochs + 1) % 10 == 0:
            progress = (epoch - warmup_epochs + 1) / num_epochs * 100
            avg_step_time = timer.get_stats('0_total_step')['mean'] * 1000
            logging.info(f"進度: {progress:.0f}% ({epoch - warmup_epochs + 1}/{num_epochs}) | "
                        f"平均步驟: {avg_step_time:.2f} ms")
    
    logging.info(f"\n✅ Profiling 完成!")
    
    # === 生成報告 ===
    timer.print_report()
    
    # 保存 JSON
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    timer.save_json(output_path)


def main():
    parser = argparse.ArgumentParser(description='簡化版訓練效能分析')
    parser.add_argument('--cfg', type=str, required=True, help='配置文件路徑')
    parser.add_argument('--epochs', type=int, default=100, help='Profiling epochs')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup epochs')
    parser.add_argument('--output', type=str, default='./results/profiling_baseline.json', 
                       help='輸出路徑')
    
    args = parser.parse_args()
    
    # 設置日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        profile_training(
            config_path=args.cfg,
            num_epochs=args.epochs,
            warmup_epochs=args.warmup,
            output_path=args.output
        )
    except Exception as e:
        logging.error(f"❌ Profiling 失敗: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
