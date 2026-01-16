"""
PDE Residual 詳細效能分析

專注於測量 PDE residual 計算的詳細時間分布
"""

import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import logging
import numpy as np

import torch
import yaml

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class DetailedTimer:
    """詳細計時器"""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.start_times: Dict[str, float] = {}
        
    def start(self, name: str):
        """開始計時"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_times[name] = time.perf_counter()
        
    def stop(self, name: str):
        """停止計時並記錄"""
        if name not in self.start_times:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start_times[name]
        self.timings[name].append(elapsed * 1000)  # 轉換為毫秒
        del self.start_times[name]
        
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """獲取統計數據"""
        stats = {}
        for name, times in self.timings.items():
            if len(times) == 0:
                continue
            stats[name] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'total': np.sum(times),
                'count': len(times)
            }
        return stats
    
    def print_report(self):
        """打印效能報告"""
        stats = self.get_stats()
        if not stats:
            print("⚠️  無計時數據")
            return
        
        # 計算總時間
        total_time = sum(s['total'] for s in stats.values())
        
        print("\n" + "=" * 100)
        print("⏱️  PDE Residual 詳細效能分析報告")
        print("=" * 100)
        print(f"{'組件名稱':<35} {'平均(ms)':<12} {'標準差(ms)':<12} {'佔比(%)':<10} {'調用次數':<10}")
        print("-" * 100)
        
        # 按總時間排序
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True)
        
        for name, stat in sorted_stats:
            percentage = (stat['total'] / total_time) * 100 if total_time > 0 else 0
            print(f"{name:<35} {stat['mean']:>10.3f}   {stat['std']:>10.3f}   "
                  f"{percentage:>8.1f}   {stat['count']:>8}")
        
        print("-" * 100)
        print(f"{'總時間':<35} {total_time:>10.3f} ms")
        print("=" * 100)


# 全局計時器實例
timer = DetailedTimer()


def patch_residual_for_profiling():
    """為 PDE residual 計算插入計時器"""
    from pinnx.losses import residuals
    
    # 保存原始函數
    original_compute_all_gradients_2d = residuals.compute_all_gradients_2d
    original_ns_residual_2d = residuals.ns_residual_2d
    
    # 包裝 compute_all_gradients_2d
    def profiled_compute_all_gradients_2d(coords, velocity, pressure):
        timer.start('residual_gradient_computation')
        result = original_compute_all_gradients_2d(coords, velocity, pressure)
        timer.stop('residual_gradient_computation')
        return result
    
    # 包裝 ns_residual_2d
    def profiled_ns_residual_2d(u_pred, coords, nu, rho, forcing=None, **kwargs):
        timer.start('residual_total')
        
        # 提取速度與壓力
        timer.start('residual_velocity_extraction')
        velocity = u_pred[:, :2]
        pressure = u_pred[:, 2:3]
        timer.stop('residual_velocity_extraction')
        
        # 計算梯度
        grads = profiled_compute_all_gradients_2d(coords, velocity, pressure)
        
        # 計算殘差
        timer.start('residual_momentum_x')
        u, v = velocity[:, 0:1], velocity[:, 1:2]
        u_grad = grads['u_grad']
        u_lap = grads['u_lap']
        p_grad = grads['p_grad']
        
        momentum_x = (
            u * u_grad[:, 0:1] + v * u_grad[:, 1:2] +
            (1.0 / rho) * p_grad[:, 0:1] -
            nu * u_lap
        )
        if forcing is not None and 'fx' in forcing:
            momentum_x = momentum_x - forcing['fx']
        timer.stop('residual_momentum_x')
        
        timer.start('residual_momentum_y')
        v_grad = grads['v_grad']
        v_lap = grads['v_lap']
        
        momentum_y = (
            u * v_grad[:, 0:1] + v * v_grad[:, 1:2] +
            (1.0 / rho) * p_grad[:, 1:2] -
            nu * v_lap
        )
        if forcing is not None and 'fy' in forcing:
            momentum_y = momentum_y - forcing['fy']
        timer.stop('residual_momentum_y')
        
        timer.start('residual_continuity')
        continuity = u_grad[:, 0:1] + v_grad[:, 1:2]
        timer.stop('residual_continuity')
        
        timer.stop('residual_total')
        
        return {
            'momentum_x': momentum_x,
            'momentum_y': momentum_y,
            'continuity': continuity
        }
    
    # 替換函數
    residuals.compute_all_gradients_2d = profiled_compute_all_gradients_2d
    residuals.ns_residual_2d = profiled_ns_residual_2d
    
    logging.info("✅ PDE Residual 已插入詳細計時器")


def patch_trainer_step_for_profiling():
    """為 Trainer.step 插入計時器"""
    from pinnx.train.trainer import Trainer
    
    original_step = Trainer.step
    
    def profiled_step(self, data_batch, epoch):
        timer.start('step_total')
        
        timer.start('step_data_transfer')
        self.optimizer.zero_grad()
        data_batch = self._transfer_batch_to_device(data_batch)
        timer.stop('step_data_transfer')
        
        timer.start('step_unpack')
        unpacked = self._unpack_data_batch(data_batch)
        timer.stop('step_unpack')
        
        timer.start('step_forward_pass')
        predictions = self._forward_pass_all_points(data_batch)
        timer.stop('step_forward_pass')
        
        timer.start('step_compute_losses')
        losses = self._compute_all_losses(predictions, data_batch, epoch)
        timer.stop('step_compute_losses')
        
        timer.start('step_combine_losses')
        total_loss, result = self._combine_and_weight_losses(losses, predictions['is_vs_pinn'], epoch)
        timer.stop('step_combine_losses')
        
        timer.start('step_backward')
        self._backward_and_optimize(total_loss, data_batch, epoch)
        timer.stop('step_backward')
        
        self._add_training_metadata(result, losses, epoch)
        
        timer.stop('step_total')
        return result
    
    Trainer.step = profiled_step
    
    logging.info("✅ Trainer.step 已插入計時器")


def run_detailed_profiling(config_path: str, num_epochs: int = 10):
    """運行詳細效能分析"""
    
    # 載入配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 確保不使用 wandb
    config['use_wandb'] = False
    if 'logging' not in config:
        config['logging'] = {}
    config['logging']['wandb'] = False
    
    # 設定訓練 epochs
    config['training']['epochs'] = num_epochs
    
    # 設定日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"🔧 使用設備: {device}")
    if torch.cuda.is_available():
        logging.info(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # 插入計時器（在建立 trainer 之前）
    patch_residual_for_profiling()
    patch_trainer_step_for_profiling()
    
    # 建立 Trainer
    from pinnx.train.trainer_builder import TrainerBuilder
    
    logging.info("🏗️  正在建立 Trainer...")
    builder = TrainerBuilder(config, device)
    trainer = builder.build()
    
    # 執行訓練
    logging.info(f"🚀 開始詳細效能分析訓練（{num_epochs} epochs）...")
    start_time = time.time()
    trainer.train()
    total_time = time.time() - start_time
    
    # 打印報告
    timer.print_report()
    
    print(f"\n總訓練時間: {total_time:.2f}s ({total_time/num_epochs:.3f}s/epoch)")
    print(f"平均每步時間: {total_time*1000/num_epochs:.2f} ms")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='PDE Residual 詳細效能分析')
    parser.add_argument('--config', type=str, required=True, help='配置文件路徑')
    parser.add_argument('--epochs', type=int, default=10, help='分析的 epoch 數')
    
    args = parser.parse_args()
    
    run_detailed_profiling(args.config, args.epochs)
