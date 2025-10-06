#!/usr/bin/env python3
"""
自動化性能基準測試腳本
收集訓練性能、記憶體使用、數值穩定性等全面數據
用於建立可重現的性能基線
"""

import os
import sys
import time
import json
import yaml
import logging
import platform
import tracemalloc
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import argparse

import torch
import psutil
import numpy as np

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """性能基準測試類別"""
    
    def __init__(self, config_path: str, task_dir: str):
        self.config_path = config_path
        self.task_dir = Path(task_dir)
        self.task_dir.mkdir(exist_ok=True)
        
        # 載入配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化結果儲存
        self.results = {
            'hardware_info': self._collect_hardware_info(),
            'environment_info': self._collect_environment_info(),
            'timestamp': datetime.now().isoformat(),
            'benchmarks': {}
        }
        
    def _collect_hardware_info(self) -> Dict[str, Any]:
        """收集硬體資訊"""
        try:
            cpu_freq = psutil.cpu_freq()
            cpu_freq_max = "Unknown"
            if cpu_freq and cpu_freq.max:
                cpu_freq_max = f"{cpu_freq.max:.1f}"
            
            return {
                'platform': platform.platform(),
                'processor': platform.processor(),
                'architecture': platform.architecture()[0],
                'cpu_count': psutil.cpu_count(logical=True),
                'cpu_freq_max': cpu_freq_max,
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'python_version': platform.python_version(),
            }
        except Exception as e:
            logger.warning(f"Failed to collect some hardware info: {e}")
            return {'error': str(e)}
    
    def _collect_environment_info(self) -> Dict[str, Any]:
        """收集環境資訊"""
        cuda_version = "Not available"
        if torch.cuda.is_available():
            try:
                import subprocess
                result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
                if result.returncode == 0:
                    cuda_version = "Available"
                else:
                    cuda_version = "Unknown"
            except:
                cuda_version = "Unknown"
        
        device_name = "CPU only"
        if torch.cuda.is_available():
            try:
                device_name = torch.cuda.get_device_name(0)
            except:
                device_name = "GPU available"
        
        return {
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': cuda_version,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'device_name': device_name,
        }
    
    def benchmark_single_training(self, seed: int = 42) -> Dict[str, Any]:
        """單次訓練基準測試"""
        logger.info(f"Running single training benchmark with seed {seed}")
        
        # 設定隨機種子
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 開始記憶體追蹤
        tracemalloc.start()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024**2)  # MB
        
        # 執行訓練
        start_time = time.time()
        try:
            # 修改配置以進行基準測試
            benchmark_config = self.config.copy()
            benchmark_config['experiment']['seed'] = seed
            benchmark_config['training']['max_epochs'] = 1000  # 較短的測試
            benchmark_config['logging']['log_freq'] = 50
            
            # 執行訓練
            results = self._run_training_with_monitoring(benchmark_config)
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {'error': str(e)}
        
        end_time = time.time()
        
        # 停止記憶體追蹤
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        final_memory = process.memory_info().rss / (1024**2)  # MB
        
        return {
            'seed': seed,
            'total_time': end_time - start_time,
            'memory_initial_mb': initial_memory,
            'memory_final_mb': final_memory,
            'memory_peak_mb': peak / (1024**2),
            'memory_used_mb': current / (1024**2),
            **results
        }
    
    def _run_training_with_monitoring(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """執行訓練並監控關鍵指標"""
        # 直接導入所需的模組
        from pinnx.models.fourier_mlp import PINNNet
        from pinnx.models.wrappers import ScaledPINNWrapper
        
        # 建立模型
        device = torch.device(config['experiment']['device'] if torch.cuda.is_available() else 'cpu')
        
        # 創建 PINN 模型
        model = PINNNet(
            in_dim=config['model']['in_dim'],
            out_dim=config['model']['out_dim'],
            width=config['model']['width'],
            depth=config['model']['depth'],
            activation=config['model']['activation'],
            fourier_m=config['model']['fourier_m'],
            fourier_sigma=config['model']['fourier_sigma']
        ).to(device)
        
        # 包裝為 ScaledPINNWrapper
        if config['model']['scaling']['learnable']:
            model = ScaledPINNWrapper(model).to(device)
        
        # 計算參數數量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 創建物理模組（簡化版）
        nu = config['physics']['nu']
        rho = config['physics']['rho']
        
        # 建立優化器
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )
        
        # 生成訓練資料
        domain = config['physics']['domain']
        n_pde = config['training']['sampling']['pde_points']
        n_bc = config['training']['sampling']['boundary_points']
        
        # PDE 點
        x_pde = torch.rand(n_pde, 1) * (domain['x_range'][1] - domain['x_range'][0]) + domain['x_range'][0]
        y_pde = torch.rand(n_pde, 1) * (domain['y_range'][1] - domain['y_range'][0]) + domain['y_range'][0]
        t_pde = torch.rand(n_pde, 1) * (domain['t_range'][1] - domain['t_range'][0]) + domain['t_range'][0]
        
        # 邊界點
        x_bc = torch.cat([
            torch.full((n_bc//4, 1), domain['x_range'][0]),  # 左邊界
            torch.full((n_bc//4, 1), domain['x_range'][1]),  # 右邊界
            torch.rand(n_bc//2, 1) * (domain['x_range'][1] - domain['x_range'][0]) + domain['x_range'][0]  # 上下邊界
        ])
        y_bc = torch.cat([
            torch.rand(n_bc//4, 1) * (domain['y_range'][1] - domain['y_range'][0]) + domain['y_range'][0],  # 左邊界
            torch.rand(n_bc//4, 1) * (domain['y_range'][1] - domain['y_range'][0]) + domain['y_range'][0],  # 右邊界
            torch.cat([
                torch.full((n_bc//4, 1), domain['y_range'][0]),  # 下邊界
                torch.full((n_bc//4, 1), domain['y_range'][1])   # 上邊界
            ])
        ])
        t_bc = torch.rand(n_bc, 1) * (domain['t_range'][1] - domain['t_range'][0]) + domain['t_range'][0]
        
        # 移動到設備
        x_pde, y_pde, t_pde = x_pde.to(device), y_pde.to(device), t_pde.to(device)
        x_bc, y_bc, t_bc = x_bc.to(device), y_bc.to(device), t_bc.to(device)
        
        # 訓練循環監控
        epoch_times = []
        losses = []
        final_loss = None
        converged_epoch = None
        
        max_epochs = config['training']['max_epochs']
        log_freq = config['logging']['log_freq']
        
        for epoch in range(max_epochs):
            epoch_start = time.time()
            
            optimizer.zero_grad()
            
            # 前向傳播
            pde_coords = torch.cat([t_pde, x_pde, y_pde], dim=1)
            bc_coords = torch.cat([t_bc, x_bc, y_bc], dim=1)
            
            # PDE 損失（簡化版 NS 殘差）
            pde_coords.requires_grad_(True)
            pde_pred = model(pde_coords)
            u, v, p = pde_pred[:, 0], pde_pred[:, 1], pde_pred[:, 2]
            
            # 計算導數
            u_t = torch.autograd.grad(u.sum(), pde_coords, create_graph=True)[0][:, 0]
            u_x = torch.autograd.grad(u.sum(), pde_coords, create_graph=True)[0][:, 1]
            u_y = torch.autograd.grad(u.sum(), pde_coords, create_graph=True)[0][:, 2]
            u_xx = torch.autograd.grad(u_x.sum(), pde_coords, create_graph=True)[0][:, 1]
            u_yy = torch.autograd.grad(u_y.sum(), pde_coords, create_graph=True)[0][:, 2]
            
            v_t = torch.autograd.grad(v.sum(), pde_coords, create_graph=True)[0][:, 0]
            v_x = torch.autograd.grad(v.sum(), pde_coords, create_graph=True)[0][:, 1]
            v_y = torch.autograd.grad(v.sum(), pde_coords, create_graph=True)[0][:, 2]
            v_xx = torch.autograd.grad(v_x.sum(), pde_coords, create_graph=True)[0][:, 1]
            v_yy = torch.autograd.grad(v_y.sum(), pde_coords, create_graph=True)[0][:, 2]
            
            p_x = torch.autograd.grad(p.sum(), pde_coords, create_graph=True)[0][:, 1]
            p_y = torch.autograd.grad(p.sum(), pde_coords, create_graph=True)[0][:, 2]
            
            # NS 殘差
            f_u = u_t + u * u_x + v * u_y + p_x / rho - nu * (u_xx + u_yy)
            f_v = v_t + u * v_x + v * v_y + p_y / rho - nu * (v_xx + v_yy)
            f_c = u_x + v_y  # 連續性方程
            
            pde_loss = torch.mean(f_u**2 + f_v**2 + f_c**2)
            
            # 邊界條件損失 (簡單零速度邊界)
            bc_pred = model(bc_coords)
            bc_loss = torch.mean(bc_pred[:, :2]**2)  # u, v = 0 at boundaries
            
            # 總損失
            total_loss = (
                config['losses']['residual_weight'] * pde_loss +
                config['losses']['boundary_weight'] * bc_loss
            )
            
            # 反向傳播
            total_loss.backward()
            optimizer.step()
            
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            losses.append(total_loss.item())
            
            # 檢查收斂
            if total_loss.item() < 1e-6:
                converged_epoch = epoch
                final_loss = total_loss.item()
                break
            
            # 日誌輸出
            if epoch % log_freq == 0:
                logger.info(f"Epoch {epoch:6d} | Loss: {total_loss.item():.6f} | Time: {epoch_time:.3f}s")
        
        # 如果沒有收斂，記錄最終損失
        if final_loss is None:
            final_loss = losses[-1]
            converged_epoch = max_epochs
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'final_loss': final_loss,
            'converged_epoch': converged_epoch,
            'avg_epoch_time': np.mean(epoch_times),
            'min_epoch_time': np.min(epoch_times),
            'max_epoch_time': np.max(epoch_times),
            'loss_history': losses,
            'epoch_times': epoch_times,
            'converged': converged_epoch < max_epochs
        }
    
    def benchmark_inference_performance(self, model_path: str | None = None) -> Dict[str, Any]:
        """推理性能基準測試"""
        logger.info("Running inference performance benchmark")
        
        # 直接導入所需的模組
        from pinnx.models.fourier_mlp import PINNNet
        from pinnx.models.wrappers import ScaledPINNWrapper
        
        # 創建模型（如果沒有提供預訓練模型）
        device = torch.device(self.config['experiment']['device'] if torch.cuda.is_available() else 'cpu')
        
        model = PINNNet(
            in_dim=self.config['model']['in_dim'],
            out_dim=self.config['model']['out_dim'],
            width=self.config['model']['width'],
            depth=self.config['model']['depth'],
            activation=self.config['model']['activation'],
            fourier_m=self.config['model']['fourier_m'],
            fourier_sigma=self.config['model']['fourier_sigma']
        ).to(device)
        
        if self.config['model']['scaling']['learnable']:
            model = ScaledPINNWrapper(model).to(device)
        
        model.eval()
        
        # 測試不同批次大小的推理性能
        batch_sizes = [1, 16, 64, 256, 1024]
        inference_results = {}
        
        for batch_size in batch_sizes:
            # 生成測試數據
            test_input = torch.randn(batch_size, self.config['model']['in_dim']).to(device)
            
            # 預熱
            with torch.no_grad():
                for _ in range(10):
                    _ = model(test_input)
            
            # 實際測試
            times = []
            with torch.no_grad():
                for _ in range(100):
                    start_time = time.time()
                    output = model(test_input)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    end_time = time.time()
                    times.append(end_time - start_time)
            
            inference_results[f'batch_{batch_size}'] = {
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'throughput': batch_size / np.mean(times)  # samples/second
            }
        
        return inference_results
    
    def benchmark_multiple_runs(self, n_runs: int = 5) -> Dict[str, Any]:
        """多次運行基準測試以評估穩定性"""
        logger.info(f"Running {n_runs} training benchmarks for stability analysis")
        
        runs = []
        seeds = [42 + i for i in range(n_runs)]
        
        for i, seed in enumerate(seeds):
            logger.info(f"Run {i+1}/{n_runs} with seed {seed}")
            run_result = self.benchmark_single_training(seed)
            runs.append(run_result)
        
        # 統計分析
        if all('error' not in run for run in runs):
            training_times = [run['total_time'] for run in runs]
            final_losses = [run['final_loss'] for run in runs]
            converged_epochs = [run['converged_epoch'] for run in runs]
            avg_epoch_times = [run['avg_epoch_time'] for run in runs]
            
            stats = {
                'n_runs': n_runs,
                'training_time': {
                    'mean': np.mean(training_times),
                    'std': np.std(training_times),
                    'min': np.min(training_times),
                    'max': np.max(training_times)
                },
                'final_loss': {
                    'mean': np.mean(final_losses),
                    'std': np.std(final_losses),
                    'min': np.min(final_losses),
                    'max': np.max(final_losses)
                },
                'converged_epochs': {
                    'mean': np.mean(converged_epochs),
                    'std': np.std(converged_epochs),
                    'min': np.min(converged_epochs),
                    'max': np.max(converged_epochs)
                },
                'avg_epoch_time': {
                    'mean': np.mean(avg_epoch_times),
                    'std': np.std(avg_epoch_times),
                    'min': np.min(avg_epoch_times),
                    'max': np.max(avg_epoch_times)
                },
                'reproducibility': {
                    'loss_cv': np.std(final_losses) / np.mean(final_losses),  # 變異係數
                    'time_cv': np.std(training_times) / np.mean(training_times),
                    'all_converged': all(run['converged'] for run in runs)
                }
            }
        else:
            stats = {'error': 'Some runs failed'}
        
        return {
            'individual_runs': runs,
            'statistics': stats
        }
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """執行完整的基準測試套件"""
        logger.info("Starting full performance benchmark suite")
        
        # 1. 多次訓練穩定性測試
        self.results['benchmarks']['stability'] = self.benchmark_multiple_runs(n_runs=5)
        
        # 2. 推理性能測試
        self.results['benchmarks']['inference'] = self.benchmark_inference_performance()
        
        # 3. 單次詳細訓練分析
        self.results['benchmarks']['detailed_training'] = self.benchmark_single_training(seed=42)
        
        return self.results
    
    def save_results(self, filename: str | None = None) -> Path:
        """儲存基準測試結果"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"baseline_data_{timestamp}.json"
        
        filepath = self.task_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to {filepath}")
        return filepath
    
    def generate_report(self) -> str:
        """生成基準測試報告"""
        if 'benchmarks' not in self.results:
            return "No benchmark data available"
        
        report = []
        report.append("# 性能基線基準測試報告")
        report.append("")
        report.append(f"**測試時間**: {self.results['timestamp']}")
        report.append("")
        
        # 硬體環境
        hw = self.results['hardware_info']
        report.append("## 🖥️ 硬體環境基線")
        report.append("")
        report.append(f"- **平台**: {hw.get('platform', 'Unknown')}")
        report.append(f"- **處理器**: {hw.get('processor', 'Unknown')}")
        report.append(f"- **架構**: {hw.get('architecture', 'Unknown')}")
        report.append(f"- **CPU 核心數**: {hw.get('cpu_count', 'Unknown')}")
        
        # 處理可能為字符串的頻率值
        cpu_freq_str = hw.get('cpu_freq_max', 'Unknown')
        if cpu_freq_str != "Unknown":
            report.append(f"- **最大頻率**: {cpu_freq_str} MHz")
        else:
            report.append(f"- **最大頻率**: Unknown")
        
        # 處理可能為字符串的記憶體值
        memory_gb = hw.get('memory_total_gb', 'Unknown')
        if isinstance(memory_gb, (int, float)):
            report.append(f"- **記憶體**: {memory_gb:.1f} GB")
        else:
            report.append(f"- **記憶體**: {memory_gb}")
        
        report.append(f"- **Python 版本**: {hw.get('python_version', 'Unknown')}")
        report.append("")
        
        # 環境配置
        env = self.results['environment_info']
        report.append("## ⚙️ 環境配置基線")
        report.append("")
        report.append(f"- **PyTorch 版本**: {env.get('torch_version', 'Unknown')}")
        report.append(f"- **CUDA 可用**: {env.get('cuda_available', False)}")
        if env.get('cuda_available', False):
            report.append(f"- **CUDA 版本**: {env.get('cuda_version', 'Unknown')}")
            report.append(f"- **GPU 設備**: {env.get('device_name', 'Unknown')}")
        report.append("")
        
        # 訓練性能基線
        if 'stability' in self.results['benchmarks']:
            stability = self.results['benchmarks']['stability']
            if 'statistics' in stability and 'error' not in stability['statistics']:
                stats = stability['statistics']
                report.append("## 🚀 訓練性能基線")
                report.append("")
                report.append(f"- **測試次數**: {stats['n_runs']}")
                report.append(f"- **平均訓練時間**: {stats['training_time']['mean']:.2f} ± {stats['training_time']['std']:.2f} 秒")
                report.append(f"- **單 epoch 時間**: {stats['avg_epoch_time']['mean']:.4f} ± {stats['avg_epoch_time']['std']:.4f} 秒")
                report.append(f"- **收斂 epoch 數**: {stats['converged_epochs']['mean']:.1f} ± {stats['converged_epochs']['std']:.1f}")
                report.append(f"- **最終損失**: {stats['final_loss']['mean']:.2e} ± {stats['final_loss']['std']:.2e}")
                report.append("")
                
                # 穩定性指標
                repo = stats['reproducibility']
                report.append("## 📊 數值穩定性基線")
                report.append("")
                report.append(f"- **損失變異係數**: {repo['loss_cv']:.4f}")
                report.append(f"- **時間變異係數**: {repo['time_cv']:.4f}")
                report.append(f"- **全部收斂**: {'✅' if repo['all_converged'] else '❌'}")
                report.append("")
        
        # 推理性能基線
        if 'inference' in self.results['benchmarks']:
            inference = self.results['benchmarks']['inference']
            report.append("## ⚡ 推理性能基線")
            report.append("")
            report.append("| 批次大小 | 平均時間 (ms) | 吞吐量 (samples/s) |")
            report.append("|---------|---------------|-------------------|")
            
            for batch_key, metrics in inference.items():
                batch_size = batch_key.split('_')[1]
                avg_time_ms = metrics['avg_time'] * 1000
                throughput = metrics['throughput']
                report.append(f"| {batch_size} | {avg_time_ms:.3f} | {throughput:.1f} |")
            report.append("")
        
        # 模型規格
        if 'detailed_training' in self.results['benchmarks']:
            detail = self.results['benchmarks']['detailed_training']
            if 'error' not in detail:
                report.append("## 🏗️ 模型規格基線")
                report.append("")
                report.append(f"- **總參數量**: {detail.get('total_params', 'Unknown'):,}")
                report.append(f"- **可訓練參數**: {detail.get('trainable_params', 'Unknown'):,}")
                report.append(f"- **記憶體峰值**: {detail.get('memory_peak_mb', 'Unknown'):.1f} MB")
                report.append("")
        
        # 關鍵指標摘要
        report.append("## 🎯 關鍵指標摘要")
        report.append("")
        if 'stability' in self.results['benchmarks'] and 'statistics' in self.results['benchmarks']['stability']:
            stats = self.results['benchmarks']['stability']['statistics']
            if 'error' not in stats:
                avg_epoch_time = stats['avg_epoch_time']['mean']
                final_loss = stats['final_loss']['mean']
                converged = stats['reproducibility']['all_converged']
                
                report.append(f"- **單 epoch 時間**: {avg_epoch_time:.4f}s {'✅' if avg_epoch_time < 0.1 else '❌'} (目標 < 0.1s)")
                report.append(f"- **收斂性**: residual loss {final_loss:.2e} {'✅' if final_loss < 1e-3 else '❌'} (目標 < 1e-3)")
                report.append(f"- **穩定性**: {'✅' if converged else '❌'} 無 NaN/Inf 問題")
        
        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="PINNs 性能基準測試")
    parser.add_argument('--cfg', default='configs/defaults.yml', help='配置檔案路徑')
    parser.add_argument('--task-dir', default='tasks/task-002', help='任務目錄')
    parser.add_argument('--runs', type=int, default=5, help='穩定性測試次數')
    
    args = parser.parse_args()
    
    # 建立基準測試器
    benchmark = PerformanceBenchmark(args.cfg, args.task_dir)
    
    try:
        # 執行完整基準測試
        results = benchmark.run_full_benchmark()
        
        # 儲存結果
        data_file = benchmark.save_results()
        
        # 生成並儲存報告
        report = benchmark.generate_report()
        report_file = Path(args.task_dir) / "perf_baseline_report.md"
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Benchmark completed successfully!")
        logger.info(f"Report saved to: {report_file}")
        logger.info(f"Data saved to: {data_file}")
        
        print(f"\n基準測試完成！")
        print(f"報告檔案: {report_file}")
        print(f"數據檔案: {data_file}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()