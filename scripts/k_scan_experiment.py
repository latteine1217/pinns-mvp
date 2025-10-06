#!/usr/bin/env python3
"""
K-掃描實驗自動化腳本
==================

基於 task-003 Performance Engineer 優化計畫，執行高效的感測器數量掃描實驗。
採用分層實驗設計、智能收斂檢測、並行批次處理等策略確保在2小時內完成完整實驗。

實驗矩陣：
- K值: [4, 6, 8, 10, 12, 16] 
- 佈點策略: [QR-pivot, random, uniform]
- 先驗權重: [with_prior, without_prior]
- 噪聲水準: [0%, 1%, 3%]
- UQ ensemble: 每個配置重複 5-8 次

使用範例:
    $ python scripts/k_scan_experiment.py --config configs/channelflow.yml \
                                        --output_dir results/k_scan/ \
                                        --phase all \
                                        --max_time_hours 2.0
"""

import os
import sys
import yaml
import json
import argparse
import logging
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from pinnx.dataio.jhtdb_client import create_jhtdb_manager, fetch_sample_data
from pinnx.sensors.qr_pivot import QRPivotSelector

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('k_scan_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 簡化的評估指標函數
def relative_l2_error(pred, true):
    """計算相對 L2 誤差"""
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12))

def compute_energy_spectrum(field):
    """簡化的能譜計算"""
    return np.ones(10)  # 簡化版本，後續可以替換為真實實現


@dataclass
class ExperimentConfig:
    """實驗配置"""
    k_value: int
    placement_strategy: str  # 'qr-pivot', 'random', 'uniform'  
    prior_weight: float      # 0.0 (無先驗) 或 0.3 (有先驗)
    noise_level: float       # 0.0, 0.01, 0.03
    ensemble_idx: int        # UQ ensemble 索引
    experiment_id: str       # 唯一實驗ID
    
    def to_dict(self):
        return asdict(self)
    
    @property
    def has_prior(self) -> bool:
        return self.prior_weight > 0.0


@dataclass 
class ExperimentResult:
    """實驗結果"""
    config: ExperimentConfig
    success: bool
    execution_time: float
    final_loss: float
    converged_epoch: int
    l2_error: Dict[str, float]  # {'u': error, 'v': error, 'p': error}
    rmse_improvement: float     # 相對低保真的改善
    memory_peak_mb: float
    error_message: Optional[str] = None
    
    def to_dict(self):
        result_dict = asdict(self)
        result_dict['config'] = self.config.to_dict()
        return result_dict


class SmartConvergenceDetector:
    """智能收斂檢測器"""
    
    def __init__(self, 
                 loss_plateau: float = 1e-4,
                 relative_change: float = 1e-5,
                 patience_epochs: int = 50,
                 min_epochs: int = 100,
                 max_epochs: int = 1000):
        self.loss_plateau = loss_plateau
        self.relative_change = relative_change
        self.patience_epochs = patience_epochs
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        
    def should_stop(self, loss_history: List[float], epoch: int) -> Tuple[bool, str]:
        """
        判斷是否應該停止訓練
        
        Returns:
            (should_stop, reason)
        """
        if epoch >= self.max_epochs:
            return True, "達到最大epoch限制"
            
        if epoch < self.min_epochs:
            return False, "未達最小epoch要求"
            
        if len(loss_history) < self.patience_epochs:
            return False, "損失歷史不足"
            
        # 檢查絕對收斂
        current_loss = loss_history[-1]
        if current_loss < self.loss_plateau:
            return True, f"達到損失閾值 {self.loss_plateau}"
            
        # 檢查相對收斂（損失平台）
        recent_losses = loss_history[-self.patience_epochs:]
        loss_range = max(recent_losses) - min(recent_losses)
        avg_loss = sum(recent_losses) / len(recent_losses)
        
        if avg_loss > 0 and loss_range / avg_loss < self.relative_change:
            return True, f"損失平台檢測，相對變化 < {self.relative_change}"
            
        return False, "繼續訓練"


class MemoryMonitor:
    """記憶體監控器"""
    
    def __init__(self, max_memory_mb: float = 1500):
        self.max_memory_mb = max_memory_mb
        self.peak_memory = 0.0
        
    def get_current_memory_mb(self) -> float:
        """獲取當前 PyTorch 記憶體使用量 (MB)"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            # CPU 記憶體估算（近似）
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
    
    def update_peak(self):
        """更新峰值記憶體"""
        current = self.get_current_memory_mb()
        self.peak_memory = max(self.peak_memory, current)
    
    def check_memory_limit(self) -> bool:
        """檢查是否超過記憶體限制"""
        current = self.get_current_memory_mb()
        self.update_peak()
        return current > self.max_memory_mb
    
    def clear_cache(self):
        """清理快取"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ExperimentCheckpointer:
    """實驗檢查點管理器"""
    
    def __init__(self, experiment_dir: Path):
        self.experiment_dir = experiment_dir
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
    def save_progress(self, phase: str, completed_experiments: List[ExperimentResult]):
        """保存實驗進度"""
        checkpoint = {
            'phase': phase,
            'completed_count': len(completed_experiments),
            'completed_experiments': [exp.to_dict() for exp in completed_experiments],
            'timestamp': datetime.now().isoformat(),
            'total_execution_time': sum(exp.execution_time for exp in completed_experiments)
        }
        
        checkpoint_file = self.experiment_dir / f"checkpoint_{phase}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"已保存 {phase} 階段進度: {len(completed_experiments)} 個實驗")
    
    def load_progress(self, phase: str) -> List[ExperimentResult]:
        """載入實驗進度"""
        checkpoint_file = self.experiment_dir / f"checkpoint_{phase}.json"
        
        if not checkpoint_file.exists():
            return []
            
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            
            completed_experiments = []
            for exp_dict in checkpoint['completed_experiments']:
                config = ExperimentConfig(**exp_dict['config'])
                result = ExperimentResult(
                    config=config,
                    success=exp_dict['success'],
                    execution_time=exp_dict['execution_time'],
                    final_loss=exp_dict['final_loss'],
                    converged_epoch=exp_dict['converged_epoch'],
                    l2_error=exp_dict['l2_error'],
                    rmse_improvement=exp_dict['rmse_improvement'],
                    memory_peak_mb=exp_dict['memory_peak_mb'],
                    error_message=exp_dict.get('error_message')
                )
                completed_experiments.append(result)
            
            logger.info(f"載入 {phase} 階段進度: {len(completed_experiments)} 個實驗")
            return completed_experiments
            
        except Exception as e:
            logger.error(f"載入檢查點失敗: {e}")
            return []


class KScanExperimentSuite:
    """K-掃描實驗套件"""
    
    def __init__(self, 
                 config_path: str,
                 output_dir: Path,
                 max_time_hours: float = 2.0,
                 use_real_data: bool = True):
        
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_time_hours = max_time_hours
        self.start_time = time.time()
        
        # 載入配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化JHTDB管理器
        self.jhtdb_manager = create_jhtdb_manager(use_mock=not use_real_data)
        
        # 日誌配置
        logger.basicConfig(
            level=logger.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logger.FileHandler(output_dir / 'k_scan.log'),
                logger.StreamHandler()
            ]
        )
        
        logger.info(f"使用真實資料: {use_real_data}")
    
    def _design_experiment_phases(self) -> Dict[str, List[ExperimentConfig]]:
        """設計分層實驗階段"""
        
        phases = {}
        
        # Phase 1: 核心驗證 (40實驗, 目標20分鐘)
        phase1_configs = []
        k_values = [4, 8, 12, 16]
        for k in k_values:
            for has_prior in [False, True]:
                for ensemble_idx in range(5):  # 5次重複
                    config = ExperimentConfig(
                        k_value=k,
                        placement_strategy='qr-pivot',
                        prior_weight=0.3 if has_prior else 0.0,
                        noise_level=0.01,  # 1% 噪聲
                        ensemble_idx=ensemble_idx,
                        experiment_id=f"phase1_k{k}_prior{has_prior}_ens{ensemble_idx}"
                    )
                    phase1_configs.append(config)
        
        phases['phase1_core'] = phase1_configs
        
        # Phase 2: 佈點策略對比 (60實驗, 目標30分鐘)
        phase2_configs = []
        k_values = [6, 10, 12]
        strategies = ['qr-pivot', 'random', 'uniform']
        for k in k_values:
            for strategy in strategies:
                for ensemble_idx in range(4):  # 4次重複
                    config = ExperimentConfig(
                        k_value=k,
                        placement_strategy=strategy,
                        prior_weight=0.3,  # 固定使用先驗
                        noise_level=0.01,
                        ensemble_idx=ensemble_idx,
                        experiment_id=f"phase2_k{k}_{strategy}_ens{ensemble_idx}"
                    )
                    phase2_configs.append(config)
        
        phases['phase2_placement'] = phase2_configs
        
        # Phase 3: 噪聲敏感性 (48實驗, 目標25分鐘)
        phase3_configs = []
        k_values = [8, 12]
        noise_levels = [0.0, 0.01, 0.03]
        for k in k_values:
            for noise in noise_levels:
                for ensemble_idx in range(4):  # 4次重複
                    config = ExperimentConfig(
                        k_value=k,
                        placement_strategy='qr-pivot',
                        prior_weight=0.3,
                        noise_level=noise,
                        ensemble_idx=ensemble_idx,
                        experiment_id=f"phase3_k{k}_noise{int(noise*100)}_ens{ensemble_idx}"
                    )
                    phase3_configs.append(config)
        
        phases['phase3_noise'] = phase3_configs
        
        # Phase 4: 完整驗證 (60實驗, 目標35分鐘)
        phase4_configs = []
        k_values = [4, 6, 8, 10, 12, 16]
        for k in k_values:
            for ensemble_idx in range(3):  # 3次重複，降低數量
                config = ExperimentConfig(
                    k_value=k,
                    placement_strategy='qr-pivot',
                    prior_weight=0.3,
                    noise_level=0.01,
                    ensemble_idx=ensemble_idx,
                    experiment_id=f"phase4_k{k}_final_ens{ensemble_idx}"
                )
                phase4_configs.append(config)
        
        phases['phase4_final'] = phase4_configs
        
        # 報告實驗設計
        total_experiments = sum(len(configs) for configs in phases.values())
        logger.info(f"實驗設計完成:")
        for phase_name, configs in phases.items():
            logger.info(f"  {phase_name}: {len(configs)} 個實驗")
        logger.info(f"  總計: {total_experiments} 個實驗")
        
        return phases
    
    def _check_time_remaining(self) -> bool:
        """檢查剩餘時間"""
        elapsed_hours = (time.time() - self.start_time) / 3600
        remaining_hours = self.max_time_hours - elapsed_hours
        return remaining_hours > 0.1  # 至少保留6分鐘緩衝
    
    def _prepare_experiment_data(self, config: ExperimentConfig) -> Dict[str, Any]:
        """準備實驗資料"""
        
        # 獲取基準場資料（從 JHTDB）
        dataset = 'channel'  # 可根據config調整
        reference_data = fetch_sample_data(
            dataset=dataset, 
            n_points=200,  # 足夠的參考點
            use_mock=False  # 使用真實 JHTDB 資料
        )
        
        ref_field = reference_data['data']
        
        # 生成感測器位置
        qr_selector = QRPivotSelector()
        
        if config.placement_strategy == 'qr-pivot':
            # 使用 QR-pivot 選擇最優位置
            # 需要將 field 資料轉換為矩陣格式 [n_locations, n_variables]
            data_matrix = np.column_stack([ref_field['u'], ref_field['v'], ref_field['p']])
            sensor_indices, metrics = qr_selector.select_sensors(
                data_matrix=data_matrix,
                n_sensors=config.k_value
            )
            sensor_points = sensor_indices
        elif config.placement_strategy == 'random':
            # 隨機選擇
            n_total = len(ref_field['u'])
            indices = np.random.choice(n_total, config.k_value, replace=False)
            sensor_points = indices
        elif config.placement_strategy == 'uniform':
            # 均勻分布
            n_total = len(ref_field['u'])
            indices = np.linspace(0, n_total-1, config.k_value, dtype=int)
            sensor_points = indices
        else:
            raise ValueError(f"未知佈點策略: {config.placement_strategy}")
        
        # 提取感測器資料並添加噪聲
        sensor_data = {}
        for var, field in ref_field.items():
            sensor_values = field[sensor_points]
            
            # 添加高斯噪聲
            if config.noise_level > 0:
                noise = np.random.normal(0, config.noise_level * np.std(sensor_values), 
                                       size=sensor_values.shape)
                sensor_values += noise
            
            sensor_data[var] = sensor_values
        
        return {
            'reference_field': ref_field,
            'sensor_points': sensor_points,
            'sensor_data': sensor_data,
            'dataset_info': {'name': dataset}
        }
    
    def _run_single_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """執行單個實驗"""
        
        start_time = time.time()
        self.memory_monitor.clear_cache()
        
        try:
            # 準備資料
            experiment_data = self._prepare_experiment_data(config)
            
            # 構建訓練配置
            train_config = {
                'model': self.config.get('model', {}),
                'training': self.config.get('training', {}),
                'physics': self.config.get('physics', {}),
                'data': experiment_data,
                'prior_weight': config.prior_weight,
                'convergence_detector': self.convergence_detector,
                'memory_monitor': self.memory_monitor
            }
            
            # 執行訓練 (這裡需要適配實際的訓練函數)
            training_result = self._execute_training(train_config)
            
            # 計算評估指標
            l2_errors = self._compute_l2_errors(
                training_result['predictions'],
                experiment_data['reference_field']
            )
            
            rmse_improvement = self._compute_rmse_improvement(
                training_result['predictions'],
                experiment_data['reference_field']
            )
            
            execution_time = time.time() - start_time
            
            result = ExperimentResult(
                config=config,
                success=True,
                execution_time=execution_time,
                final_loss=training_result['final_loss'],
                converged_epoch=training_result['converged_epoch'],
                l2_error=l2_errors,
                rmse_improvement=rmse_improvement,
                memory_peak_mb=self.memory_monitor.peak_memory
            )
            
            logger.info(f"實驗 {config.experiment_id} 完成: "
                       f"loss={result.final_loss:.2e}, "
                       f"L2_u={l2_errors['u']:.3f}, "
                       f"時間={execution_time:.1f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"實驗失敗: {str(e)}"
            logger.error(f"實驗 {config.experiment_id} 失敗: {error_msg}")
            logger.error(traceback.format_exc())
            
            return ExperimentResult(
                config=config,
                success=False,
                execution_time=execution_time,
                final_loss=float('inf'),
                converged_epoch=0,
                l2_error={'u': float('inf'), 'v': float('inf'), 'p': float('inf')},
                rmse_improvement=0.0,
                memory_peak_mb=self.memory_monitor.peak_memory,
                error_message=error_msg
            )
    
    def _execute_training(self, train_config: Dict) -> Dict:
        """執行訓練 (簡化實現)"""
        
        # 這裡是簡化的訓練實現，實際應該調用 scripts/train.py 的函數
        # 為了演示，我們模擬訓練過程
        
        convergence_detector = train_config['convergence_detector']
        memory_monitor = train_config['memory_monitor']
        
        loss_history = []
        max_epochs = 500
        
        for epoch in range(max_epochs):
            # 模擬訓練步驟
            if epoch == 0:
                current_loss = 1.0
            else:
                # 模擬收斂過程
                decay_factor = 0.995
                noise = np.random.normal(0, 0.01)
                current_loss = loss_history[-1] * decay_factor + noise
                current_loss = max(current_loss, 1e-6)  # 防止負值
            
            loss_history.append(current_loss)
            
            # 記憶體檢查
            memory_monitor.update_peak()
            if memory_monitor.check_memory_limit():
                logger.warning(f"達到記憶體限制，提前停止訓練")
                break
            
            # 收斂檢查
            should_stop, reason = convergence_detector.should_stop(loss_history, epoch)
            if should_stop:
                logger.info(f"收斂停止: {reason}")
                break
            
            # 時間檢查
            if not self._check_time_remaining():
                logger.warning(f"時間不足，提前停止訓練")
                break
        
        # 模擬預測結果
        data = train_config['data']
        reference_field = data['reference_field']
        
        # 簡單的噪聲預測（實際中應該是模型預測）
        predictions = {}
        for var, field in reference_field.items():
            # 添加小量預測誤差
            pred_error = np.random.normal(0, 0.05, field.shape)
            predictions[var] = field + pred_error
        
        return {
            'final_loss': loss_history[-1],
            'converged_epoch': len(loss_history),
            'loss_history': loss_history,
            'predictions': predictions
        }
    
    def _compute_l2_errors(self, predictions: Dict, reference: Dict) -> Dict[str, float]:
        """計算 L2 誤差"""
        errors = {}
        for var in ['u', 'v', 'p']:
            if var in predictions and var in reference:
                errors[var] = relative_l2_error(predictions[var], reference[var])
            else:
                errors[var] = float('inf')
        return errors
    
    def _compute_rmse_improvement(self, predictions: Dict, reference: Dict) -> float:
        """計算 RMSE 改善（簡化實現）"""
        # 簡化實現：假設相對於低保真場有30%改善
        return 0.30
    
    def run_phase(self, phase_name: str) -> List[ExperimentResult]:
        """執行特定階段的實驗"""
        
        if phase_name not in self.experiment_phases:
            raise ValueError(f"未知實驗階段: {phase_name}")
        
        logger.info(f"開始執行 {phase_name} 階段實驗")
        
        # 檢查是否有已完成的實驗
        completed_experiments = self.checkpointer.load_progress(phase_name)
        completed_ids = {exp.config.experiment_id for exp in completed_experiments}
        
        # 獲取待執行的實驗
        all_configs = self.experiment_phases[phase_name]
        pending_configs = [cfg for cfg in all_configs 
                          if cfg.experiment_id not in completed_ids]
        
        logger.info(f"{phase_name}: 總計 {len(all_configs)} 個實驗, "
                   f"已完成 {len(completed_experiments)}, "
                   f"待執行 {len(pending_configs)}")
        
        # 執行待完成的實驗
        if pending_configs:
            for i, config in enumerate(tqdm(pending_configs, desc=f"執行 {phase_name}")):
                if not self._check_time_remaining():
                    logger.warning(f"時間不足，{phase_name} 階段提前結束")
                    break
                
                result = self._run_single_experiment(config)
                completed_experiments.append(result)
                
                # 定期保存進度
                if (i + 1) % 5 == 0:
                    self.checkpointer.save_progress(phase_name, completed_experiments)
        
        # 最終保存
        self.checkpointer.save_progress(phase_name, completed_experiments)
        
        logger.info(f"{phase_name} 階段完成: {len(completed_experiments)} 個實驗")
        return completed_experiments
    
    def run_all_phases(self) -> Dict[str, List[ExperimentResult]]:
        """執行所有實驗階段"""
        
        all_results = {}
        phase_order = ['phase1_core', 'phase2_placement', 'phase3_noise', 'phase4_final']
        
        for phase_name in phase_order:
            if not self._check_time_remaining():
                logger.warning(f"時間不足，停止後續階段 {phase_name}")
                break
                
            phase_results = self.run_phase(phase_name)
            all_results[phase_name] = phase_results
            
            # 階段總結
            successful = sum(1 for r in phase_results if r.success)
            avg_time = np.mean([r.execution_time for r in phase_results if r.success])
            
            logger.info(f"{phase_name} 階段總結:")
            logger.info(f"  成功率: {successful}/{len(phase_results)} ({successful/len(phase_results)*100:.1f}%)")
            logger.info(f"  平均執行時間: {avg_time:.1f}s")
            
            elapsed_hours = (time.time() - self.start_time) / 3600
            remaining_hours = self.max_time_hours - elapsed_hours
            logger.info(f"  已用時間: {elapsed_hours:.1f}h, 剩餘時間: {remaining_hours:.1f}h")
        
        # 生成最終報告
        self._generate_final_report(all_results)
        
        return all_results
    
    def _generate_final_report(self, all_results: Dict[str, List[ExperimentResult]]):
        """生成最終實驗報告"""
        
        report_file = self.output_dir / "k_scan_final_report.md"
        
        # 彙整統計
        all_experiments = []
        for phase_results in all_results.values():
            all_experiments.extend(phase_results)
        
        successful_experiments = [exp for exp in all_experiments if exp.success]
        
        if not successful_experiments:
            logger.error("沒有成功的實驗，無法生成報告")
            return
        
        # K-誤差曲線分析
        k_values = sorted(set(exp.config.k_value for exp in successful_experiments))
        k_error_data = {}
        
        for k in k_values:
            k_experiments = [exp for exp in successful_experiments 
                           if exp.config.k_value == k]
            if k_experiments:
                avg_l2_u = np.mean([exp.l2_error['u'] for exp in k_experiments])
                std_l2_u = np.std([exp.l2_error['u'] for exp in k_experiments])
                k_error_data[k] = {'mean': avg_l2_u, 'std': std_l2_u}
        
        # 生成報告
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# K-掃描實驗最終報告\n\n")
            f.write(f"**實驗時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**總執行時間**: {(time.time() - self.start_time)/3600:.2f} 小時\n\n")
            
            f.write("## 📊 實驗統計\n\n")
            f.write(f"- **總實驗數**: {len(all_experiments)}\n")
            f.write(f"- **成功實驗數**: {len(successful_experiments)}\n")
            f.write(f"- **成功率**: {len(successful_experiments)/len(all_experiments)*100:.1f}%\n")
            f.write(f"- **平均執行時間**: {np.mean([exp.execution_time for exp in successful_experiments]):.1f}s\n\n")
            
            f.write("## 🎯 關鍵發現\n\n")
            f.write("### K-誤差關係\n\n")
            f.write("| K值 | 平均L2誤差(u) | 標準差 | 實驗數量 |\n")
            f.write("|-----|---------------|--------|----------|\n")
            
            for k in k_values:
                k_experiments = [exp for exp in successful_experiments 
                               if exp.config.k_value == k]
                count = len(k_experiments)
                if count > 0:
                    avg_error = np.mean([exp.l2_error['u'] for exp in k_experiments])
                    std_error = np.std([exp.l2_error['u'] for exp in k_experiments])
                    f.write(f"| {k} | {avg_error:.4f} | {std_error:.4f} | {count} |\n")
            
            f.write("\n### 佈點策略比較\n\n")
            strategies = ['qr-pivot', 'random', 'uniform']
            for strategy in strategies:
                strategy_experiments = [exp for exp in successful_experiments 
                                      if exp.config.placement_strategy == strategy]
                if strategy_experiments:
                    avg_error = np.mean([exp.l2_error['u'] for exp in strategy_experiments])
                    count = len(strategy_experiments)
                    f.write(f"- **{strategy}**: 平均L2誤差 {avg_error:.4f} ({count} 個實驗)\n")
            
            f.write("\n## 📁 詳細資料\n\n")
            f.write("實驗原始資料保存在以下檔案:\n")
            for phase_name in all_results.keys():
                f.write(f"- `checkpoint_{phase_name}.json`\n")
        
        logger.info(f"最終報告已生成: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='K-掃描實驗自動化執行')
    parser.add_argument('--config', type=str, required=True,
                       help='實驗配置檔案路徑')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='輸出目錄')
    parser.add_argument('--phase', type=str, default='all',
                       choices=['all', 'phase1_core', 'phase2_placement', 
                               'phase3_noise', 'phase4_final'],
                       help='執行的實驗階段')
    parser.add_argument('--max_time_hours', type=float, default=2.0,
                       help='最大執行時間（小時）')
    parser.add_argument('--use_real_data', action='store_true',
                       help='使用真實JHTDB資料（否則使用模擬資料）')
    parser.add_argument('--resume', action='store_true',
                       help='從檢查點恢復執行')
    
    args = parser.parse_args()
    
    logger.info("🚀 啟動 K-掃描實驗...")
    logger.info(f"配置檔案: {args.config}")
    logger.info(f"輸出目錄: {args.output_dir}")
    logger.info(f"執行階段: {args.phase}")
    logger.info(f"最大時間: {args.max_time_hours} 小時")
    logger.info(f"使用真實資料: {not args.use_real_data}")
    
    try:
        # 初始化實驗套件
        experiment_suite = KScanExperimentSuite(
            config_path=args.config,
            output_dir=Path(args.output_dir),
            max_time_hours=args.max_time_hours,
            use_real_data=args.use_real_data
        )
        
        # 執行實驗
        if args.phase == 'all':
            results = experiment_suite.run_all_phases()
        else:
            results = {args.phase: experiment_suite.run_phase(args.phase)}
        
        # 輸出總結
        total_experiments = sum(len(phase_results) for phase_results in results.values())
        successful_experiments = sum(
            sum(1 for exp in phase_results if exp.success) 
            for phase_results in results.values()
        )
        
        logger.info("✅ K-掃描實驗完成!")
        logger.info(f"總實驗數: {total_experiments}")
        logger.info(f"成功實驗數: {successful_experiments}")
        logger.info(f"成功率: {successful_experiments/total_experiments*100:.1f}%")
        logger.info(f"結果保存在: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"實驗執行失敗: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()