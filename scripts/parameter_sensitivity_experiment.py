#!/usr/bin/env python3
"""
參數敏感度實驗腳本 - 自適應殘差點採樣系統性評估

功能：
1. 網格搜索關鍵參數（keep_ratio, initial_weight, epoch_interval）
2. 自動生成配置變體並批次訓練
3. 收集並比較訓練指標（收斂速度、最終誤差、重採樣效率）
4. 可視化參數影響（熱圖、並行座標圖、收斂曲線疊加）

用法：
    # 完整參數掃描（警告：計算量大）
    python scripts/parameter_sensitivity_experiment.py \
        --base-config configs/inverse_reconstruction_main.yml \
        --output results/sensitivity_analysis/

    # 快速測試（小參數空間）
    python scripts/parameter_sensitivity_experiment.py \
        --base-config configs/inverse_reconstruction_main.yml \
        --mode quick \
        --output results/sensitivity_quick/
"""

import argparse
import copy
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from itertools import product
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 設置繪圖樣式
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


class SensitivityExperiment:
    """參數敏感度實驗管理器"""
    
    def __init__(self,
                 base_config_path: str,
                 output_dir: str,
                 mode: str = "full"):
        """
        Args:
            base_config_path: 基礎配置文件路徑
            output_dir: 輸出目錄
            mode: 實驗模式 - "full"（完整）、"quick"（快速測試）、"custom"（自定義）
        """
        self.base_config_path = Path(base_config_path)
        self.output_dir = Path(output_dir)
        self.mode = mode
        
        # 創建輸出目錄結構
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "configs").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        # 載入基礎配置
        with open(self.base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        # 定義參數空間
        self.param_space = self._define_parameter_space()
        
        # 實驗結果收集器
        self.results: List[Dict[str, Any]] = []
        
        logger.info("=" * 80)
        logger.info(f"📊 參數敏感度實驗初始化")
        logger.info(f"   基礎配置: {self.base_config_path}")
        logger.info(f"   輸出目錄: {self.output_dir}")
        logger.info(f"   實驗模式: {mode}")
        logger.info("=" * 80)
    
    def _define_parameter_space(self) -> Dict[str, List]:
        """定義參數搜索空間"""
        if self.mode == "full":
            # 完整參數空間（計算密集）
            param_space = {
                "keep_ratio": [0.5, 0.6, 0.7, 0.8],
                "initial_weight": [1.5, 2.0, 2.5, 3.0],
                "epoch_interval": [500, 1000, 1500, 2000],
                "energy_threshold": [0.95, 0.99],  # SVD 能量閾值
            }
        elif self.mode == "quick":
            # 快速測試（小參數空間）
            param_space = {
                "keep_ratio": [0.6, 0.7],
                "initial_weight": [2.0, 2.5],
                "epoch_interval": [1000, 1500],
                "energy_threshold": [0.99],
            }
        else:
            raise ValueError(f"未知模式: {self.mode}")
        
        # 計算總實驗數
        total_runs = np.prod([len(v) for v in param_space.values()])
        logger.info(f"   參數空間: {param_space}")
        logger.info(f"   總實驗數: {total_runs}")
        
        return param_space
    
    def generate_config_variants(self) -> List[Tuple[Dict, str, Dict]]:
        """
        生成所有配置變體
        
        Returns:
            [(config_dict, config_name, params_dict), ...]
        """
        variants = []
        
        # 笛卡爾積生成所有參數組合
        param_names = list(self.param_space.keys())
        param_values = list(self.param_space.values())
        
        for idx, combination in enumerate(product(*param_values)):
            # 創建參數字典
            params = dict(zip(param_names, combination))
            
            # 複製基礎配置
            config = copy.deepcopy(self.base_config)
            
            # 修改配置中的參數
            self._update_config_with_params(config, params)
            
            # 生成配置名稱
            config_name = self._generate_config_name(params, idx)
            
            variants.append((config, config_name, params))
        
        logger.info(f"✅ 生成 {len(variants)} 個配置變體")
        return variants
    
    def _update_config_with_params(self, config: Dict, params: Dict) -> None:
        """更新配置中的參數"""
        # 兼容性修正：確保 fourier 參數格式正確（train.py 期望扁平格式）
        if 'fourier' in config['model'] and isinstance(config['model']['fourier'], dict):
            fourier_cfg = config['model']['fourier']
            if 'm' in fourier_cfg:
                config['model']['fourier_m'] = fourier_cfg['m']
            if 'sigma' in fourier_cfg:
                config['model']['fourier_sigma'] = fourier_cfg['sigma']
            if 'enabled' in fourier_cfg:
                config['model']['use_fourier'] = fourier_cfg['enabled']
        
        # 兼容性修正：將 output_norm 字符串格式轉為字典格式（如果需要）
        if 'output_norm' in config['model'] and isinstance(config['model']['output_norm'], str):
            # 使用 output_scales 作為輸出範圍（如果存在）
            if 'output_scales' in config['model']:
                scales = config['model']['output_scales']
                config['model']['output_norm'] = {
                    'u': [0.0, scales.get('u', 1.0) * 20.0],  # 假設最大為 20 * scale
                    'v': [-scales.get('v', 0.1) * 5.0, scales.get('v', 0.1) * 5.0],
                    'p': [-scales.get('p', 1.0) * 100.0, scales.get('p', 1.0) * 10.0]
                }
            else:
                # 使用默認值
                config['model']['output_norm'] = {
                    'u': [0.0, 20.0],
                    'v': [-0.5, 0.5],
                    'p': [-100.0, 10.0]
                }
        
        # 自適應採樣參數
        adaptive = config["training"]["sampling"]["adaptive_collocation"]
        
        # 更新 keep_ratio
        adaptive["incremental_replace"]["keep_ratio"] = params["keep_ratio"]
        adaptive["incremental_replace"]["replace_ratio"] = 1.0 - params["keep_ratio"]
        
        # 更新重新計算的 k_new（基於 keep_ratio）
        pde_points: int = config["training"]["sampling"]["pde_points"]  # type: ignore
        k_new = int(pde_points * (1.0 - params["keep_ratio"]))
        adaptive["incremental_replace"]["k_new"] = k_new  # type: ignore
        
        # 更新 epoch_interval
        adaptive["trigger"]["epoch_interval"] = params["epoch_interval"]
        
        # 更新新點權重
        adaptive["new_point_weighting"]["initial_weight_multiplier"] = params["initial_weight"]
        
        # 更新 SVD 能量閾值
        adaptive["residual_qr"]["svd"]["energy_threshold"] = params["energy_threshold"]
        
        # 調整訓練 epoch 數（快速模式）
        if self.mode == "quick":
            config["training"]["max_epochs"] = 5000
            config["training"]["validation_freq"] = 200
            config["training"]["checkpoint_freq"] = 1000
            
            # 🔥 重要：調整 early stopping patience，確保自適應採樣有機會觸發
            # 自適應採樣最早在 epoch_interval (1000 或 1500) 時觸發
            # 所以 patience 需要 > epoch_interval 才有意義
            if "early_stopping" in config["training"]:
                current_epoch_interval = params["epoch_interval"]
                min_patience = current_epoch_interval + 500  # 至少等待一次重採樣後 500 epochs
                config["training"]["early_stopping"]["patience"] = max(
                    min_patience,
                    config["training"]["early_stopping"].get("patience", 400)
                )
                logger.info(f"🔧 Adjusted early_stopping patience to {config['training']['early_stopping']['patience']} (epoch_interval={current_epoch_interval})")
        
        
        # 兼容性修正：將多階段優化器配置轉換為 train.py 期望的扁平格式
        if "optimizer_stage1" in config["training"]:
            stage1 = config["training"]["optimizer_stage1"]
            config["training"]["lr"] = stage1.get("lr", 1e-3)
            config["training"]["weight_decay"] = stage1.get("weight_decay", 1e-6)
            config["training"]["optimizer"] = stage1.get("type", "adam")
        
        # 確保必要的訓練參數存在
        if "lr_scheduler" not in config["training"]:
            config["training"]["lr_scheduler"] = "cosine"
        if "warmup_epochs" not in config["training"]:
            config["training"]["warmup_epochs"] = 0
    
    def _generate_config_name(self, params: Dict, idx: int) -> str:
        """生成配置名稱"""
        name_parts = [f"run{idx:03d}"]
        
        for key, value in params.items():
            if key == "keep_ratio":
                name_parts.append(f"kr{value:.1f}".replace('.', 'p'))
            elif key == "initial_weight":
                name_parts.append(f"iw{value:.1f}".replace('.', 'p'))
            elif key == "epoch_interval":
                name_parts.append(f"ei{value}")
            elif key == "energy_threshold":
                name_parts.append(f"et{value:.2f}".replace('.', 'p'))
        
        return "_".join(name_parts)
    
    def save_config(self, config: Dict, config_name: str) -> Path:
        """保存配置文件（並設定輸出路徑）"""
        # 設定實驗輸出路徑
        exp_output = self.output_dir / "results" / config_name
        config["training"]["checkpoint_dir"] = str(exp_output)
        
        config_path = self.output_dir / "configs" / f"{config_name}.yml"
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        return config_path
    
    def run_single_experiment(self, 
                              config_path: Path,
                              config_name: str,
                              params: Dict) -> Dict[str, Any]:
        """
        執行單次實驗
        
        Args:
            config_path: 配置文件路徑
            config_name: 配置名稱
            params: 參數字典
            
        Returns:
            result_dict: 實驗結果
        """
        logger.info(f"▶ 開始實驗: {config_name}")
        logger.info(f"   參數: {params}")
        
        # 定義輸出路徑
        exp_output = self.output_dir / "results" / config_name
        exp_output.mkdir(exist_ok=True)
        
        # 定義日誌路徑
        log_file = self.output_dir / "logs" / f"{config_name}.log"
        
        # 構建訓練命令
        train_script = Path(__file__).parent / "train.py"
        cmd = [
            sys.executable,
            str(train_script),
            "--cfg", str(config_path)
        ]
        
        # 執行訓練
        start_time = datetime.now()
        
        try:
            with open(log_file, 'w') as log_f:
                process = subprocess.run(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    timeout=3600 * 2  # 2小時超時
                )
            
            success = (process.returncode == 0)
            
        except subprocess.TimeoutExpired:
            logger.warning(f"⚠ 實驗超時: {config_name}")
            success = False
        except Exception as e:
            logger.error(f"❌ 實驗失敗: {config_name} - {e}")
            success = False
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 提取結果（從檢查點或日誌）
        metrics = self._extract_metrics(exp_output, log_file, success)
        
        # 組合結果
        result = {
            "config_name": config_name,
            "params": params,
            "success": success,
            "duration_sec": duration,
            **metrics
        }
        
        logger.info(f"✅ 完成實驗: {config_name} ({duration:.1f}s)")
        logger.info(f"   最終損失: {metrics.get('final_loss', 'N/A')}")
        logger.info(f"   重採樣次數: {metrics.get('resample_count', 'N/A')}")
        
        return result
    
    def _extract_metrics(self, 
                         exp_output: Path,
                         log_file: Path,
                         success: bool) -> Dict[str, Any]:
        """從輸出中提取關鍵指標"""
        metrics = {
            "final_loss": None,
            "min_loss": None,
            "resample_count": None,
            "convergence_epoch": None,
            "l2_error_u": None,
            "l2_error_v": None,
            "l2_error_p": None,
        }
        
        if not success:
            return metrics
        
        # 1. 嘗試從檢查點提取
        checkpoint_files = list(exp_output.glob("checkpoint_*.pth"))
        if checkpoint_files:
            import torch
            latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
            
            try:
                ckpt = torch.load(latest_checkpoint, map_location='cpu')
                
                # 訓練歷史
                history = ckpt.get('training_history', {})
                if history and 'total_loss' in history:
                    losses = history['total_loss']
                    metrics['final_loss'] = losses[-1] if losses else None
                    metrics['min_loss'] = min(losses) if losses else None
                
                # 重採樣統計
                sampling_stats = ckpt.get('sampling_stats', {})
                metrics['resample_count'] = sampling_stats.get('resample_count', 0)
                
                # L2 誤差（如果有評估結果）
                eval_metrics = ckpt.get('eval_metrics', {})
                metrics['l2_error_u'] = eval_metrics.get('l2_error_u')
                metrics['l2_error_v'] = eval_metrics.get('l2_error_v')
                metrics['l2_error_p'] = eval_metrics.get('l2_error_p')
                
            except Exception as e:
                logger.warning(f"⚠ 無法讀取檢查點 {latest_checkpoint}: {e}")
        
        # 2. 嘗試從日誌提取（備選）
        if metrics['final_loss'] is None and log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                
                # 解析日誌中的損失（簡單正則匹配）
                import re
                loss_pattern = re.compile(r"Total Loss:\s*([\d.e+-]+)")
                resample_pattern = re.compile(r"Resampling triggered")
                
                losses = []
                resample_count = 0
                
                for line in lines:
                    loss_match = loss_pattern.search(line)
                    if loss_match:
                        losses.append(float(loss_match.group(1)))
                    
                    if resample_pattern.search(line):
                        resample_count += 1
                
                if losses:
                    metrics['final_loss'] = losses[-1]
                    metrics['min_loss'] = min(losses)
                
                metrics['resample_count'] = resample_count
                
            except Exception as e:
                logger.warning(f"⚠ 無法解析日誌 {log_file}: {e}")
        
        return metrics
    
    def run_all_experiments(self, parallel: bool = False, max_workers: int = 4) -> None:
        """
        執行所有實驗
        
        Args:
            parallel: 是否並行執行
            max_workers: 最大並行工作數
        """
        variants = self.generate_config_variants()
        
        logger.info("=" * 80)
        logger.info(f"🚀 開始批次實驗（共 {len(variants)} 個）")
        logger.info(f"   並行模式: {parallel}")
        if parallel:
            logger.info(f"   工作數: {max_workers}")
        logger.info("=" * 80)
        
        # 保存所有配置
        config_paths = []
        for config, name, params in variants:
            config_path = self.save_config(config, name)
            config_paths.append((config_path, name, params))
        
        # 執行實驗
        if parallel:
            # 並行執行
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self.run_single_experiment, 
                        config_path, 
                        name, 
                        params
                    ): name
                    for config_path, name, params in config_paths
                }
                
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        result = future.result()
                        self.results.append(result)
                    except Exception as e:
                        logger.error(f"❌ 實驗失敗 {name}: {e}")
        else:
            # 順序執行
            for config_path, name, params in config_paths:
                result = self.run_single_experiment(config_path, name, params)
                self.results.append(result)
        
        # 保存結果匯總
        self._save_results_summary()
        
        logger.info("=" * 80)
        logger.info(f"✅ 所有實驗完成！")
        logger.info(f"   成功: {sum(r['success'] for r in self.results)}/{len(self.results)}")
        logger.info("=" * 80)
    
    def _save_results_summary(self) -> None:
        """保存結果匯總"""
        # 轉為 DataFrame
        df = pd.DataFrame(self.results)
        
        # 展開 params 列
        params_df = pd.json_normalize(df['params'])
        df = pd.concat([df.drop('params', axis=1), params_df], axis=1)
        
        # 保存 CSV
        csv_path = self.output_dir / "results_summary.csv"
        df.to_csv(csv_path, index=False)
        
        # 保存 JSON
        json_path = self.output_dir / "results_summary.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"📄 結果匯總已保存:")
        logger.info(f"   CSV: {csv_path}")
        logger.info(f"   JSON: {json_path}")
    
    def analyze_and_visualize(self) -> None:
        """分析結果並生成可視化"""
        logger.info("=" * 80)
        logger.info("📊 開始結果分析與可視化...")
        logger.info("=" * 80)
        
        # 載入結果
        df = pd.DataFrame(self.results)
        params_df = pd.json_normalize(df['params'])
        df = pd.concat([df.drop('params', axis=1), params_df], axis=1)
        
        # 過濾成功的實驗
        df_success = df[df['success'] == True].copy()
        
        if df_success.empty:
            logger.warning("⚠ 沒有成功的實驗，無法生成可視化")
            return
        
        # 1. 參數熱圖（參數 vs 指標）
        self._plot_parameter_heatmaps(df_success)
        
        # 2. 並行座標圖
        self._plot_parallel_coordinates(df_success)
        
        # 3. 收斂曲線比較（需要額外數據）
        # self._plot_convergence_comparison(df_success)
        
        # 4. 統計摘要
        self._print_statistical_summary(df_success)
        
        logger.info("=" * 80)
        logger.info("✅ 分析與可視化完成！")
        logger.info(f"   圖表保存至: {self.output_dir / 'plots'}")
        logger.info("=" * 80)
    
    def _plot_parameter_heatmaps(self, df: pd.DataFrame) -> None:
        """繪製參數熱圖"""
        param_cols = ["keep_ratio", "initial_weight", "epoch_interval", "energy_threshold"]
        metric_cols = ["final_loss", "min_loss", "resample_count", "duration_sec"]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metric_cols):
            ax = axes[idx]
            
            # 針對每對參數創建熱圖
            # 這裡簡化為 keep_ratio vs initial_weight
            pivot = df.pivot_table(
                values=metric,
                index='keep_ratio',
                columns='initial_weight',
                aggfunc='mean'
            )
            
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
            ax.set_title(f'{metric} vs (keep_ratio, initial_weight)', fontweight='bold')
            ax.set_xlabel('Initial Weight Multiplier')
            ax.set_ylabel('Keep Ratio')
        
        fig.suptitle('Parameter Sensitivity Heatmaps', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / "plots" / "parameter_heatmaps.png"
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        logger.info(f"✅ 保存熱圖: {save_path}")
        plt.close(fig)
    
    def _plot_parallel_coordinates(self, df: pd.DataFrame) -> None:
        """繪製並行座標圖"""
        from pandas.plotting import parallel_coordinates
        
        # 選擇變量
        plot_cols = ["keep_ratio", "initial_weight", "epoch_interval", 
                     "energy_threshold", "final_loss"]
        
        # 標準化數值（便於可視化）
        df_norm = df[plot_cols].copy()
        for col in plot_cols:
            df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min() + 1e-8)
        
        # 根據 final_loss 分組（高/中/低）
        df_norm['loss_category'] = pd.qcut(df['final_loss'], q=3, labels=['Low', 'Medium', 'High'])
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        parallel_coordinates(
            df_norm, 
            'loss_category', 
            cols=plot_cols[:-1],
            color=['green', 'orange', 'red'],
            alpha=0.5,
            ax=ax
        )
        
        ax.set_title('Parallel Coordinates Plot: Parameter Impact on Final Loss', 
                     fontsize=14, fontweight='bold')
        ax.set_ylabel('Normalized Value')
        ax.legend(title='Loss Category', loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_dir / "plots" / "parallel_coordinates.png"
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        logger.info(f"✅ 保存並行座標圖: {save_path}")
        plt.close(fig)
    
    def _print_statistical_summary(self, df: pd.DataFrame) -> None:
        """打印統計摘要"""
        logger.info("=" * 80)
        logger.info("📈 統計摘要")
        logger.info("=" * 80)
        
        # 最佳配置
        best_idx = df['final_loss'].idxmin()
        best_config = df.loc[best_idx]
        
        logger.info("🏆 最佳配置:")
        logger.info(f"   配置名稱: {best_config['config_name']}")
        logger.info(f"   keep_ratio: {best_config['keep_ratio']}")
        logger.info(f"   initial_weight: {best_config['initial_weight']}")
        logger.info(f"   epoch_interval: {best_config['epoch_interval']}")
        logger.info(f"   energy_threshold: {best_config['energy_threshold']}")
        logger.info(f"   最終損失: {best_config['final_loss']:.6e}")
        logger.info(f"   重採樣次數: {best_config['resample_count']}")
        logger.info(f"   訓練時長: {best_config['duration_sec']:.1f}s")
        
        # 相關性分析
        logger.info("=" * 80)
        logger.info("📊 參數與最終損失的相關性:")
        
        param_cols = ["keep_ratio", "initial_weight", "epoch_interval", "energy_threshold"]
        for param in param_cols:
            corr = df[param].corr(df['final_loss'])
            logger.info(f"   {param:20s}: {corr:+.3f}")
        
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='參數敏感度實驗')
    parser.add_argument('--base-config', type=str, required=True,
                       help='基礎配置文件路徑')
    parser.add_argument('--output', type=str, default='results/sensitivity_analysis/',
                       help='輸出目錄')
    parser.add_argument('--mode', type=str, default='quick', 
                       choices=['full', 'quick'],
                       help='實驗模式（full: 完整參數空間，quick: 快速測試）')
    parser.add_argument('--parallel', action='store_true',
                       help='並行執行實驗')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='最大並行工作數')
    parser.add_argument('--analyze-only', action='store_true',
                       help='僅分析現有結果（不執行實驗）')
    
    args = parser.parse_args()
    
    # 創建實驗管理器
    experiment = SensitivityExperiment(
        base_config_path=args.base_config,
        output_dir=args.output,
        mode=args.mode
    )
    
    # 執行實驗
    if not args.analyze_only:
        experiment.run_all_experiments(
            parallel=args.parallel,
            max_workers=args.max_workers
        )
    
    # 分析與可視化
    if experiment.results or (experiment.output_dir / "results_summary.json").exists():
        # 如果是 analyze_only 模式，從文件載入結果
        if args.analyze_only and not experiment.results:
            json_path = experiment.output_dir / "results_summary.json"
            with open(json_path, 'r') as f:
                experiment.results = json.load(f)
            logger.info(f"📂 載入現有結果: {json_path}")
        
        experiment.analyze_and_visualize()
    
    logger.info("✅ 參數敏感度實驗完成！")


if __name__ == "__main__":
    main()
