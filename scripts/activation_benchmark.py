#!/usr/bin/env python3
"""
激活函數比較實驗
測試 sine、swish、tanh 在真實 JHTDB Channel Flow 數據上的效果
訓練 5000 epochs，生成完整評估報告
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import json
from datetime import datetime

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train import (
    setup_logging, set_random_seed, load_config, get_device,
    create_model, create_physics, create_loss_functions,
    create_weighters, prepare_training_data, train_step
)


class ActivationBenchmark:
    """激活函數基準測試器"""
    
    def __init__(self, base_config_path: str, output_dir: str = "activation_benchmark"):
        self.base_config = load_config(base_config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 定義測試的激活函數
        self.activations = ['sine', 'swish', 'tanh']
        
        # 結果存儲
        self.results = {}
        
        # 設置日誌
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f"benchmark_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def create_config_variant(self, activation: str) -> Dict[str, Any]:
        """為特定激活函數創建配置變體"""
        config = self.base_config.copy()
        
        # 修改模型配置
        config['model']['activation'] = activation
        
        # 修改訓練配置為 5000 epochs
        config['training']['max_epochs'] = 5000
        config['training']['checkpoint_freq'] = 1000
        config['training']['validation_freq'] = 500
        
        # 修改實驗名稱
        config['experiment']['name'] = f"activation_benchmark_{activation}"
        config['experiment']['version'] = f"5000ep_{activation}"
        
        # 禁用 early stopping 以確保完整訓練
        config['training']['early_stopping'] = False
        
        return config
    
    def train_single_activation(self, activation: str, device: torch.device) -> Dict[str, Any]:
        """訓練單個激活函數配置"""
        self.logger.info("=" * 80)
        self.logger.info(f"🚀 開始訓練: activation = {activation}")
        self.logger.info("=" * 80)
        
        # 創建配置
        config = self.create_config_variant(activation)
        
        # Sine 激活需要特殊調整以保證穩定性
        if activation == 'sine':
            config['training']['lr'] = 5e-4  # 降低學習率
            config['model']['sine_omega_0'] = 1.0  # 使用保守的 omega_0
            self.logger.info(f"⚙️  Sine 激活特殊設置: lr={config['training']['lr']}, omega_0={config['model']['sine_omega_0']}")
        
        # 設置隨機種子
        set_random_seed(config['experiment']['seed'], True)
        
        # 準備訓練數據
        training_data = prepare_training_data(config, device)
        
        # 創建模型
        model = create_model(config, device)
        physics = create_physics(config, device)
        losses = create_loss_functions(config, device)
        weighters = create_weighters(config, model, device)
        
        # 創建優化器
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )
        
        # 學習率調度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['training']['max_epochs']
        )
        
        # 訓練歷史記錄
        history = {
            'epoch': [],
            'total_loss': [],
            'residual_loss': [],
            'bc_loss': [],
            'data_loss': [],
            'momentum_x_loss': [],
            'momentum_y_loss': [],
            'continuity_loss': []
        }
        
        # 訓練循環
        start_time = time.time()
        
        for epoch in range(config['training']['max_epochs']):
            # 訓練步驟
            loss_dict = train_step(
                model, physics, losses, training_data,
                optimizer, weighters, epoch, device, config
            )
            
            # 更新學習率
            scheduler.step()
            
            # 記錄歷史
            history['epoch'].append(epoch)
            for key in ['total_loss', 'residual_loss', 'bc_loss', 'data_loss',
                       'momentum_x_loss', 'momentum_y_loss', 'continuity_loss']:
                if key in loss_dict:
                    history[key].append(loss_dict[key])
            
            # 日誌輸出
            if epoch % 100 == 0:
                elapsed = time.time() - start_time
                self.logger.info(
                    f"[{activation.upper()}] Epoch {epoch:5d} | "
                    f"Total: {loss_dict['total_loss']:.6f} | "
                    f"Residual: {loss_dict['residual_loss']:.6f} | "
                    f"BC: {loss_dict['bc_loss']:.6f} | "
                    f"Data: {loss_dict['data_loss']:.6f} | "
                    f"Time: {elapsed:.1f}s"
                )
        
        training_time = time.time() - start_time
        
        # 保存模型
        model_path = self.output_dir / f"model_{activation}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'history': history,
            'training_time': training_time
        }, model_path)
        
        self.logger.info(f"✅ {activation.upper()} 訓練完成，耗時: {training_time:.1f}s")
        self.logger.info(f"   最終 loss: {history['total_loss'][-1]:.6f}")
        self.logger.info(f"   模型已保存: {model_path}")
        
        return {
            'model': model,
            'config': config,
            'history': history,
            'training_time': training_time,
            'model_path': str(model_path)
        }
    
    def evaluate_model(self, model: torch.nn.Module, config: Dict[str, Any], 
                      activation: str, device: torch.device) -> Dict[str, Any]:
        """評估模型性能"""
        self.logger.info(f"📊 評估模型: {activation}")
        
        # 準備評估網格
        eval_cfg = config.get('evaluation', {})
        grid_res = eval_cfg.get('grid_resolution', [256, 128])
        
        # 載入 Channel Flow 數據用於對比
        from pinnx.dataio.channel_flow_loader import prepare_training_data as load_channel_flow
        
        channel_data = load_channel_flow(
            strategy=config['sensors'].get('selection_method', 'qr_pivot'),
            K=config['sensors']['K'],
            target_fields=['u', 'v', 'p']
        )
        
        full_field = channel_data['full_field']
        domain_bounds = channel_data['domain_bounds']
        
        # 創建評估網格
        x_range = domain_bounds['x']
        y_range = domain_bounds['y']
        
        x = np.linspace(x_range[0], x_range[1], grid_res[0])
        y = np.linspace(y_range[0], y_range[1], grid_res[1])
        X, Y = np.meshgrid(x, y)
        
        # 標準化座標 (與訓練一致)
        x_min, x_max = x_range[0], x_range[1]
        y_min, y_max = y_range[0], y_range[1]
        X_norm = 2.0 * (X - x_min) / (x_max - x_min) - 1.0
        Y_norm = 2.0 * (Y - y_min) / (y_max - y_min) - 1.0
        
        # 準備輸入
        coords = np.stack([X_norm.flatten(), Y_norm.flatten()], axis=1)
        coords_tensor = torch.from_numpy(coords).float().to(device)
        
        # 預測
        model.eval()
        with torch.no_grad():
            pred = model(coords_tensor).cpu().numpy()
        
        u_pred = pred[:, 0].reshape(grid_res[1], grid_res[0])
        v_pred = pred[:, 1].reshape(grid_res[1], grid_res[0])
        p_pred = pred[:, 2].reshape(grid_res[1], grid_res[0])
        
        # 提取真實場 (需要插值到評估網格)
        u_true = full_field['u']
        v_true = full_field['v']
        p_true = full_field['p']
        
        # 計算誤差指標
        def relative_l2_error(pred, true):
            return np.linalg.norm(pred - true) / np.linalg.norm(true)
        
        def rmse(pred, true):
            return np.sqrt(np.mean((pred - true)**2))
        
        u_l2_error = relative_l2_error(u_pred, u_true)
        v_l2_error = relative_l2_error(v_pred, v_true)
        p_l2_error = relative_l2_error(p_pred, p_true)
        
        u_rmse = rmse(u_pred, u_true)
        v_rmse = rmse(v_pred, v_true)
        p_rmse = rmse(p_pred, p_true)
        
        # 計算統計量
        u_mean_pred = np.mean(u_pred)
        u_mean_true = np.mean(u_true)
        u_mean_error = abs(u_mean_pred - u_mean_true) / u_mean_true
        
        metrics = {
            'u_l2_error': float(u_l2_error),
            'v_l2_error': float(v_l2_error),
            'p_l2_error': float(p_l2_error),
            'u_rmse': float(u_rmse),
            'v_rmse': float(v_rmse),
            'p_rmse': float(p_rmse),
            'u_mean_error': float(u_mean_error),
            'u_mean_pred': float(u_mean_pred),
            'u_mean_true': float(u_mean_true)
        }
        
        self.logger.info(f"   U - L2 相對誤差: {u_l2_error:.4f} ({u_l2_error*100:.2f}%)")
        self.logger.info(f"   V - L2 相對誤差: {v_l2_error:.4f} ({v_l2_error*100:.2f}%)")
        self.logger.info(f"   P - L2 相對誤差: {p_l2_error:.4f} ({p_l2_error*100:.2f}%)")
        self.logger.info(f"   U 均值誤差: {u_mean_error:.4f} ({u_mean_error*100:.2f}%)")
        
        # 保存預測場
        predictions = {
            'u': u_pred,
            'v': v_pred,
            'p': p_pred,
            'X': X,
            'Y': Y
        }
        
        return {
            'metrics': metrics,
            'predictions': predictions,
            'ground_truth': {
                'u': u_true,
                'v': v_true,
                'p': p_true
            }
        }
    
    def plot_comparison(self):
        """生成比較圖表"""
        self.logger.info("📈 生成比較圖表...")
        
        # 1. Loss 曲線比較
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for activation in self.activations:
            result = self.results[activation]
            history = result['history']
            
            # Total Loss
            axes[0, 0].plot(history['epoch'], history['total_loss'], 
                           label=activation, linewidth=2)
            
            # Residual Loss
            axes[0, 1].plot(history['epoch'], history['residual_loss'], 
                           label=activation, linewidth=2)
            
            # BC Loss
            axes[1, 0].plot(history['epoch'], history['bc_loss'], 
                           label=activation, linewidth=2)
            
            # Data Loss
            axes[1, 1].plot(history['epoch'], history['data_loss'], 
                           label=activation, linewidth=2)
        
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        axes[0, 1].set_title('Residual Loss (PDE)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
        
        axes[1, 0].set_title('Boundary Condition Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        axes[1, 1].set_title('Data Fitting Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        loss_plot_path = self.output_dir / "loss_comparison.png"
        plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"   Loss 比較圖已保存: {loss_plot_path}")
        
        # 2. 誤差指標比較 (柱狀圖)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        activations_list = list(self.activations)
        
        # U 誤差
        u_l2_errors = [self.results[act]['evaluation']['metrics']['u_l2_error'] 
                       for act in activations_list]
        axes[0].bar(activations_list, u_l2_errors, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0].set_title('U Velocity - Relative L2 Error', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Relative L2 Error', fontsize=12)
        axes[0].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(u_l2_errors):
            axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        # V 誤差
        v_l2_errors = [self.results[act]['evaluation']['metrics']['v_l2_error'] 
                       for act in activations_list]
        axes[1].bar(activations_list, v_l2_errors, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[1].set_title('V Velocity - Relative L2 Error', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Relative L2 Error', fontsize=12)
        axes[1].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(v_l2_errors):
            axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        # P 誤差
        p_l2_errors = [self.results[act]['evaluation']['metrics']['p_l2_error'] 
                       for act in activations_list]
        axes[2].bar(activations_list, p_l2_errors, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[2].set_title('Pressure - Relative L2 Error', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Relative L2 Error', fontsize=12)
        axes[2].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(p_l2_errors):
            axes[2].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        error_plot_path = self.output_dir / "error_comparison.png"
        plt.savefig(error_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"   誤差比較圖已保存: {error_plot_path}")
        
        # 3. 預測場可視化 (每個激活函數一行)
        fig, axes = plt.subplots(len(self.activations), 3, figsize=(18, 5*len(self.activations)))
        
        for i, activation in enumerate(self.activations):
            eval_result = self.results[activation]['evaluation']
            pred = eval_result['predictions']
            true = eval_result['ground_truth']
            
            # U 場
            im0 = axes[i, 0].contourf(pred['X'], pred['Y'], pred['u'], levels=20, cmap='RdBu_r')
            axes[i, 0].set_title(f'{activation.upper()} - U Velocity', fontsize=12, fontweight='bold')
            axes[i, 0].set_xlabel('x')
            axes[i, 0].set_ylabel('y')
            plt.colorbar(im0, ax=axes[i, 0])
            
            # V 場
            im1 = axes[i, 1].contourf(pred['X'], pred['Y'], pred['v'], levels=20, cmap='RdBu_r')
            axes[i, 1].set_title(f'{activation.upper()} - V Velocity', fontsize=12, fontweight='bold')
            axes[i, 1].set_xlabel('x')
            axes[i, 1].set_ylabel('y')
            plt.colorbar(im1, ax=axes[i, 1])
            
            # P 場
            im2 = axes[i, 2].contourf(pred['X'], pred['Y'], pred['p'], levels=20, cmap='RdBu_r')
            axes[i, 2].set_title(f'{activation.upper()} - Pressure', fontsize=12, fontweight='bold')
            axes[i, 2].set_xlabel('x')
            axes[i, 2].set_ylabel('y')
            plt.colorbar(im2, ax=axes[i, 2])
        
        plt.tight_layout()
        field_plot_path = self.output_dir / "field_comparison.png"
        plt.savefig(field_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"   場比較圖已保存: {field_plot_path}")
    
    def generate_report(self):
        """生成測試報告"""
        self.logger.info("📝 生成測試報告...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'base_config': str(self.base_config['experiment']['name']),
            'epochs': 5000,
            'activations_tested': self.activations,
            'results': {}
        }
        
        # 彙總結果
        for activation in self.activations:
            result = self.results[activation]
            
            report['results'][activation] = {
                'training_time': result['training_time'],
                'final_total_loss': result['history']['total_loss'][-1],
                'final_residual_loss': result['history']['residual_loss'][-1],
                'final_data_loss': result['history']['data_loss'][-1],
                'metrics': result['evaluation']['metrics'],
                'model_path': result['model_path']
            }
        
        # 保存 JSON 報告
        report_path = self.output_dir / "benchmark_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"   JSON 報告已保存: {report_path}")
        
        # 生成 Markdown 報告
        md_report = self._generate_markdown_report(report)
        md_path = self.output_dir / "BENCHMARK_REPORT.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_report)
        
        self.logger.info(f"   Markdown 報告已保存: {md_path}")
        
        return report
    
    def _generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """生成 Markdown 格式報告"""
        md = f"""# 激活函數基準測試報告

## 📋 測試配置

- **測試時間**: {report['timestamp']}
- **基礎配置**: {report['base_config']}
- **訓練輪數**: {report['epochs']} epochs
- **測試激活函數**: {', '.join(report['activations_tested'])}

## 🎯 實驗目標

評估 **sine, swish, tanh** 激活函數在 JHTDB Channel Flow Re=1000 數據上的表現：
- 訓練穩定性
- 收斂速度
- 最終精度（L2 相對誤差）
- 統計量準確性

> ⚠️ **注意**: Loss 大小不等於精度！必須以評估指標為準。

---

## 📊 結果總結

### 訓練時間對比

| 激活函數 | 訓練時間 (秒) | 訓練時間 (分鐘) |
|---------|-------------|---------------|
"""
        
        for activation in report['activations_tested']:
            t = report['results'][activation]['training_time']
            md += f"| **{activation.upper()}** | {t:.1f} | {t/60:.1f} |\n"
        
        md += "\n### 最終 Loss 對比\n\n"
        md += "| 激活函數 | Total Loss | Residual Loss | Data Loss |\n"
        md += "|---------|-----------|--------------|----------|\n"
        
        for activation in report['activations_tested']:
            res = report['results'][activation]
            md += f"| **{activation.upper()}** | {res['final_total_loss']:.6f} | {res['final_residual_loss']:.6f} | {res['final_data_loss']:.6f} |\n"
        
        md += "\n### 🎯 準確度指標 (重要！)\n\n"
        md += "| 激活函數 | U - L2 誤差 | V - L2 誤差 | P - L2 誤差 | U 均值誤差 |\n"
        md += "|---------|------------|------------|------------|----------|\n"
        
        for activation in report['activations_tested']:
            metrics = report['results'][activation]['metrics']
            md += f"| **{activation.upper()}** | {metrics['u_l2_error']:.4f} ({metrics['u_l2_error']*100:.2f}%) | {metrics['v_l2_error']:.4f} ({metrics['v_l2_error']*100:.2f}%) | {metrics['p_l2_error']:.4f} ({metrics['p_l2_error']*100:.2f}%) | {metrics['u_mean_error']:.4f} ({metrics['u_mean_error']*100:.2f}%) |\n"
        
        md += "\n### RMSE 對比\n\n"
        md += "| 激活函數 | U - RMSE | V - RMSE | P - RMSE |\n"
        md += "|---------|---------|---------|----------|\n"
        
        for activation in report['activations_tested']:
            metrics = report['results'][activation]['metrics']
            md += f"| **{activation.upper()}** | {metrics['u_rmse']:.4f} | {metrics['v_rmse']:.4f} | {metrics['p_rmse']:.4f} |\n"
        
        md += "\n---\n\n## 🏆 最佳激活函數推薦\n\n"
        
        # 找出最佳激活函數
        best_u = min(report['activations_tested'], 
                     key=lambda a: report['results'][a]['metrics']['u_l2_error'])
        best_v = min(report['activations_tested'], 
                     key=lambda a: report['results'][a]['metrics']['v_l2_error'])
        best_p = min(report['activations_tested'], 
                     key=lambda a: report['results'][a]['metrics']['p_l2_error'])
        
        md += f"- **U 速度場**: `{best_u}` (L2 誤差: {report['results'][best_u]['metrics']['u_l2_error']:.4f})\n"
        md += f"- **V 速度場**: `{best_v}` (L2 誤差: {report['results'][best_v]['metrics']['v_l2_error']:.4f})\n"
        md += f"- **壓力場**: `{best_p}` (L2 誤差: {report['results'][best_p]['metrics']['p_l2_error']:.4f})\n"
        
        md += "\n---\n\n## 📈 可視化圖表\n\n"
        md += "1. **Loss 曲線比較**: `loss_comparison.png`\n"
        md += "2. **誤差指標比較**: `error_comparison.png`\n"
        md += "3. **預測場可視化**: `field_comparison.png`\n"
        
        md += "\n---\n\n## 💾 模型文件\n\n"
        
        for activation in report['activations_tested']:
            md += f"- **{activation.upper()}**: `{report['results'][activation]['model_path']}`\n"
        
        md += "\n---\n\n## 🔬 分析與建議\n\n"
        md += "### 觀察要點\n\n"
        md += "1. **收斂穩定性**: 觀察 loss 曲線是否平穩下降\n"
        md += "2. **訓練效率**: 比較達到相同精度所需的 epochs\n"
        md += "3. **最終精度**: 以 L2 相對誤差為準（非 loss 值）\n"
        md += "4. **統計準確性**: U 均值誤差反映宏觀統計保持能力\n"
        
        md += "\n### 建議\n\n"
        md += "- 若 **sine** 表現最佳：考慮用於高頻特徵捕捉\n"
        md += "- 若 **swish** 表現最佳：平衡性能與穩定性的好選擇\n"
        md += "- 若 **tanh** 表現最佳：經典選擇，穩定可靠\n"
        
        return md
    
    def run(self):
        """執行完整基準測試"""
        self.logger.info("🚀 激活函數基準測試開始")
        self.logger.info(f"   測試激活函數: {', '.join(self.activations)}")
        self.logger.info(f"   訓練輪數: 5000 epochs")
        self.logger.info(f"   輸出目錄: {self.output_dir}")
        
        # 獲取設備
        device = get_device(self.base_config['experiment']['device'])
        self.logger.info(f"   使用設備: {device}")
        
        # 訓練所有激活函數
        for activation in self.activations:
            result = self.train_single_activation(activation, device)
            self.results[activation] = result
        
        # 評估所有模型
        for activation in self.activations:
            eval_result = self.evaluate_model(
                self.results[activation]['model'],
                self.results[activation]['config'],
                activation,
                device
            )
            self.results[activation]['evaluation'] = eval_result
        
        # 生成比較圖表
        self.plot_comparison()
        
        # 生成報告
        report = self.generate_report()
        
        self.logger.info("=" * 80)
        self.logger.info("✅ 激活函數基準測試完成！")
        self.logger.info(f"   所有結果已保存至: {self.output_dir}")
        self.logger.info("=" * 80)
        
        return report


def main():
    parser = argparse.ArgumentParser(description='激活函數基準測試')
    parser.add_argument('--config', type=str, 
                       default='configs/channel_flow_re1000_K80_wall_balanced.yml',
                       help='基礎配置文件路徑')
    parser.add_argument('--output', type=str, 
                       default='activation_benchmark',
                       help='輸出目錄')
    
    args = parser.parse_args()
    
    # 執行基準測試
    benchmark = ActivationBenchmark(args.config, args.output)
    benchmark.run()


if __name__ == "__main__":
    main()
