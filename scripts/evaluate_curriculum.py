#!/usr/bin/env python3
"""
評估課程訓練完成的檢查點
正確載入 ManualScalingWrapper 包裝的模型並計算誤差
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 無顯示模式

# 添加專案根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinnx.models import PINNNet, create_pinn_model
from pinnx.models.wrappers import ManualScalingWrapper


def setup_logging(level: str = "info") -> logging.Logger:
    """設置日誌系統"""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """載入YAML配置檔案"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_model_from_config(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """從配置創建模型（複製自 train.py）
    
    注意：必須與訓練時的模型建立邏輯完全一致，包括 wrapper
    """
    model_cfg = config['model']
    
    # 建立基礎模型
    fourier_cfg = model_cfg.get('fourier_features', {})
    fourier_m = fourier_cfg.get('fourier_m', model_cfg.get('fourier_m', 32))
    fourier_sigma = fourier_cfg.get('fourier_sigma', model_cfg.get('fourier_sigma', 1.0))
    
    if model_cfg.get('type') == 'fourier_vs_mlp':
        base_model_cfg = {
            **model_cfg,
            'fourier_m': fourier_m,
            'fourier_sigma': fourier_sigma,
            'use_fourier': model_cfg.get('use_fourier', True)
        }
        base_model = create_pinn_model(base_model_cfg).to(device)
    else:
        base_model = PINNNet(
            in_dim=model_cfg['in_dim'],
            out_dim=model_cfg['out_dim'],
            width=model_cfg['width'],
            depth=model_cfg['depth'],
            activation=model_cfg['activation'],
            use_fourier=True,
            fourier_m=fourier_m,
            fourier_sigma=fourier_sigma
        ).to(device)
    
    # 檢查是否使用 scaling wrapper
    scaling_cfg = model_cfg.get('scaling', {})
    scaling_enabled = bool(scaling_cfg)
    
    if scaling_enabled:
        # 從配置中提取輸入輸出範圍
        input_x_range = tuple(scaling_cfg.get('input_norm', {}).get('x', [0.0, 25.13]))
        input_y_range = tuple(scaling_cfg.get('input_norm', {}).get('y', [-1.0, 1.0]))
        
        input_scales = {
            'x': input_x_range,
            'y': input_y_range
        }
        
        # 輸出範圍
        output_norm = scaling_cfg.get('output_norm', {})
        output_scales = {
            'u': tuple(output_norm.get('u', [0.0, 16.5])),
            'v': tuple(output_norm.get('v', [-0.6, 0.6])),
            'p': tuple(output_norm.get('p', [-85.0, 3.0]))
        }
        
        # 建立包裝模型
        model = ManualScalingWrapper(
            base_model, 
            input_ranges=input_scales,
            output_ranges=output_scales
        ).to(device)
        
        logging.info(f"✅ Created ManualScalingWrapper")
        logging.info(f"   Input ranges: {input_scales}")
        logging.info(f"   Output ranges: {output_scales}")
    else:
        model = base_model
        logging.info("Using base model without wrapper")
    
    return model


def load_checkpoint(checkpoint_path: str, model: nn.Module) -> Tuple[int, float, Dict]:
    """載入檢查點到模型"""
    logging.info(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 載入模型權重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    config = checkpoint.get('config', {})
    
    logging.info(f"✅ Checkpoint loaded successfully")
    logging.info(f"   Epoch: {epoch}")
    logging.info(f"   Training Loss: {loss:.6f}")
    
    return epoch, loss, config


def load_sensor_data(sensor_file: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """載入感測器數據"""
    logging.info(f"Loading sensor data from: {sensor_file}")
    
    data = np.load(sensor_file)
    
    # 提取座標和真實值
    coords = data['sensor_points']  # (K, 2)
    u_true = data['sensor_u'].reshape(-1, 1)  # (K, 1)
    v_true = data['sensor_v'].reshape(-1, 1)
    p_true = data['sensor_p'].reshape(-1, 1)
    
    # 轉換為 tensor（不標準化，模型會處理）
    x = torch.from_numpy(coords[:, 0:1]).float().to(device)
    y = torch.from_numpy(coords[:, 1:2]).float().to(device)
    
    sensor_data = {
        'x': x,
        'y': y,
        'coords': torch.cat([x, y], dim=1),  # (K, 2)
        'u_true': torch.from_numpy(u_true).float().to(device),
        'v_true': torch.from_numpy(v_true).float().to(device),
        'p_true': torch.from_numpy(p_true).float().to(device)
    }
    
    logging.info(f"✅ Sensor data loaded: {len(x)} points")
    logging.info(f"   X range: [{x.min():.6f}, {x.max():.6f}]")
    logging.info(f"   Y range: [{y.min():.6f}, {y.max():.6f}]")
    logging.info(f"   U range: [{u_true.min():.6f}, {u_true.max():.6f}]")
    logging.info(f"   V range: [{v_true.min():.6f}, {v_true.max():.6f}]")
    logging.info(f"   P range: [{p_true.min():.6f}, {p_true.max():.6f}]")
    
    return sensor_data


def evaluate_model(model: nn.Module, sensor_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """評估模型在感測點上的誤差"""
    model.eval()
    
    with torch.no_grad():
        # 模型預測（直接使用原始座標，wrapper 會處理標準化）
        coords = sensor_data['coords']  # (K, 2)
        pred = model(coords)  # (K, 3)
        
        u_pred = pred[:, 0:1]
        v_pred = pred[:, 1:2]
        p_pred = pred[:, 2:3]
    
    # 提取真實值
    u_true = sensor_data['u_true']
    v_true = sensor_data['v_true']
    p_true = sensor_data['p_true']
    
    # 計算相對 L2 誤差
    def relative_l2_error(pred, true):
        numerator = torch.sqrt(torch.mean((pred - true)**2))
        denominator = torch.sqrt(torch.mean(true**2))
        return (numerator / (denominator + 1e-12)).item()
    
    u_error = relative_l2_error(u_pred, u_true)
    v_error = relative_l2_error(v_pred, v_true)
    p_error = relative_l2_error(p_pred, p_true)
    
    # 計算絕對誤差（RMSE）
    u_rmse = torch.sqrt(torch.mean((u_pred - u_true)**2)).item()
    v_rmse = torch.sqrt(torch.mean((v_pred - v_true)**2)).item()
    p_rmse = torch.sqrt(torch.mean((p_pred - p_true)**2)).item()
    
    # 計算最大誤差
    u_max_error = torch.max(torch.abs(u_pred - u_true)).item()
    v_max_error = torch.max(torch.abs(v_pred - v_true)).item()
    p_max_error = torch.max(torch.abs(p_pred - p_true)).item()
    
    results = {
        'u_rel_l2': u_error * 100,  # 轉換為百分比
        'v_rel_l2': v_error * 100,
        'p_rel_l2': p_error * 100,
        'u_rmse': u_rmse,
        'v_rmse': v_rmse,
        'p_rmse': p_rmse,
        'u_max_error': u_max_error,
        'v_max_error': v_max_error,
        'p_max_error': p_max_error
    }
    
    return results


def visualize_predictions(model: nn.Module, 
                         sensor_data: Dict[str, torch.Tensor],
                         output_dir: str = "./evaluation_results"):
    """可視化預測結果"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    
    with torch.no_grad():
        coords = sensor_data['coords']
        pred = model(coords)
        
        u_pred = pred[:, 0].cpu().numpy()
        v_pred = pred[:, 1].cpu().numpy()
        p_pred = pred[:, 2].cpu().numpy()
    
    # 提取真實值
    u_true = sensor_data['u_true'].cpu().numpy().flatten()
    v_true = sensor_data['v_true'].cpu().numpy().flatten()
    p_true = sensor_data['p_true'].cpu().numpy().flatten()
    
    # 創建對比圖
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # U 速度
    axes[0].scatter(u_true, u_pred, alpha=0.6, s=50, label='Predictions')
    u_min, u_max = u_true.min(), u_true.max()
    axes[0].plot([u_min, u_max], [u_min, u_max], 'r--', lw=2, label='Perfect fit')
    axes[0].set_xlabel('True U', fontsize=12)
    axes[0].set_ylabel('Predicted U', fontsize=12)
    axes[0].set_title('U Velocity Comparison', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # V 速度
    axes[1].scatter(v_true, v_pred, alpha=0.6, s=50, label='Predictions')
    v_min, v_max = v_true.min(), v_true.max()
    axes[1].plot([v_min, v_max], [v_min, v_max], 'r--', lw=2, label='Perfect fit')
    axes[1].set_xlabel('True V', fontsize=12)
    axes[1].set_ylabel('Predicted V', fontsize=12)
    axes[1].set_title('V Velocity Comparison', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 壓力
    axes[2].scatter(p_true, p_pred, alpha=0.6, s=50, label='Predictions')
    p_min, p_max = p_true.min(), p_true.max()
    axes[2].plot([p_min, p_max], [p_min, p_max], 'r--', lw=2, label='Perfect fit')
    axes[2].set_xlabel('True P', fontsize=12)
    axes[2].set_ylabel('Predicted P', fontsize=12)
    axes[2].set_title('Pressure Comparison', fontsize=14)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"{output_dir}/prediction_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logging.info(f"✅ Visualization saved to: {plot_path}")
    plt.close()


def main():
    """主程式"""
    parser = argparse.ArgumentParser(description='Evaluate Curriculum Training Checkpoint')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/pinnx_channel_flow_curriculum_ic_latest.pth',
                       help='Path to checkpoint file')
    parser.add_argument('--config', type=str,
                       default='configs/channel_flow_curriculum_4stage_ic.yml',
                       help='Path to config file')
    parser.add_argument('--sensor-file', type=str,
                       default='data/jhtdb/channel_flow_re1000/sensors_K80_wall_balanced.npz',
                       help='Path to sensor data file')
    parser.add_argument('--output-dir', type=str,
                       default='./evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')
    
    args = parser.parse_args()
    
    # 設置日誌
    logger = setup_logging('info')
    
    logger.info("=" * 80)
    logger.info("📊 課程訓練檢查點評估")
    logger.info("=" * 80)
    
    # 載入配置
    config = load_config(args.config)
    device = torch.device('cpu')  # 評估使用 CPU 即可
    
    # 創建模型（必須與訓練時一致）
    model = create_model_from_config(config, device)
    
    # 載入檢查點
    epoch, loss, ckpt_config = load_checkpoint(args.checkpoint, model)
    
    # 載入感測器數據
    sensor_data = load_sensor_data(args.sensor_file, device)
    
    # 評估模型
    logger.info("\n" + "=" * 80)
    logger.info("🔍 開始評估...")
    logger.info("=" * 80)
    
    results = evaluate_model(model, sensor_data)
    
    # 輸出結果
    logger.info("\n" + "=" * 80)
    logger.info("📈 評估結果")
    logger.info("=" * 80)
    logger.info(f"訓練輪數: {epoch}")
    logger.info(f"訓練損失: {loss:.6f}")
    logger.info("")
    logger.info("相對 L2 誤差 (%)：")
    logger.info(f"  U 速度: {results['u_rel_l2']:.2f}%")
    logger.info(f"  V 速度: {results['v_rel_l2']:.2f}%")
    logger.info(f"  壓力:   {results['p_rel_l2']:.2f}%")
    logger.info("")
    logger.info("RMSE 誤差：")
    logger.info(f"  U 速度: {results['u_rmse']:.6f}")
    logger.info(f"  V 速度: {results['v_rmse']:.6f}")
    logger.info(f"  壓力:   {results['p_rmse']:.6f}")
    logger.info("")
    logger.info("最大絕對誤差：")
    logger.info(f"  U 速度: {results['u_max_error']:.6f}")
    logger.info(f"  V 速度: {results['v_max_error']:.6f}")
    logger.info(f"  壓力:   {results['p_max_error']:.6f}")
    logger.info("=" * 80)
    
    # 成敗判定（目標: < 15%）
    threshold = 15.0
    all_pass = all([
        results['u_rel_l2'] < threshold,
        results['v_rel_l2'] < threshold,
        results['p_rel_l2'] < threshold
    ])
    
    if all_pass:
        logger.info("✅ 評估通過！所有場的相對誤差 < 15%")
    else:
        logger.warning("⚠️  部分場未達標（目標 < 15%）")
    
    # 可視化
    if args.visualize:
        logger.info("\n生成可視化圖表...")
        visualize_predictions(model, sensor_data, args.output_dir)
    
    # 保存結果到 JSON
    import json
    results_file = f"{args.output_dir}/evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'epoch': epoch,
            'training_loss': float(loss),
            'metrics': {k: float(v) for k, v in results.items()},
            'threshold': threshold,
            'passed': all_pass
        }, f, indent=2)
    logger.info(f"✅ 結果已保存至: {results_file}")


if __name__ == "__main__":
    main()
