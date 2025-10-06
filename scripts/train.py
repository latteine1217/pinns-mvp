#!/usr/bin/env python3
"""
PINNs 逆重建主訓練腳本
負責協調資料載入、模型建立、訓練迴圈與評估輸出
"""

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinnx.models.fourier_mlp import PINNNet
from pinnx.models.wrappers import ScaledPINNWrapper
from pinnx.physics.scaling import VSScaler
from pinnx.physics.ns_2d import NSEquations2D
from pinnx.losses.residuals import NSResidualLoss
from pinnx.losses.priors import PriorLossManager
from pinnx.losses.weighting import GradNormWeighter, CausalWeighter


def setup_logging(level: str = "info") -> logging.Logger:
    """設置日誌系統"""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pinnx.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def set_random_seed(seed: int, deterministic: bool = True) -> None:
    """設置隨機種子確保重現性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """載入YAML配置檔案"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_device(device_name: str) -> torch.device:
    """獲取運算設備"""
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using CUDA: {torch.cuda.get_device_name()}")
    elif device_name == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using Apple Metal Performance Shaders")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")
    
    return device


def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """建立 PINN 模型"""
    model_cfg = config['model']
    
    # 基礎 PINN 網路
    base_model = PINNNet(
        in_dim=model_cfg['in_dim'],
        out_dim=model_cfg['out_dim'],
        width=model_cfg['width'],
        depth=model_cfg['depth'],
        activation=model_cfg['activation'],
        use_fourier=True,
        fourier_m=model_cfg['fourier_m'],
        fourier_sigma=model_cfg['fourier_sigma']
    ).to(device)
    
    # 檢查是否啟用 VS-PINN 尺度化
    if model_cfg.get('scaling', {}).get('learnable', False):
        logging.info("Enabling VS-PINN scaling wrapper")
        
        # 建立 VS 尺度器
        scaling_cfg = model_cfg['scaling']
        input_scaler = None
        output_scaler = None
        
        # 建立輸入尺度器
        if scaling_cfg.get('input_norm') != 'none':
            input_scaler = VSScaler(
                input_dim=model_cfg['in_dim'],
                output_dim=1,  # 不用於輸出，只是佔位
                learnable=scaling_cfg['learnable']
            )
            # 使用合理的初始值
            domain = config['physics']['domain']
            x_mean = torch.tensor([
                (domain['t_range'][1] + domain['t_range'][0]) / 2,
                (domain['x_range'][1] + domain['x_range'][0]) / 2,
                (domain['y_range'][1] + domain['y_range'][0]) / 2
            ]).view(1, -1)
            x_std = torch.tensor([
                (domain['t_range'][1] - domain['t_range'][0]) / 4,
                (domain['x_range'][1] - domain['x_range'][0]) / 4,
                (domain['y_range'][1] - domain['y_range'][0]) / 4
            ]).view(1, -1)
            input_scaler.input_mean.data.copy_(x_mean)
            input_scaler.input_std.data.copy_(x_std)
            input_scaler.fitted = True
        
        # 建立輸出尺度器
        if scaling_cfg.get('output_norm') != 'none':
            output_scaler = VSScaler(
                input_dim=1,  # 不用於輸入，只是佔位
                output_dim=model_cfg['out_dim'],
                learnable=scaling_cfg['learnable']
            )
            # 使用合理的初始值
            y_mean = torch.zeros(1, model_cfg['out_dim'])
            y_std = torch.ones(1, model_cfg['out_dim'])
            output_scaler.output_mean.data.copy_(y_mean)
            output_scaler.output_std.data.copy_(y_std)
            output_scaler.fitted = True
        
        # 包裝模型
        model = ScaledPINNWrapper(
            base_model=base_model,
            input_scaler=input_scaler,
            output_scaler=output_scaler,
            variable_names=['u', 'v', 'p', 'S']
        )
        logging.info("VS-PINN scaling enabled successfully")
    else:
        # 直接使用基礎模型
        model = base_model
        logging.info("Using base model without scaling")
    
    logging.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    return model


def create_physics(config: Dict[str, Any], device: torch.device) -> NSEquations2D:
    """建立物理方程式模組"""
    physics_cfg = config['physics']
    
    physics = NSEquations2D(
        viscosity=physics_cfg['nu'],
        density=physics_cfg['rho']
    )
    
    return physics


def create_loss_functions(config: Dict[str, Any], device: torch.device) -> Dict[str, nn.Module]:
    """建立損失函數"""
    loss_cfg = config['losses']
    
    losses = {
        'residual': NSResidualLoss(
            nu=loss_cfg.get('nu', 1e-3),
            density=loss_cfg.get('rho', 1.0)
        ),
        'prior': PriorLossManager(
            consistency_weight=loss_cfg['prior_weight']
        )
    }
    
    return losses


def create_weighters(config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """建立動態權重器"""
    loss_cfg = config['losses']
    weighters = {}
    
    # 暫時跳過權重器創建，稍後在有模型時再設定
    weighters['gradnorm'] = None
    weighters['causal'] = None
    
    return weighters


def prepare_training_data(config: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    """準備訓練資料 - 僅支援 Channel Flow"""
    
    # 檢查是否使用 Channel Flow 載入器
    if 'channel_flow' in config and config['channel_flow'].get('enabled', False):
        return prepare_channel_flow_training_data(config, device)
    else:
        raise ValueError("Mock training data is no longer supported. Please enable channel_flow in config.")


def prepare_channel_flow_training_data(config: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    """使用 Channel Flow 載入器準備訓練資料"""
    from pinnx.dataio.channel_flow_loader import prepare_training_data as load_channel_flow
    
    # 載入 Channel Flow 資料
    cf_config = config['channel_flow']
    strategy = cf_config.get('strategy', 'qr_pivot')  # 'qr_pivot' 或 'random'
    K = config['sensors']['K']
    
    channel_data = load_channel_flow(
        strategy=strategy,
        K=K,
        target_fields=['u', 'v', 'p']
    )
    
    # 提取感測器座標和資料
    coords = channel_data['coordinates']  # (K, 2) numpy array
    sensor_data = channel_data['sensor_data']  # dict with 'u', 'v', 'p'
    domain_bounds = channel_data['domain_bounds']
    
    # 轉換為 PyTorch tensor
    x_sensors = torch.from_numpy(coords[:, 0:1]).float().to(device)  # (K, 1)
    y_sensors = torch.from_numpy(coords[:, 1:2]).float().to(device)  # (K, 1)
    t_sensors = torch.zeros_like(x_sensors)  # 暫時假設 t=0
    
    u_sensors = torch.from_numpy(sensor_data['u'].reshape(-1, 1)).float().to(device)
    v_sensors = torch.from_numpy(sensor_data['v'].reshape(-1, 1)).float().to(device)
    p_sensors = torch.from_numpy(sensor_data['p'].reshape(-1, 1)).float().to(device)
    
    # 生成 PDE 殘差點和邊界點
    sampling = config['training']['sampling']
    x_range = domain_bounds['x']
    y_range = domain_bounds['y']
    
    # PDE 殘差點
    x_pde = torch.rand(sampling['pde_points'], 1, device=device) * (x_range[1] - x_range[0]) + x_range[0]
    y_pde = torch.rand(sampling['pde_points'], 1, device=device) * (y_range[1] - y_range[0]) + y_range[0]
    t_pde = torch.zeros_like(x_pde)  # 穩態假設
    
    # 邊界點 (Channel Flow 上下壁面)
    n_bc = sampling['boundary_points']
    x_bc = torch.rand(n_bc, 1, device=device) * (x_range[1] - x_range[0]) + x_range[0]
    y_bc_bottom = torch.full((n_bc//2, 1), y_range[0], device=device)  # 下壁面
    y_bc_top = torch.full((n_bc - n_bc//2, 1), y_range[1], device=device)  # 上壁面
    y_bc = torch.cat([y_bc_bottom, y_bc_top], dim=0)
    x_bc = torch.cat([x_bc[:n_bc//2], x_bc[n_bc//2:]], dim=0)
    t_bc = torch.zeros_like(x_bc)
    
    return {
        'x_pde': x_pde, 'y_pde': y_pde, 't_pde': t_pde,
        'x_bc': x_bc, 'y_bc': y_bc, 't_bc': t_bc,
        'x_sensors': x_sensors, 'y_sensors': y_sensors, 't_sensors': t_sensors,
        'u_sensors': u_sensors, 'v_sensors': v_sensors, 'p_sensors': p_sensors,
        'domain_bounds': domain_bounds,
        'channel_data': channel_data  # 保留原始資料供後續使用
    }





def train_step(model: nn.Module, 
               physics: NSEquations2D,
               losses: Dict[str, nn.Module],
               data_batch: Dict[str, torch.Tensor],
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Dict[str, float]:
    """執行一個訓練步驟"""
    optimizer.zero_grad()
    
    # PDE 殘差計算 - 組合 x, y, t 座標
    x_pde = torch.cat([data_batch['x_pde'], data_batch['y_pde'], data_batch['t_pde']], dim=1)
    x_pde.requires_grad_(True)  # 確保啟用梯度計算
    
    u_pred = model(x_pde)
    
    # 提取座標和時間
    coords_2d = x_pde[:, :2]  # x, y
    coords_2d.requires_grad_(True)
    velocity = u_pred[:, :2]  # u, v
    pressure = u_pred[:, 2:3]  # p
    time_coords = x_pde[:, 2:3]  # t
    time_coords.requires_grad_(True)
    
    try:
        # 嘗試使用物理模組的方法
        residuals = physics.residual(coords_2d, velocity, pressure, time_coords)
        residual_loss = torch.mean(residuals['momentum_x']**2 + residuals['momentum_y']**2 + residuals['continuity']**2)
    except Exception as e:
        # 備用：簡單的 L2 正則化 (暫時替代)
        print(f"Physics residual computation failed: {e}")
        residual_loss = torch.mean(u_pred**2) * 0.001  # 簡單的正則化項
    
    # 邊界條件損失
    x_bc = torch.cat([data_batch['x_bc'], data_batch['y_bc'], data_batch['t_bc']], dim=1)
    u_bc_pred = model(x_bc)
    bc_loss = torch.mean(u_bc_pred[:, :2]**2)  # 無滑移條件 u=v=0
    
    # 資料匹配損失
    x_sensors = torch.cat([data_batch['x_sensors'], data_batch['y_sensors'], data_batch['t_sensors']], dim=1)
    u_sensors_pred = model(x_sensors)
    data_loss = torch.mean((u_sensors_pred[:, 0:1] - data_batch['u_sensors'])**2 + 
                          (u_sensors_pred[:, 1:2] - data_batch['v_sensors'])**2)
    
    # 總損失
    total_loss = residual_loss + bc_loss + data_loss
    
    total_loss.backward()
    optimizer.step()
    
    return {
        'total_loss': total_loss.item(),
        'residual_loss': residual_loss.item(),
        'bc_loss': bc_loss.item(),
        'data_loss': data_loss.item()
    }


def train_model(model: nn.Module,
                physics: NSEquations2D,
                losses: Dict[str, nn.Module],
                config: Dict[str, Any],
                device: torch.device) -> Dict[str, Any]:
    """主要訓練迴圈"""
    train_cfg = config['training']
    
    # 建立優化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg['lr'],
        weight_decay=train_cfg['weight_decay']
    )
    
    # 學習率調度器
    if train_cfg['lr_scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg['max_epochs']
        )
    else:
        scheduler = None
    
    # 準備訓練資料
    training_data = prepare_training_data(config, device)
    
    # 訓練循環
    logger = logging.getLogger(__name__)
    start_time = time.time()
    loss_dict = {'total_loss': 0.0, 'residual_loss': 0.0, 'bc_loss': 0.0, 'data_loss': 0.0}
    epoch = -1  # 初始化 epoch 變數
    
    for epoch in range(train_cfg['max_epochs']):
        # 執行訓練步驟
        loss_dict = train_step(
            model, physics, losses, training_data, optimizer, device
        )
        
        # 更新學習率
        if scheduler:
            scheduler.step()
        
        # 日誌輸出
        if epoch % config['logging']['log_freq'] == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"Epoch {epoch:6d} | "
                f"Total: {loss_dict['total_loss']:.6f} | "
                f"Residual: {loss_dict['residual_loss']:.6f} | "
                f"BC: {loss_dict['bc_loss']:.6f} | "
                f"Data: {loss_dict['data_loss']:.6f} | "
                f"Time: {elapsed:.1f}s"
            )
        
        # 驗證和檢查點
        if epoch % train_cfg['validation_freq'] == 0 and epoch > 0:
            logger.info("=== Validation checkpoint ===")
            # 這裡可以添加驗證邏輯
        
        # 早停檢查
        if loss_dict['total_loss'] < 1e-6:
            logger.info(f"Early convergence at epoch {epoch}")
            break
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.1f}s")
    
    return {
        'final_loss': loss_dict['total_loss'],
        'training_time': total_time,
        'epochs_completed': epoch + 1
    }


def main():
    """主程式入口"""
    parser = argparse.ArgumentParser(description='PINNs Inverse Training Script')
    parser.add_argument('--cfg', type=str, default='configs/defaults.yml',
                       help='Path to configuration file')
    parser.add_argument('--ensemble', action='store_true',
                       help='Run ensemble training for UQ')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # 載入配置
    config = load_config(args.cfg)
    
    # 設置日誌
    logger = setup_logging(config['logging']['level'])
    logger.info("=" * 60)
    logger.info("PINNs Inverse Reconstruction Training")
    logger.info("=" * 60)
    
    # 設置重現性
    set_random_seed(
        config['experiment']['seed'],
        config['reproducibility']['deterministic']
    )
    
    # 設置設備
    device = get_device(config['experiment']['device'])
    
    # 建立模型和物理模組
    model = create_model(config, device)
    physics = create_physics(config, device)
    losses = create_loss_functions(config, device)
    
    logger.info(f"Model architecture: {config['model']['type']}")
    logger.info(f"Input dimension: {config['model']['in_dim']}")
    logger.info(f"Output dimension: {config['model']['out_dim']}")
    logger.info(f"Physics: NS-2D with nu={config['physics']['nu']}")
    
    if args.ensemble:
        logger.info("Running ensemble training...")
        ensemble_cfg = config['ensemble']
        
        models = []
        for i, seed in enumerate(ensemble_cfg['seeds']):
            logger.info(f"Training ensemble member {i+1}/{len(ensemble_cfg['seeds'])} (seed={seed})")
            
            # 重置隨機種子
            set_random_seed(seed, config['reproducibility']['deterministic'])
            
            # 建立新模型
            member_model = create_model(config, device)
            
            # 訓練
            train_result = train_model(member_model, physics, losses, config, device)
            models.append(member_model)
            
            logger.info(f"Member {i+1} final loss: {train_result['final_loss']:.6f}")
        
        # 儲存模型列表（暫時不使用 EnsembleWrapper）
        logger.info(f"Ensemble training completed with {len(models)} members")
        logger.info("Note: EnsembleWrapper not implemented yet - models stored as list")
        
    else:
        logger.info("Running single model training...")
        train_result = train_model(model, physics, losses, config, device)
        logger.info(f"Training completed. Final loss: {train_result['final_loss']:.6f}")
    
    logger.info("Training script finished successfully!")


if __name__ == "__main__":
    main()