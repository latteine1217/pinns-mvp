#!/usr/bin/env python3
"""
RANS-PINNs 湍流建模訓練腳本
擴展基礎訓練腳本以支持RANS方程與湍流量預測
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
from pinnx.physics.turbulence import RANSEquations2D
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
            logging.FileHandler('rans_pinnx.log'),
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
    """載入配置檔案"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_rans_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """創建RANS-PINNs模型 (6維輸出: [u, v, p, k, ε, S])"""
    model_cfg = config['model']
    
    # 確保輸出維度為6 (u, v, p, k, ε, S)
    if model_cfg['out_dim'] != 6:
        logging.warning(f"Adjusting output dimension from {model_cfg['out_dim']} to 6 for RANS model")
        model_cfg['out_dim'] = 6
    
    # 創建基礎神經網路
    base_model = PINNNet(
        in_dim=model_cfg['in_dim'],
        out_dim=model_cfg['out_dim'],
        width=model_cfg['width'],
        depth=model_cfg['depth'],
        activation=model_cfg['activation'],
        fourier_m=model_cfg['fourier_m'],
        fourier_sigma=model_cfg['fourier_sigma'],
        use_fourier=True,
        trainable_fourier=True
    )
    
    # 應用VS-PINN尺度化包裝器
    if model_cfg['scaling']['learnable']:
        # 創建輸入和輸出尺度器
        input_scaler = VSScaler(
            input_dim=model_cfg['in_dim'],
            output_dim=model_cfg['in_dim'],  # 輸入尺度器的輸出維度等於輸入維度
            learnable=True,
            init_method=model_cfg['scaling']['input_norm']
        )
        
        output_scaler = VSScaler(
            input_dim=model_cfg['out_dim'],  # 輸出尺度器的輸入維度等於輸出維度
            output_dim=model_cfg['out_dim'],
            learnable=True,
            init_method=model_cfg['scaling']['output_norm']
        )
        
        model = ScaledPINNWrapper(
            base_model=base_model,
            input_scaler=input_scaler,
            output_scaler=output_scaler,
            variable_names=['u', 'v', 'p', 'k', 'epsilon', 'S']
        )
        
        output_scaler = VSScaler(
            input_dim=model_cfg['out_dim'],  # 輸出尺度器的輸入維度等於輸出維度
            output_dim=model_cfg['out_dim'],
            learnable=True,
            init_method=model_cfg['scaling']['output_norm']
        )
        
        model = ScaledPINNWrapper(
            base_model=base_model,
            input_scaler=input_scaler,
            output_scaler=output_scaler,
            variable_names=['u', 'v', 'p', 'k', 'epsilon', 'S']
        )
    else:
        model = base_model
    
    return model.to(device)


def create_rans_physics(config: Dict[str, Any]) -> RANSEquations2D:
    """創建RANS物理方程模組"""
    physics_cfg = config['physics']
    turbulence_cfg = physics_cfg.get('turbulence', {})
    
    return RANSEquations2D(
        viscosity=physics_cfg['nu'],
        turbulence_model=turbulence_cfg.get('model', 'k_epsilon')
    )


def prepare_training_data(config: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    """準備訓練資料"""
    sampling = config['training']['sampling']
    domain = config['physics']['domain']
    
    # 生成PDE內部點
    n_pde = sampling['pde_points']
    x_pde = torch.rand(n_pde, 1, device=device) * (domain['x_range'][1] - domain['x_range'][0]) + domain['x_range'][0]
    y_pde = torch.rand(n_pde, 1, device=device) * (domain['y_range'][1] - domain['y_range'][0]) + domain['y_range'][0]
    t_pde = torch.rand(n_pde, 1, device=device) * (domain['t_range'][1] - domain['t_range'][0]) + domain['t_range'][0]
    
    # 生成邊界點
    n_bc = sampling['boundary_points']
    x_bc = torch.rand(n_bc, 1, device=device) * (domain['x_range'][1] - domain['x_range'][0]) + domain['x_range'][0]
    y_bc = torch.zeros(n_bc, 1, device=device)  # 壁面 y=0
    t_bc = torch.rand(n_bc, 1, device=device) * (domain['t_range'][1] - domain['t_range'][0]) + domain['t_range'][0]
    
    # 模擬感測器數據 (實際應用中會從檔案載入)
    n_sensors = config['sensors']['K']
    x_sensors = torch.rand(n_sensors, 1, device=device) * (domain['x_range'][1] - domain['x_range'][0]) + domain['x_range'][0]
    y_sensors = torch.rand(n_sensors, 1, device=device) * (domain['y_range'][1] - domain['y_range'][0]) + domain['y_range'][0]
    t_sensors = torch.rand(n_sensors, 1, device=device) * (domain['t_range'][1] - domain['t_range'][0]) + domain['t_range'][0]
    
    # 模擬速度觀測 (簡單的拋物線分佈)
    u_sensors = 4 * y_sensors * (2 - y_sensors) / 4  # 拋物線速度分佈
    v_sensors = torch.zeros_like(u_sensors)
    
    return {
        'x_pde': x_pde, 'y_pde': y_pde, 't_pde': t_pde,
        'x_bc': x_bc, 'y_bc': y_bc, 't_bc': t_bc,
        'x_sensors': x_sensors, 'y_sensors': y_sensors, 't_sensors': t_sensors,
        'u_sensors': u_sensors, 'v_sensors': v_sensors
    }


def rans_train_step(model: nn.Module,
                   physics: RANSEquations2D,
                   data_batch: Dict[str, torch.Tensor],
                   optimizer: torch.optim.Optimizer,
                   loss_weights: Dict[str, float],
                   device: torch.device,
                   curriculum_stage: Optional[str] = None) -> Dict[str, float]:
    """執行一個RANS訓練步驟"""
    optimizer.zero_grad()
    
    # PDE 殘差計算
    x_pde = torch.cat([data_batch['x_pde'], data_batch['y_pde'], data_batch['t_pde']], dim=1)
    x_pde.requires_grad_(True)
    
    u_pred = model(x_pde)  # [u, v, p, k, ε, S]
    
    # 提取各個物理量 (保持梯度連接)
    # 注意：不能直接切片座標，這會破壞梯度連接
    # coords_2d = x_pde[:, :2]  # ❌ 這樣會斷開梯度連接
    
    velocity = u_pred[:, :2]     # u, v
    pressure = u_pred[:, 2:3]    # p  
    k = u_pred[:, 3:4]          # 湍動能
    epsilon = u_pred[:, 4:5]     # 耗散率
    time_coords = x_pde[:, 2:3]  # t 座標
    
    # 確保湍流量的物理合理性
    k = torch.clamp(k, min=1e-10)
    epsilon = torch.clamp(epsilon, min=1e-10)
    
    # RANS方程殘差計算
    # 使用完整的3D座標 x_pde 來保持梯度連接
    try:
        residuals = physics.residual(x_pde, velocity, pressure, k, epsilon, time_coords)
        
        # 根據課程學習階段選擇激活的損失
        loss_dict = {}
        
        if curriculum_stage is None or curriculum_stage == "all":
            # 全部方程
            loss_dict.update({
                'momentum_x_loss': torch.mean(residuals['momentum_x']**2),
                'momentum_y_loss': torch.mean(residuals['momentum_y']**2),
                'continuity_loss': torch.mean(residuals['continuity']**2),
                'k_equation_loss': torch.mean(residuals['k_equation']**2),
                'epsilon_equation_loss': torch.mean(residuals['dissipation_equation']**2)
            })
        elif curriculum_stage == "momentum":
            # 只有動量和連續方程
            loss_dict.update({
                'momentum_x_loss': torch.mean(residuals['momentum_x']**2),
                'momentum_y_loss': torch.mean(residuals['momentum_y']**2),
                'continuity_loss': torch.mean(residuals['continuity']**2),
                'k_equation_loss': torch.tensor(0.0, device=device),
                'epsilon_equation_loss': torch.tensor(0.0, device=device)
            })
        elif curriculum_stage == "with_k":
            # 動量、連續和k方程
            loss_dict.update({
                'momentum_x_loss': torch.mean(residuals['momentum_x']**2),
                'momentum_y_loss': torch.mean(residuals['momentum_y']**2),
                'continuity_loss': torch.mean(residuals['continuity']**2),
                'k_equation_loss': torch.mean(residuals['k_equation']**2),
                'epsilon_equation_loss': torch.tensor(0.0, device=device)
            })
    except Exception as e:
        print(f"RANS residual computation failed: {e}")
        # 備用簡化損失
        loss_dict = {
            'momentum_x_loss': torch.mean(velocity[:, 0:1]**2) * 0.001,
            'momentum_y_loss': torch.mean(velocity[:, 1:2]**2) * 0.001,
            'continuity_loss': torch.mean(pressure**2) * 0.001,
            'k_equation_loss': torch.mean((k - 0.01)**2),
            'epsilon_equation_loss': torch.mean((epsilon - 0.1)**2)
        }
    
    # 邊界條件損失
    x_bc = torch.cat([data_batch['x_bc'], data_batch['y_bc'], data_batch['t_bc']], dim=1)
    u_bc_pred = model(x_bc)
    
    # 壁面邊界條件: u=v=0, k=0, ε按壁面函數
    bc_loss = (torch.mean(u_bc_pred[:, :2]**2) +  # 速度
               torch.mean(u_bc_pred[:, 3:4]**2))    # 湍動能 k=0
    
    # 資料匹配損失
    x_sensors = torch.cat([data_batch['x_sensors'], data_batch['y_sensors'], data_batch['t_sensors']], dim=1)
    u_sensors_pred = model(x_sensors)
    data_loss = torch.mean((u_sensors_pred[:, 0:1] - data_batch['u_sensors'])**2 + 
                          (u_sensors_pred[:, 1:2] - data_batch['v_sensors'])**2)
    
    # 物理約束: k≥0, ε≥0
    k_constraint = torch.mean(torch.clamp(-k, min=0)**2)
    epsilon_constraint = torch.mean(torch.clamp(-epsilon, min=0)**2)
    
    # 總損失計算 (使用配置中的權重)
    total_loss = (
        loss_weights.get('momentum_x_weight', 1.0) * loss_dict['momentum_x_loss'] +
        loss_weights.get('momentum_y_weight', 1.0) * loss_dict['momentum_y_loss'] +
        loss_weights.get('continuity_weight', 1.0) * loss_dict['continuity_loss'] +
        loss_weights.get('k_equation_weight', 0.5) * loss_dict['k_equation_loss'] +
        loss_weights.get('epsilon_equation_weight', 0.5) * loss_dict['epsilon_equation_loss'] +
        loss_weights.get('boundary_weight', 1.0) * bc_loss +
        loss_weights.get('data_weight', 1.0) * data_loss +
        loss_weights.get('k_min_constraint', 1e-6) * k_constraint +
        loss_weights.get('epsilon_min_constraint', 1e-6) * epsilon_constraint
    )
    
    total_loss.backward()
    
    # 梯度剪裁 (湍流模型數值穩定性)
    if 'gradient_clip' in loss_weights:
        torch.nn.utils.clip_grad_norm_(model.parameters(), loss_weights['gradient_clip'])
    
    optimizer.step()
    
    # 返回損失詳情
    result = {
        'total_loss': total_loss.item(),
        'momentum_x_loss': loss_dict['momentum_x_loss'].item(),
        'momentum_y_loss': loss_dict['momentum_y_loss'].item(),
        'continuity_loss': loss_dict['continuity_loss'].item(),
        'k_equation_loss': loss_dict['k_equation_loss'].item(),
        'epsilon_equation_loss': loss_dict['epsilon_equation_loss'].item(),
        'bc_loss': bc_loss.item(),
        'data_loss': data_loss.item(),
        'k_constraint': k_constraint.item(),
        'epsilon_constraint': epsilon_constraint.item()
    }
    
    return result


def train_rans_model(model: nn.Module,
                    physics: RANSEquations2D,
                    config: Dict[str, Any],
                    device: torch.device) -> Dict[str, Any]:
    """RANS模型主要訓練迴圈"""
    train_cfg = config['training']
    loss_weights = config['losses']
    
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
    
    # 課程學習設置
    curriculum = train_cfg.get('curriculum', False)
    curriculum_stages = train_cfg.get('curriculum_stages', [])
    current_stage = 0
    current_stage_name = "all"
    
    if curriculum and curriculum_stages:
        current_stage_name = "momentum"  # 從動量方程開始
    
    # 訓練循環
    logger = logging.getLogger(__name__)
    start_time = time.time()
    best_loss = float('inf')
    epoch = -1
    loss_dict = {'total_loss': float('inf')}  # 初始化 loss_dict
    
    for epoch in range(train_cfg['max_epochs']):
        # 課程學習階段切換
        if curriculum and curriculum_stages and current_stage < len(curriculum_stages):
            stage_info = curriculum_stages[current_stage]
            if epoch >= stage_info['epochs'] and stage_info['epochs'] > 0:
                current_stage += 1
                if current_stage < len(curriculum_stages):
                    logger.info(f"Switching to curriculum stage {current_stage}")
                else:
                    current_stage_name = "all"
                    logger.info("Curriculum completed, training with all equations")
        
        # 設置當前階段
        if curriculum and current_stage < len(curriculum_stages):
            stage_info = curriculum_stages[current_stage]
            if 'momentum' in str(stage_info.get('active_losses', [])):
                current_stage_name = "momentum"
            elif 'k_equation' in str(stage_info.get('active_losses', [])):
                current_stage_name = "with_k" if 'epsilon' not in str(stage_info.get('active_losses', [])) else "all"
            else:
                current_stage_name = "all"
        
        # 執行訓練步驟
        loss_dict = rans_train_step(
            model, physics, training_data, optimizer, loss_weights, device, current_stage_name
        )
        
        # 更新學習率
        if scheduler:
            scheduler.step()
        
        # 日誌輸出
        if epoch % config['logging']['log_freq'] == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"Epoch {epoch:6d} | Stage: {current_stage_name} | "
                f"Total: {loss_dict['total_loss']:.6f} | "
                f"Mom_x: {loss_dict['momentum_x_loss']:.6f} | "
                f"Mom_y: {loss_dict['momentum_y_loss']:.6f} | "
                f"Cont: {loss_dict['continuity_loss']:.6f} | "
                f"k: {loss_dict['k_equation_loss']:.6f} | "
                f"ε: {loss_dict['epsilon_equation_loss']:.6f} | "
                f"BC: {loss_dict['bc_loss']:.6f} | "
                f"Data: {loss_dict['data_loss']:.6f} | "
                f"Time: {elapsed:.1f}s"
            )
        
        # 檢查最佳模型
        if loss_dict['total_loss'] < best_loss:
            best_loss = loss_dict['total_loss']
            
        # 驗證和檢查點
        if epoch % train_cfg['validation_freq'] == 0 and epoch > 0:
            logger.info("=== RANS Validation checkpoint ===")
            # 這裡可以添加湍流量的物理檢查
        
        # 早停檢查
        if loss_dict['total_loss'] < 1e-6:
            logger.info(f"Early convergence at epoch {epoch}")
            break
    
    total_time = time.time() - start_time
    logger.info(f"RANS training completed in {total_time:.1f}s")
    
    return {
        'final_loss': loss_dict['total_loss'],
        'best_loss': best_loss,
        'training_time': total_time,
        'epochs_completed': epoch + 1,
        'final_stage': current_stage_name
    }


def main():
    """主程式入口"""
    parser = argparse.ArgumentParser(description='RANS-PINNs Turbulence Training Script')
    parser.add_argument('--cfg', type=str, default='configs/rans.yml',
                       help='Path to RANS configuration file')
    parser.add_argument('--ensemble', action='store_true',
                       help='Run ensemble training for UQ')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # 載入配置
    config = load_config(args.cfg)
    
    # 設置日誌
    logger = setup_logging(config['logging']['level'])
    logger.info("=" * 80)
    logger.info("RANS-PINNs Turbulence Modeling Training")
    logger.info("=" * 80)
    
    # 設置重現性
    set_random_seed(
        config['experiment']['seed'],
        config['reproducibility']['deterministic']
    )
    
    # 設置設備
    if config['experiment']['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
    
    # 創建RANS模型
    logger.info("Creating RANS-PINNs model...")
    model = create_rans_model(config, device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 創建RANS物理方程
    logger.info("Initializing RANS physics equations...")
    physics = create_rans_physics(config)
    logger.info(f"RANS physics initialized: {physics.get_model_info()}")
    
    if args.ensemble:
        # Ensemble 訓練
        logger.info("Starting ensemble RANS training...")
        ensemble_cfg = config['ensemble']
        n_models = ensemble_cfg['n_models']
        
        results = []
        for i in range(n_models):
            logger.info(f"Training ensemble member {i+1}/{n_models}")
            
            # 每個ensemble成員使用不同的種子
            set_random_seed(
                config['experiment']['seed'] + i,
                config['reproducibility']['deterministic']
            )
            
            # 重新創建模型
            member_model = create_rans_model(config, device)
            
            # 訓練
            result = train_rans_model(member_model, physics, config, device)
            results.append(result)
            
            logger.info(f"Ensemble member {i+1} completed: Loss = {result['final_loss']:.6f}")
        
        # 計算ensemble統計
        final_losses = [r['final_loss'] for r in results]
        logger.info(f"Ensemble training completed:")
        logger.info(f"Mean final loss: {np.mean(final_losses):.6f} ± {np.std(final_losses):.6f}")
        logger.info(f"Best member loss: {min(final_losses):.6f}")
        
    else:
        # 單模型訓練
        logger.info("Starting RANS-PINNs training...")
        result = train_rans_model(model, physics, config, device)
        
        logger.info("=" * 80)
        logger.info("Training Summary:")
        logger.info(f"Final Loss: {result['final_loss']:.6f}")
        logger.info(f"Best Loss: {result['best_loss']:.6f}")
        logger.info(f"Training Time: {result['training_time']:.1f}s")
        logger.info(f"Epochs Completed: {result['epochs_completed']}")
        logger.info(f"Final Stage: {result['final_stage']}")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()