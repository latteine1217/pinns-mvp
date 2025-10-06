#!/usr/bin/env python3
"""
最終監督學習+RANS策略測試
結合尺度權重和場標準化的混合方法
"""

import sys
import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn

# 設置路徑
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import pinnx
from pinnx.dataio.channel_flow_loader import ChannelFlowLoader
from pinnx.models import PINNNet 
from pinnx.physics.ns_2d import NSEquations2D
from pinnx.losses.residuals import NSResidualLoss
from pinnx.sensors import QRPivotSelector

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def relative_L2(pred, true, dim=None):
    """計算相對L2誤差"""
    if dim is not None:
        pred_slice = pred[:, dim]
        true_slice = true[:, dim]
    else:
        pred_slice = pred.flatten()
        true_slice = true.flatten()
    
    numerator = np.sqrt(np.mean((pred_slice - true_slice)**2))
    denominator = np.sqrt(np.mean(true_slice**2))
    return numerator / denominator if denominator > 0 else float('inf')

def compute_loss_with_gradients(model, x_supervised, y_supervised, x_pde, x_boundary, x_rans, y_rans, 
                               config, ns_op, field_weights, field_stats=None):
    """計算帶梯度的損失函數"""
    # 預測
    u_pred = model(x_supervised)
    u_boundary = model(x_boundary)
    u_rans_pred = model(x_rans)
    
    # 1. 監督學習損失（可能包含標準化）
    if field_stats is not None:
        # 標準化情況下的監督損失
        supervised_loss_components = []
        for i, field in enumerate(['u', 'v', 'p']):
            field_diff = (u_pred[:, i] - y_supervised[:, i]) ** 2
            supervised_loss_components.append(field_weights[i] * torch.mean(field_diff))
    else:
        # 非標準化情況下的監督損失  
        supervised_loss_components = []
        for i in range(3):
            field_diff = (u_pred[:, i] - y_supervised[:, i]) ** 2
            supervised_loss_components.append(field_weights[i] * torch.mean(field_diff))
    
    supervised_loss = config['loss_weights']['supervised'] * sum(supervised_loss_components)
    
    # 2. PDE殘差損失
    u_pde_pred = model(x_pde)
    velocity_pred = u_pde_pred[:, :2]  # u, v
    pressure_pred = u_pde_pred[:, 2:3]  # p
    residual_dict = ns_op.residual(x_pde, velocity_pred, pressure_pred)
    pde_loss = config['loss_weights']['pde'] * sum(torch.mean(res**2) for res in residual_dict.values())
    
    # 3. RANS一致性損失 
    rans_loss = config['loss_weights']['rans'] * torch.mean((u_rans_pred - y_rans)**2)
    
    # 4. 邊界條件損失
    bc_loss = config['loss_weights']['boundary'] * torch.mean(u_boundary[:, [0,1]]**2)  # 邊界處u,v=0
    
    total_loss = supervised_loss + pde_loss + rans_loss + bc_loss
    
    return {
        'total': total_loss,
        'supervised': supervised_loss,
        'supervised_components': supervised_loss_components,
        'pde': pde_loss,
        'rans': rans_loss,
        'boundary': bc_loss
    }

def main():
    try:
        logger.info("🚀 Starting Final Supervised + RANS Strategy Test")
        logger.info("=" * 70)
        
        # 1. 配置參數
        config = {
            'name': 'Final Supervised+RANS Strategy',
            'Re': 1000,
            'supervised_points': 60,
            'rans_points': 120,
            'resolution': 'medium',
            'noise_level': 0.01,
            'width': 128,
            'depth': 6, 
            'lr': 0.001,
            'epochs': 400,
            'early_stop_patience': 80,
            'rans_weight': 0.3,
            'loss_weights': {
                'supervised': 15.0,
                'pde': 1.0,
                'rans': 0.8,
                'boundary': 5.0
            },
            'use_normalization': True,  # 啟用場標準化
            'adaptive_weights': True    # 啟用自適應權重
        }
        
        logger.info(f"Configuration: {config}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # 2. 生成增強Mock數據
        logger.info("Creating Enhanced Mock data...")
        data_loader = ChannelFlowLoader(config_path='configs/channel_flow_re1000.yml')
        mock_data = data_loader.create_enhanced_mock_data(
            resolution=config['resolution'],
            noise_level=config['noise_level']
        )
        
        logger.info(f"Mock data shape: u{mock_data['u'].shape}, v{mock_data['v'].shape}, p{mock_data['p'].shape}")
        
        # 3. 選擇監督學習點（分層採樣）
        logger.info(f"Selecting {config['supervised_points']} supervised learning points...")
        
        total_points = len(mock_data['u'])
        coordinates = mock_data['coordinates']  # 已經是 [N, 2] 格式
        
        # 分層採樣策略
        u_values = mock_data['u']
        v_values = mock_data['v']  
        p_values = mock_data['p']
        
        # 根據u場速度分層
        u_bins = np.percentile(u_values, [0, 25, 50, 75, 100])
        supervised_indices = []
        
        points_per_bin = config['supervised_points'] // 4
        for i in range(4):
            if i < 3:
                mask = (u_values >= u_bins[i]) & (u_values < u_bins[i+1])
            else:
                mask = (u_values >= u_bins[i]) & (u_values <= u_bins[i+1])
            
            bin_indices = np.where(mask)[0]
            if len(bin_indices) > 0:
                selected_count = min(points_per_bin, len(bin_indices))
                selected_idx = np.random.choice(bin_indices, size=selected_count, replace=False)
                supervised_indices.extend(selected_idx)
        
        if len(supervised_indices) < config['supervised_points']:
            remaining = config['supervised_points'] - len(supervised_indices)
            available_indices = list(set(range(total_points)) - set(supervised_indices))
            additional_indices = np.random.choice(available_indices, size=remaining, replace=False)
            supervised_indices.extend(additional_indices)
        
        supervised_indices = np.array(supervised_indices[:config['supervised_points']])
        
        supervised_points = coordinates[supervised_indices]
        supervised_data = {
            'u': mock_data['u'][supervised_indices],
            'v': mock_data['v'][supervised_indices], 
            'p': mock_data['p'][supervised_indices]
        }
        
        logger.info(f"Selected {len(supervised_points)} supervised points with stratified sampling")
        
        # 4. 創建改進的RANS先驗數據
        logger.info("Creating improved RANS prior data...")
        rans_prior = data_loader._create_mock_prior()
        
        # 生成坐標網格
        X, Y = np.meshgrid(rans_prior.coordinates['x'], rans_prior.coordinates['y'], indexing='ij')
        rans_coordinates = np.column_stack([X.flatten(), Y.flatten()])
        rans_u = rans_prior.fields['u'].flatten()
        rans_v = rans_prior.fields['v'].flatten() 
        rans_p = rans_prior.fields['p'].flatten()
        
        # 5. 創建邊界條件點
        logger.info("Creating boundary condition points...")
        boundary_points = []
        n_boundary = 40
        x_range = mock_data['x_range']
        y_range = mock_data['y_range']
        
        # 上下邊界
        for x in np.linspace(x_range[0], x_range[1], n_boundary//2):
            boundary_points.append([x, y_range[0]])  # 下邊界
            boundary_points.append([x, y_range[1]])  # 上邊界
        boundary_points = np.array(boundary_points)
        
        # 6. 創建PINN模型
        logger.info("Creating PINN model...")
        model = PINNNet(
            in_dim=2,
            out_dim=3,
            width=config['width'],
            depth=config['depth'],
            activation='tanh'
        ).to(device)
        
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        
        # 7. 準備訓練數據
        logger.info("Preparing training data...")
        
        # 計算場標準化參數（如果啟用）
        if config['use_normalization']:
            field_stats = {
                'u': {'mean': np.mean(mock_data['u']), 'std': np.std(mock_data['u'])},
                'v': {'mean': np.mean(mock_data['v']), 'std': np.std(mock_data['v'])},
                'p': {'mean': np.mean(mock_data['p']), 'std': np.std(mock_data['p'])}
            }
            logger.info(f"Field normalization stats: u(μ={field_stats['u']['mean']:.3f}, σ={field_stats['u']['std']:.3f}), "
                        f"v(μ={field_stats['v']['mean']:.6f}, σ={field_stats['v']['std']:.6f}), "
                        f"p(μ={field_stats['p']['mean']:.3f}, σ={field_stats['p']['std']:.3f})")
            
            # 監督學習點（標準化）
            x_supervised = torch.tensor(supervised_points, dtype=torch.float32).to(device)
            supervised_normalized = np.column_stack([
                (supervised_data['u'] - field_stats['u']['mean']) / field_stats['u']['std'],
                (supervised_data['v'] - field_stats['v']['mean']) / field_stats['v']['std'],
                (supervised_data['p'] - field_stats['p']['mean']) / field_stats['p']['std']
            ])
            y_supervised = torch.tensor(supervised_normalized, dtype=torch.float32).to(device)
            
            # RANS一致性點（標準化）
            rans_indices = np.random.choice(total_points, size=config['rans_points'], replace=False)
            x_rans = torch.tensor(coordinates[rans_indices], dtype=torch.float32).to(device)
            rans_normalized = np.column_stack([
                (rans_u[rans_indices] - field_stats['u']['mean']) / field_stats['u']['std'],
                (rans_v[rans_indices] - field_stats['v']['mean']) / field_stats['v']['std'],
                (rans_p[rans_indices] - field_stats['p']['mean']) / field_stats['p']['std']
            ])
            y_rans = torch.tensor(rans_normalized, dtype=torch.float32).to(device)
            
        else:
            field_stats = None
            # 監督學習點（非標準化）
            x_supervised = torch.tensor(supervised_points, dtype=torch.float32).to(device)
            y_supervised = torch.tensor(
                np.column_stack([supervised_data['u'], supervised_data['v'], supervised_data['p']]), 
                dtype=torch.float32
            ).to(device)
            
            # RANS一致性點（非標準化）
            rans_indices = np.random.choice(total_points, size=config['rans_points'], replace=False)
            x_rans = torch.tensor(coordinates[rans_indices], dtype=torch.float32).to(device)
            y_rans = torch.tensor(
                np.column_stack([
                    rans_u[rans_indices],
                    rans_v[rans_indices], 
                    rans_p[rans_indices]
                ]), 
                dtype=torch.float32
            ).to(device)
        
        # PDE殘差評估點
        x_range = np.linspace(mock_data['x_range'][0], mock_data['x_range'][1], 40)
        y_range = np.linspace(mock_data['y_range'][0], mock_data['y_range'][1], 20)
        x_grid, y_grid = np.meshgrid(x_range, y_range)
        pde_points = np.column_stack([x_grid.flatten(), y_grid.flatten()])
        x_pde = torch.tensor(pde_points, dtype=torch.float32, requires_grad=True).to(device)
        
        # 邊界條件點
        x_boundary = torch.tensor(boundary_points, dtype=torch.float32).to(device)
        
        # 計算自適應權重（如果啟用）
        if config['adaptive_weights']:
            if field_stats is not None:
                # 標準化情況下使用平衡權重
                field_weights = torch.tensor([1.0, 1.0, 1.0], device=device)
            else:
                # 非標準化情況下使用尺度調整權重
                u_scale = np.abs(np.mean(mock_data['u']))
                v_scale = np.abs(np.mean(mock_data['v'])) + 1e-8  # 避免除零
                p_scale = np.abs(np.mean(mock_data['p']))
                v_weight = u_scale / v_scale if v_scale > 1e-6 else 1000.0
                field_weights = torch.tensor([1.0, min(v_weight, 1000.0), 1.0], device=device)
        else:
            field_weights = torch.tensor([1.0, 1.0, 1.0], device=device)
        
        logger.info(f"Field weights: {field_weights.cpu().numpy()}")
        
        # 8. 創建Navier-Stokes算子
        viscosity = 1.0 / config['Re']  # 將Re轉換為viscosity
        ns_op = NSEquations2D(viscosity=viscosity)
        
        # 9. 訓練設置
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.7)
        
        # 10. 訓練迴圈
        logger.info("Starting training with Final Supervised + RANS strategy...")
        start_time = time.time()
        
        best_loss = float('inf')
        patience_count = 0
        
        for epoch in range(config['epochs']):
            model.train()
            optimizer.zero_grad()
            
            # 計算損失
            losses = compute_loss_with_gradients(
                model, x_supervised, y_supervised, x_pde, x_boundary, 
                x_rans, y_rans, config, ns_op, field_weights, field_stats
            )
            
            # 反向傳播
            losses['total'].backward()
            optimizer.step()
            
            # 記錄訓練進度
            if epoch % 50 == 0 or epoch < 10:
                u_loss = losses['supervised_components'][0].item()
                v_loss = losses['supervised_components'][1].item() 
                p_loss = losses['supervised_components'][2].item()
                
                logger.info(f"Epoch {epoch}: Total = {losses['total'].item():.4f}, "
                           f"Supervised = {losses['supervised'].item():.4f} "
                           f"(u:{u_loss:.4f}, v:{v_loss:.4f}, p:{p_loss:.4f}), "
                           f"PDE = {losses['pde'].item():.4f}, "
                           f"RANS = {losses['rans'].item():.4f}, "
                           f"BC = {losses['boundary'].item():.4f}")
            
            # 早停機制
            current_loss = losses['total'].item()
            if current_loss < best_loss:
                best_loss = current_loss
                patience_count = 0
            else:
                patience_count += 1
                
            if patience_count >= config['early_stop_patience']:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            scheduler.step(current_loss)
        
        # 11. 評估模型
        logger.info("Evaluating model...")
        model.eval()
        with torch.no_grad():
            # 在評估網格上預測
            eval_points = pde_points[:200]
            x_eval = torch.tensor(eval_points, dtype=torch.float32).to(device)
            pred_eval_raw = model(x_eval).cpu().numpy()
            
            # 反標準化預測結果（如果需要）
            if field_stats is not None:
                pred_eval = np.zeros_like(pred_eval_raw)
                pred_eval[:, 0] = pred_eval_raw[:, 0] * field_stats['u']['std'] + field_stats['u']['mean']  # u
                pred_eval[:, 1] = pred_eval_raw[:, 1] * field_stats['v']['std'] + field_stats['v']['mean']  # v
                pred_eval[:, 2] = pred_eval_raw[:, 2] * field_stats['p']['std'] + field_stats['p']['mean']  # p
            else:
                pred_eval = pred_eval_raw
            
            # 計算真實值
            true_eval = np.zeros_like(pred_eval)
            for i, point in enumerate(eval_points):
                x_idx = np.argmin(np.abs(mock_data['x_coords'] - point[0]))
                y_idx = np.argmin(np.abs(mock_data['y_coords'] - point[1]))
                flat_idx = x_idx * mock_data['ny'] + y_idx
                
                true_eval[i, 0] = mock_data['u'][flat_idx]
                true_eval[i, 1] = mock_data['v'][flat_idx]
                true_eval[i, 2] = mock_data['p'][flat_idx]
            
            # 計算L2誤差
            l2_error = relative_L2(pred_eval, true_eval)
            l2_error_u = relative_L2(pred_eval, true_eval, dim=0)
            l2_error_v = relative_L2(pred_eval, true_eval, dim=1)
            l2_error_p = relative_L2(pred_eval, true_eval, dim=2)
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f}s")
            logger.info(f"Final L2 errors - Overall: {l2_error:.4f}, u: {l2_error_u:.4f}, v: {l2_error_v:.4f}, p: {l2_error_p:.4f}")
        
        # 12. 總結結果  
        logger.info("")
        logger.info("=" * 70)
        logger.info("🎯 Final Supervised + RANS Strategy Results")
        logger.info("=" * 70)
        logger.info(f"Training Time: {training_time:.2f}s ({'✅ OK' if training_time < 30 else '⚠️ SLOW'})")
        
        # 評估成功標準
        time_ok = training_time < 30
        error_ok = l2_error < 0.5  
        excellent_improvement = l2_error < 0.3
        
        if excellent_improvement:
            logger.info(f"L2 Error: {l2_error:.4f} (🎉 EXCELLENT)")
        elif error_ok:
            logger.info(f"L2 Error: {l2_error:.4f} (✅ GOOD)")
        else:
            logger.info(f"L2 Error: {l2_error:.4f} (❌ HIGH)")
        
        # 各場評估
        logger.info(f"u field L2: {l2_error_u:.4f} ({'✅' if l2_error_u < 0.5 else '❌'})")
        logger.info(f"v field L2: {l2_error_v:.4f} ({'✅' if l2_error_v < 1.0 else '❌'})")  # v場容忍度高一些
        logger.info(f"p field L2: {l2_error_p:.4f} ({'✅' if l2_error_p < 1.0 else '❌'})")
        
        if excellent_improvement:
            logger.info("🎉 EXCELLENT! Strategy achieved target performance")
        elif error_ok:
            logger.info("📈 GOOD! Notable improvement from previous results")
        else:
            logger.info("❌ Strategy needs further refinement")
            
        # 改善比較
        previous_error = 0.6381  # 尺度權重結果
        improvement_factor = previous_error / l2_error if l2_error > 0 else float('inf')
        logger.info(f"Improvement over scale-weight result ({previous_error:.4f}): {improvement_factor:.2f}x better")
        
        return error_ok or excellent_improvement
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)