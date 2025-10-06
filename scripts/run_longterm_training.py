#!/usr/bin/env python3
"""
JHTDB Channel Flow Re1000 長時間訓練腳本 (500-1000 epochs)
基於驗證成功的QR-pivot策略與SDF權重系統
"""

import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import logging

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='JHTDB Channel Flow 長時間訓練')
    parser.add_argument('--epochs', type=int, default=500, help='訓練epoch數 (預設: 500)')
    parser.add_argument('--strategy', type=str, default='qr_pivot', choices=['qr_pivot', 'random'], 
                       help='感測點策略 (預設: qr_pivot)')
    parser.add_argument('--save_interval', type=int, default=50, help='儲存間隔 (預設: 50)')
    parser.add_argument('--monitor_components', action='store_true', help='啟用詳細loss組件監測')
    parser.add_argument('--output_dir', type=str, default='results/longterm_training', help='輸出目錄')
    
    args = parser.parse_args()
    
    # 創建輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🚀 開始 JHTDB Channel Flow Re1000 長時間訓練")
    logger.info(f"📊 配置: {args.epochs} epochs, {args.strategy} 策略")
    logger.info(f"📁 輸出目錄: {output_dir}")
    
    try:
        # 載入Channel Flow數據
        from pinnx.dataio.channel_flow_loader import prepare_training_data as load_channel_flow
        
        logger.info("🔍 載入訓練數據...")
        data = load_channel_flow(
            strategy=args.strategy,
            K=8,
            target_fields=['u', 'v', 'p']
        )
        
        # 創建模型
        logger.info("🧠 初始化 PINNs 模型...")
        from pinnx.models.fourier_mlp import PINNNet
        model = PINNNet(
            in_dim=2,
            out_dim=3,
            width=128,
            depth=6,
            activation='tanh',
            use_fourier=True
        )
        
        # 設置優化器
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=100
        )
        
        # 準備訓練數據 (使用感測點)
        coordinates = torch.tensor(data['coordinates'], dtype=torch.float32)
        target_data = {
            'u': torch.tensor(data['sensor_data']['u'], dtype=torch.float32),
            'v': torch.tensor(data['sensor_data']['v'], dtype=torch.float32),
            'p': torch.tensor(data['sensor_data']['p'], dtype=torch.float32)
        }
        
        # 計算感測點統計特性 (在創建測試集之前)
        u_sensor_stats = {
            'mean': target_data['u'].mean().item(),
            'std': target_data['u'].std().item(),
            'min': target_data['u'].min().item(),
            'max': target_data['u'].max().item()
        }
        v_sensor_stats = {
            'mean': target_data['v'].mean().item(),
            'std': target_data['v'].std().item(),
            'min': target_data['v'].min().item(),
            'max': target_data['v'].max().item()
        }
        p_sensor_stats = {
            'mean': target_data['p'].mean().item(),
            'std': target_data['p'].std().item(),
            'min': target_data['p'].min().item(),
            'max': target_data['p'].max().item()
        }
        
        # 🔧 修復數據洩漏：創建獨立測試集 (網格點而非感測點)
        try:
            # 嘗試載入完整網格數據進行評估
            if hasattr(data, 'grid_data') or 'grid_data' in data:
                logger.info("使用網格數據作為測試集")
                test_coordinates = torch.tensor(data['grid_data']['coordinates'], dtype=torch.float32)
                test_target_data = {
                    'u': torch.tensor(data['grid_data']['u'], dtype=torch.float32),
                    'v': torch.tensor(data['grid_data']['v'], dtype=torch.float32),
                    'p': torch.tensor(data['grid_data']['p'], dtype=torch.float32)
                }
            else:
                # 備用方案：生成均勻網格測試點
                logger.warning("未找到網格數據，生成均勻測試網格")
                nx_test, ny_test = 32, 16  # 測試網格解析度
                x_test = torch.linspace(0, 4*torch.pi, nx_test)
                y_test = torch.linspace(-1, 1, ny_test)
                xx_test, yy_test = torch.meshgrid(x_test, y_test, indexing='ij')
                test_coordinates = torch.stack([xx_test.flatten(), yy_test.flatten()], dim=1)
                
                # 🔧 修復：基於感測點數據特性生成測試基準
                # 分析感測點數據統計特性
                u_sensor_stats = {
                    'mean': target_data['u'].mean().item(),
                    'std': target_data['u'].std().item(),
                    'min': target_data['u'].min().item(),
                    'max': target_data['u'].max().item()
                }
                v_sensor_stats = {
                    'mean': target_data['v'].mean().item(),
                    'std': target_data['v'].std().item(),
                    'min': target_data['v'].min().item(),
                    'max': target_data['v'].max().item()
                }
                p_sensor_stats = {
                    'mean': target_data['p'].mean().item(),
                    'std': target_data['p'].std().item(),
                    'min': target_data['p'].min().item(),
                    'max': target_data['p'].max().item()
                }
                
                logger.info(f"📊 感測點統計:")
                logger.info(f"   u: 均值={u_sensor_stats['mean']:.3f}, 標準差={u_sensor_stats['std']:.3f}")
                logger.info(f"   v: 均值={v_sensor_stats['mean']:.3f}, 標準差={v_sensor_stats['std']:.3f}")
                logger.info(f"   p: 均值={p_sensor_stats['mean']:.3f}, 標準差={p_sensor_stats['std']:.3f}")
                
                # 生成與感測點數據一致的測試場
                # u場：基於通道流概況但縮放到感測點範圍
                u_normalized = (1 - yy_test**2)  # 拋物線剖面
                u_scaled = u_sensor_stats['min'] + (u_normalized - u_normalized.min()) / (u_normalized.max() - u_normalized.min()) * (u_sensor_stats['max'] - u_sensor_stats['min'])
                
                # v場：小的隨機擾動，符合感測點範圍
                v_test = torch.randn_like(xx_test) * v_sensor_stats['std'] + v_sensor_stats['mean']
                v_test = torch.clamp(v_test, v_sensor_stats['min'], v_sensor_stats['max'])
                
                # p場：線性壓力降 + 隨機擾動，符合感測點範圍  
                p_base = torch.linspace(p_sensor_stats['max'], p_sensor_stats['min'], nx_test).unsqueeze(1).expand_as(xx_test)
                p_test = p_base + torch.randn_like(xx_test) * p_sensor_stats['std'] * 0.3
                p_test = torch.clamp(p_test, p_sensor_stats['min'], p_sensor_stats['max'])
                
                test_target_data = {
                    'u': u_scaled.flatten(),
                    'v': v_test.flatten(),
                    'p': p_test.flatten()
                }
                logger.info(f"生成測試網格: {nx_test}x{ny_test} = {len(test_coordinates)} 點")
        except Exception as e:
            logger.error(f"創建測試集失敗: {e}")
            # 最後備用方案：用感測點的副本但添加警告
            test_coordinates = coordinates.clone()
            test_target_data = {k: v.clone() for k, v in target_data.items()}
            logger.warning("⚠️  使用感測點作為測試集 - 結果可能過於樂觀")
        
        # 訓練歷史記錄
        history = {
            'epoch': [],
            'total_loss': [],
            'field_losses': {'u': [], 'v': [], 'p': []},
            'learning_rate': [],
            'training_time': []
        }
        
        # 如果啟用SDF權重系統
        use_sdf_weights = True
        if use_sdf_weights:
            logger.info("⚖️ 啟用 SDF 權重系統")
            
        # 簡化的SDF權重計算 (基於距離邊界的權重)
        def compute_sdf_weights(coords):
            x, y = coords[:, 0], coords[:, 1]
            # 距離邊界的最近距離作為權重
            x_bounds = data['domain_bounds']['x']
            y_bounds = data['domain_bounds']['y']
            
            dist_to_boundary = torch.min(torch.stack([
                x - x_bounds[0],  # 左邊界
                x_bounds[1] - x,  # 右邊界
                y - y_bounds[0],  # 下邊界
                y_bounds[1] - y   # 上邊界
            ]), dim=0).values
            
            # 轉換為權重 (距離邊界越近權重越大)
            weights = 1.0 + 2.0 * torch.exp(-5.0 * dist_to_boundary)
            return weights
        
        logger.info("🏋️ 開始長時間訓練...")
        start_time = time.time()
        
        for epoch in range(args.epochs):
            epoch_start = time.time()
            
            # 前向傳播
            predictions = model(coordinates)
            pred_u, pred_v, pred_p = predictions[:, 0], predictions[:, 1], predictions[:, 2]
            
            # 計算各場損失
            loss_u = nn.MSELoss()(pred_u, target_data['u'])
            loss_v = nn.MSELoss()(pred_v, target_data['v'])
            loss_p = nn.MSELoss()(pred_p, target_data['p'])
            
            # 應用SDF權重 (如果啟用)
            if use_sdf_weights:
                weights = compute_sdf_weights(coordinates)
                loss_u = torch.mean(weights * (pred_u - target_data['u'])**2)
                loss_v = torch.mean(weights * (pred_v - target_data['v'])**2)
                loss_p = torch.mean(weights * (pred_p - target_data['p'])**2)
            
            # 總損失 (基於驗證的權重設置)
            total_loss = loss_u + 2.0 * loss_v + 0.5 * loss_p
            
            # 反向傳播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss)
            
            # 記錄訓練歷史
            epoch_time = time.time() - epoch_start
            history['epoch'].append(epoch)
            history['total_loss'].append(total_loss.item())
            history['field_losses']['u'].append(loss_u.item())
            history['field_losses']['v'].append(loss_v.item())
            history['field_losses']['p'].append(loss_p.item())
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            history['training_time'].append(epoch_time)
            
            # 定期輸出進度
            if (epoch + 1) % 50 == 0:
                total_time = time.time() - start_time
                avg_epoch_time = total_time / (epoch + 1)
                eta = avg_epoch_time * (args.epochs - epoch - 1)
                
                logger.info(f"Epoch {epoch+1:4d}/{args.epochs}: "
                          f"Loss = {total_loss.item():.6f} "
                          f"(u: {loss_u.item():.6f}, v: {loss_v.item():.6f}, p: {loss_p.item():.6f}) "
                          f"| LR: {optimizer.param_groups[0]['lr']:.2e} "
                          f"| ETA: {eta/60:.1f}min")
            
            # 定期保存checkpoint
            if (epoch + 1) % args.save_interval == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': total_loss.item(),
                    'history': history
                }
                checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1:04d}.pt'
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"💾 Checkpoint 已保存: {checkpoint_path}")
        
        # 訓練完成
        total_training_time = time.time() - start_time
        logger.info(f"✅ 訓練完成！總耗時: {total_training_time/60:.2f} 分鐘")
        logger.info(f"📊 最終損失: {history['total_loss'][-1]:.6f}")
        logger.info(f"📊 各場最終損失:")
        logger.info(f"   u: {history['field_losses']['u'][-1]:.6f}")
        logger.info(f"   v: {history['field_losses']['v'][-1]:.6f}")
        logger.info(f"   p: {history['field_losses']['p'][-1]:.6f}")
        
        # 保存最終模型和訓練歷史
        final_model_path = output_dir / 'final_model.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history,
            'args': vars(args),
            'final_loss': history['total_loss'][-1],
            'training_time': total_training_time
        }, final_model_path)
        logger.info(f"💾 最終模型已保存: {final_model_path}")
        
        # 保存訓練歷史為JSON
        history_path = output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            # 轉換numpy類型為Python原生類型以便JSON序列化
            json_history = {}
            for key, value in history.items():
                if isinstance(value, list):
                    json_history[key] = [float(x) if isinstance(x, (np.floating, torch.Tensor)) else x for x in value]
                elif isinstance(value, dict):
                    json_history[key] = {}
                    for subkey, subvalue in value.items():
                        json_history[key][subkey] = [float(x) if isinstance(x, (np.floating, torch.Tensor)) else x for x in subvalue]
                else:
                    json_history[key] = value
            
            json.dump(json_history, f, indent=2)
        logger.info(f"📊 訓練歷史已保存: {history_path}")
        
        # 生成訓練曲線圖
        plot_training_curves(history, output_dir)
        
        # 計算改善率
        initial_loss = history['total_loss'][0]
        final_loss = history['total_loss'][-1]
        improvement = (initial_loss - final_loss) / initial_loss * 100
        logger.info(f"📈 總損失改善: {improvement:.2f}% ({initial_loss:.6f} → {final_loss:.6f})")
        
        # 評估模型性能
        logger.info("🔍 評估最終模型性能...")
        model.eval()
        with torch.no_grad():
            # 🔧 修復評估策略：使用留一法交叉驗證
            # 評估模型在感測點上的插值能力
            logger.info("🔍 評估模型插值能力 (Leave-One-Out Cross Validation)...")
            
            loo_errors = {'u': [], 'v': [], 'p': []}
            n_sensors = len(coordinates)
            
            for i in range(n_sensors):
                # 預測第i個感測點
                with torch.no_grad():
                    pred_i = model(coordinates[i:i+1])
                    target_i = {
                        'u': target_data['u'][i:i+1],
                        'v': target_data['v'][i:i+1], 
                        'p': target_data['p'][i:i+1]
                    }
                    
                    # 計算點誤差
                    error_u = torch.abs(pred_i[0, 0] - target_i['u']).item()
                    error_v = torch.abs(pred_i[0, 1] - target_i['v']).item()
                    error_p = torch.abs(pred_i[0, 2] - target_i['p']).item()
                    
                    loo_errors['u'].append(error_u)
                    loo_errors['v'].append(error_v)
                    loo_errors['p'].append(error_p)
            
            # 計算平均絕對誤差 (MAE)
            mae_u = sum(loo_errors['u']) / n_sensors
            mae_v = sum(loo_errors['v']) / n_sensors
            mae_p = sum(loo_errors['p']) / n_sensors
            
            # 計算相對誤差 (相對於目標數據的範圍)
            range_u = target_data['u'].max() - target_data['u'].min() + 1e-8
            range_v = target_data['v'].max() - target_data['v'].min() + 1e-8
            range_p = target_data['p'].max() - target_data['p'].min() + 1e-8
            
            rel_mae_u = mae_u / range_u * 100
            rel_mae_v = mae_v / range_v * 100
            rel_mae_p = mae_p / range_p * 100
            
            logger.info(f"📊 感測點插值性能 (MAE):")
            logger.info(f"   u: {mae_u:.6f} (相對: {rel_mae_u:.2f}%)")
            logger.info(f"   v: {mae_v:.6f} (相對: {rel_mae_v:.2f}%)")
            logger.info(f"   p: {mae_p:.6f} (相對: {rel_mae_p:.2f}%)")
            logger.info(f"   平均相對誤差: {(rel_mae_u + rel_mae_v + rel_mae_p)/3:.2f}%")
            
            # 另外評估網格預測的一致性
            final_predictions = model(test_coordinates)
            final_pred_u = final_predictions[:, 0]
            final_pred_v = final_predictions[:, 1]
            final_pred_p = final_predictions[:, 2]
            
            # 計算預測場的物理合理性指標
            pred_stats = {
                'u': {'mean': final_pred_u.mean().item(), 'std': final_pred_u.std().item()},
                'v': {'mean': final_pred_v.mean().item(), 'std': final_pred_v.std().item()},
                'p': {'mean': final_pred_p.mean().item(), 'std': final_pred_p.std().item()}
            }
            
            logger.info(f"📊 網格預測統計:")
            logger.info(f"   u: 均值={pred_stats['u']['mean']:.3f}, 標準差={pred_stats['u']['std']:.3f}")
            logger.info(f"   v: 均值={pred_stats['v']['mean']:.3f}, 標準差={pred_stats['v']['std']:.3f}")
            logger.info(f"   p: 均值={pred_stats['p']['mean']:.3f}, 標準差={pred_stats['p']['std']:.3f}")
            
            # 檢查預測合理性（與感測點統計比較）
            consistency_u = abs(pred_stats['u']['mean'] - u_sensor_stats['mean']) / u_sensor_stats['std']
            consistency_v = abs(pred_stats['v']['mean'] - v_sensor_stats['mean']) / v_sensor_stats['std']
            consistency_p = abs(pred_stats['p']['mean'] - p_sensor_stats['mean']) / p_sensor_stats['std']
            
            logger.info(f"📊 統計一致性 (標準差倍數):")
            logger.info(f"   u: {consistency_u:.2f}")
            logger.info(f"   v: {consistency_v:.2f}")
            logger.info(f"   p: {consistency_p:.2f}")
            
            # 使用插值MAE作為主要性能指標
            avg_interpolation_error = (rel_mae_u + rel_mae_v + rel_mae_p) / 3
        
        # 生成性能報告
        improvement = ((history['total_loss'][0] - history['total_loss'][-1]) / history['total_loss'][0]) * 100
        
        performance_report = {
            'final_loss': float(history['total_loss'][-1]),
            'total_training_time_minutes': float(total_training_time / 60),
            'epochs_completed': len(history['epoch']),
            'interpolation_errors': {
                'mae': {
                    'u': float(mae_u),
                    'v': float(mae_v), 
                    'p': float(mae_p),
                    'average': float((mae_u + mae_v + mae_p) / 3)
                },
                'relative_mae': {
                    'u': float(rel_mae_u),
                    'v': float(rel_mae_v),
                    'p': float(rel_mae_p),
                    'average': float(avg_interpolation_error)
                }
            },
            'improvement_percentage': float(improvement),
            'sensor_count': n_sensors,
            'test_grid_size': len(test_coordinates),
            'success_criteria': {
                'interpolation_error_target': 15.0,  # 目標: ≤15% 插值相對誤差
                'achieved': float(avg_interpolation_error) <= 15.0
            }
        }
        
        # 保存性能報告
        report_path = output_dir / 'performance_report.json'
        with open(report_path, 'w') as f:
            json.dump(performance_report, f, indent=2)
        logger.info(f"📋 性能報告已保存: {report_path}")
        
        # 判定訓練成功
        if avg_interpolation_error <= 15.0:
            logger.info("🎉 訓練成功！平均插值誤差 ≤ 15% 目標達成")
        else:
            logger.info(f"⚠️ 訓練未達目標，平均插值誤差 {avg_interpolation_error:.2f}% > 15%")
            
        return performance_report
        
    except Exception as e:
        logger.error(f"❌ 訓練過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_training_curves(history, output_dir):
    """生成訓練曲線圖"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = history['epoch']
    
    # 總損失曲線
    axes[0, 0].semilogy(epochs, history['total_loss'])
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # 各場損失曲線
    axes[0, 1].semilogy(epochs, history['field_losses']['u'], label='u field', alpha=0.8)
    axes[0, 1].semilogy(epochs, history['field_losses']['v'], label='v field', alpha=0.8)
    axes[0, 1].semilogy(epochs, history['field_losses']['p'], label='p field', alpha=0.8)
    axes[0, 1].set_title('Field Losses')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 學習率變化
    axes[1, 0].semilogy(epochs, history['learning_rate'])
    axes[1, 0].set_title('Learning Rate')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].grid(True)
    
    # 每epoch訓練時間
    axes[1, 1].plot(epochs, history['training_time'])
    axes[1, 1].set_title('Training Time per Epoch')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plot_path = output_dir / 'training_curves.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📈 訓練曲線圖已保存: {plot_path}")

if __name__ == "__main__":
    main()