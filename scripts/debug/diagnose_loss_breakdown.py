#!/usr/bin/env python
"""
診斷訓練損失分解：分析各檢查點的損失組成
目標：理解為何損失爆炸點反而誤差更低
"""
import sys
import torch
import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

# 添加項目路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pinnx.models.fourier_mlp import PINNNet
from pinnx.models.wrappers import ManualScalingWrapper

def load_model_from_checkpoint(ckpt_path, config_path):
    """從檢查點載入模型"""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # 建立模型
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model_cfg = ckpt.get('config', {}).get('model', cfg.get('model', {}))
    
    model = PINNNet(
        in_dim=model_cfg.get('in_dim', 2),
        out_dim=model_cfg.get('out_dim', 3),
        width=model_cfg.get('width', 128),
        depth=model_cfg.get('depth', 6),
        fourier_m=model_cfg.get('fourier_m', 32),
        fourier_sigma=model_cfg.get('fourier_sigma', 1.0)
    ).to(device)
    
    # 載入權重
    state_dict = ckpt['model_state_dict']
    has_scaling_buffers = any(k in state_dict for k in ['input_min', 'input_max'])
    
    if has_scaling_buffers:
        in_dim = model_cfg.get('in_dim', 2)
        out_dim = model_cfg.get('out_dim', 3)
        in_ranges = {f'in_{i}': (0.0, 1.0) for i in range(in_dim)}
        out_ranges = {f'out_{i}': (0.0, 1.0) for i in range(out_dim)}
        
        wrapper = ManualScalingWrapper(base_model=model, input_ranges=in_ranges, output_ranges=out_ranges).to(device)
        
        has_base_prefix = any(k.startswith('base_model.') for k in state_dict.keys())
        if not has_base_prefix:
            mapped_state = {}
            for k, v in state_dict.items():
                if k in ['input_min', 'input_max', 'output_min', 'output_max']:
                    mapped_state[k] = v
                else:
                    mapped_state[f'base_model.{k}'] = v
            state_dict = mapped_state
        
        wrapper.load_state_dict(state_dict, strict=False)
        model = wrapper
    else:
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    return model, device, cfg

def compute_loss_components(model, data, cfg, device):
    """計算各損失分量"""
    losses = {}
    
    # 獲取物理參數
    physics_cfg = cfg.get('physics', {})
    nu = physics_cfg.get('nu', 1.0e-3)
    
    # 準備數據 - 處理網格格式
    x_1d = torch.from_numpy(data['x']).float().to(device)  # (Nx,)
    y_1d = torch.from_numpy(data['y']).float().to(device)  # (Ny,)
    
    # 創建網格並展平為散點
    X, Y = torch.meshgrid(x_1d, y_1d, indexing='ij')  # (Nx, Ny)
    x_flat = X.reshape(-1)  # (Nx*Ny,)
    y_flat = Y.reshape(-1)  # (Nx*Ny,)
    
    u_true = torch.from_numpy(data['u']).float().to(device).reshape(-1)  # (Nx*Ny,)
    v_true = torch.from_numpy(data['v']).float().to(device).reshape(-1)
    p_true = torch.from_numpy(data['p']).float().to(device).reshape(-1)
    
    # 1. 數據損失（不需要梯度）
    xy_no_grad = torch.stack([x_flat, y_flat], dim=-1)  # (Nx*Ny, 2)
    with torch.no_grad():
        pred = model(xy_no_grad)
    
    u_pred = pred[:, 0]
    v_pred = pred[:, 1]
    p_pred = pred[:, 2] if pred.shape[1] > 2 else torch.zeros_like(u_pred)
    
    losses['data_u'] = torch.mean((u_pred - u_true) ** 2).item()
    losses['data_v'] = torch.mean((v_pred - v_true) ** 2).item()
    losses['data_p'] = torch.mean((p_pred - p_true) ** 2).item()
    losses['data_total'] = losses['data_u'] + losses['data_v'] + losses['data_p']
    
    # 2. PDE 殘差（需要梯度）
    xy = torch.stack([x_flat, y_flat], dim=-1)
    xy.requires_grad_(True)
    
    pred = model(xy)
    u = pred[:, 0]
    v = pred[:, 1]
    p = pred[:, 2] if pred.shape[1] > 2 else torch.zeros_like(u)
    
    # 計算一階導數
    u_grads = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
    u_x = u_grads[:, 0]
    u_y = u_grads[:, 1]
    
    v_grads = torch.autograd.grad(v.sum(), xy, create_graph=True)[0]
    v_x = v_grads[:, 0]
    v_y = v_grads[:, 1]
    
    p_grads = torch.autograd.grad(p.sum(), xy, create_graph=True)[0]
    p_x = p_grads[:, 0]
    p_y = p_grads[:, 1]
    
    # 計算二階導數
    u_xx = torch.autograd.grad(u_x.sum(), xy, create_graph=True)[0][:, 0]
    u_yy = torch.autograd.grad(u_y.sum(), xy, create_graph=True)[0][:, 1]
    v_xx = torch.autograd.grad(v_x.sum(), xy, create_graph=True)[0][:, 0]
    v_yy = torch.autograd.grad(v_y.sum(), xy, create_graph=True)[0][:, 1]
    
    # NS 方程殘差
    momentum_x = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    momentum_y = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    continuity = u_x + v_y
    
    losses['pde_momentum_x'] = torch.mean(momentum_x ** 2).item()
    losses['pde_momentum_y'] = torch.mean(momentum_y ** 2).item()
    losses['pde_continuity'] = torch.mean(continuity ** 2).item()
    losses['pde_total'] = losses['pde_momentum_x'] + losses['pde_momentum_y'] + losses['pde_continuity']
    
    # 3. 邊界條件（壁面）- 簡化版本，使用數據點中最接近壁面的點
    wall_threshold = 0.05  # 距離壁面 0.05 範圍內
    top_wall_mask = (y_flat > 1.0 - wall_threshold).cpu().numpy()
    bot_wall_mask = (y_flat < -1.0 + wall_threshold).cpu().numpy()
    
    with torch.no_grad():
        if top_wall_mask.sum() > 0:
            u_wall_top = u_pred[top_wall_mask]
            v_wall_top = v_pred[top_wall_mask]
            bc_u_top = torch.mean(u_wall_top ** 2).item()
            bc_v_top = torch.mean(v_wall_top ** 2).item()
        else:
            bc_u_top = bc_v_top = 0.0
        
        if bot_wall_mask.sum() > 0:
            u_wall_bot = u_pred[bot_wall_mask]
            v_wall_bot = v_pred[bot_wall_mask]
            bc_u_bot = torch.mean(u_wall_bot ** 2).item()
            bc_v_bot = torch.mean(v_wall_bot ** 2).item()
        else:
            bc_u_bot = bc_v_bot = 0.0
    
    losses['bc_wall_u'] = bc_u_top + bc_u_bot
    losses['bc_wall_v'] = bc_v_top + bc_v_bot
    losses['bc_total'] = losses['bc_wall_u'] + losses['bc_wall_v']
    
    return losses

def diagnose_checkpoints(checkpoint_epochs, config_path, data_path):
    """診斷多個檢查點"""
    print("=" * 80)
    print("  損失分解診斷")
    print("=" * 80)
    
    # 載入數據
    data = np.load(data_path)
    data_dict = {
        'x': data['x'],
        'y': data['y'],
        'u': data['u'],
        'v': data['v'],
        'p': data['p']
    }
    print(f"\n✅ 載入數據: {data_path}")
    print(f"   數據點數: {len(data_dict['x'])}")
    
    results = {}
    
    for epoch in checkpoint_epochs:
        ckpt_path = f"checkpoints/curriculum_adam_baseline_epoch_{epoch}.pth"
        print(f"\n{'='*80}")
        print(f"📊 診斷 Epoch {epoch}")
        print(f"{'='*80}")
        
        # 載入模型
        model, device, cfg = load_model_from_checkpoint(ckpt_path, config_path)
        
        # 計算損失分量
        losses = compute_loss_components(model, data_dict, cfg, device)
        results[epoch] = losses
        
        # 顯示結果
        print(f"\n📈 損失分量（原始值）:")
        print(f"  【數據損失】 Total: {losses['data_total']:.4f}")
        print(f"    ├─ U: {losses['data_u']:.4f}")
        print(f"    ├─ V: {losses['data_v']:.4f}")
        print(f"    └─ P: {losses['data_p']:.4f}")
        
        print(f"\n  【PDE 殘差】 Total: {losses['pde_total']:.4f}")
        print(f"    ├─ Momentum X: {losses['pde_momentum_x']:.4f}")
        print(f"    ├─ Momentum Y: {losses['pde_momentum_y']:.4f}")
        print(f"    └─ Continuity: {losses['pde_continuity']:.4f}")
        
        print(f"\n  【邊界條件】 Total: {losses['bc_total']:.4f}")
        print(f"    ├─ Wall U: {losses['bc_wall_u']:.4f}")
        print(f"    └─ Wall V: {losses['bc_wall_v']:.4f}")
        
        # 應用配置權重
        loss_cfg = cfg.get('losses', {})
        w_data = loss_cfg.get('data_weight', 1.0)
        w_mom_x = loss_cfg.get('momentum_x_weight', 1.0)
        w_mom_y = loss_cfg.get('momentum_y_weight', 1.0)
        w_cont = loss_cfg.get('continuity_weight', 2.0)
        w_bc = loss_cfg.get('wall_constraint_weight', 5.0)
        
        weighted_data = w_data * losses['data_total']
        weighted_pde = (w_mom_x * losses['pde_momentum_x'] + 
                       w_mom_y * losses['pde_momentum_y'] + 
                       w_cont * losses['pde_continuity'])
        weighted_bc = w_bc * losses['bc_total']
        total_weighted = weighted_data + weighted_pde + weighted_bc
        
        print(f"\n📊 加權損失（配置權重）:")
        print(f"  數據損失 (×{w_data}):     {weighted_data:8.4f}  ({weighted_data/total_weighted*100:.1f}%)")
        print(f"  PDE 殘差 (×混合):         {weighted_pde:8.4f}  ({weighted_pde/total_weighted*100:.1f}%)")
        print(f"  邊界條件 (×{w_bc}):     {weighted_bc:8.4f}  ({weighted_bc/total_weighted*100:.1f}%)")
        print(f"  {'─'*40}")
        print(f"  總加權損失:              {total_weighted:8.4f}")
        
        results[epoch]['weighted'] = {
            'data': weighted_data,
            'pde': weighted_pde,
            'bc': weighted_bc,
            'total': total_weighted
        }
    
    # 繪製對比圖
    plot_loss_comparison(results)
    
    return results

def plot_loss_comparison(results):
    """繪製損失對比圖"""
    epochs = sorted(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 原始損失分量對比
    ax = axes[0, 0]
    data_losses = [results[e]['data_total'] for e in epochs]
    pde_losses = [results[e]['pde_total'] for e in epochs]
    bc_losses = [results[e]['bc_total'] for e in epochs]
    
    x = np.arange(len(epochs))
    width = 0.25
    ax.bar(x - width, data_losses, width, label='Data Loss', alpha=0.8)
    ax.bar(x, pde_losses, width, label='PDE Residual', alpha=0.8)
    ax.bar(x + width, bc_losses, width, label='BC Loss', alpha=0.8)
    ax.set_xlabel('Checkpoint Epoch')
    ax.set_ylabel('Loss Value')
    ax.set_title('Raw Loss Components')
    ax.set_xticks(x)
    ax.set_xticklabels(epochs)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 2. 加權損失對比
    ax = axes[0, 1]
    weighted_data = [results[e]['weighted']['data'] for e in epochs]
    weighted_pde = [results[e]['weighted']['pde'] for e in epochs]
    weighted_bc = [results[e]['weighted']['bc'] for e in epochs]
    
    ax.bar(x - width, weighted_data, width, label='Weighted Data', alpha=0.8)
    ax.bar(x, weighted_pde, width, label='Weighted PDE', alpha=0.8)
    ax.bar(x + width, weighted_bc, width, label='Weighted BC', alpha=0.8)
    ax.set_xlabel('Checkpoint Epoch')
    ax.set_ylabel('Weighted Loss Value')
    ax.set_title('Weighted Loss Components')
    ax.set_xticks(x)
    ax.set_xticklabels(epochs)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 3. 損失佔比（堆疊圖）
    ax = axes[1, 0]
    totals = [results[e]['weighted']['total'] for e in epochs]
    data_ratios = [results[e]['weighted']['data'] / t * 100 for e, t in zip(epochs, totals)]
    pde_ratios = [results[e]['weighted']['pde'] / t * 100 for e, t in zip(epochs, totals)]
    bc_ratios = [results[e]['weighted']['bc'] / t * 100 for e, t in zip(epochs, totals)]
    
    ax.bar(x, data_ratios, width=0.6, label='Data %', alpha=0.8)
    ax.bar(x, pde_ratios, width=0.6, bottom=data_ratios, label='PDE %', alpha=0.8)
    bottom = [d + p for d, p in zip(data_ratios, pde_ratios)]
    ax.bar(x, bc_ratios, width=0.6, bottom=bottom, label='BC %', alpha=0.8)
    ax.set_xlabel('Checkpoint Epoch')
    ax.set_ylabel('Loss Contribution (%)')
    ax.set_title('Loss Composition Ratio')
    ax.set_xticks(x)
    ax.set_xticklabels(epochs)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # 4. 數據損失細分（U/V/P）
    ax = axes[1, 1]
    u_losses = [results[e]['data_u'] for e in epochs]
    v_losses = [results[e]['data_v'] for e in epochs]
    p_losses = [results[e]['data_p'] for e in epochs]
    
    ax.bar(x - width, u_losses, width, label='U Error', alpha=0.8)
    ax.bar(x, v_losses, width, label='V Error', alpha=0.8)
    ax.bar(x + width, p_losses, width, label='P Error', alpha=0.8)
    ax.set_xlabel('Checkpoint Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Data Loss Breakdown (U/V/P)')
    ax.set_xticks(x)
    ax.set_xticklabels(epochs)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = 'results/loss_breakdown_diagnosis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 圖表已保存: {output_path}")
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='診斷訓練損失分解')
    parser.add_argument('--config', type=str, 
                       default='configs/curriculum_adam_vs_soap_adam.yml',
                       help='配置文件路徑')
    parser.add_argument('--data', type=str,
                       default='data/jhtdb/channel_flow_re1000/cutout_128x64.npz',
                       help='評估數據路徑')
    parser.add_argument('--epochs', nargs='+', type=int,
                       default=[7000, 7500, 8000],
                       help='要診斷的檢查點 epoch')
    
    args = parser.parse_args()
    
    results = diagnose_checkpoints(args.epochs, args.config, args.data)
    
    print("\n" + "=" * 80)
    print("  診斷完成")
    print("=" * 80)
