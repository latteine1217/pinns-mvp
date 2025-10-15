"""
簡單的 PirateNet 評估腳本
用於快速驗證訓練後的模型性能
"""

import torch
import numpy as np
import yaml
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import pinnx
from pinnx.models.fourier_mlp import create_enhanced_pinn
from pinnx.physics.vs_pinn_channel_flow import VSPINNChannelFlow
from pinnx.train.config_loader import load_config

def load_model_and_physics(checkpoint_path, config_path, device='mps'):
    """載入模型和物理求解器"""
    
    # 載入配置
    cfg = load_config(config_path)
    
    # 載入檢查點
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"✅ 載入檢查點: epoch {checkpoint.get('epoch', 'N/A')}")
    print(f"   訓練損失: {checkpoint.get('loss', 'N/A'):.2f}")
    
    # 創建模型
    model_cfg = cfg['model']
    model = create_enhanced_pinn(
        input_dim=3,
        output_dim=4,
        hidden_dim=model_cfg.get('hidden_dim', 256),
        num_layers=model_cfg.get('num_layers', 4),
        activation=model_cfg.get('activation', 'swish'),
        use_fourier=model_cfg.get('use_fourier', True),
        fourier_m=model_cfg.get('fourier_m', 32),
        fourier_sigma=model_cfg.get('fourier_sigma', 2.0),
        use_rwf=model_cfg.get('use_rwf', True),
        rwf_rank=model_cfg.get('rwf_rank', 32)
    )
    
    # 載入權重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"✅ 模型已載入到設備: {device}")
    
    # 創建物理求解器
    physics_cfg = cfg['physics']
    domain = cfg['data']['jhtdb_config']['domain']
    physics = VSPINNChannelFlow(
        domain_x=(domain['x'][0], domain['x'][1]),
        domain_y=(domain['y'][0], domain['y'][1]),
        domain_z=(domain['z'][0], domain['z'][1]),
        nu=physics_cfg.get('nu', 5e-5),
        dpdx=physics_cfg.get('dpdx', 0.0025),
        rho=physics_cfg.get('rho', 1.0),
        N_x=physics_cfg['vs_pinn']['N_x'],
        N_y=physics_cfg['vs_pinn']['N_y'],
        N_z=physics_cfg['vs_pinn']['N_z'],
        enable_loss_norm=physics_cfg.get('enable_loss_norm', True),
        warmup_epochs=physics_cfg.get('warmup_epochs', 5)
    )
    physics.set_device(device)
    print(f"✅ 物理求解器已創建")
    
    return model, physics, cfg

def generate_test_grid(domain, nx=64, ny=32, nz=64):
    """生成測試網格"""
    x = np.linspace(domain['x'][0], domain['x'][1], nx)
    y = np.linspace(domain['y'][0], domain['y'][1], ny)
    z = np.linspace(domain['z'][0], domain['z'][1], nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    coords = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    
    return coords, (nx, ny, nz)

def evaluate_physics_residuals(model, physics, coords, batch_size=4096, device='mps'):
    """評估物理殘差"""
    
    model.eval()
    n_points = len(coords)
    all_residuals = {
        'momentum_x': [],
        'momentum_y': [],
        'momentum_z': [],
        'continuity': []
    }
    
    print(f"\n🔍 評估物理殘差 ({n_points} 個測試點)...")
    
    for i in range(0, n_points, batch_size):
        batch = coords[i:i+batch_size]
        coords_tensor = torch.tensor(batch, dtype=torch.float32, device=device)
        coords_tensor.requires_grad_(True)
        
        # 前向傳播
        output = model(coords_tensor)
        
        # 計算殘差
        residuals = physics.compute_pde_residuals(coords_tensor, output)
        
        # 提取各項殘差
        all_residuals['momentum_x'].append(residuals['momentum_x'].detach().cpu().numpy())
        all_residuals['momentum_y'].append(residuals['momentum_y'].detach().cpu().numpy())
        all_residuals['momentum_z'].append(residuals['momentum_z'].detach().cpu().numpy())
        all_residuals['continuity'].append(residuals['continuity'].detach().cpu().numpy())
    
    # 合併所有批次
    combined = {
        'momentum_x': np.concatenate(all_residuals['momentum_x']),
        'momentum_y': np.concatenate(all_residuals['momentum_y']),
        'momentum_z': np.concatenate(all_residuals['momentum_z']),
        'continuity': np.concatenate(all_residuals['continuity'])
    }
    
    return combined

def compute_statistics(residuals):
    """計算殘差統計"""
    stats = {}
    
    for name, values in residuals.items():
        abs_values = np.abs(values)
        stats[name] = {
            'mean': float(np.mean(abs_values)),
            'std': float(np.std(abs_values)),
            'max': float(np.max(abs_values)),
            'median': float(np.median(abs_values)),
            'rmse': float(np.sqrt(np.mean(values**2)))
        }
    
    return stats

def print_results(stats):
    """打印評估結果"""
    print("\n" + "="*70)
    print("📊 物理殘差統計")
    print("="*70)
    
    print(f"\n{'方程式':<20} {'RMSE':>12} {'Mean':>12} {'Std':>12} {'Max':>12}")
    print("-"*70)
    
    for name, s in stats.items():
        print(f"{name:<20} {s['rmse']:>12.2e} {s['mean']:>12.2e} {s['std']:>12.2e} {s['max']:>12.2e}")
    
    print("="*70)
    
    # 綜合評分
    total_rmse = np.mean([s['rmse'] for s in stats.values()])
    print(f"\n✅ 綜合 RMSE: {total_rmse:.2e}")
    
    # 判定準則
    if total_rmse < 1.0:
        print("🎉 優秀！物理殘差控制良好")
    elif total_rmse < 5.0:
        print("✅ 良好！物理約束基本滿足")
    elif total_rmse < 10.0:
        print("⚠️  尚可，建議繼續訓練")
    else:
        print("❌ 需要改進，物理殘差偏大")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='簡單的 PirateNet 評估')
    parser.add_argument('--checkpoint', type=str, required=True, help='檢查點路徑')
    parser.add_argument('--config', type=str, required=True, help='配置文件路徑')
    parser.add_argument('--device', type=str, default='mps', help='計算設備')
    parser.add_argument('--nx', type=int, default=64, help='X 方向網格數')
    parser.add_argument('--ny', type=int, default=32, help='Y 方向網格數')
    parser.add_argument('--nz', type=int, default=64, help='Z 方向網格數')
    parser.add_argument('--batch-size', type=int, default=4096, help='批次大小')
    args = parser.parse_args()
    
    print("="*70)
    print("  PirateNet 簡易評估工具")
    print("="*70)
    
    # 載入模型
    model, physics, cfg = load_model_and_physics(
        args.checkpoint, 
        args.config, 
        device=args.device
    )
    
    # 生成測試網格
    domain = cfg['data']['jhtdb_config']['domain']
    coords, grid_shape = generate_test_grid(domain, args.nx, args.ny, args.nz)
    print(f"\n📐 測試網格: {grid_shape} = {len(coords)} 個點")
    
    # 評估物理殘差
    residuals = evaluate_physics_residuals(
        model, physics, coords, 
        batch_size=args.batch_size, 
        device=args.device
    )
    
    # 計算統計
    stats = compute_statistics(residuals)
    
    # 打印結果
    print_results(stats)
    
    # 保存結果
    output_dir = Path('results/piratenet_quick_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'physics_residuals.npz'
    np.savez(
        output_file,
        momentum_x=residuals['momentum_x'],
        momentum_y=residuals['momentum_y'],
        momentum_z=residuals['momentum_z'],
        continuity=residuals['continuity'],
        grid_shape=grid_shape,
        coords=coords
    )
    print(f"\n💾 殘差數據已保存: {output_file}")
    
    # 保存統計數據為 JSON
    import json
    stats_file = output_dir / 'physics_residuals_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"💾 統計數據已保存: {stats_file}")
    
    print("="*70)

if __name__ == '__main__':
    main()
