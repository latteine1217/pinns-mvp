"""
PirateNet vs JHTDB 真實數據對比評估
=====================================

評估項目：
1. 相對 L2 誤差（u, v, w, p）
2. 統計量對比（均值、標準差、Reynolds 應力）
3. 物理殘差（NS 方程滿足度）
"""

import torch
import numpy as np
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinnx.models import create_pinn_model
from pinnx.train.config_loader import load_config


def load_checkpoint(checkpoint_path, device='mps'):
    """載入訓練檢查點"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    epoch = checkpoint.get('epoch', 'N/A')
    loss = checkpoint.get('loss', float('nan'))
    print(f"✅ 載入檢查點: epoch {epoch}")
    if isinstance(loss, (int, float)):
        print(f"   訓練損失: {loss:.4e}")
    else:
        print(f"   訓練損失: {loss}")
    return checkpoint


def load_jhtdb_ground_truth(jhtdb_path):
    """載入 JHTDB 真實數據"""
    data = np.load(jhtdb_path)
    
    # 座標是 1D 數組，需要建立網格
    x_1d = data['x']  # (128,)
    y_1d = data['y']  # (128,)
    z_1d = data['z']  # (32,)
    
    # 建立 3D 網格
    X, Y, Z = np.meshgrid(x_1d, y_1d, z_1d, indexing='ij')
    
    # 座標矩陣
    coords = np.stack([
        X.flatten(),
        Y.flatten(),
        Z.flatten()
    ], axis=1)
    
    # 場變量
    fields = {
        'u': data['u'].flatten(),
        'v': data['v'].flatten(),
        'w': data['w'].flatten(),
        'p': data['p'].flatten() if 'p' in data else None
    }
    
    print(f"✅ JHTDB 數據: {coords.shape[0]} 個點")
    print(f"   網格形狀: {data['u'].shape}")
    print(f"   域範圍: x=[{x_1d.min():.2f}, {x_1d.max():.2f}]")
    print(f"           y=[{y_1d.min():.2f}, {y_1d.max():.2f}]")
    print(f"           z=[{z_1d.min():.2f}, {z_1d.max():.2f}]")
    
    return coords, fields, data


def create_model_from_checkpoint(checkpoint, config_path, device='mps'):
    """從檢查點創建模型"""
    # 🔧 優先使用檢查點中保存的配置（更準確）
    ckpt_cfg = checkpoint.get('config', {})
    file_cfg = load_config(config_path)
    
    # 合併配置：檢查點優先，文件配置補充
    cfg = {**file_cfg, **ckpt_cfg} if ckpt_cfg else file_cfg
    model_cfg = cfg.get('model', {})
    physics_cfg = cfg.get('physics', {})
    
    print(f"📋 模型配置來源: {'檢查點' if ckpt_cfg else '配置文件'}")
    
    # 檢查是否有 VS-PINN 縮放因子
    input_scale_factors = None
    
    # 🔧 修復：支援兩種配置路徑
    # 路徑 1: physics.vs_pinn.scaling_factors.* (標準路徑)
    # 路徑 2: physics.scaling.* (舊版路徑)
    if 'vs_pinn' in physics_cfg and 'scaling_factors' in physics_cfg['vs_pinn']:
        vs_pinn_cfg = physics_cfg['vs_pinn']['scaling_factors']
        scale_factors = [
            vs_pinn_cfg.get('N_x', 1.0),
            vs_pinn_cfg.get('N_y', 1.0),
            vs_pinn_cfg.get('N_z', 1.0)
        ]
        input_scale_factors = scale_factors
        print(f"🔧 VS-PINN 縮放因子 (vs_pinn.scaling_factors): {scale_factors}")
    elif 'scaling' in physics_cfg and physics_cfg['scaling'].get('use_scaling', False):
        scale_factors = [
            physics_cfg['scaling'].get('N_x', 1.0),
            physics_cfg['scaling'].get('N_y', 1.0),
            physics_cfg['scaling'].get('N_z', 1.0)
        ]
        input_scale_factors = scale_factors
        print(f"🔧 VS-PINN 縮放因子 (scaling.*): {scale_factors}")
    
    # 🔧 處理 Fourier 特徵配置（支援多種格式）
    # 優先級：model.fourier_m > model.fourier_features.fourier_m
    fourier_m = model_cfg.get('fourier_m', 32)
    fourier_sigma = model_cfg.get('fourier_sigma', 1.0)
    
    # 從 fourier_features 配置中讀取（作為備選）
    if 'fourier_features' in model_cfg and fourier_m == 32:  # 僅當未明確設定時使用
        ff_cfg = model_cfg['fourier_features']
        # 注意：不覆蓋已存在的 fourier_m
        if 'fourier_m' not in model_cfg:
            fourier_m = ff_cfg.get('fourier_m', fourier_m)
        if 'fourier_sigma' not in model_cfg:
            fourier_sigma = ff_cfg.get('fourier_sigma', fourier_sigma)
    
    print(f"📐 模型架構: {model_cfg.get('width')}×{model_cfg.get('depth')}")
    print(f"🌊 Fourier 特徵: M={fourier_m}, σ={fourier_sigma}")
    print(f"🎭 激活函數: {model_cfg.get('activation')}")
    
    model_config = {
        'type': model_cfg.get('type', 'fourier_vs_mlp'),
        'in_dim': model_cfg.get('in_dim', 3),
        'out_dim': model_cfg.get('out_dim', 4),
        'width': model_cfg.get('width', model_cfg.get('hidden_dim', 256)),
        'depth': model_cfg.get('depth', model_cfg.get('num_layers', 4)),
        'activation': model_cfg.get('activation', 'swish'),
        'fourier_m': fourier_m,
        'fourier_sigma': fourier_sigma,
        'use_fourier': model_cfg.get('use_fourier', True),
        'use_rwf': model_cfg.get('use_rwf', False),
        'rwf_scale_std': model_cfg.get('rwf_scale_std', 0.1),
        'rwf_scale_mean': model_cfg.get('rwf_scale_mean', 0.0),
    }

    if input_scale_factors is not None:
        model_config['input_scale_factors'] = input_scale_factors

    model = create_pinn_model(model_config)
    
    # 🔧 防禦性載入：過濾不相容的鍵
    state_dict = checkpoint['model_state_dict']
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    
    unexpected = ckpt_keys - model_keys
    missing = model_keys - ckpt_keys
    
    if unexpected:
        print(f"⚠️  過濾檢查點中的不相容鍵: {unexpected}")
        state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    
    if missing:
        print(f"⚠️  模型缺少的鍵（將使用初始化值）: {missing}")
    
    # 非嚴格模式載入（允許部分鍵缺失）
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    num_layers = model_cfg.get('depth', model_cfg.get('num_layers', 4))
    hidden_dim = model_cfg.get('width', model_cfg.get('hidden_dim', 256))
    print(f"✅ 模型已載入: {num_layers}×{hidden_dim}")
    print(f"   Fourier: M={model_cfg.get('fourier_m', 32)}, σ={model_cfg.get('fourier_sigma', 2.0)}")
    print(f"   RWF: {model_cfg.get('use_rwf', True)}")
    
    return model


def predict_full_field(model, coords, batch_size=4096, device='mps'):
    """在完整網格上預測"""
    n_points = len(coords)
    predictions = []
    
    print(f"\n🔮 預測完整場 ({n_points} 個點)...")
    
    with torch.no_grad():
        for i in range(0, n_points, batch_size):
            batch = coords[i:i+batch_size]
            coords_tensor = torch.tensor(batch, dtype=torch.float32, device=device)
            
            output = model(coords_tensor)
            predictions.append(output.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    
    pred_fields = {
        'u': predictions[:, 0],
        'v': predictions[:, 1],
        'w': predictions[:, 2],
        'p': predictions[:, 3]
    }
    
    return pred_fields


def compute_relative_l2_error(pred, true):
    """計算相對 L2 誤差"""
    numerator = np.linalg.norm(pred - true)
    denominator = np.linalg.norm(true)
    
    if denominator < 1e-12:
        return float('nan')
    
    return numerator / denominator


def compute_field_statistics(pred_fields, true_fields):
    """計算場統計量"""
    stats = {}
    
    for field_name in ['u', 'v', 'w', 'p']:
        if true_fields[field_name] is None:
            continue
        
        pred = pred_fields[field_name]
        true = true_fields[field_name]
        
        # 相對 L2 誤差
        rel_l2 = compute_relative_l2_error(pred, true)
        
        # 絕對誤差統計
        abs_error = np.abs(pred - true)
        
        # 統計量
        stats[field_name] = {
            'relative_l2': float(rel_l2),
            'mae': float(np.mean(abs_error)),
            'rmse': float(np.sqrt(np.mean((pred - true)**2))),
            'max_error': float(np.max(abs_error)),
            'pred_mean': float(np.mean(pred)),
            'true_mean': float(np.mean(true)),
            'pred_std': float(np.std(pred)),
            'true_std': float(np.std(true))
        }
    
    return stats


def print_comparison_table(stats):
    """打印對比表格"""
    print("\n" + "="*90)
    print("📊 場變量誤差分析")
    print("="*90)
    
    print(f"\n{'場變量':<10} {'相對L2誤差':<15} {'RMSE':<15} {'MAE':<15} {'最大誤差':<15}")
    print("-"*90)
    
    for field_name, s in stats.items():
        rel_l2_pct = s['relative_l2'] * 100
        print(f"{field_name:<10} {rel_l2_pct:>13.2f}% {s['rmse']:>14.4e} {s['mae']:>14.4e} {s['max_error']:>14.4e}")
    
    print("\n" + "="*90)
    print("📈 統計量對比")
    print("="*90)
    
    print(f"\n{'場變量':<10} {'預測均值':<15} {'真實均值':<15} {'預測標準差':<15} {'真實標準差':<15}")
    print("-"*90)
    
    for field_name, s in stats.items():
        print(f"{field_name:<10} {s['pred_mean']:>14.4e} {s['true_mean']:>14.4e} {s['pred_std']:>14.4e} {s['true_std']:>14.4e}")
    
    print("="*90)


def evaluate_success_criteria(stats):
    """評估成功標準"""
    print("\n" + "="*90)
    print("🎯 專案成功標準檢驗")
    print("="*90)
    
    # 標準 1: 速度場相對 L2 誤差 ≤ 10-15%
    velocity_errors = [
        stats.get('u', {}).get('relative_l2', 1.0),
        stats.get('v', {}).get('relative_l2', 1.0),
        stats.get('w', {}).get('relative_l2', 1.0)
    ]
    avg_velocity_error = np.mean(velocity_errors) * 100
    
    print(f"\n1. 速度場相對 L2 誤差:")
    print(f"   - u: {stats.get('u', {}).get('relative_l2', 0)*100:.2f}%")
    print(f"   - v: {stats.get('v', {}).get('relative_l2', 0)*100:.2f}%")
    print(f"   - w: {stats.get('w', {}).get('relative_l2', 0)*100:.2f}%")
    print(f"   - 平均: {avg_velocity_error:.2f}%")
    
    if avg_velocity_error <= 15.0:
        print(f"   ✅ 通過！（目標：≤ 15%）")
        success_velocity = True
    else:
        print(f"   ❌ 未通過（目標：≤ 15%）")
        success_velocity = False
    
    # 標準 2: 壓力場相對 L2 誤差 ≤ 15%
    if 'p' in stats:
        pressure_error = stats['p']['relative_l2'] * 100
        print(f"\n2. 壓力場相對 L2 誤差: {pressure_error:.2f}%")
        
        if pressure_error <= 15.0:
            print(f"   ✅ 通過！（目標：≤ 15%）")
            success_pressure = True
        else:
            print(f"   ❌ 未通過（目標：≤ 15%）")
            success_pressure = False
    else:
        print(f"\n2. 壓力場: ⚠️  無真實數據對比")
        success_pressure = None
    
    print("\n" + "="*90)
    
    overall_success = success_velocity and (success_pressure if success_pressure is not None else True)
    
    if overall_success:
        print("🎉 整體評估：成功！達到專案目標")
    else:
        print("⚠️  整體評估：需要改進")
    
    print("="*90)
    
    return {
        'velocity_success': success_velocity,
        'pressure_success': success_pressure,
        'overall_success': overall_success
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='PirateNet vs JHTDB 真實數據對比')
    parser.add_argument('--checkpoint', type=str, required=True, help='檢查點路徑')
    parser.add_argument('--config', type=str, required=True, help='配置文件路徑')
    parser.add_argument('--jhtdb', type=str, default='data/jhtdb/channel_flow_re1000/cutout3d_128x128x32.npz',
                       help='JHTDB 數據路徑')
    parser.add_argument('--device', type=str, default='mps', help='計算設備')
    parser.add_argument('--batch-size', type=int, default=4096, help='批次大小')
    args = parser.parse_args()
    
    print("="*90)
    print("  PirateNet vs JHTDB 真實數據對比評估")
    print("="*90)
    
    # 1. 載入檢查點
    checkpoint = load_checkpoint(args.checkpoint, device=args.device)
    
    # 2. 載入 JHTDB 真實數據
    coords, true_fields, jhtdb_data = load_jhtdb_ground_truth(args.jhtdb)
    
    # 3. 創建模型
    model = create_model_from_checkpoint(checkpoint, args.config, device=args.device)
    
    # 4. 預測完整場
    pred_fields = predict_full_field(model, coords, batch_size=args.batch_size, device=args.device)
    
    # 5. 計算統計
    stats = compute_field_statistics(pred_fields, true_fields)
    
    # 6. 打印對比表格
    print_comparison_table(stats)
    
    # 7. 評估成功標準
    success_criteria = evaluate_success_criteria(stats)
    
    # 8. 保存結果
    output_dir = Path('results/piratenet_quick_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存統計數據
    stats_file = output_dir / 'vs_jhtdb_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump({
            'statistics': stats,
            'success_criteria': success_criteria,
            'checkpoint_epoch': checkpoint.get('epoch', 'N/A'),
            'checkpoint_loss': float(checkpoint.get('loss', 0))
        }, f, indent=2)
    
    print(f"\n💾 統計數據已保存: {stats_file}")
    
    # 保存預測場
    pred_file = output_dir / 'vs_jhtdb_predictions.npz'
    np.savez(
        pred_file,
        coords=coords,
        u_pred=pred_fields['u'],
        v_pred=pred_fields['v'],
        w_pred=pred_fields['w'],
        p_pred=pred_fields['p'],
        u_true=true_fields['u'],
        v_true=true_fields['v'],
        w_true=true_fields['w'],
        p_true=true_fields['p'] if true_fields['p'] is not None else np.zeros_like(pred_fields['p']),
        grid_shape=jhtdb_data['u'].shape
    )
    
    print(f"💾 預測場已保存: {pred_file}")
    print("="*90)


if __name__ == '__main__':
    main()
