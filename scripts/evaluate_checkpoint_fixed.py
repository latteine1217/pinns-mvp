#!/usr/bin/env python
"""Phase 6B 評估腳本 - 修復版（支援輸出反標準化）"""
import sys
import json
import torch
import numpy as np
import yaml
from pathlib import Path

# 添加項目路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pinnx.train.factory import create_model

def load_checkpoint_and_model(checkpoint_path, config_path, device):
    """載入檢查點並創建模型（包含正確的標準化配置）"""
    print(f"📂 載入檢查點: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 載入配置（優先使用檢查點中的配置）
    if 'config' in checkpoint:
        config = checkpoint['config']
        print("  ✅ 使用檢查點內嵌配置")
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("  ⚠️  檢查點無配置，使用外部配置文件")
    
    # 創建模型（使用 factory 函數）
    model = create_model(config, device)
    
    # 載入權重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        raise KeyError("檢查點中缺少 'model_state_dict'")
    
    model.eval()
    print(f"✅ 模型已載入並設置為評估模式")
    
    return model, config, checkpoint


def get_denormalization_factors(config):
    """從配置中提取反標準化因子"""
    scaling_cfg = config.get('model', {}).get('scaling', {})
    output_norm = scaling_cfg.get('output_norm', None)
    
    if output_norm == 'friction_velocity':
        # 速度標準化為 u_tau 的倍數
        u_tau = config.get('physics', {}).get('channel_flow', {}).get('u_tau', 0.04997)
        velocity_scale = u_tau  # 訓練時：u_real / u_tau → u_normalized
        pressure_scale = u_tau  # 壓力通常也用相同尺度（待確認）
        
        print(f"\n🔧 檢測到輸出標準化配置:")
        print(f"  - output_norm: {output_norm}")
        print(f"  - u_tau: {u_tau}")
        print(f"  - 反標準化因子: {velocity_scale} (速度), {pressure_scale} (壓力)")
        
        return {
            'u': velocity_scale,
            'v': velocity_scale,
            'w': velocity_scale,
            'p': pressure_scale
        }
    else:
        print(f"\n⚠️  未檢測到標準化配置 (output_norm={output_norm})，不進行反標準化")
        return None


def evaluate_checkpoint(checkpoint_path, config_path, data_path, output_dir=None):
    """評估檢查點並計算流場誤差"""
    print("=" * 70)
    print("  Phase 6B 評估腳本 - 修復版")
    print("=" * 70)
    
    # 設備
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  設備: {device}")
    
    # 載入模型
    model, config, checkpoint = load_checkpoint_and_model(checkpoint_path, config_path, device)
    
    # 獲取反標準化因子
    denorm_factors = get_denormalization_factors(config)
    
    # 載入評估資料
    print(f"\n📁 載入評估資料: {data_path}")
    data = np.load(data_path)
    
    # 提取座標和真實場
    x, y, z = data['x'], data['y'], data['z']
    u_true = data['u']  # (Nx, Ny, Nz)
    v_true = data['v']
    w_true = data['w']
    p_true = data['p']
    
    # 建立網格座標
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    
    print(f"  網格尺寸: {u_true.shape}")
    print(f"  總點數: {len(coords):,}")
    print(f"  座標範圍:")
    print(f"    x ∈ [{x.min():.3f}, {x.max():.3f}]")
    print(f"    y ∈ [{y.min():.3f}, {y.max():.3f}]")
    print(f"    z ∈ [{z.min():.3f}, {z.max():.3f}]")
    
    # 預測（分批處理避免記憶體溢出）
    print(f"\n🔮 模型預測...")
    batch_size = 8192
    n_points = len(coords)
    n_batches = (n_points + batch_size - 1) // batch_size
    
    predictions = []
    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_points)
            
            batch_coords = torch.tensor(coords[start_idx:end_idx], 
                                       dtype=torch.float32, device=device)
            batch_pred = model(batch_coords).cpu().numpy()
            predictions.append(batch_pred)
            
            if (i + 1) % 10 == 0 or (i + 1) == n_batches:
                print(f"  進度: {i+1}/{n_batches} 批次")
    
    pred = np.concatenate(predictions, axis=0)
    print(f"✅ 預測完成: {pred.shape}")
    
    # 反標準化預測值
    u_pred_raw = pred[:, 0]
    v_pred_raw = pred[:, 1]
    w_pred_raw = pred[:, 2]
    p_pred_raw = pred[:, 3]
    
    if denorm_factors is not None:
        print(f"\n🔄 執行反標準化...")
        u_pred = u_pred_raw / denorm_factors['u']
        v_pred = v_pred_raw / denorm_factors['v']
        w_pred = w_pred_raw / denorm_factors['w']
        p_pred = p_pred_raw / denorm_factors['p']
        
        print(f"  u: {u_pred_raw.mean():.4f} → {u_pred.mean():.4f} (×{1/denorm_factors['u']:.2f})")
        print(f"  v: {v_pred_raw.mean():.4f} → {v_pred.mean():.4f}")
        print(f"  w: {w_pred_raw.mean():.4f} → {w_pred.mean():.4f}")
        print(f"  p: {p_pred_raw.mean():.4f} → {p_pred.mean():.4f}")
    else:
        u_pred, v_pred, w_pred, p_pred = u_pred_raw, v_pred_raw, w_pred_raw, p_pred_raw
    
    # Reshape 為原始網格形狀
    u_pred = u_pred.reshape(u_true.shape)
    v_pred = v_pred.reshape(v_true.shape)
    w_pred = w_pred.reshape(w_true.shape)
    p_pred = p_pred.reshape(p_true.shape)
    
    # 計算誤差
    def relative_l2_error(pred, true):
        return np.linalg.norm(pred - true) / np.linalg.norm(true)
    
    u_error = relative_l2_error(u_pred, u_true)
    v_error = relative_l2_error(v_pred, v_true)
    w_error = relative_l2_error(w_pred, w_true)
    p_error = relative_l2_error(p_pred, p_true)
    mean_error = (u_error + v_error + w_error) / 3
    
    # 計算逐點誤差統計
    u_pointwise = np.abs(u_pred - u_true)
    v_pointwise = np.abs(v_pred - v_true)
    w_pointwise = np.abs(w_pred - w_true)
    
    # 統計資訊
    results = {
        'checkpoint': str(checkpoint_path),
        'epoch': checkpoint.get('epoch', -1),
        'data_file': str(data_path),
        'grid_size': list(u_true.shape),
        'total_points': int(np.prod(u_true.shape)),
        'denormalization_applied': denorm_factors is not None,
        'denorm_factors': {k: float(v) for k, v in denorm_factors.items()} if denorm_factors else None,
        'relative_l2_errors': {
            'u': float(u_error),
            'v': float(v_error),
            'w': float(w_error),
            'p': float(p_error)
        },
        'mean_l2_error': float(mean_error),
        'pointwise_errors': {
            'u': {
                'median': float(np.median(u_pointwise)),
                'p95': float(np.percentile(u_pointwise, 95)),
                'max': float(np.max(u_pointwise))
            },
            'v': {
                'median': float(np.median(v_pointwise)),
                'p95': float(np.percentile(v_pointwise, 95)),
                'max': float(np.max(v_pointwise))
            },
            'w': {
                'median': float(np.median(w_pointwise)),
                'p95': float(np.percentile(w_pointwise, 95)),
                'max': float(np.max(w_pointwise))
            }
        },
        'predictions_stats': {
            'u': {
                'min': float(u_pred.min()),
                'max': float(u_pred.max()),
                'mean': float(u_pred.mean()),
                'std': float(u_pred.std())
            },
            'v': {
                'min': float(v_pred.min()),
                'max': float(v_pred.max()),
                'mean': float(v_pred.mean()),
                'std': float(v_pred.std())
            },
            'w': {
                'min': float(w_pred.min()),
                'max': float(w_pred.max()),
                'mean': float(w_pred.mean()),
                'std': float(w_pred.std())
            },
            'p': {
                'min': float(p_pred.min()),
                'max': float(p_pred.max()),
                'mean': float(p_pred.mean()),
                'std': float(p_pred.std())
            }
        },
        'true_field_stats': {
            'u': {
                'min': float(u_true.min()),
                'max': float(u_true.max()),
                'mean': float(u_true.mean()),
                'std': float(u_true.std())
            },
            'v': {
                'min': float(v_true.min()),
                'max': float(v_true.max()),
                'mean': float(v_true.mean()),
                'std': float(v_true.std())
            },
            'w': {
                'min': float(w_true.min()),
                'max': float(w_true.max()),
                'mean': float(w_true.mean()),
                'std': float(w_true.std())
            }
        }
    }
    
    # 打印結果
    print("\n" + "=" * 70)
    print("  評估結果")
    print("=" * 70)
    print(f"\n相對 L2 誤差:")
    print(f"  u: {u_error*100:.2f}%  {'✅' if u_error < 0.15 else '❌'} (目標 < 15%)")
    print(f"  v: {v_error*100:.2f}%  {'✅' if v_error < 0.15 else '❌'}")
    print(f"  w: {w_error*100:.2f}%  {'✅' if w_error < 0.15 else '❌'}")
    print(f"  平均: {mean_error*100:.2f}%")
    
    print(f"\n預測值統計:")
    print(f"  u: [{u_pred.min():.3f}, {u_pred.max():.3f}], 均值 {u_pred.mean():.3f}")
    print(f"  v: [{v_pred.min():.3f}, {v_pred.max():.3f}], 均值 {v_pred.mean():.3f}")
    print(f"  w: [{w_pred.min():.3f}, {w_pred.max():.3f}], 均值 {w_pred.mean():.3f}")
    
    print(f"\n真實場統計:")
    print(f"  u: [{u_true.min():.3f}, {u_true.max():.3f}], 均值 {u_true.mean():.3f}")
    print(f"  v: [{v_true.min():.3f}, {v_true.max():.3f}], 均值 {v_true.mean():.3f}")
    print(f"  w: [{w_true.min():.3f}, {w_true.max():.3f}], 均值 {w_true.mean():.3f}")
    
    # 保存結果
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        metrics_file = output_path / 'metrics_fixed.json'
        with open(metrics_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 結果已保存至: {metrics_file}")
    
    print("=" * 70)
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 6B 評估腳本 - 修復版')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='檢查點路徑')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路徑')
    parser.add_argument('--data', type=str, required=True,
                       help='評估資料路徑（3D cutout .npz）')
    parser.add_argument('--output', type=str, default=None,
                       help='輸出目錄（默認不保存）')
    
    args = parser.parse_args()
    
    evaluate_checkpoint(args.checkpoint, args.config, args.data, args.output)
