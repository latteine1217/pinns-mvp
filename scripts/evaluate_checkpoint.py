#!/usr/bin/env python
"""快速評估檢查點的流場誤差"""
import sys
import torch
import numpy as np
import yaml
from pathlib import Path

# 添加項目路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pinnx.models.fourier_mlp import PINNNet
from pinnx.models.wrappers import ManualScalingWrapper

def load_checkpoint(ckpt_path):
    """載入檢查點"""
    print(f"📂 載入檢查點: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    return checkpoint

def evaluate_model(checkpoint_path, config_path, data_path=None):
    """評估模型"""
    print("=" * 70)
    print("  PINNs Checkpoint 快速評估")
    print("=" * 70)
    
    # 載入配置
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 載入檢查點
    ckpt = load_checkpoint(checkpoint_path)
    
    # 提取訓練資訊
    print("\n📊 訓練資訊：")
    print("-" * 70)
    if 'epoch' in ckpt:
        print(f"  訓練輪數: {ckpt['epoch']}")
    if 'loss' in ckpt:
        print(f"  最終損失: {ckpt['loss']:.6f}")
    if 'config' in ckpt:
        print(f"  配置已嵌入: ✅")
    
    # 建立模型
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 從配置或檢查點中獲取模型參數
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
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        print("⚠️  無法找到模型權重")
        return
    
    # 檢測是否包含 ManualScalingWrapper 的緩衝區
    has_scaling_buffers = any(k in state_dict for k in ['input_min', 'input_max', 'output_min', 'output_max'])
    has_base_prefix = any(k.startswith('base_model.') for k in state_dict.keys())
    
    if has_scaling_buffers:
        print("  檢測到尺度化緩衝區，使用 ManualScalingWrapper...")
        # 建立佔位範圍（會被 state_dict 覆蓋）
        in_dim = model_cfg.get('in_dim', 2)
        out_dim = model_cfg.get('out_dim', 3)
        in_ranges = {f'in_{i}': (0.0, 1.0) for i in range(in_dim)}
        out_ranges = {f'out_{i}': (0.0, 1.0) for i in range(out_dim)}
        
        wrapper = ManualScalingWrapper(base_model=model, input_ranges=in_ranges, output_ranges=out_ranges).to(device)
        
        # 處理鍵映射
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
        print("  ✅ ManualScalingWrapper 載入成功")
    else:
        # 處理可能的 base_model 前綴
        if has_base_prefix:
            state_dict = {k.replace('base_model.', ''): v for k, v in state_dict.items() 
                         if k not in ['input_min', 'input_max', 'output_min', 'output_max']}
        
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"✅ 模型已載入（設備: {device}）")
    
    # 載入測試數據
    if data_path is None:
        data_path = cfg.get('data', {}).get('sensors_cache', 'data/jhtdb/channel_qr_K80_sensors.npz')
    
    print("\n📁 載入測試數據...")
    print("-" * 70)
    
    try:
        data = np.load(data_path, allow_pickle=True)
        
        # 提取座標和場值（支持多種格式）
        if 'coords' in data:
            # 格式1: 直接的 coords, u, v, p
            coords = data['coords']
            u_true = data['u'].reshape(-1, 1)
            v_true = data['v'].reshape(-1, 1)
            w_true = data['w'].reshape(-1, 1) if 'w' in data else None
            p_true = data['p'].reshape(-1, 1)
        elif 'sensor_points' in data:
            # 格式2: 感測點格式
            coords = data['sensor_points']
            u_true = data['sensor_u'].reshape(-1, 1)
            v_true = data['sensor_v'].reshape(-1, 1)
            w_true = data['sensor_w'].reshape(-1, 1) if 'sensor_w' in data else None
            p_true = data['sensor_p'].reshape(-1, 1)
        elif 'x' in data and 'y' in data:
            # 格式3: 網格格式 (支持 2D 和 3D)
            x = data['x']  # (Nx,)
            y = data['y']  # (Ny,)
            
            # 檢測是否為 3D 數據
            if 'z' in data and data['z'].ndim == 1:
                # 3D 網格格式
                z = data['z']  # (Nz,)
                u = data['u']  # (Nx, Ny, Nz)
                v = data['v']
                w = data.get('w', None)  # 可能沒有 w
                p = data['p']
                
                # 建立網格座標
                X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
                coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
                u_true = u.ravel().reshape(-1, 1)
                v_true = v.ravel().reshape(-1, 1)
                w_true = w.ravel().reshape(-1, 1) if w is not None else None
                p_true = p.ravel().reshape(-1, 1)
            else:
                # 2D 網格格式
                u = data['u']  # (Nx, Ny)
                v = data['v']
                p = data['p']
                
                # 建立網格座標
                X, Y = np.meshgrid(x, y, indexing='ij')
                coords = np.stack([X.ravel(), Y.ravel()], axis=1)
                u_true = u.ravel().reshape(-1, 1)
                v_true = v.ravel().reshape(-1, 1)
                w_true = None
                p_true = p.ravel().reshape(-1, 1)
        else:
            print(f"❌ 無法識別的數據格式。可用鍵: {list(data.keys())}")
            return
        
        print(f"  數據點數: {len(coords)}")
        print(f"  座標維度: {coords.shape}")
        print(f"  座標範圍:")
        for i, name in enumerate(['x', 'y', 'z'][:coords.shape[1]]):
            print(f"    {name} ∈ [{coords[:, i].min():.3f}, {coords[:, i].max():.3f}]")
        
    except FileNotFoundError:
        print(f"❌ 數據文件不存在: {data_path}")
        return
    except Exception as e:
        print(f"❌ 載入數據失敗: {e}")
        return
    
    # 預測
    print("\n🔮 模型預測...")
    print("-" * 70)
    with torch.no_grad():
        # 確保座標維度與模型輸入一致
        if coords.shape[1] < model_cfg.get('in_dim', 2):
            print(f"⚠️  座標維度 ({coords.shape[1]}) 小於模型輸入維度 ({model_cfg.get('in_dim', 2)})")
            # 補零或使用默認值
            coords_input = np.pad(coords, ((0, 0), (0, model_cfg.get('in_dim', 2) - coords.shape[1])), 
                                 mode='constant', constant_values=0)
        else:
            coords_input = coords[:, :model_cfg.get('in_dim', 2)]
        
        coords_tensor = torch.FloatTensor(coords_input).to(device)
        pred = model(coords_tensor)
        u_pred = pred[:, 0:1].cpu().numpy()
        v_pred = pred[:, 1:2].cpu().numpy()
        
        # 根據輸出維度決定 w 和 p 的位置
        if pred.shape[1] == 4:  # 3D: (u, v, w, p)
            w_pred = pred[:, 2:3].cpu().numpy()
            p_pred = pred[:, 3:4].cpu().numpy()
        elif pred.shape[1] == 3:  # 2D: (u, v, p)
            w_pred = None
            p_pred = pred[:, 2:3].cpu().numpy()
        else:
            w_pred = None
            p_pred = np.zeros_like(u_pred)
    
    # 計算誤差
    def relative_l2_error(pred, true):
        """相對 L2 誤差"""
        return np.linalg.norm(pred - true) / np.linalg.norm(true)
    
    def mean_absolute_error(pred, true):
        """平均絕對誤差"""
        return np.mean(np.abs(pred - true))
    
    u_error = relative_l2_error(u_pred, u_true) * 100
    v_error = relative_l2_error(v_pred, v_true) * 100
    p_error = relative_l2_error(p_pred, p_true) * 100
    
    u_mae = mean_absolute_error(u_pred, u_true)
    v_mae = mean_absolute_error(v_pred, v_true)
    p_mae = mean_absolute_error(p_pred, p_true)
    
    # 如果有 w 速度，也計算誤差
    if w_true is not None and w_pred is not None:
        w_error = relative_l2_error(w_pred, w_true) * 100
        w_mae = mean_absolute_error(w_pred, w_true)
    else:
        w_error = None
        w_mae = None
    
    print("\n🎯 評估結果：")
    print("=" * 70)
    print(f"  U 速度場：")
    print(f"    - 相對 L2 誤差: {u_error:.2f}%")
    print(f"    - 平均絕對誤差: {u_mae:.6f}")
    print(f"    - 預測範圍: [{u_pred.min():.3f}, {u_pred.max():.3f}]")
    print(f"    - 真實範圍: [{u_true.min():.3f}, {u_true.max():.3f}]")
    print()
    print(f"  V 速度場：")
    print(f"    - 相對 L2 誤差: {v_error:.2f}%")
    print(f"    - 平均絕對誤差: {v_mae:.6f}")
    print(f"    - 預測範圍: [{v_pred.min():.3f}, {v_pred.max():.3f}]")
    print(f"    - 真實範圍: [{v_true.min():.3f}, {v_true.max():.3f}]")
    print()
    
    # 如果有 W 速度，顯示其誤差
    if w_error is not None and w_pred is not None and w_true is not None:
        print(f"  W 速度場：")
        print(f"    - 相對 L2 誤差: {w_error:.2f}%")
        print(f"    - 平均絕對誤差: {w_mae:.6f}")
        print(f"    - 預測範圍: [{w_pred.min():.3f}, {w_pred.max():.3f}]")
        print(f"    - 真實範圍: [{w_true.min():.3f}, {w_true.max():.3f}]")
        print()
    
    print(f"  壓力場：")
    print(f"    - 相對 L2 誤差: {p_error:.2f}%")
    print(f"    - 平均絕對誤差: {p_mae:.6f}")
    print(f"    - 預測範圍: [{p_pred.min():.3f}, {p_pred.max():.3f}]")
    print(f"    - 真實範圍: [{p_true.min():.3f}, {p_true.max():.3f}]")
    
    print("\n" + "=" * 70)
    print("🏆 成功指標檢查（目標: < 15%）：")
    print("=" * 70)
    
    success_count = 0
    total_metrics = 3  # 基本: u, v, p
    
    if u_error < 15.0:
        print(f"  ✅ U 速度場: {u_error:.2f}% < 15%")
        success_count += 1
    else:
        print(f"  ❌ U 速度場: {u_error:.2f}% >= 15%")
    
    if v_error < 15.0:
        print(f"  ✅ V 速度場: {v_error:.2f}% < 15%")
        success_count += 1
    else:
        print(f"  ❌ V 速度場: {v_error:.2f}% >= 15%")
    
    # 如果有 W 速度，也檢查
    if w_error is not None:
        total_metrics = 4
        if w_error < 15.0:
            print(f"  ✅ W 速度場: {w_error:.2f}% < 15%")
            success_count += 1
        else:
            print(f"  ❌ W 速度場: {w_error:.2f}% >= 15%")
    
    if p_error < 15.0:
        print(f"  ✅ 壓力場: {p_error:.2f}% < 15%")
        success_count += 1
    else:
        print(f"  ❌ 壓力場: {p_error:.2f}% >= 15%")
    
    print("\n" + "=" * 70)
    if success_count == total_metrics:
        print("  🎉 所有指標均達標！")
    elif success_count >= total_metrics * 0.67:
        print(f"  ⚠️  部分指標達標 ({success_count}/{total_metrics})，需進一步優化")
    else:
        print(f"  ❌ 大部分指標未達標 ({success_count}/{total_metrics})，需重新訓練")
    print("=" * 70)
    
    result = {
        'u_error': u_error,
        'v_error': v_error,
        'p_error': p_error,
        'success_count': success_count,
        'total_metrics': total_metrics
    }
    
    if w_error is not None:
        result['w_error'] = w_error
    
    return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='快速評估 PINNs 檢查點')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/curriculum_adam_baseline_latest.pth',
                       help='檢查點路徑')
    parser.add_argument('--config', type=str,
                       default='configs/curriculum_adam_vs_soap_adam.yml',
                       help='配置文件路徑')
    parser.add_argument('--data', type=str, default=None,
                       help='測試數據路徑（默認使用配置中的路徑）')
    
    args = parser.parse_args()
    
    evaluate_model(args.checkpoint, args.config, args.data)
