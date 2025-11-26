#!/usr/bin/env python
"""快速評估檢查點的流場誤差"""
import sys
import os
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
    
    # 建立模型（使用 factory 以支援所有模型類型）
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 從配置或檢查點中獲取完整配置（優先檢查點內嵌配置）
    if 'config' in ckpt:
        full_cfg = ckpt['config']
        print("  使用檢查點內嵌配置")
    else:
        full_cfg = cfg
        print("  使用外部配置文件")
    
    model_cfg = full_cfg.get('model', {})
    
    # 使用 factory 創建模型
    from pinnx.train.factory import create_model
    try:
        model = create_model(full_cfg, device)
        print(f"✅ 模型已創建（類型: {type(model).__name__}，設備: {device}）")
    except Exception as e:
        print(f"❌ 使用 factory 創建模型失敗: {e}")
        print("⚠️  回退到簡單 PINNNet...")
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
    
    # 載入狀態字典
    try:
        model.load_state_dict(state_dict, strict=False)
        print("  ✅ 模型權重載入成功")
    except Exception as e:
        print(f"  ⚠️  權重載入警告: {e}")
        print("  嘗試部分載入...")
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    print(f"✅ 模型已準備完成")
    
    # 載入測試數據
    if data_path is None:
        # 優先從 sensors 配置讀取，否則使用 data 配置
        sensor_file = cfg.get('sensors', {}).get('sensor_file', None)
        print(f"🔍 Debug - sensors.sensor_file: {sensor_file}")
        if sensor_file is None:
            sensor_file = cfg.get('data', {}).get('sensors_cache', 'data/jhtdb/sensors_kf8_qr_K100.npz')
            print(f"🔍 Debug - data.sensors_cache (fallback): {sensor_file}")
        data_path = sensor_file
    
    print("\n📁 載入測試數據...")
    print("-" * 70)
    print(f"📍 數據路徑: {data_path}")
    print(f"📍 檔案存在: {os.path.exists(data_path)}")
    print(f"📍 檔案存在: {os.path.exists(data_path)}")
    
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
            # 格式2: 感測點格式 (3D Channel Flow)
            coords = data['sensor_points']
            u_true = data['sensor_u'].reshape(-1, 1)
            v_true = data['sensor_v'].reshape(-1, 1)
            w_true = data['sensor_w'].reshape(-1, 1) if 'sensor_w' in data else None
            p_true = data['sensor_p'].reshape(-1, 1)
        elif 'sensor_x' in data and 'sensor_y' in data:
            # 格式2.5: Kolmogorov flow 感測點格式 (2D)
            sensor_x = data['sensor_x']
            sensor_y = data['sensor_y']
            coords = np.stack([sensor_x, sensor_y], axis=1)
            u_true = data['sensor_u'].reshape(-1, 1)
            v_true = data['sensor_v'].reshape(-1, 1)
            w_true = None  # 2D flow 沒有 w
            p_true = None  # Kolmogorov 感測器可能沒有壓力
            print(f"✅ Kolmogorov 感測器格式 (2D): {len(sensor_x)} 個感測點")
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
        if true is None or pred is None:
            return None
        return np.linalg.norm(pred - true) / np.linalg.norm(true)
    
    def mean_absolute_error(pred, true):
        """平均絕對誤差"""
        if true is None or pred is None:
            return None
        return np.mean(np.abs(pred - true))
    
    u_error_raw = relative_l2_error(u_pred, u_true)
    v_error_raw = relative_l2_error(v_pred, v_true)
    p_error_raw = relative_l2_error(p_pred, p_true)
    
    u_error = u_error_raw * 100 if u_error_raw is not None else None
    v_error = v_error_raw * 100 if v_error_raw is not None else None
    p_error = p_error_raw * 100 if p_error_raw is not None else None
    
    u_mae = mean_absolute_error(u_pred, u_true)
    v_mae = mean_absolute_error(v_pred, v_true)
    p_mae = mean_absolute_error(p_pred, p_true)
    
    # 如果有 w 速度，也計算誤差
    if w_true is not None and w_pred is not None:
        w_error_raw = relative_l2_error(w_pred, w_true)
        w_error = w_error_raw * 100 if w_error_raw is not None else None
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
    if p_error is not None and p_true is not None:
        print(f"    - 相對 L2 誤差: {p_error:.2f}%")
        print(f"    - 平均絕對誤差: {p_mae:.6f}")
        print(f"    - 預測範圍: [{p_pred.min():.3f}, {p_pred.max():.3f}]")
        print(f"    - 真實範圍: [{p_true.min():.3f}, {p_true.max():.3f}]")
    else:
        print(f"    - ⚠️ 無壓力場資料（Kolmogorov 感測器無壓力）")
    
    print("\n" + "=" * 70)
    print("🏆 成功指標檢查（目標: < 15%）：")
    print("=" * 70)
    
    success_count = 0
    total_metrics = 2  # 基本: u, v (壓力場可能沒有)
    
    if u_error is not None and u_error < 15.0:
        print(f"  ✅ U 速度場: {u_error:.2f}% < 15%")
        success_count += 1
    elif u_error is not None:
        print(f"  ❌ U 速度場: {u_error:.2f}% >= 15%")
    
    if v_error is not None and v_error < 15.0:
        print(f"  ✅ V 速度場: {v_error:.2f}% < 15%")
        success_count += 1
    elif v_error is not None:
        print(f"  ❌ V 速度場: {v_error:.2f}% >= 15%")
    
    # 如果有 W 速度，也檢查
    if w_error is not None:
        total_metrics += 1
        if w_error < 15.0:
            print(f"  ✅ W 速度場: {w_error:.2f}% < 15%")
            success_count += 1
        else:
            print(f"  ❌ W 速度場: {w_error:.2f}% >= 15%")
    
    # 如果有壓力場，也檢查
    if p_error is not None:
        total_metrics += 1
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
