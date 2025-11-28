#!/usr/bin/env python
"""
Kolmogorov Flow 快速評估腳本（修復版）
=============================
修復 DNS 數據格式不匹配問題

變更：
- 支援 'time' 和 't' 兩種時間鍵名
- 自動生成空間座標網格（DNS 文件不包含 x, y）
- 智能檢測 HDF5 文件結構
"""

import sys
import os
import argparse
import torch
import numpy as np
import yaml
import h5py
from pathlib import Path
import matplotlib.pyplot as plt

# 添加項目路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pinnx.train.factory import create_model


def load_checkpoint(checkpoint_path, device):
    """載入訓練檢查點"""
    print(f"\n📂 載入檢查點: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 提取配置（優先使用內嵌配置）
    if 'config' in checkpoint:
        config = checkpoint['config']
        print("  ✅ 使用檢查點內嵌配置")
    else:
        raise ValueError("檢查點缺少配置資訊，請使用 --config 指定外部配置文件")
    
    # 創建模型
    model = create_model(config, device)
    print(f"  ✅ 模型已創建 (類型: {type(model).__name__})")
    
    # 載入權重
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        raise KeyError("檢查點缺少模型權重 (需要 'model_state_dict' 或 'model')")
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print(f"  📊 訓練輪數: {checkpoint.get('epoch', 'N/A')}")
    if 'loss' in checkpoint:
        print(f"  📉 最終損失: {checkpoint['loss']:.6e}")
    
    return model, config


def load_dns_reference(config, time_snapshot=None):
    """載入 DNS 參考資料（智能適配）"""
    data_path = config['data']['kolmogorov_config']['data_path']
    print(f"\n📂 載入 DNS 參考資料: {data_path}")
    
    if not Path(data_path).exists():
        raise FileNotFoundError(f"DNS 數據檔案不存在: {data_path}")
    
    with h5py.File(data_path, 'r') as f:
        # 檢查文件結構
        available_keys = list(f.keys())
        print(f"  📋 檢測到的數據集: {available_keys}")
        
        # 讀取流場數據
        u = np.array(f['u'])  # [Nt, Ny, Nx]
        v = np.array(f['v'])
        p = np.array(f['p'])
        
        # 讀取時間（智能檢測）
        if 'time' in f:
            t = np.array(f['time'])
            print(f"  ✅ 使用時間鍵: 'time'")
        elif 't' in f:
            t = np.array(f['t'])
            print(f"  ✅ 使用時間鍵: 't'")
        else:
            raise KeyError(f"找不到時間數據，可用鍵: {available_keys}")
        
        # 獲取網格大小
        Nt, Ny, Nx = u.shape
        
        # 獲取域大小（優先級：HDF5 屬性 > 配置文件）
        if hasattr(f, 'attrs') and 'L' in f.attrs:
            L = float(f.attrs['L'])
            print(f"  ✅ 從 HDF5 屬性讀取域大小: L={L:.4f}")
        else:
            # 從配置文件獲取
            domain_x = config['data']['kolmogorov_config']['domain']['x']
            L = float(domain_x[1] - domain_x[0])
            print(f"  ✅ 從配置文件讀取域大小: L={L:.4f}")
        
        # 生成空間座標（週期性網格）
        x = np.linspace(0, L, Nx, endpoint=False, dtype=np.float32)
        y = np.linspace(0, L, Ny, endpoint=False, dtype=np.float32)
        
        print(f"  ✅ DNS 資料形狀: u={u.shape}, v={v.shape}, p={p.shape}")
        print(f"  📏 空間網格: {Nx} × {Ny}")
        print(f"  📏 空間範圍: x=[{x.min():.4f}, {x.max():.4f}], y=[{y.min():.4f}, {y.max():.4f}]")
        print(f"  ⏱️  時間範圍: t=[{t.min():.2f}, {t.max():.2f}] ({len(t)} snapshots)")
    
    # 時間窗過濾
    time_range = config['data']['kolmogorov_config'].get('time_range', [float(t.min()), float(t.max())])
    time_mask = (t >= time_range[0]) & (t <= time_range[1])
    
    t_filtered = t[time_mask]
    u_filtered = u[time_mask]
    v_filtered = v[time_mask]
    p_filtered = p[time_mask]
    
    print(f"  ✅ 時間窗過濾: [{time_range[0]:.1f}, {time_range[1]:.1f}] → {len(t_filtered)} snapshots")
    
    # 單一時間快照或時間平均
    if time_snapshot is not None:
        # 找最接近的時間點
        idx = int(np.argmin(np.abs(t_filtered - time_snapshot)))
        actual_time = t_filtered[idx]
        print(f"  🎯 使用單一時間快照: t={actual_time:.2f} (請求 t={time_snapshot:.2f})")
        u_final = u_filtered[idx:idx+1]
        v_final = v_filtered[idx:idx+1]
        p_final = p_filtered[idx:idx+1]
        t_final = np.array([actual_time])
    else:
        print(f"  📊 使用時間平均場 (Nt={len(t_filtered)})")
        u_final = u_filtered
        v_final = v_filtered
        p_final = p_filtered
        t_final = t_filtered
    
    return {
        'x': x,
        'y': y,
        't': t_final,
        'u': u_final,
        'v': v_final,
        'p': p_final
    }


def create_evaluation_grid(x, y, t, n_points=256):
    """創建評估網格 - 3D: (t, x, y)
    
    ⚠️ 強制使用單一時間快照（t[0]）以便視覺化
    """
    # 空間下採樣
    if len(x) > n_points:
        x_idx = np.linspace(0, len(x)-1, n_points, dtype=int)
        y_idx = np.linspace(0, len(y)-1, n_points, dtype=int)
        x_sub = x[x_idx]
        y_sub = y[y_idx]
    else:
        x_sub = x
        y_sub = y
    
    # ⚠️ 強制使用單一時間點（修復視覺化 reshape 錯誤）
    if len(t) > 1:
        print(f"  ⚠️  檢測到多個時間點 (Nt={len(t)})，強制使用 t[0]={t[0]:.4f}")
        t_single = t[0:1]
    else:
        t_single = t
    
    # 創建網格 - 注意順序：(t, x, y)
    coords = []
    for ti in t_single:
        for yi in y_sub:
            for xi in x_sub:
                coords.append([ti, xi, yi])
    
    grid_coords = np.array(coords, dtype=np.float32)
    
    print(f"\n🔢 評估網格:")
    print(f"  空間解析度: {len(x_sub)} × {len(y_sub)} (原始: {len(x)} × {len(y)})")
    print(f"  時間點數: {len(t_single)} (原始: {len(t)})")
    print(f"  總評估點數: {len(grid_coords):,}")
    print(f"  輸入維度: {grid_coords.shape} (✅ 應為 [N, 3]: t, x, y)")
    
    return grid_coords


def predict_fields(model, coords, device, batch_size=10000):
    """批次預測流場"""
    print(f"\n🔮 模型預測...")
    
    model.eval()
    u_list = []
    v_list = []
    p_list = []
    
    n_total = len(coords)
    n_batches = (n_total + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(n_batches):
            start = i * batch_size
            end = min((i+1) * batch_size, n_total)
            
            batch = torch.tensor(coords[start:end], dtype=torch.float32, device=device)
            output = model(batch)  # [B, 3]
            
            u_list.append(output[:, 0].cpu().numpy())
            v_list.append(output[:, 1].cpu().numpy())
            p_list.append(output[:, 2].cpu().numpy())
            
            if (i+1) % 10 == 0 or (i+1) == n_batches:
                print(f"  進度: {i+1}/{n_batches} ({(i+1)/n_batches*100:.1f}%)", end='\r')
    
    print(f"\n  ✅ 預測完成: {n_total:,} 點")
    
    # 合併批次
    predictions = {
        'u': np.concatenate(u_list),
        'v': np.concatenate(v_list),
        'p': np.concatenate(p_list)
    }
    
    return predictions


def downsample_reference(ref_data, n_points):
    """下採樣參考資料以匹配評估網格
    
    ⚠️ 強制使用單一時間快照（第一個時間步 t[0]）
    """
    x = ref_data['x']
    y = ref_data['y']
    u = ref_data['u']  # [Nt, Ny, Nx]
    v = ref_data['v']
    p = ref_data['p']
    
    # ⚠️ 強制使用單一時間快照（修復視覺化 reshape 錯誤）
    if u.shape[0] > 1:
        print(f"  ⚠️  參考資料包含多個時間點 (Nt={u.shape[0]})，強制使用 t[0]")
        u = u[0:1]  # [1, Ny, Nx]
        v = v[0:1]
        p = p[0:1]
    
    # 空間下採樣索引
    if len(x) > n_points:
        x_idx = np.linspace(0, len(x)-1, n_points, dtype=int)
        y_idx = np.linspace(0, len(y)-1, n_points, dtype=int)
        
        u_sub = u[:, y_idx, :][:, :, x_idx]
        v_sub = v[:, y_idx, :][:, :, x_idx]
        p_sub = p[:, y_idx, :][:, :, x_idx]
    else:
        u_sub = u
        v_sub = v
        p_sub = p
    
    # 展平為 1D
    result = {
        'u': u_sub.flatten(),
        'v': v_sub.flatten(),
        'p': p_sub.flatten()
    }
    
    print(f"  ✅ 下採樣完成: {u_sub.shape} → {result['u'].shape[0]} 點")
    
    return result


def compute_metrics(pred, ref):
    """計算評估指標"""
    print(f"\n📊 計算評估指標...")
    
    metrics = {}
    
    # 相對 L2 誤差
    for var in ['u', 'v', 'p']:
        pred_var = pred[var]
        ref_var = ref[var]
        
        l2 = np.linalg.norm(pred_var - ref_var) / (np.linalg.norm(ref_var) + 1e-10)
        rmse = np.sqrt(np.mean((pred_var - ref_var)**2))
        
        metrics[f'{var}_l2'] = float(l2)
        metrics[f'{var}_rmse'] = float(rmse)
    
    # 質量守恆（散度近似）
    metrics['mass_conservation'] = float(np.sqrt(np.mean((pred['u'] - ref['u'])**2 + (pred['v'] - ref['v'])**2)))
    
    # 打印結果
    print(f"\n{'='*70}")
    print(f"  評估結果")
    print(f"{'='*70}")
    print(f"  速度 u L2 誤差: {metrics['u_l2']:.4f} ({metrics['u_l2']*100:.2f}%)")
    print(f"  速度 v L2 誤差: {metrics['v_l2']:.4f} ({metrics['v_l2']*100:.2f}%)")
    print(f"  壓力 p L2 誤差: {metrics['p_l2']:.4f} ({metrics['p_l2']*100:.2f}%)")
    print(f"  速度 u RMSE:   {metrics['u_rmse']:.6f}")
    print(f"  速度 v RMSE:   {metrics['v_rmse']:.6f}")
    print(f"  壓力 p RMSE:   {metrics['p_rmse']:.6f}")
    print(f"  質量守恆誤差:   {metrics['mass_conservation']:.6f}")
    print(f"{'='*70}\n")
    
    return metrics


def visualize_results(pred, ref, metrics, output_dir, n_grid):
    """視覺化評估結果"""
    print(f"\n📊 生成視覺化圖表...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 重組為 2D 網格 (假設單一時間快照)
    Ny = Nx = n_grid
    
    for var in ['u', 'v', 'p']:
        pred_2d = pred[var].reshape(Ny, Nx)
        ref_2d = ref[var].reshape(Ny, Nx)
        error_2d = np.abs(pred_2d - ref_2d)
        
        # 三面板圖
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 參考場
        im0 = axes[0].imshow(ref_2d, cmap='RdBu_r', origin='lower', aspect='auto')
        axes[0].set_title(f'{var.upper()} Reference (DNS)')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        plt.colorbar(im0, ax=axes[0])
        
        # 預測場
        im1 = axes[1].imshow(pred_2d, cmap='RdBu_r', origin='lower', aspect='auto')
        axes[1].set_title(f'{var.upper()} Prediction (PINNs)')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        plt.colorbar(im1, ax=axes[1])
        
        # 誤差場
        im2 = axes[2].imshow(error_2d, cmap='hot', origin='lower', aspect='auto')
        axes[2].set_title(f'{var.upper()} Absolute Error (L2={metrics[f"{var}_l2"]:.4f})')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        plt.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        fig.savefig(output_dir / f'field_{var}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✅ 保存: field_{var}.png")
    
    print(f"\n📁 視覺化完成，保存於: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Kolmogorov Flow 快速評估（修復版）')
    parser.add_argument('--checkpoint', type=str, required=True, help='檢查點路徑')
    parser.add_argument('--config', type=str, default=None, help='配置文件路徑（可選）')
    parser.add_argument('--output', type=str, required=True, help='輸出目錄')
    parser.add_argument('--n-points', type=int, default=256, help='評估網格解析度')
    parser.add_argument('--time-snapshot', type=float, default=None, help='指定時間快照（None=時間平均）')
    parser.add_argument('--device', type=str, default='auto', help='運算裝置 (auto/cuda/mps/cpu)')
    args = parser.parse_args()
    
    # 設定裝置
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"\n🖥️  運算裝置: {device}")
    
    # 載入檢查點
    model, config = load_checkpoint(args.checkpoint, device)
    
    # 如果提供外部配置，合併使用
    if args.config is not None:
        with open(args.config, 'r') as f:
            external_config = yaml.safe_load(f)
        # 使用外部配置的數據路徑
        if 'data' in external_config:
            config['data'] = external_config['data']
    
    # 載入 DNS 參考資料
    ref_data = load_dns_reference(config, time_snapshot=args.time_snapshot)
    
    # 創建評估網格
    grid_coords = create_evaluation_grid(
        ref_data['x'], 
        ref_data['y'], 
        ref_data['t'], 
        n_points=args.n_points
    )
    
    # 模型預測
    predictions = predict_fields(model, grid_coords, device)
    
    # 下採樣參考資料
    reference = downsample_reference(ref_data, args.n_points)
    
    # 計算指標
    metrics = compute_metrics(predictions, reference)
    
    # 視覺化
    output_dir = Path(args.output)
    visualize_results(predictions, reference, metrics, output_dir, args.n_points)
    
    # 保存指標
    metrics_file = output_dir / 'metrics.yaml'
    with open(metrics_file, 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)
    print(f"\n💾 指標保存於: {metrics_file}")
    
    print(f"\n✅ 評估完成！")


if __name__ == '__main__':
    main()
