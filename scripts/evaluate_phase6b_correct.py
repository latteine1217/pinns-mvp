#!/usr/bin/env python
"""Phase 6B 評估腳本 - 正確版本（無輸出縮放）

關鍵修復：
1. VS-PINN 模式下模型輸出**未經過任何標準化**
2. 直接使用模型輸出，無需反標準化
3. 與真實物理場直接比較
"""
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
    """載入檢查點並創建模型"""
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
    
    # 檢查是否使用了 VS-PINN
    vs_pinn_enabled = config.get('physics', {}).get('vs_pinn', {}).get('scaling_factors', None) is not None
    if vs_pinn_enabled:
        print("  🎯 檢測到 VS-PINN 模式：模型輸出為物理尺度，無需反標準化")
    
    return model, config, checkpoint


def evaluate_checkpoint(checkpoint_path, config_path, data_path, output_dir=None, subsample=None):
    """評估檢查點並計算流場誤差
    
    Args:
        checkpoint_path: 檢查點路徑
        config_path: 配置文件路徑
        data_path: DNS 資料路徑
        output_dir: 輸出目錄（可選）
        subsample: 子採樣間隔（可選，例如 4 表示每4個點取1個）
    """
    print("=" * 70)
    print("  Phase 6B 評估腳本 - 正確版本（無輸出縮放）")
    print("=" * 70)
    
    # 設備
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  設備: {device}")
    
    # 載入模型
    model, config, checkpoint = load_checkpoint_and_model(checkpoint_path, config_path, device)
    
    # 載入評估資料
    print(f"\n📁 載入評估資料: {data_path}")
    data = np.load(data_path)
    
    # 提取座標和真實場
    x_1d = data['x']  # (Nx,)
    y_1d = data['y']  # (Ny,)
    z_1d = data['z']  # (Nz,)
    u_true = data['u']  # (Nx, Ny, Nz)
    v_true = data['v']
    w_true = data['w']
    p_true = data['p']
    
    print(f"  原始資料形狀: u {u_true.shape}")
    print(f"  座標範圍: x=[{x_1d.min():.2f}, {x_1d.max():.2f}], "
          f"y=[{y_1d.min():.2f}, {y_1d.max():.2f}], z=[{z_1d.min():.2f}, {z_1d.max():.2f}]")
    
    # 子採樣（減少記憶體使用）
    if subsample and subsample > 1:
        print(f"\n⚙️  子採樣: 每 {subsample} 個點取 1 個")
        x_1d = x_1d[::subsample]
        y_1d = y_1d[::subsample]
        z_1d = z_1d[::subsample]
        u_true = u_true[::subsample, ::subsample, ::subsample]
        v_true = v_true[::subsample, ::subsample, ::subsample]
        w_true = w_true[::subsample, ::subsample, ::subsample]
        p_true = p_true[::subsample, ::subsample, ::subsample]
        print(f"  子採樣後形狀: u {u_true.shape}")
    
    # 建立網格
    x_grid, y_grid, z_grid = np.meshgrid(x_1d, y_1d, z_1d, indexing='ij')
    
    # 展平為 (N, 1) 格式
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()
    z_flat = z_grid.ravel()
    u_flat = u_true.ravel()
    v_flat = v_true.ravel()
    w_flat = w_true.ravel()
    p_flat = p_true.ravel()
    
    N = x_flat.shape[0]
    print(f"\n📊 評估點數: {N:,}")
    
    # 準備模型輸入
    coords = torch.tensor(
        np.stack([x_flat, y_flat, z_flat], axis=1),
        dtype=torch.float32,
        device=device
    )
    
    # 批次推理（避免記憶體溢出）
    batch_size = 50000
    num_batches = (N + batch_size - 1) // batch_size
    
    print(f"\n🔮 開始推理 (batch_size={batch_size}, num_batches={num_batches})...")
    
    u_pred_list = []
    v_pred_list = []
    w_pred_list = []
    p_pred_list = []
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, N)
            
            coords_batch = coords[start_idx:end_idx]
            
            # 模型前向傳播
            pred = model(coords_batch)  # (batch, 4) -> [u, v, w, p]
            
            # 🔧 關鍵修復：直接使用模型輸出，無需反標準化
            u_pred_list.append(pred[:, 0].cpu().numpy())
            v_pred_list.append(pred[:, 1].cpu().numpy())
            w_pred_list.append(pred[:, 2].cpu().numpy())
            p_pred_list.append(pred[:, 3].cpu().numpy())
            
            if (i + 1) % 10 == 0 or (i + 1) == num_batches:
                print(f"  進度: {i+1}/{num_batches} batches ({100*(i+1)/num_batches:.1f}%)")
    
    # 拼接結果
    u_pred = np.concatenate(u_pred_list)
    v_pred = np.concatenate(v_pred_list)
    w_pred = np.concatenate(w_pred_list)
    p_pred = np.concatenate(p_pred_list)
    
    print("✅ 推理完成")
    
    # 計算誤差指標
    print("\n" + "=" * 70)
    print("  流場重建誤差")
    print("=" * 70)
    
    def rel_l2_error(pred, true):
        """相對 L2 誤差"""
        return np.linalg.norm(pred - true) / np.linalg.norm(true)
    
    def rmse(pred, true):
        """均方根誤差"""
        return np.sqrt(np.mean((pred - true) ** 2))
    
    # 計算各場誤差
    u_rel_l2 = rel_l2_error(u_pred, u_flat)
    v_rel_l2 = rel_l2_error(v_pred, v_flat)
    w_rel_l2 = rel_l2_error(w_pred, w_flat)
    p_rel_l2 = rel_l2_error(p_pred, p_flat)
    
    u_rmse = rmse(u_pred, u_flat)
    v_rmse = rmse(v_pred, v_flat)
    w_rmse = rmse(w_pred, w_flat)
    p_rmse = rmse(p_pred, p_flat)
    
    # 統計資訊
    print("\n📊 真實場統計：")
    print(f"  u: mean={u_flat.mean():.4f}, std={u_flat.std():.4f}, range=[{u_flat.min():.4f}, {u_flat.max():.4f}]")
    print(f"  v: mean={v_flat.mean():.4f}, std={v_flat.std():.4f}, range=[{v_flat.min():.4f}, {v_flat.max():.4f}]")
    print(f"  w: mean={w_flat.mean():.4f}, std={w_flat.std():.4f}, range=[{w_flat.min():.4f}, {w_flat.max():.4f}]")
    print(f"  p: mean={p_flat.mean():.4f}, std={p_flat.std():.4f}, range=[{p_flat.min():.4f}, {p_flat.max():.4f}]")
    
    print("\n📊 預測場統計：")
    print(f"  u: mean={u_pred.mean():.4f}, std={u_pred.std():.4f}, range=[{u_pred.min():.4f}, {u_pred.max():.4f}]")
    print(f"  v: mean={v_pred.mean():.4f}, std={v_pred.std():.4f}, range=[{v_pred.min():.4f}, {v_pred.max():.4f}]")
    print(f"  w: mean={w_pred.mean():.4f}, std={w_pred.std():.4f}, range=[{w_pred.min():.4f}, {w_pred.max():.4f}]")
    print(f"  p: mean={p_pred.mean():.4f}, std={p_pred.std():.4f}, range=[{p_pred.min():.4f}, {p_pred.max():.4f}]")
    
    print("\n📏 相對 L2 誤差：")
    print(f"  u: {u_rel_l2*100:.2f}%")
    print(f"  v: {v_rel_l2*100:.2f}%")
    print(f"  w: {w_rel_l2*100:.2f}%")
    print(f"  p: {p_rel_l2*100:.2f}%")
    print(f"  平均: {(u_rel_l2 + v_rel_l2 + w_rel_l2)*100/3:.2f}% (速度場)")
    
    print("\n📏 RMSE：")
    print(f"  u: {u_rmse:.6f}")
    print(f"  v: {v_rmse:.6f}")
    print(f"  w: {w_rmse:.6f}")
    print(f"  p: {p_rmse:.6f}")
    
    # 判斷成敗
    velocity_avg_error = (u_rel_l2 + v_rel_l2 + w_rel_l2) / 3
    success_threshold = 0.15  # 15%
    
    print("\n" + "=" * 70)
    if velocity_avg_error <= success_threshold:
        print(f"  ✅ 評估通過！速度場平均誤差 {velocity_avg_error*100:.2f}% ≤ {success_threshold*100:.0f}%")
    else:
        print(f"  ❌ 評估失敗！速度場平均誤差 {velocity_avg_error*100:.2f}% > {success_threshold*100:.0f}%")
    print("=" * 70)
    
    # 保存結果
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'metrics': {
                'u_rel_l2': float(u_rel_l2),
                'v_rel_l2': float(v_rel_l2),
                'w_rel_l2': float(w_rel_l2),
                'p_rel_l2': float(p_rel_l2),
                'u_rmse': float(u_rmse),
                'v_rmse': float(v_rmse),
                'w_rmse': float(w_rmse),
                'p_rmse': float(p_rmse),
                'velocity_avg_error': float(velocity_avg_error),
                'success': velocity_avg_error <= success_threshold
            },
            'checkpoint': str(checkpoint_path),
            'config': str(config_path),
            'data': str(data_path),
            'num_points': int(N)
        }
        
        results_path = output_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 結果已保存至: {results_path}")
    
    return {
        'u_rel_l2': u_rel_l2,
        'v_rel_l2': v_rel_l2,
        'w_rel_l2': w_rel_l2,
        'p_rel_l2': p_rel_l2,
        'velocity_avg_error': velocity_avg_error,
        'success': velocity_avg_error <= success_threshold
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 6B 評估腳本')
    parser.add_argument('--checkpoint', type=str, required=True, help='檢查點路徑')
    parser.add_argument('--config', type=str, required=True, help='配置文件路徑')
    parser.add_argument('--data', type=str, required=True, help='DNS 資料路徑 (.npz)')
    parser.add_argument('--output', type=str, default=None, help='輸出目錄')
    parser.add_argument('--subsample', type=int, default=None, help='子採樣間隔（例如 4）')
    
    args = parser.parse_args()
    
    evaluate_checkpoint(
        args.checkpoint,
        args.config,
        args.data,
        args.output,
        args.subsample
    )
