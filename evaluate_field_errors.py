#!/usr/bin/env python3
"""評估模型的場域誤差（與 DNS 真值比較）"""
import sys
import torch
import numpy as np
import yaml
from pathlib import Path

# 添加項目路徑
sys.path.insert(0, str(Path(__file__).parent))

from pinnx.models.fourier_mlp import PINNNet
from pinnx.train.factory import create_model

def load_dns_data(dns_path='data/kolmogorov_dns/snapshot_re50_for_eval.npz'):
    """載入 DNS 參考資料"""
    print(f"📂 載入 DNS 資料: {dns_path}")
    dns = np.load(dns_path)
    
    # 建立網格
    x = dns['x']
    y = dns['y']
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # 真值場
    u_true = dns['u']  # (256, 256)
    v_true = dns['v']
    p_true = dns['p']
    
    print(f"  網格大小: {X.shape}")
    print(f"  u 範圍: [{u_true.min():.4f}, {u_true.max():.4f}]")
    print(f"  v 範圍: [{v_true.min():.4f}, {v_true.max():.4f}]")
    print(f"  p 範圍: [{p_true.min():.4f}, {p_true.max():.4f}]")
    
    return X, Y, u_true, v_true, p_true

def evaluate_model(checkpoint_path, config_path, dns_path='data/kolmogorov_dns/snapshot_re50_for_eval.npz'):
    """評估模型"""
    print("=" * 80)
    print("  場域誤差評估 (Field Error Evaluation)")
    print("=" * 80)
    
    # 載入配置
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 載入檢查點
    print(f"\n📂 載入檢查點: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # 提取訓練資訊
    print("\n📊 訓練資訊:")
    print("-" * 80)
    if 'epoch' in ckpt:
        print(f"  訓練輪數: {ckpt['epoch']}")
    if 'loss' in ckpt:
        print(f"  最終訓練損失: {ckpt['loss']:.6f}")
    
    # 使用內嵌配置（如果有）
    if 'config' in ckpt:
        full_cfg = ckpt['config']
        print("  使用檢查點內嵌配置")
    else:
        full_cfg = cfg
        print("  使用外部配置文件")
    
    # 創建模型
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n🔧 創建模型 (device={device})...")
    
    try:
        model = create_model(full_cfg, device)
        print(f"  ✅ 使用 factory 創建模型: {type(model).__name__}")
    except Exception as e:
        print(f"  ⚠️  Factory 失敗: {e}")
        model_cfg = full_cfg.get('model', {})
        model = PINNNet(
            in_dim=model_cfg.get('in_dim', 2),
            out_dim=model_cfg.get('out_dim', 3),
            width=model_cfg.get('width', 128),
            depth=model_cfg.get('depth', 6),
            fourier_m=model_cfg.get('fourier_m', 32),
            fourier_sigma=model_cfg.get('fourier_sigma', 1.0)
        ).to(device)
        print(f"  ✅ 使用簡單 PINNNet")
    
    # 載入權重
    state_dict = ckpt.get('model_state_dict') or ckpt.get('model')
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("  ✅ 模型權重載入完成")
    
    # 載入 DNS 資料
    print("\n📂 載入 DNS 參考資料...")
    print("-" * 80)
    X, Y, u_true, v_true, p_true = load_dns_data(dns_path)
    
    # 建立輸入張量
    N = X.shape[0] * X.shape[1]
    x_flat = X.flatten()
    y_flat = Y.flatten()
    
    # 檢查模型輸入維度
    model_in_dim = full_cfg.get('model', {}).get('in_dim', 2)
    if model_in_dim == 3:
        # 3D 輸入 (x, y, t) - 使用時間 t=0
        t_flat = np.zeros_like(x_flat)
        xyt = torch.tensor(np.stack([x_flat, y_flat, t_flat], axis=1), dtype=torch.float32).to(device)
        print(f"  使用 3D 輸入 (x, y, t), t=0")
    else:
        # 2D 輸入 (x, y)
        xyt = torch.tensor(np.stack([x_flat, y_flat], axis=1), dtype=torch.float32).to(device)
        print(f"  使用 2D 輸入 (x, y)")
    
    xy = xyt
    
    print(f"\n🔮 模型預測 (N={N} 個點)...")
    print("-" * 80)
    
    # 檢查是否有標準化參數
    norm_info = ckpt.get('normalization', None)
    if norm_info:
        print(f"  檢測到標準化: {norm_info['norm_type']}")
        means = norm_info.get('means', {})
        stds = norm_info.get('stds', {})
        var_order = norm_info.get('variable_order', ['u', 'v', 'p'])
        print(f"  變數順序: {var_order}")
        print(f"  均值: u={means.get('u', 0):.4f}, v={means.get('v', 0):.4f}, p={means.get('p', 0):.4f}")
        print(f"  標準差: u={stds.get('u', 1):.4f}, v={stds.get('v', 1):.4f}, p={stds.get('p', 1):.4f}")
    else:
        print("  未檢測到標準化")
        means = {}
        stds = {}
        var_order = ['u', 'v', 'p']
    
    # 分批預測（避免 OOM）
    batch_size = 4096
    predictions = []
    
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = xy[i:i+batch_size]
            pred = model(batch)
            predictions.append(pred.cpu())
    
    predictions = torch.cat(predictions, dim=0).numpy()  # (N, out_dim)
    
    # 反標準化（如果有）
    if norm_info and means and stds:
        print("  進行反標準化...")
        for idx, var in enumerate(var_order[:predictions.shape[1]]):
            if var in means and var in stds:
                predictions[:, idx] = predictions[:, idx] * stds[var] + means[var]
                print(f"    {var}: 已反標準化")
    
    # Reshape 回網格
    u_pred = predictions[:, 0].reshape(X.shape)
    v_pred = predictions[:, 1].reshape(X.shape)
    p_pred = predictions[:, 2].reshape(X.shape)
    
    print("  ✅ 預測完成")
    print(f"  u_pred 範圍: [{u_pred.min():.4f}, {u_pred.max():.4f}]")
    print(f"  v_pred 範圍: [{v_pred.min():.4f}, {v_pred.max():.4f}]")
    print(f"  p_pred 範圍: [{p_pred.min():.4f}, {p_pred.max():.4f}]")
    
    # 計算誤差指標
    print("\n📊 誤差分析:")
    print("=" * 80)
    
    # 1. L2 相對誤差
    def rel_l2(pred, true):
        return np.linalg.norm(pred - true) / np.linalg.norm(true)
    
    u_rel_l2 = rel_l2(u_pred, u_true)
    v_rel_l2 = rel_l2(v_pred, v_true)
    p_rel_l2 = rel_l2(p_pred, p_true)
    
    print("\n1️⃣  相對 L2 誤差 (Relative L2 Error):")
    print("-" * 80)
    print(f"  u: {u_rel_l2:.4%}")
    print(f"  v: {v_rel_l2:.4%}")
    print(f"  p: {p_rel_l2:.4%}")
    print(f"  平均: {np.mean([u_rel_l2, v_rel_l2, p_rel_l2]):.4%}")
    
    # 2. RMSE (Root Mean Square Error)
    def rmse(pred, true):
        return np.sqrt(np.mean((pred - true)**2))
    
    u_rmse = rmse(u_pred, u_true)
    v_rmse = rmse(v_pred, v_true)
    p_rmse = rmse(p_pred, p_true)
    
    print("\n2️⃣  RMSE (均方根誤差):")
    print("-" * 80)
    print(f"  u: {u_rmse:.6f}")
    print(f"  v: {v_rmse:.6f}")
    print(f"  p: {p_rmse:.6f}")
    
    # 3. 最大絕對誤差
    u_max_err = np.abs(u_pred - u_true).max()
    v_max_err = np.abs(v_pred - v_true).max()
    p_max_err = np.abs(p_pred - p_true).max()
    
    print("\n3️⃣  最大絕對誤差 (Max Absolute Error):")
    print("-" * 80)
    print(f"  u: {u_max_err:.6f}")
    print(f"  v: {v_max_err:.6f}")
    print(f"  p: {p_max_err:.6f}")
    
    # 4. 平均絕對誤差
    u_mae = np.mean(np.abs(u_pred - u_true))
    v_mae = np.mean(np.abs(v_pred - v_true))
    p_mae = np.mean(np.abs(p_pred - p_true))
    
    print("\n4️⃣  平均絕對誤差 (Mean Absolute Error):")
    print("-" * 80)
    print(f"  u: {u_mae:.6f}")
    print(f"  v: {v_mae:.6f}")
    print(f"  p: {p_mae:.6f}")
    
    # 5. R² 係數
    def r2_score(pred, true):
        ss_res = np.sum((true - pred)**2)
        ss_tot = np.sum((true - np.mean(true))**2)
        return 1 - (ss_res / ss_tot)
    
    u_r2 = r2_score(u_pred, u_true)
    v_r2 = r2_score(v_pred, v_true)
    p_r2 = r2_score(p_pred, p_true)
    
    print("\n5️⃣  R² 係數 (決定係數, 1.0=完美):")
    print("-" * 80)
    print(f"  u: {u_r2:.6f}")
    print(f"  v: {v_r2:.6f}")
    print(f"  p: {p_r2:.6f}")
    print(f"  平均: {np.mean([u_r2, v_r2, p_r2]):.6f}")
    
    # 6. 場域統計比較
    print("\n6️⃣  場域統計比較:")
    print("-" * 80)
    print(f"  u_true: mean={u_true.mean():.4f}, std={u_true.std():.4f}")
    print(f"  u_pred: mean={u_pred.mean():.4f}, std={u_pred.std():.4f}")
    print(f"  v_true: mean={v_true.mean():.4f}, std={v_true.std():.4f}")
    print(f"  v_pred: mean={v_pred.mean():.4f}, std={v_pred.std():.4f}")
    print(f"  p_true: mean={p_true.mean():.4f}, std={p_true.std():.4f}")
    print(f"  p_pred: mean={p_pred.mean():.4f}, std={p_pred.std():.4f}")
    
    # 7. 散度檢查 (∇·u)
    print("\n7️⃣  散度檢查 (∇·u = ∂u/∂x + ∂v/∂y):")
    print("-" * 80)
    
    dx = X[1,0] - X[0,0]
    dy = Y[0,1] - Y[0,0]
    
    du_dx = np.gradient(u_pred, dx, axis=0)
    dv_dy = np.gradient(v_pred, dy, axis=1)
    divergence = du_dx + dv_dy
    
    div_mean = np.abs(divergence).mean()
    div_max = np.abs(divergence).max()
    div_std = divergence.std()
    
    print(f"  |∇·u| 平均: {div_mean:.6e}")
    print(f"  |∇·u| 最大: {div_max:.6e}")
    print(f"  ∇·u 標準差: {div_std:.6e}")
    
    # 質量守恆評級
    if div_mean < 1e-3:
        grade = "⭐⭐⭐⭐⭐ Excellent"
    elif div_mean < 5e-3:
        grade = "⭐⭐⭐⭐ Very Good"
    elif div_mean < 1e-2:
        grade = "⭐⭐⭐ Good"
    elif div_mean < 5e-2:
        grade = "⭐⭐ Fair"
    else:
        grade = "⭐ Poor"
    
    print(f"  質量守恆評級: {grade}")
    
    # 總結
    print("\n" + "=" * 80)
    print("  總結 (Summary)")
    print("=" * 80)
    
    avg_rel_l2 = np.mean([u_rel_l2, v_rel_l2, p_rel_l2])
    avg_r2 = np.mean([u_r2, v_r2, p_r2])
    
    print(f"  平均相對 L2 誤差: {avg_rel_l2:.4%}")
    print(f"  平均 R² 係數: {avg_r2:.6f}")
    print(f"  散度平均: {div_mean:.6e}")
    
    # 總體評級
    if avg_rel_l2 < 0.05 and avg_r2 > 0.99:
        overall = "🏆 Outstanding"
    elif avg_rel_l2 < 0.10 and avg_r2 > 0.95:
        overall = "⭐⭐⭐⭐⭐ Excellent"
    elif avg_rel_l2 < 0.15 and avg_r2 > 0.90:
        overall = "⭐⭐⭐⭐ Very Good"
    elif avg_rel_l2 < 0.25 and avg_r2 > 0.80:
        overall = "⭐⭐⭐ Good"
    elif avg_rel_l2 < 0.40 and avg_r2 > 0.60:
        overall = "⭐⭐ Fair"
    else:
        overall = "⭐ Poor"
    
    print(f"  總體評級: {overall}")
    print("=" * 80)
    
    return {
        'rel_l2': {'u': u_rel_l2, 'v': v_rel_l2, 'p': p_rel_l2, 'avg': avg_rel_l2},
        'rmse': {'u': u_rmse, 'v': v_rmse, 'p': p_rmse},
        'max_err': {'u': u_max_err, 'v': v_max_err, 'p': p_max_err},
        'mae': {'u': u_mae, 'v': v_mae, 'p': p_mae},
        'r2': {'u': u_r2, 'v': v_r2, 'p': p_r2, 'avg': avg_r2},
        'divergence': {'mean': div_mean, 'max': div_max, 'std': div_std},
        'overall_grade': overall
    }

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='評估模型場域誤差')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint 路徑')
    parser.add_argument('--config', required=True, help='配置文件路徑')
    parser.add_argument('--dns', default='data/kolmogorov_dns/snapshot_re50_for_eval.npz', 
                        help='DNS 參考資料路徑')
    
    args = parser.parse_args()
    
    results = evaluate_model(args.checkpoint, args.config, args.dns)
