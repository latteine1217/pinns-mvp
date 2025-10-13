#!/usr/bin/env python3
"""
預測場與真實場可視化對比腳本

生成專業級的對比圖表，幫助診斷模型學習效果：
1. 2D 切片對比圖（真實 | 預測 | 絕對誤差）
2. 統計分佈對比（直方圖、散點圖）
3. 剖面線對比（通道高度方向）

使用方式:
    python scripts/visualize_prediction_comparison.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import seaborn as sns

# 配置 matplotlib
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ============================================================
# 配置
# ============================================================

BASELINE_CONFIG = "configs/vs_pinn_baseline_1k.yml"
FOURIER_CONFIG = "configs/vs_pinn_fourier_1k.yml"

BASELINE_CHECKPOINT = "checkpoints/vs_pinn_baseline_1k_latest.pth"
FOURIER_CHECKPOINT = "checkpoints/vs_pinn_fourier_1k_latest.pth"

TEST_DATA = "data/jhtdb/channel_flow_re1000/cutout3d_128x128x32.npz"

OUTPUT_DIR = Path("results/field_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 工具函數
# ============================================================

def load_config(config_path):
    """載入 YAML 配置"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_test_data(data_path):
    """載入測試資料"""
    print(f"\n📂 載入測試資料: {data_path}")
    data = np.load(data_path)
    
    # 檢查資料格式
    if 'coords' in data:
        # 格式 A: 已有座標陣列
        coords = data['coords']
        u_true = data['u']
        v_true = data['v']
        w_true = data['w']
        p_true = data['p']
        
        # 推斷網格形狀
        n_points = coords.shape[0]
        x_unique = np.unique(coords[:, 0])
        y_unique = np.unique(coords[:, 1])
        z_unique = np.unique(coords[:, 2])
        nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)
    else:
        # 格式 B: 分離的座標與場數據 (x, y, z, u, v, w, p)
        x = data['x']  # [nx]
        y = data['y']  # [ny]
        z = data['z']  # [nz]
        u_3d = data['u']  # [nx, ny, nz]
        v_3d = data['v']
        w_3d = data['w']
        p_3d = data['p']
        
        nx, ny, nz = len(x), len(y), len(z)
        
        # 生成網格座標
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # [N, 3]
        
        # 展平場數據
        u_true = u_3d.ravel()
        v_true = v_3d.ravel()
        w_true = w_3d.ravel()
        p_true = p_3d.ravel()
    
    n_points = len(u_true)
    
    print(f"  ✅ 網格形狀: ({nx}, {ny}, {nz})")
    print(f"  ✅ 總點數: {n_points}")
    print(f"  ✅ 座標範圍:")
    print(f"     X: [{coords[:, 0].min():.4f}, {coords[:, 0].max():.4f}]")
    print(f"     Y: [{coords[:, 1].min():.4f}, {coords[:, 1].max():.4f}]")
    print(f"     Z: [{coords[:, 2].min():.4f}, {coords[:, 2].max():.4f}]")
    
    # 計算真實值範圍（用於後處理縮放）
    true_ranges = {
        'u': (float(u_true.min()), float(u_true.max())),
        'v': (float(v_true.min()), float(v_true.max())),
        'w': (float(w_true.min()), float(w_true.max())),
        'p': (float(p_true.min()), float(p_true.max())),
    }
    
    print(f"  ✅ 場範圍:")
    for var in ['u', 'v', 'w', 'p']:
        vmin, vmax = true_ranges[var]
        print(f"     {var}: [{vmin:.4f}, {vmax:.4f}]")
    
    return {
        'coords': coords,
        'u': u_true,
        'v': v_true,
        'w': w_true,
        'p': p_true,
        'grid_shape': (nx, ny, nz),
        'true_ranges': true_ranges
    }

def load_model(config_path, checkpoint_path, device):
    """載入模型"""
    from pinnx.models.fourier_mlp import PINNNet
    
    print(f"\n🔧 載入模型: {checkpoint_path}")
    
    config = load_config(config_path)
    model_cfg = config['model']
    
    # 提取正確的參數
    in_dim = model_cfg.get('in_dim', 3)
    out_dim = model_cfg.get('out_dim', 4)
    width = model_cfg.get('width', 200)
    depth = model_cfg.get('depth', 8)
    activation = model_cfg.get('activation', 'tanh')
    
    # Fourier 特徵相關參數（使用正確的參數名）
    use_fourier = model_cfg.get('use_fourier', False)
    fourier_m = model_cfg.get('fourier_m', 32)
    fourier_sigma = model_cfg.get('fourier_sigma', 5.0)
    
    print(f"  ℹ️  模型配置:")
    print(f"     - 輸入/輸出維度: {in_dim} → {out_dim}")
    print(f"     - 網路結構: {depth} 層 × {width} 寬度")
    print(f"     - 激活函數: {activation}")
    print(f"     - Fourier Features: {'啟用' if use_fourier else '禁用'}")
    if use_fourier:
        print(f"       · fourier_m: {fourier_m}")
        print(f"       · fourier_sigma: {fourier_sigma}")
    
    # 創建模型（使用正確的參數名）
    model = PINNNet(
        in_dim=in_dim,
        out_dim=out_dim,
        width=width,
        depth=depth,
        activation=activation,
        use_fourier=use_fourier,
        fourier_m=fourier_m,
        fourier_sigma=fourier_sigma
    )
    
    # 載入檢查點
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 處理狀態字典
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 嘗試載入（可能有額外參數）
    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"  ✅ 成功載入所有參數")
    except RuntimeError as e:
        # 過濾掉額外的參數
        model_keys = set(model.state_dict().keys())
        state_keys = set(state_dict.keys())
        extra_keys = state_keys - model_keys
        missing_keys = model_keys - state_keys
        
        if extra_keys:
            print(f"  ⚠️  檢查點中有額外參數 ({len(extra_keys)} 個): {list(extra_keys)[:5]}...")
        if missing_keys:
            print(f"  ⚠️  模型中缺少參數 ({len(missing_keys)} 個): {list(missing_keys)[:5]}...")
        
        # 嘗試寬鬆載入
        filtered_state = {k: v for k, v in state_dict.items() if k in model_keys}
        model.load_state_dict(filtered_state, strict=False)
        print(f"  ✅ 成功載入 {len(filtered_state)} / {len(model_keys)} 個參數")
    
    model.to(device)
    model.eval()
    
    print(f"  ✅ 模型總參數量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, config

@torch.no_grad()
def predict(model, coords, device, batch_size=8192):
    """批次預測"""
    model.eval()
    coords_tensor = torch.FloatTensor(coords).to(device)
    
    n_points = coords_tensor.shape[0]
    predictions = []
    
    for i in range(0, n_points, batch_size):
        batch = coords_tensor[i:i+batch_size]
        pred = model(batch)
        predictions.append(pred.cpu().numpy())
    
    return np.vstack(predictions)

# ============================================================
# 可視化函數
# ============================================================

def plot_2d_slice_comparison(true_field, pred_field, coords, grid_shape, 
                            var_name, slice_axis='z', slice_idx=None,
                            save_path=None):
    """
    繪製 2D 切片對比圖（真實 | 預測 | 絕對誤差）
    
    Args:
        true_field: 真實場 [N]
        pred_field: 預測場 [N]
        coords: 座標 [N, 3]
        grid_shape: 網格形狀 (nx, ny, nz)
        var_name: 變量名稱 (u, v, w, p)
        slice_axis: 切片軸 ('x', 'y', 'z')
        slice_idx: 切片索引（None 則使用中間）
        save_path: 保存路徑
    """
    nx, ny, nz = grid_shape
    
    # 重塑為 3D 網格
    true_3d = true_field.reshape(nx, ny, nz)
    pred_3d = pred_field.reshape(nx, ny, nz)
    error_3d = np.abs(pred_3d - true_3d)
    
    # 提取切片
    if slice_axis == 'x':
        idx = slice_idx if slice_idx is not None else nx // 2
        true_slice = true_3d[idx, :, :]
        pred_slice = pred_3d[idx, :, :]
        error_slice = error_3d[idx, :, :]
        xlabel, ylabel = 'Y', 'Z'
    elif slice_axis == 'y':
        idx = slice_idx if slice_idx is not None else ny // 2
        true_slice = true_3d[:, idx, :]
        pred_slice = pred_3d[:, idx, :]
        error_slice = error_3d[:, idx, :]
        xlabel, ylabel = 'X', 'Z'
    else:  # z
        idx = slice_idx if slice_idx is not None else nz // 2
        true_slice = true_3d[:, :, idx]
        pred_slice = pred_3d[:, :, idx]
        error_slice = error_3d[:, :, idx]
        xlabel, ylabel = 'X', 'Y'
    
    # 創建圖表
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 共享色標範圍（真實 & 預測）
    vmin = min(true_slice.min(), pred_slice.min())
    vmax = max(true_slice.max(), pred_slice.max())
    
    # 真實場
    im1 = axes[0].imshow(true_slice.T, origin='lower', cmap='RdBu_r', 
                         vmin=vmin, vmax=vmax, aspect='auto')
    axes[0].set_title(f'Ground Truth: {var_name}')
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    plt.colorbar(im1, ax=axes[0])
    
    # 預測場
    im2 = axes[1].imshow(pred_slice.T, origin='lower', cmap='RdBu_r', 
                         vmin=vmin, vmax=vmax, aspect='auto')
    axes[1].set_title(f'Prediction: {var_name}')
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    plt.colorbar(im2, ax=axes[1])
    
    # 絕對誤差
    im3 = axes[2].imshow(error_slice.T, origin='lower', cmap='hot', 
                         vmin=0, vmax=error_slice.max(), aspect='auto')
    axes[2].set_title(f'Absolute Error: {var_name}')
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel(ylabel)
    plt.colorbar(im3, ax=axes[2])
    
    plt.suptitle(f'{var_name.upper()} Field Comparison ({slice_axis.upper()}-slice at idx={idx})', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ 已保存: {save_path}")
    
    plt.close()

def plot_statistical_comparison(true_field, pred_field, var_name, save_path=None):
    """
    繪製統計分佈對比（直方圖 + 散點圖）
    
    Args:
        true_field: 真實場 [N]
        pred_field: 預測場 [N]
        var_name: 變量名稱
        save_path: 保存路徑
    """
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig)
    
    # 1. 直方圖對比
    ax1 = fig.add_subplot(gs[0])
    ax1.hist(true_field, bins=50, alpha=0.6, label='Ground Truth', color='blue', density=True)
    ax1.hist(pred_field, bins=50, alpha=0.6, label='Prediction', color='red', density=True)
    ax1.set_xlabel(f'{var_name}')
    ax1.set_ylabel('Density')
    ax1.set_title(f'{var_name.upper()} Distribution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 散點圖（預測 vs 真實）
    ax2 = fig.add_subplot(gs[1])
    
    # 下採樣（避免點太多）
    n_samples = min(10000, len(true_field))
    indices = np.random.choice(len(true_field), n_samples, replace=False)
    
    ax2.scatter(true_field[indices], pred_field[indices], alpha=0.3, s=1, color='blue')
    
    # 理想線 y=x
    min_val = min(true_field.min(), pred_field.min())
    max_val = max(true_field.max(), pred_field.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (y=x)', linewidth=2)
    
    ax2.set_xlabel(f'{var_name} (Ground Truth)')
    ax2.set_ylabel(f'{var_name} (Prediction)')
    ax2.set_title(f'{var_name.upper()} Scatter Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    # 3. 誤差分佈直方圖
    ax3 = fig.add_subplot(gs[2])
    error = pred_field - true_field
    ax3.hist(error, bins=50, alpha=0.7, color='green', density=True)
    ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax3.set_xlabel(f'Error ({var_name})')
    ax3.set_ylabel('Density')
    ax3.set_title(f'{var_name.upper()} Error Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 添加統計信息
    error_mean = error.mean()
    error_std = error.std()
    ax3.text(0.05, 0.95, f'Mean: {error_mean:.4f}\nStd: {error_std:.4f}', 
             transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ 已保存: {save_path}")
    
    plt.close()

def plot_profile_comparison(true_field, pred_field, coords, grid_shape, 
                           var_name, profile_axis='y', save_path=None):
    """
    繪製剖面線對比（沿通道高度）
    
    Args:
        true_field: 真實場 [N]
        pred_field: 預測場 [N]
        coords: 座標 [N, 3]
        grid_shape: 網格形狀 (nx, ny, nz)
        var_name: 變量名稱
        profile_axis: 剖面軸 ('x', 'y', 'z')
        save_path: 保存路徑
    """
    nx, ny, nz = grid_shape
    
    # 重塑為 3D 網格
    true_3d = true_field.reshape(nx, ny, nz)
    pred_3d = pred_field.reshape(nx, ny, nz)
    coords_3d = coords.reshape(nx, ny, nz, 3)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    if profile_axis == 'y':
        # 沿 Y 軸（通道高度）的剖面
        # 平均 X 和 Z 方向
        true_profile = true_3d.mean(axis=(0, 2))
        pred_profile = pred_3d.mean(axis=(0, 2))
        y_coords = coords_3d[0, :, 0, 1]  # Y 座標
        
        axes[0].plot(y_coords, true_profile, 'b-', label='Ground Truth', linewidth=2)
        axes[0].plot(y_coords, pred_profile, 'r--', label='Prediction', linewidth=2)
        axes[0].set_xlabel('Y (Channel Height)')
        axes[0].set_ylabel(f'{var_name}')
        axes[0].set_title(f'{var_name.upper()} Profile (averaged over X, Z)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 誤差
        error = np.abs(pred_profile - true_profile)
        axes[1].plot(y_coords, error, 'g-', linewidth=2)
        axes[1].set_xlabel('Y (Channel Height)')
        axes[1].set_ylabel(f'Absolute Error ({var_name})')
        axes[1].set_title(f'{var_name.upper()} Profile Error')
        axes[1].grid(True, alpha=0.3)
    
    elif profile_axis == 'x':
        # 沿 X 軸的剖面
        true_profile = true_3d.mean(axis=(1, 2))
        pred_profile = pred_3d.mean(axis=(1, 2))
        x_coords = coords_3d[:, 0, 0, 0]
        
        axes[0].plot(x_coords, true_profile, 'b-', label='Ground Truth', linewidth=2)
        axes[0].plot(x_coords, pred_profile, 'r--', label='Prediction', linewidth=2)
        axes[0].set_xlabel('X (Streamwise)')
        axes[0].set_ylabel(f'{var_name}')
        axes[0].set_title(f'{var_name.upper()} Profile (averaged over Y, Z)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        error = np.abs(pred_profile - true_profile)
        axes[1].plot(x_coords, error, 'g-', linewidth=2)
        axes[1].set_xlabel('X (Streamwise)')
        axes[1].set_ylabel(f'Absolute Error ({var_name})')
        axes[1].set_title(f'{var_name.upper()} Profile Error')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ 已保存: {save_path}")
    
    plt.close()

# ============================================================
# 主函數
# ============================================================

def main():
    print("=" * 60)
    print("📊 預測場與真實場可視化對比")
    print("=" * 60)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n🖥️  使用設備: {device}")
    
    # ----------------------------------------
    # 1. 載入測試資料
    # ----------------------------------------
    test_data = load_test_data(TEST_DATA)
    
    # ----------------------------------------
    # 2. 載入模型並預測
    # ----------------------------------------
    print("\n" + "=" * 60)
    print("🔮 載入模型並預測...")
    print("=" * 60)
    
    # Baseline
    baseline_model, baseline_config = load_model(BASELINE_CONFIG, BASELINE_CHECKPOINT, device)
    print("\n📊 Baseline 預測中...")
    baseline_pred = predict(baseline_model, test_data['coords'], device)
    
    # 後處理縮放
    from pinnx.utils.denormalization import denormalize_output
    baseline_pred = denormalize_output(
        baseline_pred, 
        baseline_config, 
        output_norm_type='post_scaling',
        true_ranges=test_data['true_ranges'],
        verbose=False
    )
    
    # Fourier
    fourier_model, fourier_config = load_model(FOURIER_CONFIG, FOURIER_CHECKPOINT, device)
    print("\n📊 Fourier 預測中...")
    fourier_pred = predict(fourier_model, test_data['coords'], device)
    
    # 後處理縮放
    fourier_pred = denormalize_output(
        fourier_pred, 
        fourier_config, 
        output_norm_type='post_scaling',
        true_ranges=test_data['true_ranges'],
        verbose=False
    )
    
    # ----------------------------------------
    # 3. 生成可視化
    # ----------------------------------------
    print("\n" + "=" * 60)
    print("🎨 生成可視化圖表...")
    print("=" * 60)
    
    var_names = ['u', 'v', 'w', 'p']
    var_indices = [0, 1, 2, 3]
    
    for i, var_name in enumerate(var_names):
        print(f"\n📈 處理變量: {var_name.upper()}")
        
        true_field = test_data[var_name]
        baseline_field = baseline_pred[:, i]
        fourier_field = fourier_pred[:, i]
        
        # 3.1 2D 切片對比 (Baseline)
        print("  - 生成 2D 切片對比圖 (Baseline)...")
        plot_2d_slice_comparison(
            true_field, baseline_field, test_data['coords'], test_data['grid_shape'],
            var_name, slice_axis='z', slice_idx=None,
            save_path=OUTPUT_DIR / f"{var_name}_baseline_slice_z.png"
        )
        
        # 3.2 2D 切片對比 (Fourier)
        print("  - 生成 2D 切片對比圖 (Fourier)...")
        plot_2d_slice_comparison(
            true_field, fourier_field, test_data['coords'], test_data['grid_shape'],
            var_name, slice_axis='z', slice_idx=None,
            save_path=OUTPUT_DIR / f"{var_name}_fourier_slice_z.png"
        )
        
        # 3.3 統計分佈對比 (Baseline)
        print("  - 生成統計分佈對比圖 (Baseline)...")
        plot_statistical_comparison(
            true_field, baseline_field, var_name,
            save_path=OUTPUT_DIR / f"{var_name}_baseline_statistics.png"
        )
        
        # 3.4 統計分佈對比 (Fourier)
        print("  - 生成統計分佈對比圖 (Fourier)...")
        plot_statistical_comparison(
            true_field, fourier_field, var_name,
            save_path=OUTPUT_DIR / f"{var_name}_fourier_statistics.png"
        )
        
        # 3.5 剖面線對比 (Baseline)
        print("  - 生成剖面線對比圖 (Baseline)...")
        plot_profile_comparison(
            true_field, baseline_field, test_data['coords'], test_data['grid_shape'],
            var_name, profile_axis='y',
            save_path=OUTPUT_DIR / f"{var_name}_baseline_profile_y.png"
        )
        
        # 3.6 剖面線對比 (Fourier)
        print("  - 生成剖面線對比圖 (Fourier)...")
        plot_profile_comparison(
            true_field, fourier_field, test_data['coords'], test_data['grid_shape'],
            var_name, profile_axis='y',
            save_path=OUTPUT_DIR / f"{var_name}_fourier_profile_y.png"
        )
    
    print("\n" + "=" * 60)
    print("✅ 可視化完成！")
    print("=" * 60)
    print(f"\n📁 結果位置: {OUTPUT_DIR}")
    print("\n生成的圖表：")
    print("  - *_slice_z.png: 2D 切片對比圖（Z 平面）")
    print("  - *_statistics.png: 統計分佈對比圖")
    print("  - *_profile_y.png: 沿通道高度的剖面線對比")
    print("  - Baseline 和 Fourier 各自的對比圖")
    print()

if __name__ == "__main__":
    main()
