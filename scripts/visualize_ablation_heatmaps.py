#!/usr/bin/env python3
"""
PDE 約束消融實驗 - 中心層熱圖視覺化
生成預測場 vs 真實場的 2D 熱圖對比（聚焦中心層誤差分析）
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

# 添加項目路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinnx.train.factory import create_model, get_device
from pinnx.train.config_loader import load_config

# ============================================================================
# 常數定義
# ============================================================================
# 中心層定義（基於歸一化座標 y/h，h=1）
CENTER_LAYER_Y_RANGE = (0.0, 0.2)

# ============================================================================
# 載入模型與數據
# ============================================================================
def load_experiment_model(
    checkpoint_path: Path,
    config_path: Path,
    device: torch.device
) -> torch.nn.Module:
    """載入實驗模型"""
    config = load_config(str(config_path))
    model = create_model(config, device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', {}))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model


def load_jhtdb_2d_slice(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    載入 JHTDB 2D 切片真實數據
    
    Returns:
        x, y: 1D 座標陣列
        coords: (N, 3) 座標數組 [x, y, z]
        fields: (N, 4) 場數據 [u, v, w, p]
    """
    velocity_file = data_dir / "raw" / "JHU Turbulence Channel_velocity_t1.h5"
    pressure_file = data_dir / "raw" / "JHU Turbulence Channel_pressure_t1.h5"
    
    with h5py.File(velocity_file, 'r') as fv:
        velocity = np.array(fv['Velocity_0001'])  # (nx, ny, nz, 3)
        x = np.array(fv['xcoor'])
        y = np.array(fv['ycoor'])
        z = np.array(fv['zcoor'])
        
        # 取中心 Z 切片
        z_idx = velocity.shape[2] // 2
        z_center = z[z_idx]
        
        u_2d = velocity[:, :, z_idx, 0]  # (nx, ny)
        v_2d = velocity[:, :, z_idx, 1]
        w_2d = velocity[:, :, z_idx, 2]
    
    # 載入壓力場
    if pressure_file.exists():
        with h5py.File(pressure_file, 'r') as fp:
            pressure = np.array(fp['Pressure_0001'])  # (nx, ny, nz, 1)
            p_2d = pressure[:, :, z_idx, 0]
    else:
        p_2d = np.zeros_like(u_2d)
    
    # 創建網格
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = np.full_like(X, z_center)
    
    # 展平
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    fields = np.stack([u_2d.ravel(), v_2d.ravel(), w_2d.ravel(), p_2d.ravel()], axis=1)
    
    return x, y, coords, fields


def predict_full_field(
    model: torch.nn.Module,
    coords: np.ndarray,
    device: torch.device,
    batch_size: int = 8192
) -> np.ndarray:
    """批次預測完整場"""
    model.eval()
    n_points = coords.shape[0]
    n_batches = (n_points + batch_size - 1) // batch_size
    
    predictions = []
    
    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_points)
            
            batch_coords = torch.from_numpy(coords[start_idx:end_idx]).float().to(device)
            batch_pred = model(batch_coords)
            predictions.append(batch_pred.cpu().numpy())
    
    return np.vstack(predictions)


# ============================================================================
# 熱圖視覺化
# ============================================================================
def visualize_heatmaps(
    x: np.ndarray,
    y: np.ndarray,
    true_fields: np.ndarray,
    exp_predictions: Dict[str, np.ndarray],
    output_dir: Path
):
    """
    生成中心層 U 速度熱圖對比
    
    Args:
        x, y: 1D 座標陣列
        true_fields: (N, 4) 真實場數據
        exp_predictions: {exp_name: (N, 4) 預測場數據}
        output_dir: 輸出目錄
    """
    # 設定中文字體
    mpl.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    mpl.rcParams['axes.unicode_minus'] = False
    
    # 提取中心層數據
    nx, ny = len(x), len(y)
    y_norm = np.abs(y)
    center_mask = (y_norm >= CENTER_LAYER_Y_RANGE[0]) & (y_norm < CENTER_LAYER_Y_RANGE[1])
    center_indices = np.where(center_mask)[0]
    
    # 重塑為 2D 網格
    U_true_2d = true_fields[:, 0].reshape(nx, ny)
    
    # 只顯示中心層（y ∈ [0, 0.2]）
    U_true_center = U_true_2d[:, center_indices]
    y_center = y[center_indices]
    
    # 建立實驗列表
    exp_names = list(exp_predictions.keys())
    n_exp = len(exp_names)
    
    # ========================================================================
    # 圖 1: 真實場 + 三個實驗預測場（4x1 網格）
    # ========================================================================
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(4, 1, hspace=0.3, figure=fig)
    
    # 計算全域色階範圍（保持一致性）
    vmin = U_true_center.min()
    vmax = U_true_center.max()
    
    # 子圖 1: 真實場
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(
        U_true_center.T,
        aspect='auto',
        origin='lower',
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax,
        extent=(x.min(), x.max(), y_center.min(), y_center.max())
    )
    ax0.set_title('Ground Truth (JHTDB DNS)', fontsize=14, fontweight='bold')
    ax0.set_ylabel('Y (Center Layer)', fontsize=12)
    cbar0 = plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    cbar0.set_label('U Velocity', fontsize=10)
    
    # 子圖 2-4: 實驗預測場
    last_ax = None
    for idx, exp_name in enumerate(exp_names):
        ax = fig.add_subplot(gs[idx + 1, 0])
        
        # 提取預測場
        U_pred_2d = exp_predictions[exp_name][:, 0].reshape(nx, ny)
        U_pred_center = U_pred_2d[:, center_indices]
        
        im = ax.imshow(
            U_pred_center.T,
            aspect='auto',
            origin='lower',
            cmap='RdBu_r',
            vmin=vmin,
            vmax=vmax,
            extent=(x.min(), x.max(), y_center.min(), y_center.max())
        )
        
        # 計算相對誤差
        rel_error = np.linalg.norm(U_pred_center - U_true_center) / np.linalg.norm(U_true_center)
        
        ax.set_title(f'{exp_name} (Rel. L2 Error: {rel_error:.4f})', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y (Center Layer)', fontsize=12)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('U Velocity', fontsize=10)
        
        last_ax = ax  # 記錄最後一個子圖
    
    # 設定最後一行的 X 軸標籤
    if last_ax is not None:
        last_ax.set_xlabel('X (Streamwise)', fontsize=12)
    
    plt.savefig(output_dir / 'center_layer_u_velocity_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 熱圖 1 已保存: {output_dir / 'center_layer_u_velocity_heatmaps.png'}")
    
    # ========================================================================
    # 圖 2: 誤差場熱圖（3x1 網格）
    # ========================================================================
    fig2 = plt.figure(figsize=(20, 9))
    gs2 = GridSpec(3, 1, hspace=0.3, figure=fig2)
    
    last_ax2 = None
    for idx, exp_name in enumerate(exp_names):
        ax = fig2.add_subplot(gs2[idx, 0])
        
        # 提取預測場
        U_pred_2d = exp_predictions[exp_name][:, 0].reshape(nx, ny)
        U_pred_center = U_pred_2d[:, center_indices]
        
        # 計算絕對誤差
        error = np.abs(U_pred_center - U_true_center)
        
        im = ax.imshow(
            error.T,
            aspect='auto',
            origin='lower',
            cmap='hot_r',
            extent=(x.min(), x.max(), y_center.min(), y_center.max())
        )
        
        # 統計誤差
        mean_error = error.mean()
        max_error = error.max()
        rel_error = np.linalg.norm(U_pred_center - U_true_center) / np.linalg.norm(U_true_center)
        
        ax.set_title(
            f'{exp_name} - Absolute Error (Mean: {mean_error:.4f}, Max: {max_error:.4f}, Rel. L2: {rel_error:.4f})',
            fontsize=14,
            fontweight='bold'
        )
        ax.set_ylabel('Y (Center Layer)', fontsize=12)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('|Error|', fontsize=10)
        
        last_ax2 = ax  # 記錄最後一個子圖
    
    # 設定最後一行的 X 軸標籤
    if last_ax2 is not None:
        last_ax2.set_xlabel('X (Streamwise)', fontsize=12)
    
    plt.savefig(output_dir / 'center_layer_u_velocity_error_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 熱圖 2 已保存: {output_dir / 'center_layer_u_velocity_error_heatmaps.png'}")
    
    # ========================================================================
    # 圖 3: 沿流向平均剖面（1D 曲線）
    # ========================================================================
    fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左圖: 平均速度剖面
    ax_left = axes[0]
    ax_left.plot(y_center, U_true_center.mean(axis=0), 'k-', linewidth=2.5, label='Ground Truth', marker='o', markersize=4)
    
    for exp_name in exp_names:
        U_pred_2d = exp_predictions[exp_name][:, 0].reshape(nx, ny)
        U_pred_center = U_pred_2d[:, center_indices]
        ax_left.plot(y_center, U_pred_center.mean(axis=0), linewidth=2, label=exp_name, marker='s', markersize=3, alpha=0.8)
    
    ax_left.set_xlabel('Y (Center Layer)', fontsize=12)
    ax_left.set_ylabel('Streamwise-Averaged U', fontsize=12)
    ax_left.set_title('Averaged Velocity Profile', fontsize=14, fontweight='bold')
    ax_left.legend(fontsize=10)
    ax_left.grid(alpha=0.3, linestyle='--')
    
    # 右圖: 平均絕對誤差剖面
    ax_right = axes[1]
    
    for exp_name in exp_names:
        U_pred_2d = exp_predictions[exp_name][:, 0].reshape(nx, ny)
        U_pred_center = U_pred_2d[:, center_indices]
        error = np.abs(U_pred_center - U_true_center).mean(axis=0)
        ax_right.plot(y_center, error, linewidth=2, label=exp_name, marker='s', markersize=3, alpha=0.8)
    
    ax_right.set_xlabel('Y (Center Layer)', fontsize=12)
    ax_right.set_ylabel('Mean Absolute Error', fontsize=12)
    ax_right.set_title('Error Profile', fontsize=14, fontweight='bold')
    ax_right.legend(fontsize=10)
    ax_right.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'center_layer_u_velocity_profiles.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 熱圖 3 已保存: {output_dir / 'center_layer_u_velocity_profiles.png'}")


# ============================================================================
# 主流程
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="PDE 約束消融實驗 - 中心層熱圖視覺化")
    parser.add_argument(
        '--exp1-checkpoint',
        type=str,
        default='./checkpoints/ablation_exp1_qr_no_pde/best_model.pth',
        help='實驗 1 檢查點（QR No-PDE）'
    )
    parser.add_argument(
        '--exp2-checkpoint',
        type=str,
        default='./checkpoints/ablation_exp2_qr_weak_pde/best_model.pth',
        help='實驗 2 檢查點（QR Weak-PDE）'
    )
    parser.add_argument(
        '--exp3-checkpoint',
        type=str,
        default='./checkpoints/ablation_exp3_wall_no_center/best_model.pth',
        help='實驗 3 檢查點（Wall No-Center）'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data/jhtdb/channel_flow_re1000',
        help='JHTDB 資料目錄'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results/ablation_experiments',
        help='輸出目錄'
    )
    parser.add_argument('--device', type=str, default='auto', help='計算設備')
    
    args = parser.parse_args()
    
    # 路徑轉換
    project_root = Path(__file__).parent.parent
    exp1_checkpoint = project_root / args.exp1_checkpoint
    exp2_checkpoint = project_root / args.exp2_checkpoint
    exp3_checkpoint = project_root / args.exp3_checkpoint
    data_dir = project_root / args.data_dir
    output_dir = project_root / args.output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("PDE 約束消融實驗 - 中心層熱圖視覺化")
    print("=" * 80)
    
    # 獲取設備
    device = get_device(args.device)
    print(f"使用設備: {device}")
    
    # 定義實驗
    experiments = {
        'Exp1 (QR No-PDE)': {
            'checkpoint': exp1_checkpoint,
            'config': project_root / 'configs/ablation_pde_constraint/exp1_qr_no_pde.yml'
        },
        'Exp2 (QR Weak-PDE)': {
            'checkpoint': exp2_checkpoint,
            'config': project_root / 'configs/ablation_pde_constraint/exp2_qr_weak_pde.yml'
        },
        'Exp3 (Wall No-Center)': {
            'checkpoint': exp3_checkpoint,
            'config': project_root / 'configs/ablation_pde_constraint/exp3_wall_no_center.yml'
        }
    }
    
    # 載入真實數據
    print("\n步驟 1: 載入 JHTDB 真實數據...")
    x, y, coords, true_fields = load_jhtdb_2d_slice(data_dir)
    print(f"  ✅ 載入完成: {coords.shape[0]:,} 個網格點")
    
    # 預測每個實驗
    exp_predictions = {}
    
    for exp_name, exp_info in experiments.items():
        print(f"\n步驟 2: 評估 {exp_name}...")
        
        # 載入模型
        model = load_experiment_model(
            exp_info['checkpoint'],
            exp_info['config'],
            device
        )
        
        # 預測完整場
        print("  預測完整場...")
        pred_fields = predict_full_field(model, coords, device)
        exp_predictions[exp_name] = pred_fields
        print("  ✅ 預測完成")
    
    # 生成熱圖
    print("\n步驟 3: 生成熱圖視覺化...")
    visualize_heatmaps(x, y, true_fields, exp_predictions, output_dir)
    
    print("\n" + "=" * 80)
    print("✅ 視覺化完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
