#!/usr/bin/env python
"""
分層誤差評估腳本 - 驗證 QR-Pivot 中心層缺失假設
目的：比較 Wall-Clustered 和 QR-Pivot 在三個物理層的重建誤差
"""
import sys
import torch
import numpy as np
import yaml
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional
import json

# 添加項目路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pinnx.models.fourier_mlp import PINNNet, create_pinn_model  # 使用統一模型與工廠函數
from pinnx.utils.normalization import DataNormalizer, InputNormalizer  # 正確的標準化器位置

# ============================================================================
# 常數定義
# ============================================================================
LAYER_DEFINITIONS = {
    'wall': {'range': (0.8, 1.0), 'name': '壁面層', 'description': 'High shear, y+ < 100'},
    'log': {'range': (0.2, 0.8), 'name': '對數層', 'description': 'Turbulent core, 100 < y+ < 800'},
    'center': {'range': (0.0, 0.2), 'name': '中心層', 'description': 'Low gradient, y+ > 800'}
}

# ============================================================================
# 載入模型與數據
# ============================================================================
def load_checkpoint_and_model(checkpoint_path: Path, config_path: Path, device: torch.device) -> Tuple[torch.nn.Module, Dict, DataNormalizer]:
    """載入檢查點與模型"""
    print(f"📂 載入檢查點: {checkpoint_path.name}")
    
    # 載入配置
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 載入檢查點
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 建立標準化器（從 checkpoint metadata 恢復）
    if 'normalization' in checkpoint:
        normalizer = DataNormalizer.from_metadata(checkpoint['normalization'])
        print(f"  ✅ 標準化器已載入: type={normalizer.norm_type}")
    else:
        print(f"  ⚠️  未找到標準化器狀態，使用預設值")
        normalizer = DataNormalizer(norm_type='none')
    
    # 建立模型（優先使用檢查點中的配置，因為它包含完整參數）
    if 'config' in checkpoint and 'model' in checkpoint['config']:
        model_cfg = checkpoint['config']['model'].copy()
        print(f"  ✅ 使用檢查點中的模型配置")
        
        # 🔧 修復：從權重推斷 use_input_projection（如果配置中缺失）
        if 'use_input_projection' not in model_cfg or model_cfg['use_input_projection'] is None:
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', {}))
            first_layer_key = 'hidden_layers.0.linear.weight'
            if first_layer_key in state_dict:
                first_layer_shape = state_dict[first_layer_key].shape
                width = model_cfg.get('width', 256)
                fourier_m = model_cfg.get('fourier_m', 32)
                expected_dim_with_proj = width
                expected_dim_without_proj = 2 * fourier_m if model_cfg.get('use_fourier', True) else model_cfg.get('in_dim', 3)
                
                if first_layer_shape[1] == expected_dim_with_proj:
                    model_cfg['use_input_projection'] = True
                    print(f"  🔧 推斷 use_input_projection=True（第一層輸入={first_layer_shape[1]}）")
                elif first_layer_shape[1] == expected_dim_without_proj:
                    model_cfg['use_input_projection'] = False
                    print(f"  🔧 推斷 use_input_projection=False（第一層輸入={first_layer_shape[1]}）")
    else:
        model_cfg = cfg['model']
        print(f"  ⚠️  檢查點中無配置，使用 YAML 配置")
    
    model = create_pinn_model(model_cfg).to(device)
    
    # 載入權重
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', {}))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print(f"  訓練輪數: {checkpoint.get('epoch', 'N/A')}")
    print(f"  最終損失: {checkpoint.get('loss', 'N/A'):.6e}" if 'loss' in checkpoint else "  最終損失: N/A")
    
    return model, cfg, normalizer


def load_jhtdb_ground_truth(data_path: Path, domain_bounds: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    載入 JHTDB 真實數據（2D/3D 切片）
    
    Args:
        data_path: HDF5 文件路徑
        domain_bounds: 域範圍 {'x': [min, max], 'y': [min, max], 'z': [min, max]}（用於重建座標網格）
    
    Returns:
        coords: (N, 3) 座標數組（x, y, z）
        fields: (N, n_vars) 場數據（u, v, w, p）
    """
    print(f"📂 載入 JHTDB 真實數據: {data_path.name}")
    
    with h5py.File(data_path, 'r') as f:
        # 讀取速度場（不使用 squeeze 以保留所有維度）
        u = np.array(f['u'])  # (nx, ny, nz)
        v = np.array(f['v'])
        w = np.array(f['w'])
        
        # 檢查是否有壓力
        if 'p' in f:
            p = np.array(f['p'])
        else:
            p = None
        
        # 確定網格維度
        shape_3d = u.shape
        if len(shape_3d) == 2:
            # 2D 數據，添加 z 維度
            nx, ny = shape_3d
            nz = 1
            u = u[:, :, np.newaxis]
            v = v[:, :, np.newaxis]
            w = w[:, :, np.newaxis]
            if p is not None:
                p = p[:, :, np.newaxis]
        else:
            nx, ny, nz = shape_3d
        
        # 嘗試讀取座標（如果存在）
        if 'x' in f and 'y' in f and 'z' in f:
            x = np.array(f['x'])
            y = np.array(f['y'])
            z = np.array(f['z'])
        else:
            # 從場數據形狀推斷網格尺寸，並使用配置的域範圍
            if domain_bounds is None:
                # 使用 JHTDB Channel Flow 標準域範圍
                domain_bounds = {
                    'x': [0.0, 8.0 * np.pi],   # 8πh ≈ 25.13
                    'y': [-1.0, 1.0],          # [-h, +h]
                    'z': [0.0, 3.0 * np.pi]    # 3πh ≈ 9.42
                }
            x = np.linspace(domain_bounds['x'][0], domain_bounds['x'][1], nx)
            y = np.linspace(domain_bounds['y'][0], domain_bounds['y'][1], ny)
            z = np.linspace(domain_bounds['z'][0], domain_bounds['z'][1], nz)
            print(f"  ⚠️  未找到座標，從域範圍重建: x=[{x[0]:.2f}, {x[-1]:.2f}], y=[{y[0]:.2f}, {y[-1]:.2f}], z=[{z[0]:.2f}, {z[-1]:.2f}]")
    
    # 組合座標網格 (nx, ny, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # (N, 3)
    
    # 組合速度場 (N, n_vars)
    if p is not None:
        fields = np.stack([u.ravel(), v.ravel(), w.ravel(), p.ravel()], axis=1)  # (N, 4)
    else:
        fields = np.stack([u.ravel(), v.ravel(), w.ravel()], axis=1)  # (N, 3)
    
    print(f"  網格尺寸: {u.shape}")
    print(f"  座標範圍: x=[{x.min():.2f}, {x.max():.2f}], y=[{y.min():.2f}, {y.max():.2f}], z=[{z.min():.2f}, {z.max():.2f}]")
    print(f"  速度範圍: u=[{u.min():.3f}, {u.max():.3f}], v=[{v.min():.3f}, {v.max():.3f}], w=[{w.min():.3f}, {w.max():.3f}]")
    print(f"  場變量: {fields.shape[-1]} 個 ({'u,v,w,p' if p is not None else 'u,v,w'})")
    
    return coords, fields


def predict_full_field(model: torch.nn.Module, coords: np.ndarray, normalizer: DataNormalizer, 
                       device: torch.device, batch_size: int = 4096) -> np.ndarray:
    """預測全場（支援新版 DataNormalizer API）"""
    print(f"🔮 模型預測全場...")
    
    n_points = coords.shape[0]
    predictions = []
    
    # ⚠️ 注意：coords 是空間坐標 (x, y)，不需要用 DataNormalizer 處理
    # DataNormalizer 僅處理輸出變量 (u, v, w, p)
    # 假設模型已在訓練時對輸入進行正確處理（或不需要標準化）
    
    with torch.no_grad():
        for i in range(0, n_points, batch_size):
            batch_coords = coords[i:i+batch_size]
            
            # 轉換為 Tensor（輸入坐標通常不需要標準化，或已在模型內部處理）
            batch_coords_tensor = torch.tensor(batch_coords, dtype=torch.float32).to(device)
            
            # 預測（模型輸出是標準化空間）
            batch_pred_norm = model(batch_coords_tensor)
            
            # 反標準化輸出（使用批次 API）
            var_order = ['u', 'v', 'w', 'p'][:batch_pred_norm.shape[-1]]  # 自動適配輸出維度
            batch_pred = normalizer.denormalize_batch(batch_pred_norm.cpu(), var_order=var_order)
            
            # 轉換為 numpy（如果是 Tensor）
            if isinstance(batch_pred, torch.Tensor):
                batch_pred = batch_pred.numpy()
            
            predictions.append(batch_pred)
    
    predictions = np.vstack(predictions)
    print(f"  預測完成: {predictions.shape}")
    
    return predictions


# ============================================================================
# 分層誤差計算
# ============================================================================
def compute_layer_wise_errors(coords: np.ndarray, predictions: np.ndarray, 
                              ground_truth: np.ndarray) -> Dict:
    """計算分層誤差"""
    print(f"\n📊 計算分層誤差...")
    
    # 提取 y 座標（歸一化到 [0, 1]）
    y_coords = np.abs(coords[:, 1])  # 取絕對值（對稱通道流）
    
    results = {}
    
    for layer_name, layer_info in LAYER_DEFINITIONS.items():
        y_min, y_max = layer_info['range']
        
        # 選擇該層的點
        mask = (y_coords >= y_min) & (y_coords < y_max)
        n_points = mask.sum()
        
        if n_points == 0:
            print(f"  ⚠️  {layer_info['name']} ({layer_name}): 無數據點")
            results[layer_name] = {
                'n_points': 0,
                'l2_error': np.nan,
                'relative_l2': np.nan,
                'u_error': np.nan,
                'v_error': np.nan,
                'w_error': np.nan
            }
            continue
        
        # 提取該層數據
        pred_layer = predictions[mask]
        gt_layer = ground_truth[mask]
        
        # 處理變量數量不匹配（預測可能有 p，真實數據可能沒有）
        n_vars = min(pred_layer.shape[1], gt_layer.shape[1])
        pred_layer = pred_layer[:, :n_vars]
        gt_layer = gt_layer[:, :n_vars]
        
        # 計算 L2 誤差
        diff = pred_layer - gt_layer
        l2_error = np.linalg.norm(diff, axis=0)  # 每個變量
        gt_norm = np.linalg.norm(gt_layer, axis=0)
        relative_l2 = l2_error / (gt_norm + 1e-10)
        
        # 總體相對 L2
        total_l2 = np.linalg.norm(diff)
        total_gt = np.linalg.norm(gt_layer)
        total_relative = total_l2 / (total_gt + 1e-10)
        
        # 構建結果字典（動態支援 2/3/4 個變量）
        result_dict = {
            'n_points': int(n_points),
            'y_range': layer_info['range'],
            'l2_error': float(total_l2),
            'relative_l2': float(total_relative),
            'u_error': float(relative_l2[0]),
            'v_error': float(relative_l2[1]) if n_vars > 1 else np.nan,
            'w_error': float(relative_l2[2]) if n_vars > 2 else np.nan,
            'p_error': float(relative_l2[3]) if n_vars > 3 else np.nan,
            'name': layer_info['name'],
            'description': layer_info['description']
        }
        results[layer_name] = result_dict
        
        # 動態生成誤差輸出字符串
        var_names = ['u', 'v', 'w', 'p'][:n_vars]
        error_str = ' / '.join([f"{relative_l2[i]:.4f}" for i in range(n_vars)])
        var_label = ' / '.join(var_names)
        
        print(f"  {layer_info['name']} ({layer_name}):")
        print(f"    點數: {n_points:,} ({n_points/len(y_coords)*100:.1f}%)")
        print(f"    相對 L2: {total_relative:.4f}")
        print(f"    {var_label} 誤差: {error_str}")
    
    return results


def compare_strategies(results_wall: Dict, results_qr: Dict) -> Dict:
    """比較兩種策略"""
    print(f"\n🔍 策略比較分析...")
    
    comparison = {}
    
    for layer_name in LAYER_DEFINITIONS.keys():
        wall_error = results_wall[layer_name]['relative_l2']
        qr_error = results_qr[layer_name]['relative_l2']
        
        if np.isnan(wall_error) or np.isnan(qr_error):
            ratio = np.nan
            advantage = "N/A"
        else:
            ratio = qr_error / wall_error
            if ratio < 0.95:
                advantage = "QR-Pivot 優勢"
            elif ratio > 1.05:
                advantage = "Wall-Clustered 優勢"
            else:
                advantage = "相當"
        
        comparison[layer_name] = {
            'wall_error': float(wall_error) if not np.isnan(wall_error) else None,
            'qr_error': float(qr_error) if not np.isnan(qr_error) else None,
            'ratio': float(ratio) if not np.isnan(ratio) else None,
            'advantage': advantage,
            'name': LAYER_DEFINITIONS[layer_name]['name']
        }
        
        print(f"  {LAYER_DEFINITIONS[layer_name]['name']}:")
        print(f"    Wall-Clustered: {wall_error:.4f}")
        print(f"    QR-Pivot: {qr_error:.4f}")
        print(f"    比值 (QR/Wall): {ratio:.4f} ({advantage})")
    
    return comparison


# ============================================================================
# 視覺化
# ============================================================================
def plot_layer_wise_errors(results_wall: Dict, results_qr: Dict, output_dir: Path):
    """繪製分層誤差對比圖"""
    print(f"\n📈 生成視覺化圖表...")
    
    # 準備數據
    layers = list(LAYER_DEFINITIONS.keys())
    layer_names = [LAYER_DEFINITIONS[l]['name'] for l in layers]
    
    wall_errors = [results_wall[l]['relative_l2'] for l in layers]
    qr_errors = [results_qr[l]['relative_l2'] for l in layers]
    
    # 創建圖表
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # === 圖 1: 分層誤差對比 ===
    ax = axes[0]
    x = np.arange(len(layers))
    width = 0.35
    
    ax.bar(x - width/2, wall_errors, width, label='Wall-Clustered', alpha=0.8, color='#3498db')
    ax.bar(x + width/2, qr_errors, width, label='QR-Pivot', alpha=0.8, color='#e74c3c')
    
    ax.set_xlabel('Physical Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Relative L2 Error', fontsize=12, fontweight='bold')
    ax.set_title('Layer-wise Reconstruction Error Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # === 圖 2: 誤差比值 ===
    ax = axes[1]
    ratios = [qr_errors[i] / wall_errors[i] if wall_errors[i] > 0 else np.nan for i in range(len(layers))]
    colors = ['#27ae60' if r < 1.0 else '#e67e22' if r > 1.0 else '#95a5a6' for r in ratios]
    
    ax.bar(x, ratios, color=colors, alpha=0.8)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Equal Performance')
    
    ax.set_xlabel('Physical Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error Ratio (QR / Wall)', fontsize=12, fontweight='bold')
    ax.set_title('QR-Pivot vs Wall-Clustered Error Ratio', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 標註數值
    for i, ratio in enumerate(ratios):
        if not np.isnan(ratio):
            ax.text(i, ratio + 0.05, f'{ratio:.2f}x', ha='center', fontsize=10, fontweight='bold')
    
    # === 圖 3: 速度分量誤差 ===
    ax = axes[2]
    
    # Wall-Clustered
    wall_u = [results_wall[l]['u_error'] for l in layers]
    wall_v = [results_wall[l]['v_error'] for l in layers]
    wall_w = [results_wall[l]['w_error'] for l in layers]
    
    # QR-Pivot
    qr_u = [results_qr[l]['u_error'] for l in layers]
    qr_v = [results_qr[l]['v_error'] for l in layers]
    qr_w = [results_qr[l]['w_error'] for l in layers]
    
    width = 0.12
    x_pos = np.arange(len(layers))
    
    ax.bar(x_pos - 1.5*width, wall_u, width, label='Wall u', alpha=0.8, color='#3498db')
    ax.bar(x_pos - 0.5*width, wall_v, width, label='Wall v', alpha=0.8, color='#2ecc71')
    ax.bar(x_pos + 0.5*width, wall_w, width, label='Wall w', alpha=0.8, color='#9b59b6')
    
    ax.bar(x_pos + 1.5*width, qr_u, width, label='QR u', alpha=0.8, color='#e74c3c', hatch='//')
    ax.bar(x_pos + 2.5*width, qr_v, width, label='QR v', alpha=0.8, color='#f39c12', hatch='//')
    ax.bar(x_pos + 3.5*width, qr_w, width, label='QR w', alpha=0.8, color='#e67e22', hatch='//')
    
    ax.set_xlabel('Physical Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Relative L2 Error', fontsize=12, fontweight='bold')
    ax.set_title('Velocity Component Errors by Layer', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(layer_names)
    ax.legend(ncol=2, fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'layer_wise_error_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ 已儲存: {output_path}")
    
    plt.close()


def plot_spatial_error_distribution(coords: np.ndarray, predictions_wall: np.ndarray, 
                                    predictions_qr: np.ndarray, ground_truth: np.ndarray, 
                                    output_dir: Path):
    """繪製空間誤差分佈熱圖"""
    print(f"  生成空間誤差分佈圖...")
    
    # 處理變量數量不匹配
    n_vars = min(predictions_wall.shape[1], ground_truth.shape[1])
    pred_wall_aligned = predictions_wall[:, :n_vars]
    pred_qr_aligned = predictions_qr[:, :n_vars]
    gt_aligned = ground_truth[:, :n_vars]
    
    # 計算誤差
    error_wall = np.linalg.norm(pred_wall_aligned - gt_aligned, axis=1)
    error_qr = np.linalg.norm(pred_qr_aligned - gt_aligned, axis=1)
    
    # 重塑為 2D 網格（假設 coords 來自 meshgrid）
    x_unique = np.unique(coords[:, 0])
    y_unique = np.unique(coords[:, 1])
    nx, ny = len(x_unique), len(y_unique)
    
    error_wall_2d = error_wall.reshape(nx, ny)
    error_qr_2d = error_qr.reshape(nx, ny)
    
    # 創建圖表
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # === Wall-Clustered 誤差 ===
    ax = axes[0]
    im1 = ax.imshow(error_wall_2d.T, origin='lower', extent=[x_unique.min(), x_unique.max(), 
                                                              y_unique.min(), y_unique.max()],
                   cmap='viridis', aspect='auto')
    ax.set_title('Wall-Clustered: Spatial Error Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    plt.colorbar(im1, ax=ax, label='L2 Error')
    
    # 標註物理層
    ax.axhline(y=0.8, color='white', linestyle='--', linewidth=1.5, label='Log Layer')
    ax.axhline(y=0.2, color='white', linestyle='--', linewidth=1.5)
    ax.axhline(y=-0.8, color='white', linestyle='--', linewidth=1.5)
    ax.axhline(y=-0.2, color='white', linestyle='--', linewidth=1.5)
    
    # === QR-Pivot 誤差 ===
    ax = axes[1]
    im2 = ax.imshow(error_qr_2d.T, origin='lower', extent=[x_unique.min(), x_unique.max(), 
                                                            y_unique.min(), y_unique.max()],
                   cmap='viridis', aspect='auto')
    ax.set_title('QR-Pivot: Spatial Error Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    plt.colorbar(im2, ax=ax, label='L2 Error')
    
    # 標註物理層
    ax.axhline(y=0.8, color='white', linestyle='--', linewidth=1.5)
    ax.axhline(y=0.2, color='white', linestyle='--', linewidth=1.5)
    ax.axhline(y=-0.8, color='white', linestyle='--', linewidth=1.5)
    ax.axhline(y=-0.2, color='white', linestyle='--', linewidth=1.5)
    
    # === 誤差差異 (QR - Wall) ===
    ax = axes[2]
    error_diff = error_qr_2d - error_wall_2d
    vmax = np.abs(error_diff).max()
    im3 = ax.imshow(error_diff.T, origin='lower', extent=[x_unique.min(), x_unique.max(), 
                                                          y_unique.min(), y_unique.max()],
                   cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_title('Error Difference (QR - Wall)\nRed = QR worse, Blue = QR better', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    plt.colorbar(im3, ax=ax, label='Error Difference')
    
    # 標註物理層
    ax.axhline(y=0.8, color='black', linestyle='--', linewidth=1.5)
    ax.axhline(y=0.2, color='black', linestyle='--', linewidth=1.5)
    ax.axhline(y=-0.8, color='black', linestyle='--', linewidth=1.5)
    ax.axhline(y=-0.2, color='black', linestyle='--', linewidth=1.5)
    
    plt.tight_layout()
    output_path = output_dir / 'spatial_error_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ 已儲存: {output_path}")
    
    plt.close()


# ============================================================================
# 主函數
# ============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='分層誤差評估')
    parser.add_argument('--wall-checkpoint', type=str, required=True, help='Wall-Clustered 檢查點路徑')
    parser.add_argument('--wall-config', type=str, required=True, help='Wall-Clustered 配置文件')
    parser.add_argument('--qr-checkpoint', type=str, required=True, help='QR-Pivot 檢查點路徑')
    parser.add_argument('--qr-config', type=str, required=True, help='QR-Pivot 配置文件')
    parser.add_argument('--jhtdb-data', type=str, required=True, help='JHTDB 真實數據 (HDF5)')
    parser.add_argument('--output', type=str, default='./results/layer_wise_analysis', help='輸出目錄')
    args = parser.parse_args()
    
    # 路徑處理
    wall_ckpt = Path(args.wall_checkpoint)
    wall_cfg = Path(args.wall_config)
    qr_ckpt = Path(args.qr_checkpoint)
    qr_cfg = Path(args.qr_config)
    jhtdb_data = Path(args.jhtdb_data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("  分層誤差評估 - 驗證 QR-Pivot 中心層缺失假設")
    print("=" * 80)
    
    # 設備
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  使用設備: {device}")
    
    # === 1. 載入真實數據 ===
    print(f"\n{'='*80}")
    print("步驟 1: 載入 JHTDB 真實數據")
    print("="*80)
    coords, ground_truth = load_jhtdb_ground_truth(jhtdb_data)
    
    # === 2. 載入 Wall-Clustered 模型並預測 ===
    print(f"\n{'='*80}")
    print("步驟 2: 載入 Wall-Clustered 模型")
    print("="*80)
    model_wall, cfg_wall, norm_wall = load_checkpoint_and_model(wall_ckpt, wall_cfg, device)
    predictions_wall = predict_full_field(model_wall, coords, norm_wall, device)
    
    # === 3. 載入 QR-Pivot 模型並預測 ===
    print(f"\n{'='*80}")
    print("步驟 3: 載入 QR-Pivot 模型")
    print("="*80)
    model_qr, cfg_qr, norm_qr = load_checkpoint_and_model(qr_ckpt, qr_cfg, device)
    predictions_qr = predict_full_field(model_qr, coords, norm_qr, device)
    
    # === 4. 計算分層誤差 ===
    print(f"\n{'='*80}")
    print("步驟 4: 計算分層誤差")
    print("="*80)
    
    print(f"\n--- Wall-Clustered ---")
    results_wall = compute_layer_wise_errors(coords, predictions_wall, ground_truth)
    
    print(f"\n--- QR-Pivot ---")
    results_qr = compute_layer_wise_errors(coords, predictions_qr, ground_truth)
    
    # === 5. 比較分析 ===
    print(f"\n{'='*80}")
    print("步驟 5: 策略比較")
    print("="*80)
    comparison = compare_strategies(results_wall, results_qr)
    
    # === 6. 視覺化 ===
    print(f"\n{'='*80}")
    print("步驟 6: 視覺化")
    print("="*80)
    plot_layer_wise_errors(results_wall, results_qr, output_dir)
    plot_spatial_error_distribution(coords, predictions_wall, predictions_qr, ground_truth, output_dir)
    
    # === 7. 儲存結果 ===
    print(f"\n{'='*80}")
    print("步驟 7: 儲存結果")
    print("="*80)
    
    results = {
        'wall_clustered': results_wall,
        'qr_pivot': results_qr,
        'comparison': comparison,
        'metadata': {
            'wall_checkpoint': str(wall_ckpt),
            'qr_checkpoint': str(qr_ckpt),
            'jhtdb_data': str(jhtdb_data),
            'n_points': int(coords.shape[0]),
            'device': str(device)
        }
    }
    
    json_path = output_dir / 'layer_wise_error_analysis.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✅ JSON 報告已儲存: {json_path}")
    
    # 生成 Markdown 報告
    md_path = output_dir / 'layer_wise_error_report.md'
    with open(md_path, 'w') as f:
        f.write("# 分層誤差分析報告\n\n")
        f.write("## 執行摘要\n\n")
        f.write("本報告驗證 **QR-Pivot 中心層缺失假設**，比較兩種感測點策略在三個物理層的重建誤差。\n\n")
        
        f.write("## 關鍵發現\n\n")
        f.write("| 物理層 | Wall-Clustered | QR-Pivot | 誤差比值 (QR/Wall) | 優勢 |\n")
        f.write("|--------|----------------|----------|-------------------|------|\n")
        for layer_name, comp in comparison.items():
            f.write(f"| {comp['name']} | {comp['wall_error']:.4f} | {comp['qr_error']:.4f} | "
                   f"{comp['ratio']:.2f}× | {comp['advantage']} |\n")
        
        f.write("\n## 詳細分析\n\n")
        for layer_name, layer_info in LAYER_DEFINITIONS.items():
            f.write(f"### {layer_info['name']} ({layer_info['description']})\n\n")
            f.write(f"**Wall-Clustered**:\n")
            f.write(f"- 點數: {results_wall[layer_name]['n_points']:,}\n")
            f.write(f"- 相對 L2: {results_wall[layer_name]['relative_l2']:.4f}\n")
            f.write(f"- u/v/w 誤差: {results_wall[layer_name]['u_error']:.4f} / "
                   f"{results_wall[layer_name]['v_error']:.4f} / {results_wall[layer_name]['w_error']:.4f}\n\n")
            
            f.write(f"**QR-Pivot**:\n")
            f.write(f"- 點數: {results_qr[layer_name]['n_points']:,}\n")
            f.write(f"- 相對 L2: {results_qr[layer_name]['relative_l2']:.4f}\n")
            f.write(f"- u/v/w 誤差: {results_qr[layer_name]['u_error']:.4f} / "
                   f"{results_qr[layer_name]['v_error']:.4f} / {results_qr[layer_name]['w_error']:.4f}\n\n")
        
        f.write("## 結論\n\n")
        f.write("根據分層誤差分析，")
        center_comp = comparison['center']
        if center_comp['ratio'] and center_comp['ratio'] > 1.05:
            f.write(f"**中心層假設得到驗證**：QR-Pivot 的中心層誤差比 Wall-Clustered 高 "
                   f"{(center_comp['ratio']-1)*100:.1f}%，證實了因缺乏感測點導致的誤差累積。\n")
        else:
            f.write(f"**中心層假設未得到驗證**：QR-Pivot 的中心層誤差與 Wall-Clustered 相當或更低，"
                   f"表明其他機制（如 PDE 約束）有效補償了感測點缺失。\n")
    
    print(f"  ✅ Markdown 報告已儲存: {md_path}")
    
    print(f"\n{'='*80}")
    print("✅ 分層誤差評估完成！")
    print(f"輸出目錄: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
