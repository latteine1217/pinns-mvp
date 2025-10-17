#!/usr/bin/env python3
"""
PDE 約束補償機制消融實驗 - 分層誤差評估
目的：驗證 PDE 約束是否為 QR-Pivot 中心層誤差補償的關鍵機制
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
import json

import numpy as np
import torch
import yaml
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

# 添加項目路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinnx.train.factory import create_model, get_device
from pinnx.train.config_loader import load_config

# ============================================================================
# 常數定義
# ============================================================================
# 通道流分層定義（基於歸一化座標 y/h，h=1）
LAYER_DEFINITIONS = {
    'wall': {'y_range': (0.8, 1.0), 'name': '壁面層', 'color': '#e74c3c'},
    'log': {'y_range': (0.2, 0.8), 'name': '對數層', 'color': '#3498db'},
    'center': {'y_range': (0.0, 0.2), 'name': '中心層', 'color': '#2ecc71'}
}

# ============================================================================
# 日誌設定
# ============================================================================
def setup_logging(output_dir: Path, verbose: bool = False):
    """設定日誌"""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_file = output_dir / "evaluation_log.txt"
    
    # 清除現有處理器
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 設定格式
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

# ============================================================================
# 載入模型與數據
# ============================================================================
def load_experiment_model(
    checkpoint_path: Path,
    config_path: Path,
    device: torch.device
) -> Tuple[torch.nn.Module, Dict]:
    """
    載入實驗檢查點與模型
    
    Returns:
        (model, config): 已載入權重的模型與配置
    """
    logging.info(f"載入實驗: {checkpoint_path.parent.name}")
    logging.info(f"  檢查點: {checkpoint_path.name}")
    logging.info(f"  配置: {config_path.name}")
    
    # 載入配置
    config = load_config(str(config_path))
    
    # 創建模型（使用訓練腳本的工廠函數）
    model = create_model(config, device)
    
    # 載入檢查點
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 載入模型權重
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', {}))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    epoch = checkpoint.get('epoch', 'N/A')
    loss = checkpoint.get('loss', float('nan'))
    
    logging.info(f"  ✅ 模型載入成功")
    logging.info(f"     訓練輪數: {epoch}")
    logging.info(f"     最終損失: {loss:.6e}")
    
    return model, config


def load_jhtdb_2d_slice(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    載入 JHTDB 2D 切片真實數據（從原始 HDF5 提取 XY 平面）
    
    Returns:
        coords: (N, 3) 座標數組 [x, y, z]
        fields: (N, 4) 場數據 [u, v, w, p]
    """
    velocity_file = data_dir / "raw" / "JHU Turbulence Channel_velocity_t1.h5"
    pressure_file = data_dir / "raw" / "JHU Turbulence Channel_pressure_t1.h5"
    
    if not velocity_file.exists():
        raise FileNotFoundError(f"速度場文件未找到: {velocity_file}")
    
    logging.info("載入 JHTDB 2D 切片數據...")
    
    with h5py.File(velocity_file, 'r') as fv:
        # 檢查檔案結構
        if 'Velocity_0001' in fv:
            # JHTDB 格式：(nx, ny, nz, 3)
            velocity = np.array(fv['Velocity_0001'])  # (nx, ny, nz, 3)
            x = np.array(fv['xcoor'])
            y = np.array(fv['ycoor'])
            z = np.array(fv['zcoor'])
            
            # 取中心 Z 切片
            z_idx = velocity.shape[2] // 2
            z_center = z[z_idx] if len(z) > z_idx else 3.0 * np.pi / 2
            
            u_2d = velocity[:, :, z_idx, 0]  # (nx, ny)
            v_2d = velocity[:, :, z_idx, 1]
            w_2d = velocity[:, :, z_idx, 2]
            
            logging.info(f"  速度場形狀: {velocity.shape} -> 2D 切片: {u_2d.shape}")
        else:
            raise KeyError("無法識別 HDF5 檔案格式")
    
    # 載入壓力場
    if pressure_file.exists():
        with h5py.File(pressure_file, 'r') as fp:
            if 'Pressure_0001' in fp:
                pressure = np.array(fp['Pressure_0001'])  # (nx, ny, nz, 1)
                p_2d = pressure[:, :, z_idx, 0]
            else:
                logging.warning("  ⚠️  壓力場格式未知，使用零值替代")
                p_2d = np.zeros_like(u_2d)
    else:
        logging.warning("  ⚠️  壓力場文件未找到，使用零值替代")
        p_2d = np.zeros_like(u_2d)
    
    # 使用實際座標（JHTDB 提供）
    nx, ny = u_2d.shape
    
    # 創建 XY 網格（Z 維度已切片，使用中心位置）
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = np.full_like(X, z_center)
    
    logging.info(f"  網格座標範圍:")
    logging.info(f"    X: [{x.min():.4f}, {x.max():.4f}]")
    logging.info(f"    Y: [{y.min():.4f}, {y.max():.4f}]")
    logging.info(f"    Z: {z_center:.4f} (中心切片)")
    
    # 展平為 (N, 3) 與 (N, 4)
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # (N, 3)
    fields = np.stack([u_2d.ravel(), v_2d.ravel(), w_2d.ravel(), p_2d.ravel()], axis=1)  # (N, 4)
    
    logging.info(f"  ✅ 載入完成: {coords.shape[0]:,} 個網格點")
    
    return coords, fields


# ============================================================================
# 分層誤差計算
# ============================================================================
def compute_layer_wise_errors(
    coords: np.ndarray,
    pred: np.ndarray,
    true: np.ndarray,
    layer_defs: Dict,
    variables: list = ['u', 'v', 'w', 'p']
) -> Dict:
    """
    計算分層誤差（基於歸一化 Y 座標）
    
    Args:
        coords: (N, 3) 座標 [x, y, z]
        pred: (N, n_vars) 預測場
        true: (N, n_vars) 真實場
        layer_defs: 分層定義字典
        variables: 變量名稱列表
    
    Returns:
        results: {layer_name: {var: rel_l2_error, ...}, ...}
    """
    y_norm = np.abs(coords[:, 1])  # 歸一化座標 |y/h|，h=1
    
    results = {}
    
    for layer_name, layer_info in layer_defs.items():
        y_min, y_max = layer_info['y_range']
        mask = (y_norm >= y_min) & (y_norm < y_max)
        n_points = mask.sum()
        
        if n_points == 0:
            logging.warning(f"  ⚠️  {layer_info['name']} 無數據點")
            continue
        
        layer_errors = {'n_points': n_points}
        
        for i, var in enumerate(variables):
            pred_var = pred[mask, i]
            true_var = true[mask, i]
            
            # 相對 L2 誤差
            l2_diff = np.linalg.norm(pred_var - true_var)
            l2_norm = np.linalg.norm(true_var)
            rel_l2 = (l2_diff / l2_norm) if l2_norm > 1e-12 else float('nan')
            
            # 平均絕對誤差（MAE）
            mae = np.mean(np.abs(pred_var - true_var))
            
            layer_errors[var] = {
                'rel_l2': rel_l2,
                'mae': mae
            }
        
        results[layer_name] = layer_errors
        
        logging.info(
            f"  {layer_info['name']} ({layer_name}): "
            f"y∈[{y_min:.1f}, {y_max:.1f}], "
            f"n={n_points:,}, "
            f"u_err={layer_errors['u']['rel_l2']:.4f}"
        )
    
    return results


def predict_full_field(
    model: torch.nn.Module,
    coords: np.ndarray,
    device: torch.device,
    batch_size: int = 8192
) -> np.ndarray:
    """
    批次預測完整場（避免 OOM）
    
    Args:
        model: PINN 模型
        coords: (N, 3) 座標數組
        device: 計算設備
        batch_size: 批次大小
    
    Returns:
        predictions: (N, n_vars) 預測場
    """
    model.eval()
    n_points = coords.shape[0]
    n_batches = (n_points + batch_size - 1) // batch_size
    
    predictions = []
    
    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_points)
            
            # 準備批次
            batch_coords = torch.from_numpy(coords[start_idx:end_idx]).float().to(device)
            
            # 預測
            batch_pred = model(batch_coords)
            predictions.append(batch_pred.cpu().numpy())
            
            if (i + 1) % 10 == 0 or (i + 1) == n_batches:
                logging.info(f"    預測進度: {i + 1}/{n_batches} 批次")
    
    return np.vstack(predictions)


# ============================================================================
# 視覺化
# ============================================================================
def visualize_layer_errors(results: Dict, output_dir: Path):
    """
    視覺化三個實驗的分層誤差比較
    
    Args:
        results: {exp_name: {layer_name: {var: {rel_l2, mae}, ...}, ...}, ...}
        output_dir: 輸出目錄
    """
    # 設定中文字體（macOS）
    mpl.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    mpl.rcParams['axes.unicode_minus'] = False
    
    exp_names = list(results.keys())
    layer_names = ['wall', 'log', 'center']
    layer_labels = [LAYER_DEFINITIONS[ln]['name'] for ln in layer_names]
    variables = ['u', 'v', 'w']  # 不繪製壓力（可能為零）
    
    # 提取數據
    data = {}
    for var in variables:
        data[var] = np.zeros((len(exp_names), len(layer_names)))
        for i, exp_name in enumerate(exp_names):
            for j, layer_name in enumerate(layer_names):
                if layer_name in results[exp_name]:
                    data[var][i, j] = results[exp_name][layer_name][var]['rel_l2']
                else:
                    data[var][i, j] = np.nan
    
    # 繪製分組柱狀圖
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    x = np.arange(len(layer_names))
    width = 0.25
    
    for ax, var in zip(axes, variables):
        for i, exp_name in enumerate(exp_names):
            offset = (i - 1) * width
            bars = ax.bar(
                x + offset,
                data[var][i, :],
                width,
                label=exp_name,
                alpha=0.8
            )
            
            # 標註數值
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 0.01,
                        f'{height:.3f}',
                        ha='center',
                        va='bottom',
                        fontsize=8
                    )
        
        ax.set_xlabel('物理層', fontsize=12)
        ax.set_ylabel('相對 L2 誤差', fontsize=12)
        ax.set_title(f'{var.upper()} 速度分量', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(layer_labels)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'layer_wise_errors_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"  ✅ 圖表已保存: {output_dir / 'layer_wise_errors_comparison.png'}")


# ============================================================================
# 主評估流程
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="PDE 約束消融實驗 - 分層誤差評估")
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
    parser.add_argument('--verbose', action='store_true', help='詳細日誌')
    
    args = parser.parse_args()
    
    # 路徑轉換
    project_root = Path(__file__).parent.parent
    exp1_checkpoint = project_root / args.exp1_checkpoint
    exp2_checkpoint = project_root / args.exp2_checkpoint
    exp3_checkpoint = project_root / args.exp3_checkpoint
    data_dir = project_root / args.data_dir
    output_dir = project_root / args.output_dir
    
    # 創建輸出目錄
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 設定日誌
    setup_logging(output_dir, args.verbose)
    
    logging.info("=" * 80)
    logging.info("PDE 約束補償機制消融實驗 - 分層誤差評估")
    logging.info("=" * 80)
    
    # 獲取設備
    device = get_device(args.device)
    logging.info(f"使用設備: {device}")
    
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
    logging.info("\n" + "=" * 80)
    logging.info("步驟 1: 載入 JHTDB 真實數據")
    logging.info("=" * 80)
    coords, true_fields = load_jhtdb_2d_slice(data_dir)
    
    # 評估每個實驗
    all_results = {}
    
    for exp_name, exp_info in experiments.items():
        logging.info("\n" + "=" * 80)
        logging.info(f"步驟 2: 評估 {exp_name}")
        logging.info("=" * 80)
        
        # 載入模型
        model, config = load_experiment_model(
            exp_info['checkpoint'],
            exp_info['config'],
            device
        )
        
        # 預測完整場
        logging.info("  預測完整場...")
        pred_fields = predict_full_field(model, coords, device)
        
        # 計算分層誤差
        logging.info("  計算分層誤差...")
        layer_errors = compute_layer_wise_errors(
            coords,
            pred_fields,
            true_fields,
            LAYER_DEFINITIONS
        )
        
        all_results[exp_name] = layer_errors
    
    # 保存結果
    logging.info("\n" + "=" * 80)
    logging.info("步驟 3: 保存結果")
    logging.info("=" * 80)
    
    # 轉換為 JSON 可序列化格式
    json_results = {}
    for exp_name, layers in all_results.items():
        json_results[exp_name] = {}
        for layer_name, layer_data in layers.items():
            json_results[exp_name][layer_name] = {}
            for key, value in layer_data.items():
                if key == 'n_points':
                    json_results[exp_name][layer_name][key] = int(value)
                elif isinstance(value, dict):
                    json_results[exp_name][layer_name][key] = {
                        k: float(v) for k, v in value.items()
                    }
                else:
                    json_results[exp_name][layer_name][key] = float(value)
    
    results_file = output_dir / 'layer_wise_errors.json'
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logging.info(f"  ✅ 結果已保存: {results_file}")
    
    # 視覺化
    logging.info("  生成視覺化圖表...")
    visualize_layer_errors(all_results, output_dir)
    
    # 打印摘要
    logging.info("\n" + "=" * 80)
    logging.info("評估摘要 - 中心層相對 L2 誤差（U 速度）")
    logging.info("=" * 80)
    for exp_name in all_results.keys():
        if 'center' in all_results[exp_name]:
            center_u_err = all_results[exp_name]['center']['u']['rel_l2']
            logging.info(f"  {exp_name:25s}: {center_u_err:.4f}")
        else:
            logging.info(f"  {exp_name:25s}: N/A")
    
    logging.info("\n✅ 評估完成！")


if __name__ == '__main__':
    main()
