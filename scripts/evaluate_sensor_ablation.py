#!/usr/bin/env python
"""
評估感測點策略對比實驗結果

本腳本專為 ablation_sensor_qr_K50 與 ablation_sensor_stratified_K50 實驗設計，
使用工廠方法正確載入 enhanced_fourier_mlp 模型，並生成完整的對比分析。

功能：
1. 從配置文件動態創建模型（支援所有模型類型）
2. 載入檢查點並評估流場誤差（L2, 逐點誤差）
3. 生成視覺化對比圖（速度剖面、誤差分布、能譜）
4. 輸出結構化結果報告（JSON + Markdown）

使用方式：
    python scripts/evaluate_sensor_ablation.py \
        --checkpoint checkpoints/ablation_sensor_qr_K50/best_model.pth \
        --config configs/ablation_sensor_qr_K50.yml \
        --output results/ablation_sensor_qr_K50
"""

import sys
import argparse
import logging
from pathlib import Path
import json

import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt

# 添加項目路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pinnx.train.factory import create_model, get_device


def compute_error_metrics(pred, true):
    """
    計算誤差指標
    
    Args:
        pred: 預測字典 {'u': tensor, 'v': tensor, 'w': tensor, 'p': tensor}
        true: 真實字典 {'u': tensor, 'v': tensor, 'w': tensor, 'p': tensor}
    
    Returns:
        metrics: 誤差指標字典
    """
    metrics = {}
    
    for var in ['u', 'v', 'w', 'p']:
        p = pred[var]
        t = true[var]
        
        # L2 誤差
        l2_error = torch.norm(p - t, p=2).item()
        
        # 相對 L2 誤差
        rel_l2_error = l2_error / (torch.norm(t, p=2).item() + 1e-12)
        
        # 最大誤差
        max_error = torch.max(torch.abs(p - t)).item()
        
        # 平均誤差
        mean_error = torch.mean(torch.abs(p - t)).item()
        
        metrics[var] = {
            'l2_error': l2_error,
            'rel_l2_error': rel_l2_error,
            'max_error': max_error,
            'mean_error': mean_error
        }
    
    return metrics


def setup_logging(log_level="INFO"):
    """設置日誌"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )


def load_checkpoint(checkpoint_path, device):
    """載入檢查點"""
    logging.info(f"📂 載入檢查點: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 提取訓練資訊
    epoch = checkpoint.get('epoch', 'unknown')
    loss = checkpoint.get('loss', 'unknown')
    logging.info(f"   Epoch: {epoch} | Loss: {loss}")
    
    # 提取標準化參數（用於反標準化）
    if 'normalization' in checkpoint:
        norm = checkpoint['normalization']
        logging.info(f"   ✅ 找到標準化參數: {norm.get('type', 'unknown')}")
        if norm.get('type') == 'training_data_norm':
            means = norm.get('means', {})
            scales = norm.get('scales', {})
            logging.info(f"      均值: u={means.get('u', 0):.3f}, v={means.get('v', 0):.6f}, "
                        f"w={means.get('w', 0):.6f}, p={means.get('p', 0):.3f}")
            logging.info(f"      標準差: u={scales.get('u', 1):.3f}, v={scales.get('v', 1):.6f}, "
                        f"w={scales.get('w', 1):.6f}, p={scales.get('p', 1):.3f}")
    else:
        logging.warning("   ⚠️ 檢查點中未找到標準化參數，假設無標準化")
    
    return checkpoint


def load_test_data(data_path, device):
    """載入測試資料（JHTDB 2D slice）"""
    logging.info(f"📦 載入測試資料: {data_path}")
    
    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"測試資料未找到: {data_path}\n"
            f"請先執行 scripts/fetch_channel_flow.py 獲取 JHTDB 資料"
        )
    
    data = np.load(data_path)
    
    # 提取坐標（1D 陣列）
    x_1d = data['x']  # [Nx]
    y_1d = data['y']  # [Ny]
    
    # 提取流場（2D 網格）
    u_2d = data['u']  # [Nx, Ny]
    v_2d = data['v']
    w_2d = data['w']
    p_2d = data['p']
    
    logging.info(f"   網格形狀: {u_2d.shape}")
    logging.info(f"   x 範圍: [{x_1d.min():.3f}, {x_1d.max():.3f}]")
    logging.info(f"   y 範圍: [{y_1d.min():.3f}, {y_1d.max():.3f}]")
    
    # 建立網格座標 meshgrid
    X_mesh, Y_mesh = np.meshgrid(x_1d, y_1d, indexing='ij')
    
    # 展平為 1D 陣列 [Nx*Ny]
    x_flat = X_mesh.ravel()
    y_flat = Y_mesh.ravel()
    u_flat = u_2d.ravel()
    v_flat = v_2d.ravel()
    w_flat = w_2d.ravel()
    p_flat = p_2d.ravel()
    
    # 構建輸入張量 [N, 3] (x, y, z=0)
    N = len(x_flat)
    z_flat = np.zeros_like(x_flat)
    
    X_test = torch.tensor(
        np.stack([x_flat, y_flat, z_flat], axis=1),
        dtype=torch.float32,
        device=device
    )
    
    u_true = torch.tensor(u_flat, dtype=torch.float32, device=device)
    v_true = torch.tensor(v_flat, dtype=torch.float32, device=device)
    w_true = torch.tensor(w_flat, dtype=torch.float32, device=device)
    p_true = torch.tensor(p_flat, dtype=torch.float32, device=device)
    
    logging.info(f"   輸入張量形狀: {X_test.shape}")
    
    return {
        'X': X_test,
        'u': u_true,
        'v': v_true,
        'w': w_true,
        'p': p_true,
        'x_1d': x_1d,
        'y_1d': y_1d,
        'grid_shape': u_2d.shape
    }


def denormalize_output(pred, normalization):
    """
    反標準化模型輸出
    
    Args:
        pred: 標準化的預測值 [N, 4] (u, v, w, p)
        normalization: 標準化參數字典
    
    Returns:
        denorm_pred: 反標準化的預測值
    """
    if normalization is None or normalization.get('type') != 'training_data_norm':
        logging.warning("⚠️ 無標準化參數，跳過反標準化步驟")
        return pred
    
    means = normalization.get('means', {})
    scales = normalization.get('scales', {})
    
    # 反標準化公式: y = y_normalized * std + mean
    denorm_pred = pred.clone()
    
    for i, var in enumerate(['u', 'v', 'w', 'p']):
        mean = means.get(var, 0.0)
        scale = scales.get(var, 1.0)
        denorm_pred[:, i] = pred[:, i] * scale + mean
    
    logging.info("✅ 輸出已反標準化")
    return denorm_pred


def evaluate_model(model, test_data, checkpoint, device):
    """評估模型預測誤差（含反標準化）"""
    logging.info("🔬 開始評估模型...")
    
    model.eval()
    with torch.no_grad():
        # 前向推理
        X = test_data['X'].to(device)
        pred_normalized = model(X)  # [N, 4] (u, v, w, p) - 標準化空間
        
        # ⭐ 反標準化模型輸出
        normalization = checkpoint.get('normalization', None)
        pred = denormalize_output(pred_normalized, normalization)
        
        u_pred = pred[:, 0]
        v_pred = pred[:, 1]
        w_pred = pred[:, 2]
        p_pred = pred[:, 3]
    
    # 檢查反標準化後的數值範圍
    logging.info("=" * 70)
    logging.info("🔍 預測值範圍檢查（反標準化後）：")
    logging.info("-" * 70)
    for i, var in enumerate(['u', 'v', 'w', 'p']):
        pred_val = pred[:, i]
        true_val = test_data[var].to(device)
        logging.info(f"  {var}: 預測 [{pred_val.min():.3f}, {pred_val.max():.3f}] | "
                    f"真實 [{true_val.min():.3f}, {true_val.max():.3f}]")
    logging.info("=" * 70)
    
    # 計算誤差指標
    u_true = test_data['u'].to(device)
    v_true = test_data['v'].to(device)
    w_true = test_data['w'].to(device)
    p_true = test_data['p'].to(device)
    
    metrics = compute_error_metrics(
        pred={'u': u_pred, 'v': v_pred, 'w': w_pred, 'p': p_pred},
        true={'u': u_true, 'v': v_true, 'w': w_true, 'p': p_true}
    )
    
    logging.info("📊 評估結果：")
    logging.info("-" * 70)
    for var in ['u', 'v', 'w', 'p']:
        logging.info(
            f"  {var}: L2 Error = {metrics[var]['l2_error']:.4f}, "
            f"Rel. L2 = {metrics[var]['rel_l2_error']:.2%}"
        )
    logging.info("=" * 70)
    
    # 轉換為 numpy 用於視覺化
    results = {
        'metrics': metrics,
        'predictions': {
            'u': u_pred.cpu().numpy(),
            'v': v_pred.cpu().numpy(),
            'w': w_pred.cpu().numpy(),
            'p': p_pred.cpu().numpy()
        },
        'ground_truth': {
            'u': u_true.cpu().numpy(),
            'v': v_true.cpu().numpy(),
            'w': w_true.cpu().numpy(),
            'p': p_true.cpu().numpy()
        }
    }
    
    return results


def visualize_results(results, test_data, output_dir):
    """生成視覺化圖表"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"📈 生成視覺化圖表至: {output_dir}")
    
    pred = results['predictions']
    true = results['ground_truth']
    x_1d = test_data['x_1d']
    y_1d = test_data['y_1d']
    grid_shape = test_data['grid_shape']  # (Nx, Ny)
    
    # Reshape 回 2D 網格
    def reshape_to_grid(arr):
        return arr.reshape(grid_shape)
    
    # === 1. 速度剖面對比（沿 y 方向，x 平均）===
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, var in enumerate(['u', 'v', 'w']):
        ax = axes[i]
        
        # Reshape 為 2D 並沿 x 方向平均
        pred_2d = reshape_to_grid(pred[var])
        true_2d = reshape_to_grid(true[var])
        
        pred_mean_y = pred_2d.mean(axis=0)  # [Ny]
        true_mean_y = true_2d.mean(axis=0)  # [Ny]
        
        ax.plot(y_1d, true_mean_y, 'k-', label='JHTDB Ground Truth', linewidth=2)
        ax.plot(y_1d, pred_mean_y, 'r--', label='PINN Prediction', linewidth=1.5)
        
        ax.set_xlabel('y (wall-normal)', fontsize=12)
        ax.set_ylabel(f'{var} velocity', fontsize=12)
        ax.set_title(f'{var.upper()} Profile (Rel. L2: {results["metrics"][var]["rel_l2_error"]:.2%})')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'velocity_profiles.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logging.info("   ✅ 速度剖面圖已保存")
    
    # === 2. 誤差分布熱圖 ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for i, var in enumerate(['u', 'v', 'w', 'p']):
        ax = axes.flat[i]
        
        pred_2d = reshape_to_grid(pred[var])
        true_2d = reshape_to_grid(true[var])
        error_2d = np.abs(pred_2d - true_2d)
        
        im = ax.imshow(error_2d.T, origin='lower', aspect='auto',
                       extent=[x_1d.min(), x_1d.max(), y_1d.min(), y_1d.max()],
                       cmap='hot')
        ax.set_xlabel('x (streamwise)')
        ax.set_ylabel('y (wall-normal)')
        ax.set_title(f'{var.upper()} Error (Rel. L2: {results["metrics"][var]["rel_l2_error"]:.2%})')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'error_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logging.info("   ✅ 誤差熱圖已保存")
    
    # === 3. 壓力場對比 ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    p_pred_2d = reshape_to_grid(pred['p'])
    p_true_2d = reshape_to_grid(true['p'])
    
    p_pred_mean_y = p_pred_2d.mean(axis=0)
    p_true_mean_y = p_true_2d.mean(axis=0)
    
    axes[0].plot(y_1d, p_true_mean_y, 'k-', label='Ground Truth', linewidth=2)
    axes[0].plot(y_1d, p_pred_mean_y, 'r--', label='Prediction', linewidth=1.5)
    axes[0].set_xlabel('y (wall-normal)')
    axes[0].set_ylabel('Pressure')
    axes[0].set_title('Pressure Profile (x-averaged)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 誤差分布
    p_error_mean_y = np.abs(p_pred_2d - p_true_2d).mean(axis=0)
    axes[1].plot(y_1d, p_error_mean_y, 'b-', linewidth=1.5)
    axes[1].set_xlabel('y (wall-normal)')
    axes[1].set_ylabel('Absolute Error (x-averaged)')
    axes[1].set_title(f'Pressure Error (Rel. L2: {results["metrics"]["p"]["rel_l2_error"]:.2%})')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'pressure_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logging.info("   ✅ 壓力對比圖已保存")


def save_results(results, checkpoint_info, output_dir):
    """保存結構化結果"""
    output_dir = Path(output_dir)
    
    # === JSON 格式 ===
    results_json = {
        'checkpoint': checkpoint_info,
        'metrics': {}
    }
    
    for var in ['u', 'v', 'w', 'p']:
        results_json['metrics'][var] = {
            'l2_error': float(results['metrics'][var]['l2_error']),
            'rel_l2_error': float(results['metrics'][var]['rel_l2_error']),
            'max_error': float(results['metrics'][var]['max_error']),
            'mean_error': float(results['metrics'][var]['mean_error'])
        }
    
    json_path = output_dir / 'evaluation_results.json'
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    logging.info(f"💾 結果已保存至: {json_path}")
    
    # === Markdown 報告 ===
    md_lines = [
        "# 評估結果報告",
        "",
        f"**檢查點**: `{checkpoint_info['path']}`  ",
        f"**Epoch**: {checkpoint_info['epoch']}  ",
        f"**訓練 Loss**: {checkpoint_info['loss']}",
        "",
        "## 誤差指標",
        "",
        "| 變量 | L2 Error | Relative L2 | Max Error | Mean Error |",
        "|------|----------|-------------|-----------|------------|"
    ]
    
    for var in ['u', 'v', 'w', 'p']:
        m = results['metrics'][var]
        md_lines.append(
            f"| {var} | {m['l2_error']:.4f} | {m['rel_l2_error']:.2%} | "
            f"{m['max_error']:.4f} | {m['mean_error']:.4f} |"
        )
    
    md_path = output_dir / 'evaluation_report.md'
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    
    logging.info(f"📝 Markdown 報告已保存至: {md_path}")


def main():
    parser = argparse.ArgumentParser(description='評估感測點策略對比實驗')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='檢查點路徑（.pth 文件）')
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路徑（.yml 文件）')
    parser.add_argument('--data', type=str,
                        default='data/jhtdb/channel/2d_slice_z0_normalized.npz',
                        help='測試資料路徑')
    parser.add_argument('--output', type=str, required=True,
                        help='結果輸出目錄')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='日誌等級')
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    logging.info("=" * 70)
    logging.info("  感測點策略評估腳本")
    logging.info("=" * 70)
    
    # === 1. 載入配置 ===
    logging.info(f"📄 載入配置: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # === 2. 設備選擇 ===
    device = get_device(config.get('experiment', {}).get('device', 'auto'))
    logging.info(f"🖥️  使用設備: {device}")
    
    # === 3. 創建模型（使用工廠方法）===
    logging.info("🏗️  創建模型...")
    model = create_model(config, device)
    
    # === 4. 載入檢查點 ===
    checkpoint = load_checkpoint(args.checkpoint, device)
    
    # 提取模型權重
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        raise KeyError("檢查點中未找到模型權重（'model_state_dict' 或 'model'）")
    
    # 載入權重
    model.load_state_dict(state_dict, strict=False)
    logging.info("✅ 模型權重載入成功")
    
    # === 5. 載入測試資料 ===
    test_data = load_test_data(args.data, device)
    
    # === 6. 評估模型 ===
    results = evaluate_model(model, test_data, checkpoint, device)
    
    # === 7. 視覺化 ===
    visualize_results(results, test_data, args.output)
    
    # === 8. 保存結果 ===
    checkpoint_info = {
        'path': args.checkpoint,
        'epoch': checkpoint.get('epoch', 'unknown'),
        'loss': checkpoint.get('loss', 'unknown')
    }
    save_results(results, checkpoint_info, args.output)
    
    logging.info("=" * 70)
    logging.info("✅ 評估完成！")
    logging.info("=" * 70)


if __name__ == '__main__':
    main()
