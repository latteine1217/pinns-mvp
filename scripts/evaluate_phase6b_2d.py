"""
評估 Phase 6B 模型（2D 切片版本）
============================================

問題根因：訓練時使用 2D 切片感測點（w=0），但評估時使用 3D 資料（w≠0），導致巨大誤差。

本腳本使用與訓練一致的 2D 切片資料進行評估，驗證模型在 u, v, p 上的實際性能。

使用方法：
    python scripts/evaluate_phase6b_2d.py
"""

import sys
import os
import numpy as np
import torch
import logging
from pathlib import Path

# 添加專案根目錄到路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pinnx.models.fourier_mlp import create_enhanced_pinn


def setup_logging():
    """設置日誌"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """載入檢查點"""
    logger = logging.getLogger(__name__)
    logger.info(f"載入檢查點: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    logger.info(f"檢查點 Epoch: {checkpoint['epoch']}")
    
    return checkpoint


def load_2d_evaluation_data(data_file: str, subsample: int = 1):
    """
    載入 2D 切片評估資料
    
    Args:
        data_file: 資料檔案路徑
        subsample: 子採樣因子（1=無子採樣）
    
    Returns:
        coords: (N, 3) - [x, y, z=0]
        true_fields: dict - {'u': (N, 1), 'v': (N, 1), 'p': (N, 1)}
    """
    logger = logging.getLogger(__name__)
    logger.info(f"載入 2D 評估資料: {data_file}")
    
    data = np.load(data_file)
    logger.info(f"資料鍵: {list(data.keys())}")
    
    # 讀取座標和場
    x = data['x']  # (nx,)
    y = data['y']  # (ny,)
    u_2d = data['u']  # (nx, ny)
    v_2d = data['v']  # (nx, ny)
    p_2d = data['p']  # (nx, ny)
    
    logger.info(f"原始網格大小: x={x.shape}, y={y.shape}, u={u_2d.shape}")
    
    # 子採樣
    if subsample > 1:
        x = x[::subsample]
        y = y[::subsample]
        u_2d = u_2d[::subsample, ::subsample]
        v_2d = v_2d[::subsample, ::subsample]
        p_2d = p_2d[::subsample, ::subsample]
        logger.info(f"子採樣後網格大小: x={x.shape}, y={y.shape}")
    
    # 創建網格
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # 展平為 (N, 3) - [x, y, z=0]
    coords = np.stack([
        X.ravel(),
        Y.ravel(),
        np.zeros_like(X.ravel())  # z=0（2D 切片）
    ], axis=1)
    
    # 展平場
    true_fields = {
        'u': u_2d.ravel().reshape(-1, 1).astype(np.float32),
        'v': v_2d.ravel().reshape(-1, 1).astype(np.float32),
        'p': p_2d.ravel().reshape(-1, 1).astype(np.float32)
    }
    
    logger.info(f"評估點數: {coords.shape[0]}")
    logger.info(f"座標範圍: x=[{coords[:, 0].min():.4f}, {coords[:, 0].max():.4f}], "
                f"y=[{coords[:, 1].min():.4f}, {coords[:, 1].max():.4f}]")
    
    for key, val in true_fields.items():
        logger.info(f"{key} 範圍: [{val.min():.4f}, {val.max():.4f}]")
    
    return coords, true_fields


def predict_fields(model: torch.nn.Module, coords: np.ndarray, device: torch.device, batch_size: int = 4096):
    """
    使用模型預測場（包含反標準化）
    
    Args:
        model: PyTorch 模型
        coords: (N, 3) numpy array
        device: 設備
        batch_size: 批次大小
    
    Returns:
        predictions: dict - {'u': (N, 1), 'v': (N, 1), 'w': (N, 1), 'p': (N, 1)}
        
    Note:
        模型輸出是標準化的值，需要反標準化回物理空間
    """
    logger = logging.getLogger(__name__)
    logger.info("開始預測...")
    
    # ✅ TASK-008 Phase 6B 修復：使用與訓練相同的標準化因子
    # 這些因子來自 pinnx/train/trainer.py 的 step() 方法
    u_scale = 9.841839   # DNS 完整場 U 均值
    v_scale = 0.188766   # DNS 完整場 V 標準差
    w_scale = 3.865396   # DNS 完整場 W 標準差
    p_scale = 35.655934  # DNS 完整場 P 均值絕對值
    
    logger.info("使用訓練時的標準化因子:")
    logger.info(f"  u_scale = {u_scale:.6f}")
    logger.info(f"  v_scale = {v_scale:.6f}")
    logger.info(f"  w_scale = {w_scale:.6f}")
    logger.info(f"  p_scale = {p_scale:.6f}")
    
    model.eval()
    n_points = coords.shape[0]
    n_batches = (n_points + batch_size - 1) // batch_size
    
    predictions = {'u': [], 'v': [], 'w': [], 'p': []}  # type: dict[str, list[np.ndarray]]
    
    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_points)
            
            # 轉換為 tensor
            coords_batch = torch.from_numpy(coords[start_idx:end_idx]).float().to(device)
            
            # 預測（標準化空間）
            output = model(coords_batch)  # (batch, 4) - [u, v, w, p] (normalized)
            
            # ✅ 反標準化到物理空間
            u_pred = output[:, 0:1] * u_scale
            v_pred = output[:, 1:2] * v_scale
            w_pred = output[:, 2:3] * w_scale
            p_pred = output[:, 3:4] * p_scale
            
            # 分離各分量（已反標準化）
            predictions['u'].append(u_pred.cpu().numpy())
            predictions['v'].append(v_pred.cpu().numpy())
            predictions['w'].append(w_pred.cpu().numpy())
            predictions['p'].append(p_pred.cpu().numpy())
            
            if (i + 1) % 10 == 0:
                logger.info(f"預測進度: {i + 1}/{n_batches}")
    
    # 合併批次
    predictions_final = {}  # type: dict[str, np.ndarray]
    for key in predictions:
        predictions_final[key] = np.concatenate(predictions[key], axis=0)
    
    logger.info("預測完成（已反標準化到物理空間）")
    return predictions_final


def compute_metrics(predictions: dict, true_fields: dict):
    """
    計算評估指標
    
    Args:
        predictions: 預測場
        true_fields: 真值場
    
    Returns:
        metrics: dict
    """
    logger = logging.getLogger(__name__)
    
    metrics = {}
    
    # 計算相對 L2 誤差（只計算 u, v, p，忽略 w）
    for key in ['u', 'v', 'p']:
        pred = predictions[key]
        true = true_fields[key]
        
        l2_error = np.linalg.norm(pred - true) / np.linalg.norm(true)
        metrics[f'{key}_l2_error'] = l2_error * 100  # 百分比
        
        logger.info(f"{key} 相對 L2 誤差: {l2_error * 100:.2f}%")
    
    # 平均速度場誤差
    avg_velocity_error = (metrics['u_l2_error'] + metrics['v_l2_error']) / 2
    metrics['avg_velocity_l2_error'] = avg_velocity_error
    
    logger.info(f"平均速度場誤差: {avg_velocity_error:.2f}%")
    
    return metrics


def main():
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("Phase 6B 模型評估（2D 切片版本）")
    logger.info("=" * 80)
    
    # 配置
    checkpoint_path = "checkpoints/test_rans_phase6b/epoch_100.pth"
    data_file = "data/jhtdb/channel_flow_re1000/cutout_128x64.npz"
    subsample = 2  # 子採樣因子（加快評估）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"設備: {device}")
    
    # 載入檢查點
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint['config']
    
    # 創建模型（使用 create_enhanced_pinn）
    logger.info("創建模型...")
    base_model = create_enhanced_pinn(
        in_dim=config['model']['in_dim'],
        out_dim=config['model']['out_dim'],
        width=config['model']['width'],
        depth=config['model']['depth'],
        activation=config['model']['activation'],
        use_fourier=True,
        fourier_m=config['model']['fourier_features']['fourier_m'],
        fourier_sigma=config['model']['fourier_features']['fourier_sigma']
    ).to(device)
    
    # 載入權重（忽略不匹配的鍵）
    base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    logger.info("模型權重載入完成（strict=False）")
    
    # 載入 2D 評估資料
    coords, true_fields = load_2d_evaluation_data(data_file, subsample=subsample)
    
    # 預測
    predictions = predict_fields(base_model, coords, device)
    
    # 顯示預測範圍
    logger.info("\n" + "=" * 80)
    logger.info("預測場範圍:")
    for key in ['u', 'v', 'w', 'p']:
        pred = predictions[key]
        logger.info(f"  {key}: [{pred.min():.4f}, {pred.max():.4f}]")
    
    logger.info("\n真值場範圍:")
    for key in ['u', 'v', 'p']:
        true = true_fields[key]
        logger.info(f"  {key}: [{true.min():.4f}, {true.max():.4f}]")
    
    # 計算指標
    logger.info("\n" + "=" * 80)
    logger.info("評估指標:")
    logger.info("=" * 80)
    metrics = compute_metrics(predictions, true_fields)
    
    # 成功/失敗判斷
    logger.info("\n" + "=" * 80)
    threshold = 15.0  # 目標: < 15%
    if metrics['avg_velocity_l2_error'] < threshold:
        logger.info(f"✅ 評估通過！平均速度場誤差 {metrics['avg_velocity_l2_error']:.2f}% < {threshold}%")
    else:
        logger.info(f"❌ 評估失敗！平均速度場誤差 {metrics['avg_velocity_l2_error']:.2f}% > {threshold}%")
    
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
