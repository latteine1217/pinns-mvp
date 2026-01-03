#!/usr/bin/env python3
"""
PINNs 統一評估腳本
支援配置驅動的模組化評估流程，修復座標標準化問題

使用範例：
    python scripts/evaluate.py --checkpoint checkpoints/model.pth --config configs/model.yml
    python scripts/evaluate.py --checkpoint checkpoints/model.pth --reference data/jhtdb/full_field.npz --output results/eval
"""

import argparse
import datetime
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import yaml

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinnx.evals.metrics import relative_L2, rmse_metrics, conservation_error
from pinnx.evals.visualizer import Visualizer
from pinnx.models.wrappers import ManualScalingWrapper

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """設置日誌系統"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """載入配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"✅ 配置載入成功: {config_path}")
    return config


def load_checkpoint(checkpoint_path: str, device: str = 'cpu') -> Tuple[torch.nn.Module, Dict]:
    """載入模型檢查點（相容多種 state_dict 結構）"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 從檢查點中恢復配置
    config = checkpoint.get('config', {})

    # 重建模型（需要從配置中獲取架構資訊）
    from pinnx.models.fourier_mlp import PINNNet

    model_cfg = config.get('model', {})
    in_dim = model_cfg.get('in_dim', 2)  # (x, y) for 2D channel flow
    out_dim = model_cfg.get('out_dim', 3)  # (u, v, p)
    width = model_cfg.get('width', 128)
    depth = model_cfg.get('depth', 6)
    ff_cfg = model_cfg.get('fourier_features')
    if not isinstance(ff_cfg, dict) or 'type' not in ff_cfg:
        raise KeyError("Missing required config key: model.fourier_features.type")

    ff_type = ff_cfg.get('type', 'disabled')
    use_fourier = ff_type != 'disabled'
    fourier_m = int(ff_cfg.get('fourier_m', 0)) if use_fourier else 0
    fourier_sigma = float(ff_cfg.get('fourier_sigma', 0.0)) if use_fourier else 0.0

    model = PINNNet(
        in_dim=in_dim,
        out_dim=out_dim,
        width=width,
        depth=depth,
        use_fourier=use_fourier,
        fourier_m=fourier_m,
        fourier_sigma=fourier_sigma
    ).to(device)

    # 取得原始 state_dict（可能包含包裝器與標準化緩衝區）
    raw_state = checkpoint.get('model_state_dict', {})
    if not raw_state:
        # fallback: 一些舊檔可能直接存放在 'state_dict' 或根層
        raw_state = checkpoint.get('state_dict', checkpoint)

    # 檢測是否包含 ManualScalingWrapper 的緩衝區
    has_scaling_buffers = any(k in raw_state for k in ['input_min', 'input_max', 'output_min', 'output_max'])
    has_base_prefix = any(k.startswith('base_model.') for k in raw_state.keys())

    if has_scaling_buffers:
        # 使用 ManualScalingWrapper 還原完整模型（以佔位範圍初始化，隨後由 state_dict 覆蓋）
        try:
            # 建立佔位範圍（名稱無關，僅需維度正確）
            in_ranges = {f'in_{i}': (0.0, 1.0) for i in range(in_dim)}
            out_ranges = {name: (0.0, 1.0) for name in (['u', 'v', 'p'][:out_dim] + [f'out_{i}' for i in range(max(0, out_dim-3))])}
            # 若 out_dim > 3，上面會產生多餘名稱，統一重建為連續鍵
            out_ranges = {f'out_{i}': (0.0, 1.0) for i in range(out_dim)}

            wrapper = ManualScalingWrapper(base_model=model, input_ranges=in_ranges, output_ranges=out_ranges).to(device)

            # 如果權重鍵沒有 'base_model.' 前綴，且是裸模型權重，則需要將鍵映射到 wrapper.base_model 下
            if not has_base_prefix:
                mapped_state = {}
                for k, v in raw_state.items():
                    if k in ['input_min', 'input_max', 'output_min', 'output_max']:
                        mapped_state[k] = v
                    else:
                        mapped_state[f'base_model.{k}'] = v
                raw_state_to_load = mapped_state
            else:
                raw_state_to_load = raw_state

            missing, unexpected = wrapper.load_state_dict(raw_state_to_load, strict=False)
            if missing:
                logger.warning(f"⚠️ 載入包裝模型時缺少鍵: {missing}")
            if unexpected:
                logger.warning(f"⚠️ 載入包裝模型時存在未使用鍵: {unexpected}")

            model = wrapper
            logger.info("✅ 偵測到尺度化緩衝區，已使用 ManualScalingWrapper 還原模型")
        except Exception as e:
            logger.warning(f"⚠️ ManualScalingWrapper 還原失敗，回退至裸模型載入：{e}")
            # 回退：按裸模型流程過濾並載入
            state_no_buffers = {k: v for k, v in raw_state.items() if k not in ['input_min', 'input_max', 'output_min', 'output_max']}
            if any(k.startswith('base_model.') for k in state_no_buffers):
                state_no_buffers = {k.replace('base_model.', '', 1): v for k, v in state_no_buffers.items()}
            model_keys = set(model.state_dict().keys())
            filtered_state = {k: v for k, v in state_no_buffers.items() if k in model_keys}
            missing, unexpected = model.load_state_dict(filtered_state, strict=False)
            if missing:
                logger.warning(f"⚠️ 載入權重時缺少鍵: {missing}")
            if unexpected:
                logger.warning(f"⚠️ 載入權重時存在未使用鍵: {unexpected}")
    else:
        # 裸模型：去除包裝前綴與非參數鍵，並過濾到匹配的鍵
        state_no_buffers = {k: v for k, v in raw_state.items() if k not in ['input_min', 'input_max', 'output_min', 'output_max']}
        if has_base_prefix:
            state_no_buffers = {k.replace('base_model.', '', 1): v for k, v in state_no_buffers.items()}
        model_keys = set(model.state_dict().keys())
        filtered_state = {k: v for k, v in state_no_buffers.items() if k in model_keys}
        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        if missing:
            logger.warning(f"⚠️ 載入權重時缺少鍵: {missing}")
        if unexpected:
            logger.warning(f"⚠️ 載入權重時存在未使用鍵: {unexpected}")

    model.eval()

    logger.info(f"✅ 模型載入成功: {checkpoint_path}")
    logger.info(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    loss_val = checkpoint.get('loss', None)
    if isinstance(loss_val, (int, float)):
        logger.info(f"   Loss: {loss_val:.6e}")
    else:
        logger.info("   Loss: N/A")
    fourier_info = f", Fourier M={fourier_m}" if use_fourier else ", Fourier disabled"
    logger.info(f"   架構: {width}×{depth}{fourier_info}")

    return model, config


def load_reference_data(ref_path: str) -> Dict[str, np.ndarray]:
    """載入參考數據（真值）"""
    data = np.load(ref_path, allow_pickle=True)
    
    # 處理兩種可能的數據格式
    if 'coordinates' in data.keys():
        # 格式 1: coordinates (dict) + (u, v, p) 網格
        coords = data['coordinates'].item()  # 從 object array 解包
        x_1d = coords['x']
        y_1d = coords['y']
        
        # 生成 2D 網格
        X, Y = np.meshgrid(x_1d, y_1d, indexing='ij')
        
        ref_data = {
            'x': X,
            'y': Y,
            'u': data['u'],
            'v': data['v'],
            'p': data['p']
        }
    else:
        # 格式 2: 直接的 x, y 陣列（已經是網格）
        ref_data = {
            'x': data['x'],
            'y': data['y'],
            'u': data['u'],
            'v': data['v'],
            'p': data['p']
        }
    
    logger.info(f"✅ 參考數據載入成功: {ref_path}")
    logger.info(f"   數據形狀: {ref_data['u'].shape}")
    logger.info(f"   x 範圍: [{ref_data['x'].min():.4f}, {ref_data['x'].max():.4f}]")
    logger.info(f"   y 範圍: [{ref_data['y'].min():.4f}, {ref_data['y'].max():.4f}]")
    
    return ref_data


def normalize_coords(x: np.ndarray, y: np.ndarray, 
                     x_range: Tuple[float, float], 
                     y_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    座標標準化到 [-1, 1]（與訓練時一致）
    
    Args:
        x, y: 原始座標
        x_range, y_range: 座標範圍 (min, max)
        
    Returns:
        標準化後的座標
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    x_norm = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    y_norm = 2.0 * (y - y_min) / (y_max - y_min) - 1.0
    
    return x_norm, y_norm


def predict_on_grid(model: torch.nn.Module,
                    ref_data: Dict[str, np.ndarray],
                    config: Dict[str, Any],
                    device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """
    在參考數據網格上進行預測
    
    Args:
        model: PINNs 模型
        ref_data: 參考數據（包含 x, y 座標）
        config: 配置字典
        device: 計算設備
        
    Returns:
        預測結果字典 {u, v, p}
    """
    # 提取座標
    x_raw = ref_data['x'].flatten()
    y_raw = ref_data['y'].flatten()
    
    # 從配置中獲取域範圍
    domain_cfg = config.get('domain', {})
    x_range = domain_cfg.get('x_range', [x_raw.min(), x_raw.max()])
    y_range = domain_cfg.get('y_range', [y_raw.min(), y_raw.max()])
    
    # 檢查是否需要標準化
    normalize = config.get('normalize', True)
    
    if normalize:
        logger.info("🔄 應用座標標準化（與訓練一致）")
        x_norm, y_norm = normalize_coords(x_raw, y_raw, x_range, y_range)
    else:
        logger.info("⚠️  未進行座標標準化")
        x_norm, y_norm = x_raw, y_raw
    
    # 構建輸入張量 (x, y) - 2D 通道流
    coords = np.stack([x_norm, y_norm], axis=1)  # (N, 2)
    coords_tensor = torch.from_numpy(coords).float().to(device)
    
    logger.info(f"📊 輸入座標範圍檢查:")
    logger.info(f"   x: [{coords_tensor[:, 0].min():.4f}, {coords_tensor[:, 0].max():.4f}]")
    logger.info(f"   y: [{coords_tensor[:, 1].min():.4f}, {coords_tensor[:, 1].max():.4f}]")
    
    # 批次預測（避免記憶體溢出）
    batch_size = 4096
    n_points = len(coords_tensor)
    predictions = []
    
    with torch.no_grad():
        for i in range(0, n_points, batch_size):
            batch = coords_tensor[i:i+batch_size]
            pred = model(batch)
            predictions.append(pred)
    
    # 拼接結果
    pred_full = torch.cat(predictions, dim=0)  # (N, 3) -> [u, v, p]
    
    pred_data = {
        'u': pred_full[:, 0],
        'v': pred_full[:, 1],
        'p': pred_full[:, 2]
    }
    
    logger.info(f"✅ 預測完成，數據點數: {n_points}")
    
    return pred_data


def calibrate_scale_from_sensors(
    model: torch.nn.Module,
    pred_data: Dict[str, torch.Tensor],
    sensor_file: str,
    device: str = 'cpu'
) -> Dict[str, torch.Tensor]:
    """
    基於感測點數據校正預測場的尺度
    
    策略：
        1. 載入感測點座標與真實值
        2. 在感測點位置預測模型輸出
        3. 計算 affine 變換：pred_corrected = pred * scale + shift
        4. 應用到整個場
    
    Args:
        model: 訓練好的 PINN 模型
        pred_data: 原始預測數據 {'u', 'v', 'p'}
        sensor_file: 感測點數據路徑（.npz）
        device: 計算設備
    
    Returns:
        校正後的預測數據
    """
    logger.info(f"🔧 開始基於感測點的尺度校正...")
    
    # 1. 載入感測點數據
    sensor_data = np.load(sensor_file, allow_pickle=True)
    sensor_coords = sensor_data['sensor_points']  # (K, 2)
    sensor_true = sensor_data['sensor_data'].item()  # dict: {'u', 'v', 'p'}
    
    K = sensor_coords.shape[0]
    logger.info(f"   感測點數量: {K}")
    
    # 2. 在感測點位置預測
    with torch.no_grad():
        coords_tensor = torch.from_numpy(sensor_coords).float().to(device)
        sensor_pred = model(coords_tensor)  # (K, 3)
    
    # 3. 計算每個場的尺度變換參數
    corrected_data = {}
    
    for i, field in enumerate(['u', 'v', 'p']):
        # 感測點處的預測與真值
        pred_at_sensors = sensor_pred[:, i].cpu().numpy()
        true_at_sensors = sensor_true[field]
        
        # 計算 affine 變換（最小二乘法擬合）
        # pred_corrected = a * pred + b
        # 使用統計方法：匹配均值與標準差
        pred_mean = pred_at_sensors.mean()
        pred_std = pred_at_sensors.std() + 1e-10
        true_mean = true_at_sensors.mean()
        true_std = true_at_sensors.std()
        
        scale = true_std / pred_std
        shift = true_mean - pred_mean * scale
        
        # 應用到整個場
        corrected_field = pred_data[field] * scale + shift
        corrected_data[field] = corrected_field
        
        logger.info(f"   {field.upper()}: scale={scale:.4f}, shift={shift:.4f}")
        logger.info(f"      原始範圍: [{pred_data[field].min():.3f}, {pred_data[field].max():.3f}]")
        logger.info(f"      校正範圍: [{corrected_field.min():.3f}, {corrected_field.max():.3f}]")
        logger.info(f"      真實範圍: [{true_at_sensors.min():.3f}, {true_at_sensors.max():.3f}]")
    
    logger.info("✅ 尺度校正完成！")
    return corrected_data


def compute_metrics(pred_data: Dict[str, torch.Tensor],
                    ref_data: Dict[str, np.ndarray],
                    coords: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    計算評估指標
    
    Returns:
        指標字典
    """
    metrics = {}
    
    # 轉換參考數據為 Tensor
    ref_tensors = {
        'u': torch.from_numpy(ref_data['u'].flatten()).float(),
        'v': torch.from_numpy(ref_data['v'].flatten()).float(),
        'p': torch.from_numpy(ref_data['p'].flatten()).float()
    }
    
    # 計算相對 L2 誤差（在CPU上避免裝置不一致問題，且不影響自動微分圖）
    for field in ['u', 'v', 'p']:
        pred_cpu = pred_data[field].detach().cpu()
        ref_cpu = ref_tensors[field].detach().cpu()
        rel_l2 = relative_L2(pred_cpu, ref_cpu).item()
        metrics[f'rel_L2_{field}'] = rel_l2
        logger.info(f"   {field.upper()} 相對 L2 誤差: {rel_l2:.6f} {'✅' if rel_l2 <= 0.15 else '❌'}")

    # 計算 RMSE（轉為 numpy）
    for field in ['u', 'v', 'p']:
        pred = pred_data[field].detach().cpu().numpy()
        ref = ref_tensors[field].detach().cpu().numpy()
        rmse = np.sqrt(np.mean((pred - ref)**2))
        ref_std = ref.std()
        rel_rmse = rmse / (ref_std + 1e-10)
        metrics[f'rmse_{field}'] = float(rmse)
        metrics[f'rel_rmse_{field}'] = float(rel_rmse)
    
    # 質量守恆檢查（如果提供座標）
    if coords is not None:
        try:
            div = conservation_error(pred_data['u'], pred_data['v'], coords)
            metrics['divergence_mean'] = div
            logger.info(f"   散度誤差: {div:.2e}")
        except Exception as e:
            logger.warning(f"⚠️  無法計算散度: {e}")
    
    return metrics


def generate_evaluation(checkpoint_path: str,
                       reference_path: str,
                       config_path: Optional[str] = None,
                       output_dir: str = "evaluation_results",
                       device: str = "cpu",
                       sensor_file: Optional[str] = None,
                       apply_scale_calibration: bool = False) -> None:
    """Generate comprehensive evaluation report for a trained model."""
    # 這個函數尚未實現，使用main函數替代
    print("⚠️  generate_evaluation function not implemented. Use main() instead.")
    pass
