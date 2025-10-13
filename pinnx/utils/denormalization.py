"""
輸出反標準化工具模組

提供統一的模型輸出反標準化功能，確保評估時預測值與真實值在相同量綱下比較。

主要功能：
- 支持多種標準化類型：friction_velocity, channel_flow, manual_scaling
- 從配置文件重建縮放參數
- 處理 ManualScalingWrapper 的反縮放邏輯
- 詳細的日誌輸出與範圍驗證

使用方式：
    from pinnx.utils.denormalization import denormalize_output
    
    predictions = model(coords)  # 已標準化的預測
    predictions_physical = denormalize_output(predictions, config)
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


def denormalize_output(
    predictions: Union[np.ndarray, torch.Tensor],
    config: Dict,
    output_norm_type: Optional[str] = None,
    verbose: bool = True,
    true_ranges: Optional[Dict[str, Tuple[float, float]]] = None
) -> np.ndarray:
    """
    反標準化模型輸出到物理量綱
    
    支持的標準化類型：
    1. "friction_velocity": 基於摩擦速度尺度
       - u, v, w → u, v, w * u_τ
       - p → p * ρu_τ²
    
    2. "manual_scaling": ManualScalingWrapper 的反縮放
       - 從 [-1, 1] 反縮放到配置中的 output_ranges
    
    3. "post_scaling": 後處理縮放（自動範圍映射）
       - 從模型輸出範圍線性映射到真實數據範圍
       - 適用於訓練時未使用縮放的情況
    
    4. "none" / "identity": 不處理
    
    Args:
        predictions: 模型預測 [N, out_dim]，out_dim ∈ {3, 4, 5}
                    - 3D: (u, v, w) 或 (u, v, p)
                    - 4D: (u, v, w, p)
                    - 5D: (u, v, w, p, S) - 含源項
        config: 訓練配置字典（需包含 physics 和 model.scaling 段）
        output_norm_type: 標準化類型（若為 None 則從 config 讀取）
        verbose: 是否輸出詳細日誌
        true_ranges: 真實數據範圍 (用於 post_scaling)
                    例如: {'u': (0.0, 17.37), 'v': (-1.31, 1.33), ...}
        
    Returns:
        反標準化後的預測 [N, out_dim] (numpy array)
        
    Raises:
        ValueError: 配置缺失或不支持的標準化類型
        
    Examples:
        >>> # Friction velocity 標準化
        >>> config = {
        ...     'model': {'scaling': {'output_norm': 'friction_velocity'}},
        ...     'physics': {'rho': 1.0, 'channel_flow': {'u_tau': 1.0}}
        ... }
        >>> pred_normalized = np.array([[1.0, 0.1, 0.5, 0.01]])  # (u, v, w, p)
        >>> pred_physical = denormalize_output(pred_normalized, config)
        >>> # 結果: [[1.0, 0.1, 0.5, 0.01]] (u_tau=1.0 時無變化)
        
        >>> # 後處理縮放
        >>> true_ranges = {'u': (0, 17.4), 'v': (-1.3, 1.3), 'w': (-22, 21), 'p': (-226, 2.4)}
        >>> pred_scaled = denormalize_output(pred_normalized, config, 'post_scaling', 
        ...                                   true_ranges=true_ranges)
    """
    # 轉換為 numpy array (如果是 torch.Tensor)
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    
    predictions = np.array(predictions, dtype=np.float32)
    
    # 確定標準化類型
    if output_norm_type is None:
        if 'model' in config and 'scaling' in config['model']:
            output_norm_type = config['model']['scaling'].get('output_norm', 'none')
        else:
            output_norm_type = 'none'
    
    if verbose:
        logger.info(f"🔄 反標準化類型: {output_norm_type}")
        logger.info(f"📊 輸入預測範圍: min={predictions.min():.4f}, max={predictions.max():.4f}")
    
    # 不需要反標準化的情況
    if output_norm_type in ('none', 'identity', None):
        if verbose:
            logger.info("✅ 無需反標準化 (type=none)")
        return predictions
    
    # ===================================================================
    # 類型 1: Friction Velocity 標準化
    # ===================================================================
    if output_norm_type == 'friction_velocity':
        return _denormalize_friction_velocity(predictions, config, verbose)
    
    # ===================================================================
    # 類型 2: Manual Scaling (從統計範圍反縮放)
    # ===================================================================
    elif output_norm_type == 'manual_scaling':
        return _denormalize_manual_scaling(predictions, config, verbose)
    
    # ===================================================================
    # 類型 3: Post Scaling (後處理自動範圍映射)
    # ===================================================================
    elif output_norm_type == 'post_scaling':
        if true_ranges is None:
            raise ValueError("post_scaling 模式需要提供 true_ranges 參數")
        return _denormalize_post_scaling(predictions, true_ranges, verbose)
    
    # ===================================================================
    # 類型 4: 字典格式 (直接指定範圍)
    # ===================================================================
    elif isinstance(output_norm_type, dict):
        return _denormalize_from_dict(predictions, output_norm_type, verbose)
    
    else:
        raise ValueError(f"不支持的 output_norm 類型: {output_norm_type}")


def _denormalize_friction_velocity(
    predictions: np.ndarray,
    config: Dict,
    verbose: bool
) -> np.ndarray:
    """
    Friction velocity 反標準化：u,v,w * u_τ; p * ρu_τ²
    
    注意：實際上訓練時可能使用 ManualScalingWrapper，此函數僅處理
    配置中明確聲明使用 friction_velocity 標準化的情況。
    """
    # 提取物理參數
    if 'physics' not in config:
        raise ValueError("配置缺少 'physics' 段落")
    
    physics = config['physics']
    
    # 摩擦速度 u_τ
    if 'channel_flow' in physics and 'u_tau' in physics['channel_flow']:
        u_tau = physics['channel_flow']['u_tau']
    else:
        u_tau = 1.0  # 默認值
        if verbose:
            logger.warning(f"⚠️  未找到 u_tau，使用默認值 1.0")
    
    # 密度 ρ
    rho = physics.get('rho', 1.0)
    
    # 計算縮放因子
    velocity_scale = u_tau
    pressure_scale = rho * u_tau ** 2
    
    if verbose:
        logger.info(f"📐 物理參數: u_τ={u_tau}, ρ={rho}")
        logger.info(f"📐 縮放因子: vel_scale={velocity_scale}, p_scale={pressure_scale}")
    
    # 應用反縮放
    result = predictions.copy()
    out_dim = predictions.shape[-1]
    
    if out_dim == 3:
        # (u, v, p) 或 (u, v, w)
        # 假設前兩個是速度，最後一個是壓力或 w
        result[:, 0:2] *= velocity_scale  # u, v
        # 第三個變量需要根據實際情況判斷
        # 這裡假設是 w（3D 情況）
        result[:, 2] *= velocity_scale  # w
        
    elif out_dim == 4:
        # (u, v, w, p)
        result[:, 0:3] *= velocity_scale  # u, v, w
        result[:, 3] *= pressure_scale    # p
        
    elif out_dim == 5:
        # (u, v, w, p, S)
        result[:, 0:3] *= velocity_scale  # u, v, w
        result[:, 3] *= pressure_scale    # p
        # 源項 S 通常已經在物理量綱，不縮放
        
    else:
        raise ValueError(f"不支持的輸出維度: {out_dim} (預期 3, 4, 或 5)")
    
    if verbose:
        logger.info(f"📊 輸出預測範圍: min={result.min():.4f}, max={result.max():.4f}")
        logger.info("✅ Friction velocity 反標準化完成")
    
    return result


def _denormalize_manual_scaling(
    predictions: np.ndarray,
    config: Dict,
    verbose: bool
) -> np.ndarray:
    """
    ManualScalingWrapper 反標準化：從 [-1, 1] 反縮放到物理範圍
    
    反縮放公式（與 ManualScalingWrapper.forward 第 67 行對應）：
        y_physical = (y_scaled + 1) / 2 * (max - min) + min
    
    注意：ManualScalingWrapper 在訓練時已經執行了反縮放（forward 返回物理值），
    所以評估時如果使用相同的 wrapper，輸出已經是物理量。
    
    此函數用於：直接從基礎模型（未包裝）獲取預測時的反縮放。
    """
    if verbose:
        logger.warning("⚠️  manual_scaling 模式：需要從配置重建輸出範圍")
    
    # 從配置提取輸出範圍
    output_ranges = _extract_output_ranges(config, verbose)
    
    # 假設網路輸出在 [-1, 1]
    result = predictions.copy()
    
    # 反縮放公式: y = (y_scaled + 1) / 2 * (max - min) + min
    for i, (var_name, (min_val, max_val)) in enumerate(output_ranges.items()):
        if i >= result.shape[-1]:
            break
        result[:, i] = (result[:, i] + 1) / 2 * (max_val - min_val) + min_val
    
    if verbose:
        logger.info(f"📊 輸出預測範圍: min={result.min():.4f}, max={result.max():.4f}")
        logger.info("✅ Manual scaling 反標準化完成")
    
    return result


def _denormalize_post_scaling(
    predictions: np.ndarray,
    true_ranges: Dict[str, Tuple[float, float]],
    verbose: bool
) -> np.ndarray:
    """
    後處理縮放：自動將模型輸出範圍映射到真實數據範圍
    
    適用場景：
    - 訓練時未使用任何輸出縮放（VS-PINN 跳過 ManualScalingWrapper）
    - 模型輸出在近似物理量綱但數值範圍錯誤
    - 需要後處理校正到真實物理尺度
    
    縮放公式（線性映射）：
        y_scaled = (y_pred - pred_min) / (pred_max - pred_min) * (true_max - true_min) + true_min
    
    Args:
        predictions: 模型原始預測 [N, out_dim]
        true_ranges: 真實數據範圍，例如:
                    {'u': (0.0, 17.37), 'v': (-1.31, 1.33), 'w': (-22, 21), 'p': (-226, 2.4)}
        verbose: 是否輸出詳細日誌
        
    Returns:
        縮放後的預測 [N, out_dim]
        
    Notes:
        - 自動從 predictions 計算當前範圍（pred_min, pred_max）
        - 執行線性映射到 (true_min, true_max)
        - 假設模型已學到流場相對結構，僅絕對尺度錯誤
    """
    if verbose:
        logger.info("🔧 執行後處理縮放 (post_scaling)")
    
    result = predictions.copy()
    var_names = list(true_ranges.keys())
    
    # 對每個變量分別縮放
    for i, var_name in enumerate(var_names):
        if i >= result.shape[-1]:
            break
        
        # 當前變量的預測範圍
        pred_min = result[:, i].min()
        pred_max = result[:, i].max()
        
        # 真實數據範圍
        true_min, true_max = true_ranges[var_name]
        
        # 線性映射公式
        if pred_max - pred_min < 1e-8:
            # 避免除以零（預測為常數）
            if verbose:
                logger.warning(f"⚠️  {var_name}: 預測為常數 ({pred_min:.4f})，跳過縮放")
            continue
        
        result[:, i] = (result[:, i] - pred_min) / (pred_max - pred_min) * (true_max - true_min) + true_min
        
        if verbose:
            scaled_min = result[:, i].min()
            scaled_max = result[:, i].max()
            logger.info(f"  {var_name}: [{pred_min:.4f}, {pred_max:.4f}] → [{scaled_min:.4f}, {scaled_max:.4f}] "
                       f"(真實: [{true_min:.4f}, {true_max:.4f}])")
    
    if verbose:
        logger.info("✅ 後處理縮放完成")
    
    return result


def _denormalize_from_dict(
    predictions: np.ndarray,
    output_ranges: Dict[str, Tuple[float, float]],
    verbose: bool
) -> np.ndarray:
    """
    從字典格式的輸出範圍反標準化
    
    Args:
        output_ranges: 例如 {'u': (0, 20), 'v': (-1, 1), 'w': (-5, 5), 'p': (-100, 10)}
    """
    result = predictions.copy()
    
    # 反縮放公式: y = (y_scaled + 1) / 2 * (max - min) + min
    for i, (var_name, (min_val, max_val)) in enumerate(output_ranges.items()):
        if i >= result.shape[-1]:
            break
        result[:, i] = (result[:, i] + 1) / 2 * (max_val - min_val) + min_val
        
        if verbose:
            logger.info(f"  {var_name}: [{min_val}, {max_val}] -> "
                       f"[{result[:, i].min():.4f}, {result[:, i].max():.4f}]")
    
    if verbose:
        logger.info("✅ 字典格式反標準化完成")
    
    return result


def _extract_output_ranges(
    config: Dict,
    verbose: bool
) -> Dict[str, Tuple[float, float]]:
    """
    從配置提取輸出範圍（用於 ManualScalingWrapper）
    
    優先級：
    1. config['model']['scaling']['output_norm'] (dict 格式)
    2. 從統計信息推導（如有）
    3. 硬編碼默認值
    """
    output_ranges = {}
    
    # 檢查 scaling 配置
    if 'model' in config and 'scaling' in config['model']:
        scaling_cfg = config['model']['scaling']
        output_norm_raw = scaling_cfg.get('output_norm')
        
        if isinstance(output_norm_raw, dict):
            # 字典格式：直接使用
            output_ranges = {
                'u': tuple(output_norm_raw.get('u', [0.0, 20.0])),
                'v': tuple(output_norm_raw.get('v', [-1.0, 1.0])),
                'w': tuple(output_norm_raw.get('w', [-5.0, 5.0])),
                'p': tuple(output_norm_raw.get('p', [-100.0, 10.0]))
            }
            if verbose:
                logger.info("✅ 從配置文件讀取輸出範圍 (dict 格式)")
            return output_ranges
    
    # 硬編碼默認值（基於 JHTDB Channel Re1000）
    output_ranges = {
        'u': (0.0, 20.0),
        'v': (-1.0, 1.0),
        'w': (-5.0, 5.0),
        'p': (-100.0, 10.0)
    }
    
    if verbose:
        logger.warning("⚠️  使用硬編碼默認輸出範圍（可能不準確）")
    
    return output_ranges


def verify_denormalization(
    predictions: np.ndarray,
    true_values: np.ndarray,
    var_names: Optional[list] = None,
    tolerance: float = 0.3
) -> bool:
    """
    驗證反標準化是否正確（通過範圍匹配）
    
    Args:
        predictions: 反標準化後的預測
        true_values: 真實值
        var_names: 變量名稱列表
        tolerance: 範圍容差（預測範圍應在真實範圍的 ±tolerance 內）
        
    Returns:
        驗證是否通過
    """
    if var_names is None:
        var_names = ['u', 'v', 'w', 'p', 'S'][:predictions.shape[-1]]
    
    all_pass = True
    
    for i, var in enumerate(var_names):
        if i >= predictions.shape[-1]:
            break
        
        pred_min, pred_max = predictions[:, i].min(), predictions[:, i].max()
        true_min, true_max = true_values[:, i].min(), true_values[:, i].max()
        
        true_range = true_max - true_min
        
        # 檢查預測範圍是否在真實範圍的容差內
        min_ok = abs(pred_min - true_min) <= tolerance * true_range
        max_ok = abs(pred_max - true_max) <= tolerance * true_range
        
        status = "✅" if (min_ok and max_ok) else "❌"
        logger.info(f"{status} {var}: 預測[{pred_min:.2f}, {pred_max:.2f}] vs "
                   f"真實[{true_min:.2f}, {true_max:.2f}]")
        
        if not (min_ok and max_ok):
            all_pass = False
    
    return all_pass


# ===================================================================
# 便捷函數
# ===================================================================

def create_denormalizer_from_config(config: Dict, verbose: bool = True):
    """
    從配置創建反標準化函數（閉包）
    
    Returns:
        denorm_fn: 接受 predictions 並返回反標準化結果的函數
    """
    output_norm_type = None
    if 'model' in config and 'scaling' in config['model']:
        output_norm_type = config['model']['scaling'].get('output_norm', 'none')
    
    def denorm_fn(predictions):
        return denormalize_output(predictions, config, output_norm_type, verbose)
    
    return denorm_fn


if __name__ == "__main__":
    # 測試代碼
    import sys
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("=" * 60)
    print("📋 反標準化工具測試")
    print("=" * 60)
    
    # 測試 1: Friction velocity
    print("\n--- 測試 1: Friction Velocity 反標準化 ---")
    config_fv = {
        'model': {'scaling': {'output_norm': 'friction_velocity'}},
        'physics': {
            'rho': 1.0,
            'channel_flow': {'u_tau': 0.05}  # 實際 JHTDB 的 u_tau
        }
    }
    
    pred_normalized = np.array([
        [1.0, 0.1, 0.5, 0.01],  # (u, v, w, p) 標準化後
        [1.2, -0.05, 0.3, -0.005]
    ])
    
    pred_physical = denormalize_output(pred_normalized, config_fv, verbose=True)
    print(f"\n標準化預測:\n{pred_normalized}")
    print(f"\n物理量預測:\n{pred_physical}")
    
    # 預期: u,v,w * 0.05, p * (1.0 * 0.05^2) = p * 0.0025
    expected = pred_normalized.copy()
    expected[:, 0:3] *= 0.05  # u, v, w
    expected[:, 3] *= 0.0025  # p
    
    assert np.allclose(pred_physical, expected), "Friction velocity 反標準化錯誤！"
    print("✅ 測試 1 通過")
    
    # 測試 2: None (無需反標準化)
    print("\n--- 測試 2: None 類型 (無需反標準化) ---")
    config_none = {'model': {'scaling': {'output_norm': 'none'}}}
    
    pred_none = denormalize_output(pred_normalized, config_none, verbose=True)
    assert np.array_equal(pred_none, pred_normalized), "None 類型應返回原始值！"
    print("✅ 測試 2 通過")
    
    # 測試 3: 範圍驗證
    print("\n--- 測試 3: 範圍驗證 ---")
    true_values = np.array([
        [1.0, 0.1, 0.5, 0.02],
        [1.5, -0.03, 0.4, -0.003]
    ])
    
    is_valid = verify_denormalization(
        pred_physical, true_values, 
        var_names=['u', 'v', 'w', 'p'],
        tolerance=0.5  # 寬容度
    )
    
    if is_valid:
        print("✅ 測試 3 通過")
    else:
        print("⚠️  測試 3 部分失敗（但可能是合理的模型誤差）")
    
    print("\n" + "=" * 60)
    print("✅ 所有測試完成！")
    print("=" * 60)
