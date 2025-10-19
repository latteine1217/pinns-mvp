"""
檢查點管理模組

提供模型訓練過程中的檢查點保存與載入功能。
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import Optimizer
import logging

from pinnx.physics.validators import compute_physics_metrics

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    loss: float,
    config: Dict[str, Any],
    checkpoint_dir: str = "checkpoints"
) -> str:
    """
    保存模型檢查點。
    
    自動保存兩份檢查點：
    1. 帶 epoch 編號的檢查點（用於歷史記錄）
    2. 最新檢查點（方便快速恢復訓練）
    
    Parameters
    ----------
    model : nn.Module
        要保存的模型。
    optimizer : Optimizer
        優化器（會保存其狀態以便恢復訓練）。
    epoch : int
        當前訓練 epoch 數。
    loss : float
        當前損失值。
    config : Dict[str, Any]
        完整的訓練配置字典（需包含 'experiment.name' 欄位）。
    checkpoint_dir : str, optional
        檢查點保存目錄，預設為 "checkpoints"。
    
    Returns
    -------
    str
        保存的檢查點檔案路徑（帶 epoch 編號的版本）。
    
    Raises
    ------
    KeyError
        若 config 中缺少 'experiment.name' 欄位。
    
    Examples
    --------
    >>> model = FourierMLP(...)
    >>> optimizer = torch.optim.Adam(model.parameters())
    >>> config = {'experiment': {'name': 'test_run'}}
    >>> path = save_checkpoint(model, optimizer, epoch=100, loss=0.01, config=config)
    >>> print(path)
    'checkpoints/test_run_epoch_100.pth'
    
    Notes
    -----
    檢查點包含以下信息：
    - epoch: 訓練輪數
    - model_state_dict: 模型參數
    - optimizer_state_dict: 優化器狀態
    - loss: 損失值
    - config: 完整配置
    """
    # 確保目錄存在
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # 構建檢查點字典
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }
    
    # 獲取實驗名稱
    try:
        experiment_name = config['experiment']['name']
    except KeyError as e:
        raise KeyError("config 必須包含 'experiment.name' 欄位") from e
    
    # 保存帶 epoch 編號的檢查點
    checkpoint_path = os.path.join(checkpoint_dir, f"{experiment_name}_epoch_{epoch}.pth")
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"檢查點已保存: {checkpoint_path}")
    
    # 保存最新檢查點（覆蓋式）
    latest_path = os.path.join(checkpoint_dir, f"{experiment_name}_latest.pth")
    torch.save(checkpoint, latest_path)
    logger.debug(f"最新檢查點已更新: {latest_path}")
    
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    device: str = "cpu"
) -> Tuple[int, float, Optional[Dict[str, Any]]]:
    """
    載入模型檢查點。
    
    Parameters
    ----------
    checkpoint_path : str
        檢查點檔案路徑。
    model : nn.Module
        要載入參數的模型（會 in-place 修改）。
    optimizer : Optimizer, optional
        要載入狀態的優化器（若為 None 則不載入優化器狀態）。
    device : str, optional
        載入到的設備，預設為 "cpu"。
        可選值: "cpu", "cuda", "cuda:0", "mps" 等。
    
    Returns
    -------
    epoch : int
        檢查點保存時的 epoch 數。
    loss : float
        檢查點保存時的損失值。
    config : Dict[str, Any] or None
        訓練配置（若檢查點中不包含則返回 None）。
    
    Raises
    ------
    FileNotFoundError
        若檢查點檔案不存在。
    RuntimeError
        若模型參數載入失敗（例如架構不匹配）。
    
    Examples
    --------
    >>> model = FourierMLP(...)
    >>> optimizer = torch.optim.Adam(model.parameters())
    >>> epoch, loss, config = load_checkpoint(
    ...     "checkpoints/test_run_latest.pth", 
    ...     model, 
    ...     optimizer,
    ...     device="cuda"
    ... )
    >>> print(f"從 epoch {epoch} 恢復訓練，損失: {loss:.6f}")
    
    Notes
    -----
    - 模型和優化器會被 in-place 修改
    - 若只需推理，可不傳入 optimizer
    - 使用 map_location 確保跨設備載入安全
    """
    # 檢查檔案是否存在
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"檢查點檔案不存在: {checkpoint_path}")
    
    # 載入檢查點
    logger.info(f"正在載入檢查點: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 載入模型參數
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("模型參數載入成功")
    except RuntimeError as e:
        logger.error(f"模型參數載入失敗: {e}")
        raise
    
    # 載入優化器狀態（可選）
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("優化器狀態載入成功")
        except Exception as e:
            logger.warning(f"優化器狀態載入失敗（將使用初始狀態）: {e}")
    
    # 提取元數據
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    config = checkpoint.get('config', None)
    
    logger.info(f"檢查點載入完成 - Epoch: {epoch}, Loss: {loss:.6f}")

    return epoch, loss, config


def validate_physics_before_save(
    model: nn.Module,
    coords: torch.Tensor,
    config: Dict[str, Any],
    device: torch.device
) -> Tuple[bool, Dict[str, Any]]:
    """
    在保存檢查點前執行物理一致性驗證

    Args:
        model: PINN 模型
        coords: 驗證點座標 [N, 2] 或 [N, 3]
        config: 訓練配置
        device: 計算設備

    Returns:
        (驗證是否通過, 物理指標字典)
    """
    # 檢查是否啟用物理驗證
    physics_val_cfg = config.get('physics_validation', {})
    if not physics_val_cfg.get('enabled', True):
        logger.info("物理驗證已禁用，跳過驗證")
        return True, {}

    # 獲取驗證閾值
    thresholds = physics_val_cfg.get('thresholds', {
        'mass_conservation': 1e-2,
        'momentum_conservation': 1e-1,
        'boundary_condition': 1e-3
    })

    # 獲取物理參數
    physics_params = {
        'nu': config.get('physics', {}).get('nu', 5e-5),
        'wall_positions': config.get('domain', {}).get('wall_positions', (0.0, 2.0))
    }

    # 確保座標在正確的設備上且需要梯度
    coords = coords.to(device)
    if not coords.requires_grad:
        coords = coords.clone().detach().requires_grad_(True)

    try:
        # 使用模型進行預測
        model.eval()
        with torch.no_grad():
            predictions_raw = model(coords)

        # 重新預測以計算梯度（用於物理驗證）
        model.eval()
        coords_grad = coords.clone().detach().requires_grad_(True)
        predictions_grad = model(coords_grad)

        # 提取預測場（支援 2D 和 3D）
        if predictions_grad.shape[1] == 3:  # 2D: u, v, p
            u, v, p = predictions_grad[:, 0:1], predictions_grad[:, 1:2], predictions_grad[:, 2:3]
            predictions = {'u': u, 'v': v, 'p': p}
        elif predictions_grad.shape[1] == 4:  # 3D: u, v, w, p
            u, v, w, p = predictions_grad[:, 0:1], predictions_grad[:, 1:2], predictions_grad[:, 2:3], predictions_grad[:, 3:4]
            predictions = {'u': u, 'v': v, 'w': w, 'p': p}
        else:
            logger.warning(f"未知的輸出維度: {predictions_grad.shape[1]}，跳過物理驗證")
            return True, {}

        # 計算物理指標
        metrics = compute_physics_metrics(
            coords_grad,
            predictions,
            physics_params=physics_params,
            validation_thresholds=thresholds
        )

        # 檢查驗證結果
        validation_passed = metrics['validation_passed']
        trivial_solution = metrics.get('trivial_solution', {})

        # === 診斷性輸出（記錄但不拒絕） ===
        logger.info("=" * 60)
        logger.info("📊 物理診斷報告")
        logger.info("=" * 60)

        # 質量守恆
        mass_status = "✓" if metrics['mass_conservation_passed'] else "✗"
        logger.info(f"質量守恆誤差: {metrics['mass_conservation_error']:.6e} "
                   f"(閾值: {thresholds['mass_conservation']:.6e}) [{mass_status}]")

        # 動量守恆
        momentum_status = "✓" if metrics['momentum_conservation_passed'] else "✗"
        logger.info(f"動量守恆誤差: {metrics['momentum_conservation_error']:.6e} "
                   f"(閾值: {thresholds['momentum_conservation']:.6e}) [{momentum_status}]")

        # 邊界條件
        bc_status = "✓" if metrics['boundary_condition_passed'] else "✗"
        logger.info(f"邊界條件誤差: {metrics['boundary_condition_error']:.6e} "
                   f"(閾值: {thresholds['boundary_condition']:.6e}) [{bc_status}]")

        # Trivial Solution 檢測（這個是嚴重警告）
        if trivial_solution.get('is_trivial', False):
            logger.warning("=" * 60)
            logger.warning("🚨 警告：檢測到 Trivial Solution！")
            logger.warning(f"   類型: {trivial_solution['type']}")
            logger.warning(f"   詳情: {trivial_solution['details']}")
            logger.warning("=" * 60)
            logger.warning("建議檢查：")
            logger.warning("  1. PDE Loss Ratio 是否過低（< 10%）")
            logger.warning("  2. 資料損失權重是否過高（壓制物理約束）")
            logger.warning("  3. 學習率是否過低（無法逃離局部最小值）")
            logger.warning("  4. 初始化是否合理（檢查 Fourier features）")
            logger.warning("=" * 60)

        # 整體評估
        if not validation_passed:
            if trivial_solution.get('is_trivial', False):
                logger.warning("⚠️  物理診斷：Trivial Solution（建議立即檢查）")
            else:
                logger.info("ℹ️  物理診斷：約束未滿足（訓練初期正常）")
        else:
            logger.info("✓ 物理診斷：約束滿足")

        logger.info("=" * 60)

        # 僅在 strict_mode 時才拒絕保存
        strict_mode = config.get('physics_validation', {}).get('strict_mode', False)

        if strict_mode and trivial_solution.get('is_trivial', False):
            logger.error("❌ Strict Mode: 檢測到 Trivial Solution，拒絕保存")
            return False, metrics

        # 預設：總是允許保存（僅診斷）
        return True, metrics

    except Exception as e:
        logger.error(f"物理驗證過程中發生錯誤: {str(e)}")
        logger.warning("由於驗證錯誤，將允許保存檢查點（請手動檢查模型）")
        return True, {'validation_error': str(e)}
