"""
感測器數據驗證模組
"""

import logging
from typing import Dict, Any

import numpy as np
import torch


def validate_sensor_data_quality(sensor_data: Dict[str, Any], logger) -> None:
    """
    驗證感測器數據具有合理的物理統計特性
    
    防止使用損壞的 RANS prior（v/w/p 標準差 ~10⁻⁷）進行標準化
    
    Args:
        sensor_data: 包含 u/v/w/p 的字典
        logger: logging.Logger 實例
    
    Raises:
        ValueError: 如果檢測到不合理的數據統計
    """
    logger.info("\n" + "="*80)
    logger.info("🔍 Sensor Data Quality Check")
    logger.info("="*80)
    
    for var in ['u', 'v', 'w', 'p']:
        if var not in sensor_data:
            continue
            
        data = sensor_data[var]
        if isinstance(data, torch.Tensor):
            data_np = data.cpu().numpy().flatten()
        else:
            data_np = np.asarray(data).flatten()
        
        mean = float(data_np.mean())
        std = float(data_np.std())
        data_min = float(data_np.min())
        data_max = float(data_np.max())
        
        logger.info(f"  {var}: mean={mean:.6e}, std={std:.6e}, range=[{data_min:.6e}, {data_max:.6e}]")
        
        # 檢查異常小的標準差（數值噪聲）
        if var in ['v', 'w'] and std < 1e-3:
            raise ValueError(
                f"\n❌ CRITICAL: {var} has std={std:.2e} (< 1e-3)\n"
                f"   This indicates numerical noise, NOT physical data!\n"
                f"   \n"
                f"   Likely causes:\n"
                f"     1. Sensor file has no actual DNS values (coordinates only)\n"
                f"     2. Using corrupted RANS prior for normalization (steady RANS → v/w ≈ 0)\n"
                f"     3. Data loading pipeline bug\n"
                f"   \n"
                f"   Solutions:\n"
                f"     - Option A: Generate sensor file with DNS values extracted from cutout\n"
                f"     - Option B: Force normalization from DNS cutout (not sensors)\n"
                f"   \n"
                f"   See: results/channel_flow_evaluation/ROOT_CAUSE_FINAL.md"
            )
        
        if var == 'p' and std < 1e-4:
            raise ValueError(
                f"\n❌ CRITICAL: {var} has std={std:.2e} (< 1e-4)\n"
                f"   Pressure field appears to be gauge pressure residuals!\n"
                f"   Check if RANS data is being used instead of DNS."
            )
        
        # 檢查全零或 NaN
        if np.abs(data_np).max() < 1e-10:
            raise ValueError(f"❌ {var} field is all zeros (max magnitude < 1e-10)!")
        
        if not np.isfinite(data_np).all():
            raise ValueError(f"❌ {var} field contains NaN or Inf!")
    
    logger.info("✅ Sensor data quality check PASSED")
    logger.info("="*80 + "\n")
