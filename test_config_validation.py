#!/usr/bin/env python
"""
快速驗證簡化後的配置文件
檢查 DNS NPY 參數載入和優先級
"""

import sys
import logging
from pathlib import Path

# 添加專案根目錄到路徑
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import torch
import yaml

from pinnx.train.config_loader import load_config
from pinnx.train.model_physics_factory import create_model, create_physics
from pinnx.dataio.loaders.kolmogorov import prepare_kolmogorov_training_data

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def validate_config(config_path: str):
    """驗證配置文件載入和參數來源"""
    logger.info("="*70)
    logger.info("🔍 配置文件驗證測試")
    logger.info("="*70)
    logger.info(f"\n📄 配置文件: {config_path}\n")

    # 1. 載入配置
    logger.info("步驟 1: 載入配置文件...")
    config = load_config(config_path)
    logger.info("✅ 配置載入成功")

    # 2. 載入 DNS 數據（觸發自動回填）
    logger.info("\n步驟 2: 載入 DNS 數據...")
    device = torch.device('cpu')  # 驗證時使用 CPU
    training_data = prepare_kolmogorov_training_data(config, device)
    logger.info("✅ DNS 數據載入成功")

    # 3. 檢查回填的物理參數
    logger.info("\n步驟 3: 檢查回填的物理參數...")
    kol_cfg = config.get('data', {}).get('kolmogorov_config', {})
    dns_params = kol_cfg.get('physics_params', {})

    if dns_params:
        logger.info("✅ DNS NPY 參數已回填:")
        for key, value in dns_params.items():
            logger.info(f"   {key}: {value}")
    else:
        logger.warning("⚠️  未找到回填的 physics_params")

    # 4. 創建物理模塊（觸發參數優先級邏輯）
    logger.info("\n步驟 4: 創建物理模塊（檢查參數來源）...")
    logger.info("-" * 70)
    physics = create_physics(config, training_data)
    logger.info("-" * 70)
    logger.info("✅ 物理模塊創建成功")

    # 5. 驗證物理參數
    logger.info("\n步驟 5: 驗證物理參數...")
    logger.info(f"   nu (黏性係數): {physics.nu}")
    logger.info(f"   rho (密度): {physics.rho}")
    if hasattr(physics, 'forcing_params'):
        logger.info(f"   forcing_params: {physics.forcing_params}")

    # 6. 創建模型（快速驗證）
    logger.info("\n步驟 6: 創建模型...")
    model = create_model(config, training_data)
    logger.info(f"✅ 模型創建成功")
    logger.info(f"   參數總數: {sum(p.numel() for p in model.parameters()):,}")

    logger.info("\n" + "="*70)
    logger.info("✅ 配置驗證測試通過")
    logger.info("="*70)

    return True


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='驗證簡化後的配置文件')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/kolmogorov_re100_kf4_K100.yml',
        help='配置文件路徑'
    )

    args = parser.parse_args()

    try:
        validate_config(args.config)
    except Exception as e:
        logger.error(f"\n❌ 驗證失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
