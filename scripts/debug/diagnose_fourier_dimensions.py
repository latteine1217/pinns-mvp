"""
診斷 Fourier Annealing 訓練中的維度不匹配問題
"""

import torch
import sys
import logging
from pathlib import Path

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 添加專案根目錄到路徑
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from pinnx.models.axis_selective_fourier import AxisSelectiveFourierFeatures
from pinnx.train.factory import create_model
from pinnx.train.config_loader import load_config

def diagnose_model_creation():
    """診斷模型創建過程中的維度"""
    
    # 載入配置
    config_path = project_root / "configs" / "channel_flow_fourier_annealing_demo.yml"
    config = load_config(str(config_path))
    
    logging.info("=" * 60)
    logging.info("📋 配置資訊")
    logging.info("=" * 60)
    
    model_cfg = config['model']
    logging.info(f"  模型類型: {model_cfg['type']}")
    logging.info(f"  輸入維度: {model_cfg['in_dim']}")
    logging.info(f"  輸出維度: {model_cfg['out_dim']}")
    logging.info(f"  MLP 寬度: {model_cfg['width']}")
    logging.info(f"  MLP 深度: {model_cfg['depth']}")
    
    fourier_cfg = model_cfg['fourier_features']
    axes_config = fourier_cfg['axes_config']
    logging.info(f"\n  Fourier 軸配置:")
    for axis, freqs in axes_config.items():
        logging.info(f"    {axis}: {freqs} → {len(freqs)} 頻率 → {2*len(freqs)} 維")
    
    expected_dim = sum(2 * len(freqs) for freqs in axes_config.values())
    logging.info(f"  預期 Fourier 輸出維度: {expected_dim}")
    
    # 創建模型
    logging.info("\n" + "=" * 60)
    logging.info("🔧 創建模型")
    logging.info("=" * 60)
    
    device = torch.device('cpu')
    model = create_model(config, device)
    
    # 檢查模型結構
    logging.info("\n" + "=" * 60)
    logging.info("🔍 模型結構檢查")
    logging.info("=" * 60)
    
    if hasattr(model, 'fourier_features'):
        ff = model.fourier_features
        logging.info(f"  Fourier Features 實例:")
        logging.info(f"    類型: {type(ff).__name__}")
        logging.info(f"    輸入維度: {ff.in_dim}")
        logging.info(f"    輸出維度: {ff.out_dim}")
        logging.info(f"    B 矩陣形狀: {ff.B.shape}")
        logging.info(f"    活躍頻率: {ff.get_active_frequencies()}")
    else:
        logging.error("❌ 模型沒有 fourier_features 屬性！")
        return
    
    if hasattr(model, 'hidden_layers'):
        first_layer = model.hidden_layers[0]
        logging.info(f"\n  第一層 MLP:")
        if hasattr(first_layer, 'linear'):
            weight = first_layer.linear.weight
            logging.info(f"    權重形狀: {weight.shape}")
            logging.info(f"    期望輸入維度: {weight.shape[1]}")
        elif hasattr(first_layer, 'weight'):
            weight = first_layer.weight
            logging.info(f"    權重形狀: {weight.shape}")
            logging.info(f"    期望輸入維度: {weight.shape[1]}")
        else:
            logging.warning("    無法找到權重矩陣")
    else:
        logging.error("❌ 模型沒有 hidden_layers 屬性！")
        return
    
    # 測試前向傳播
    logging.info("\n" + "=" * 60)
    logging.info("🚀 測試前向傳播")
    logging.info("=" * 60)
    
    batch_size = 10
    x = torch.randn(batch_size, 3)  # 3D 座標
    
    logging.info(f"  輸入座標形狀: {x.shape}")
    
    # Step 1: Fourier 編碼
    logging.info("\n  【Step 1】Fourier 特徵編碼")
    features = model.fourier_features(x)
    logging.info(f"    輸出形狀: {features.shape}")
    logging.info(f"    ✅ 預期: ({batch_size}, {ff.out_dim})")
    
    # Step 2: 第一層前向傳播
    logging.info("\n  【Step 2】第一層 MLP")
    try:
        h1 = model.hidden_layers[0](features)
        logging.info(f"    輸出形狀: {h1.shape}")
        logging.info(f"    ✅ 成功通過第一層")
    except RuntimeError as e:
        logging.error(f"    ❌ 錯誤: {e}")
        logging.error(f"    維度不匹配:")
        logging.error(f"      輸入: {features.shape}")
        logging.error(f"      期望: ({batch_size}, {weight.shape[1]})")
        return
    
    # Step 3: 完整前向傳播
    logging.info("\n  【Step 3】完整模型前向傳播")
    try:
        output = model(x)
        logging.info(f"    輸出形狀: {output.shape}")
        logging.info(f"    ✅ 預期: ({batch_size}, {model_cfg['out_dim']})")
        logging.info("\n✅ 所有測試通過！模型結構正確。")
    except RuntimeError as e:
        logging.error(f"    ❌ 完整前向傳播失敗: {e}")
        return
    
    # 統計參數量
    logging.info("\n" + "=" * 60)
    logging.info("📊 模型統計")
    logging.info("=" * 60)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"  總參數量: {total_params:,}")
    logging.info(f"  可訓練參數: {trainable_params:,}")

if __name__ == "__main__":
    diagnose_model_creation()
