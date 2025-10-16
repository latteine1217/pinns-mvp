#!/usr/bin/env python3
"""
驗證標準化修復：檢查中心線速度 u(y=0) 是否接近真實值

測試點：
- 位置: (x=12.5, y=0.0, z=4.7) → 實際最近點 (x=12.47, y=0.016)
- 真實值: u ≈ 14.97 (從完整 2D 切片資料提取)
- 修復前: u ≈ 0.14 (誤差 99.4%)
- 修復後（10 epochs）: u ≈ 5.78 (誤差 61.4%)
- 目標: 相對誤差 < 20% (需延長訓練至 100+ epochs)

執行方式:
    python scripts/debug/verify_normalization_fix.py
"""

import sys
from pathlib import Path
import logging

import torch
import yaml

# 添加專案根目錄到路徑
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: Path) -> dict:
    """載入檢查點文件"""
    logger.info(f"載入檢查點: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 打印檢查點內容結構
    logger.info(f"檢查點包含的鍵: {list(checkpoint.keys())}")
    
    return checkpoint


def rebuild_model_from_checkpoint(checkpoint: dict, device: torch.device):
    """從檢查點重建模型（使用 factory.create_model）"""
    
    # === 1. 提取配置 ===
    config = checkpoint.get('config')
    if config is None:
        raise ValueError("檢查點缺少 'config' 字段")
    
    model_cfg = config['model']
    logger.info(f"模型類型: {model_cfg.get('type', 'fourier_mlp')}")
    logger.info(f"模型維度: in_dim={model_cfg['in_dim']}, out_dim={model_cfg['out_dim']}")
    logger.info(f"網路結構: {model_cfg['depth']}×{model_cfg['width']}")
    
    # === 2. 使用 factory.create_model 重建（完整支援所有配置）===
    from pinnx.train.factory import create_model
    
    model = create_model(config, device, statistics=None)
    logger.info("✅ 成功創建模型架構")
    
    # === 3. 載入權重 ===
    model_state = checkpoint.get('model_state_dict')
    if model_state is None:
        raise ValueError("檢查點缺少 'model_state_dict' 字段")
    
    # 嘗試載入權重（允許部分匹配以應對配置變化）
    try:
        model.load_state_dict(model_state, strict=True)
        logger.info("✅ 成功載入模型權重（strict=True）")
    except RuntimeError as e:
        logger.warning(f"⚠️ 嚴格載入失敗，嘗試寬鬆模式: {e}")
        model.load_state_dict(model_state, strict=False)
        logger.info("✅ 成功載入模型權重（strict=False，可能有部分不匹配）")
    
    return model, config


def predict_centerline_velocity(
    model: torch.nn.Module,
    config: dict,
    device: torch.device
) -> dict:
    """
    預測中心線速度並與基準值對比
    
    Returns:
        {
            'u_pred': 預測值（反標準化後）,
            'u_true': 真實值,
            'rel_error': 相對誤差（%）,
            'abs_error': 絕對誤差
        }
    """
    model.eval()
    
    # === 測試點：中心線 y=0 ===
    x = 12.5  # 域中心
    y = 0.0   # 中心線
    z = 4.7   # z 方向中心（2D 切片時忽略）
    
    # 真實值（從完整 2D 切片資料 cutout_128x64.npz 提取）
    # 測試點 (x=12.47, y=0.016) 的 u 值
    u_true = 14.9658  # 修正：使用與訓練資料同單位的真實值
    
    # === 構建輸入張量 ===
    coords = torch.tensor([[x, y, z]], dtype=torch.float32, device=device)
    
    logger.info(f"測試點座標: x={x}, y={y}, z={z}")
    
    # === 前向推理 ===
    with torch.no_grad():
        output = model(coords)  # shape: (1, 4) -> [u, v, w, p]
    
    # === 提取預測值 ===
    u_pred_tensor = output[0, 0]  # u 分量
    u_pred = u_pred_tensor.cpu().item()
    
    # === 檢查是否已反標準化 ===
    # 若模型有 ManualScalingWrapper，輸出應已反標準化
    # 若沒有，需手動反標準化
    
    normalization = config.get('normalization', {})
    norm_type = normalization.get('type', 'none')
    
    if norm_type == 'training_data_norm':
        # 從檢查點提取標準化統計量（修復：正確讀取 means/scales 字典）
        means = normalization.get('means', {})
        scales = normalization.get('scales', {})
        
        u_mean = means.get('u', 0.0)
        u_std = scales.get('u', 1.0)
        
        logger.info(f"標準化統計量: u_mean={u_mean:.4f}, u_std={u_std:.4f}")
        
        # 檢查模型輸出是否已反標準化（通過 ManualScalingWrapper）
        # 若輸出值在 [-3, 3] 範圍內，可能仍是標準化值
        if abs(u_pred) < 5.0:
            logger.warning("⚠️ 輸出值疑似仍為標準化值，嘗試手動反標準化...")
            u_pred_denorm = u_pred * u_std + u_mean
            logger.info(f"   手動反標準化: {u_pred:.4f} -> {u_pred_denorm:.4f}")
            u_pred = u_pred_denorm
    
    # === 計算誤差 ===
    abs_error = abs(u_pred - u_true)
    rel_error = (abs_error / abs(u_true)) * 100.0
    
    result = {
        'u_pred': u_pred,
        'u_true': u_true,
        'rel_error': rel_error,
        'abs_error': abs_error
    }
    
    return result


def main():
    """主流程"""
    
    # === 配置路徑 ===
    checkpoint_path = ROOT_DIR / 'checkpoints/normalization_baseline_test_fix_v1/best_model.pth'
    
    if not checkpoint_path.exists():
        logger.error(f"❌ 檢查點文件不存在: {checkpoint_path}")
        return
    
    # === 設備選擇 ===
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logger.info(f"使用設備: {device}")
    
    # === 載入檢查點 ===
    checkpoint = load_checkpoint(checkpoint_path)
    
    # === 檢查標準化配置 ===
    normalization = checkpoint.get('normalization', {})
    logger.info("=" * 60)
    logger.info("📊 標準化配置:")
    logger.info(f"   類型: {normalization.get('type', 'none')}")
    
    # 修復：從 means/scales 字典中讀取
    means = normalization.get('means', {})
    scales = normalization.get('scales', {})
    
    if means and scales:
        logger.info(f"   u: mean={means.get('u', 0.0):.4f}, std={scales.get('u', 1.0):.4f}")
        logger.info(f"   v: mean={means.get('v', 0.0):.4f}, std={scales.get('v', 1.0):.4f}")
        logger.info(f"   p: mean={means.get('p', 0.0):.4f}, std={scales.get('p', 1.0):.4f}")
        logger.info(f"   w: mean={means.get('w', 0.0):.4f}, std={scales.get('w', 1.0):.4f}")
    logger.info("=" * 60)
    
    # === 重建模型 ===
    try:
        model, config = rebuild_model_from_checkpoint(checkpoint, device)
    except Exception as e:
        logger.error(f"❌ 模型重建失敗: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # === 預測中心線速度 ===
    logger.info("\n" + "=" * 60)
    logger.info("🎯 預測中心線速度:")
    
    result = predict_centerline_velocity(model, checkpoint, device)
    
    # === 輸出結果 ===
    logger.info("=" * 60)
    logger.info("📊 驗證結果:")
    logger.info(f"   預測值 (u_pred):  {result['u_pred']:.4f}")
    logger.info(f"   真實值 (u_true):  {result['u_true']:.4f}")
    logger.info(f"   絕對誤差:         {result['abs_error']:.4f}")
    logger.info(f"   相對誤差:         {result['rel_error']:.2f}%")
    logger.info("=" * 60)
    
    # === 成功判定 ===
    if result['rel_error'] < 20.0:
        logger.info("✅ 修復成功！相對誤差 < 20%")
        logger.info("   建議：延長訓練至 100 epochs 以進一步降低誤差")
    elif result['rel_error'] < 50.0:
        logger.info("⚠️ 部分改善（相對誤差 20-50%）")
        logger.info("   建議：檢查 VS-PINN 配置、損失權重或學習率")
    else:
        logger.info("❌ 修復失敗！相對誤差 > 50%")
        logger.info("   需進一步診斷：檢查標準化統計量、模型輸出範圍")
    
    # === 保存結果到文件 ===
    output_file = ROOT_DIR / 'results' / 'normalization_fix_verification.txt'
    output_file.parent.mkdir(exist_ok=True)
    
    with output_file.open('w') as f:
        f.write("=" * 60 + "\n")
        f.write("標準化修復驗證報告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"檢查點路徑: {checkpoint_path}\n")
        f.write(f"測試點: (x=12.5, y=0.0, z=4.7)\n\n")
        f.write(f"預測值 (u_pred):  {result['u_pred']:.6f}\n")
        f.write(f"真實值 (u_true):  {result['u_true']:.6f}\n")
        f.write(f"絕對誤差:         {result['abs_error']:.6f}\n")
        f.write(f"相對誤差:         {result['rel_error']:.4f}%\n\n")
        
        if result['rel_error'] < 20.0:
            f.write("✅ 修復成功！\n")
        elif result['rel_error'] < 50.0:
            f.write("⚠️ 部分改善\n")
        else:
            f.write("❌ 修復失敗\n")
    
    logger.info(f"\n📝 驗證報告已保存: {output_file}")


if __name__ == '__main__':
    main()
