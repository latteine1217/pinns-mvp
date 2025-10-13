#!/usr/bin/env python3
"""
測試邊界條件修復效果
快速訓練 100 epochs，驗證壁面約束是否生效
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import yaml
import numpy as np
from pathlib import Path
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # 載入配置
    config_path = Path("configs/channel_flow_re1000_K80_wall_balanced.yml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 修改訓練參數：快速測試
    config['training']['max_epochs'] = 100
    config['training']['validation_freq'] = 25
    config['logging']['log_freq'] = 25
    
    # 強制使用 CPU（避免 CUDA 問題）
    device = torch.device('cpu')
    config['experiment']['device'] = 'cpu'
    
    logging.info("="*60)
    logging.info("🧪 邊界條件修復驗證測試")
    logging.info("="*60)
    logging.info(f"配置檔案: {config_path}")
    logging.info(f"訓練輪數: {config['training']['max_epochs']}")
    logging.info(f"設備: {device}")
    logging.info(f"壁面約束權重: {config['losses']['wall_constraint_weight']}")
    logging.info(f"週期性權重: {config['losses']['periodicity_weight']}")
    logging.info(f"邊界採樣點數: {config['training']['sampling']['boundary_points']}")
    logging.info("="*60)
    
    # 導入訓練函數
    from train import train
    
    # 開始訓練
    try:
        model, trainer_state = train(config)
        
        logging.info("\n" + "="*60)
        logging.info("✅ 訓練完成！準備診斷邊界條件...")
        logging.info("="*60)
        
        # 診斷邊界條件
        diagnose_boundary_conditions(model, config, device)
        
    except Exception as e:
        logging.error(f"❌ 訓練失敗: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def diagnose_boundary_conditions(model, config, device):
    """診斷邊界條件是否滿足"""
    
    # 生成壁面測試點
    n_test = 200
    x_range = config['physics']['domain']['x_range']
    
    # 標準化函數
    def normalize_x(x):
        return 2.0 * (x - x_range[0]) / (x_range[1] - x_range[0]) - 1.0
    
    def normalize_y(y):
        return y  # y 已經在 [-1, 1]
    
    # 下壁面測試點 (y = -1.0)
    x_bottom = np.linspace(x_range[0], x_range[1], n_test)
    y_bottom = np.full(n_test, -1.0)
    coords_bottom = np.stack([normalize_x(x_bottom), normalize_y(y_bottom)], axis=1)
    
    # 上壁面測試點 (y = +1.0)
    x_top = np.linspace(x_range[0], x_range[1], n_test)
    y_top = np.full(n_test, 1.0)
    coords_top = np.stack([normalize_x(x_top), normalize_y(y_top)], axis=1)
    
    # 轉換為 Tensor
    coords_bottom_t = torch.from_numpy(coords_bottom).float().to(device)
    coords_top_t = torch.from_numpy(coords_top).float().to(device)
    
    # 模型預測
    model.eval()
    with torch.no_grad():
        pred_bottom = model(coords_bottom_t).cpu().numpy()
        pred_top = model(coords_top_t).cpu().numpy()
    
    # 提取速度分量
    u_bottom = pred_bottom[:, 0]
    v_bottom = pred_bottom[:, 1]
    u_top = pred_top[:, 0]
    v_top = pred_top[:, 1]
    
    # 計算統計量
    logging.info("\n" + "="*60)
    logging.info("📊 邊界條件診斷結果")
    logging.info("="*60)
    
    logging.info(f"\n🔽 下壁面 (y = -1.0, {n_test} 點):")
    logging.info(f"  U_mean = {u_bottom.mean():+.6f}  (應為 0)")
    logging.info(f"  U_max  = {np.abs(u_bottom).max():+.6f}")
    logging.info(f"  U_std  = {u_bottom.std():.6f}")
    logging.info(f"  V_mean = {v_bottom.mean():+.6f}  (應為 0)")
    logging.info(f"  V_max  = {np.abs(v_bottom).max():+.6f}")
    logging.info(f"  V_std  = {v_bottom.std():.6f}")
    
    logging.info(f"\n🔼 上壁面 (y = +1.0, {n_test} 點):")
    logging.info(f"  U_mean = {u_top.mean():+.6f}  (應為 0)")
    logging.info(f"  U_max  = {np.abs(u_top).max():+.6f}")
    logging.info(f"  U_std  = {u_top.std():.6f}")
    logging.info(f"  V_mean = {v_top.mean():+.6f}  (應為 0)")
    logging.info(f"  V_max  = {np.abs(v_top).max():+.6f}")
    logging.info(f"  V_std  = {v_top.std():.6f}")
    
    # 判斷是否滿足條件
    tolerance = 0.01  # 容差
    
    checks = {
        'bottom_u': np.abs(u_bottom.mean()) < tolerance,
        'bottom_v': np.abs(v_bottom.mean()) < tolerance,
        'top_u': np.abs(u_top.mean()) < tolerance,
        'top_v': np.abs(v_top.mean()) < tolerance,
    }
    
    logging.info("\n" + "="*60)
    logging.info("🎯 驗收結果 (容差 = 0.01):")
    logging.info("="*60)
    
    for key, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logging.info(f"  {key:15s}: {status}")
    
    total_passed = sum(checks.values())
    total_checks = len(checks)
    
    logging.info(f"\n總計: {total_passed}/{total_checks} 通過")
    
    if total_passed == total_checks:
        logging.info("\n🎉 所有邊界條件檢查通過！")
    else:
        logging.warning(f"\n⚠️ 有 {total_checks - total_passed} 項檢查未通過，需要進一步調整")
    
    logging.info("="*60)


if __name__ == '__main__':
    sys.exit(main())
