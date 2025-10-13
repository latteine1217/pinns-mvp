#!/usr/bin/env python3
"""
Fourier Phase 分佈驗證腳本

功能：
1. 驗證 Fourier 特徵的 phase 分佈是否合理
2. 檢查是否存在雙重標準化問題
3. 輸出統計量：Mean, Std, Min, Max

使用方法：
    python scripts/debug/validate_fourier_phase.py --config configs/vs_pinn_test_quick_FIXED_v3.yml --epochs 50
"""

import argparse
import sys
import torch
import numpy as np
import logging
from pathlib import Path

# 添加專案根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pinnx.models.fourier_mlp import PINNNet


def setup_logging():
    """設置日誌"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def validate_fourier_phase(model: PINNNet, input_coords: torch.Tensor, coord_ranges: dict):
    """
    驗證 Fourier phase 分佈
    
    Args:
        model: PINN 模型
        input_coords: 輸入座標 (N, in_dim)
        coord_ranges: 座標範圍字典 {'x': (min, max), 'y': (min, max), ...}
    """
    if not hasattr(model, 'fourier') or model.fourier is None:
        logging.warning("⚠️ 模型沒有 Fourier 特徵層，跳過驗證")
        return
    
    logging.info("=" * 80)
    logging.info("Fourier Phase 分佈驗證")
    logging.info("=" * 80)
    
    # 獲取 Fourier 投影矩陣
    B = model.fourier.B  # (in_dim, fourier_m)
    fourier_m = B.shape[1]
    fourier_sigma = model.fourier.sigma if hasattr(model.fourier, 'sigma') else 1.0
    
    logging.info(f"\n📊 Fourier 配置:")
    logging.info(f"   - Fourier dim (m): {fourier_m}")
    logging.info(f"   - Fourier sigma (σ): {fourier_sigma}")
    logging.info(f"   - 投影矩陣形狀: {B.shape}")
    
    # 檢查輸入座標範圍
    logging.info(f"\n📏 輸入座標統計:")
    for i, (coord_name, (coord_min, coord_max)) in enumerate(coord_ranges.items()):
        if i < input_coords.shape[1]:
            actual_min = input_coords[:, i].min().item()
            actual_max = input_coords[:, i].max().item()
            logging.info(f"   - {coord_name}: 預期 [{coord_min:.4f}, {coord_max:.4f}], "
                        f"實際 [{actual_min:.4f}, {actual_max:.4f}]")
            
            # 檢查是否匹配（允許小誤差）
            if abs(actual_min - coord_min) > 0.1 or abs(actual_max - coord_max) > 0.1:
                logging.warning(f"     ⚠️ 座標範圍不匹配！可能存在標準化問題")
    
    # 計算 Fourier phase: z = 2π * x @ B
    with torch.no_grad():
        z = 2.0 * np.pi * input_coords @ B  # (N, fourier_m)
        
        logging.info(f"\n🌊 Fourier Phase 統計 (z = 2π x @ B):")
        logging.info(f"   - Mean: {z.mean().item():.4f}")
        logging.info(f"   - Std:  {z.std().item():.4f}")
        logging.info(f"   - Min:  {z.min().item():.4f}")
        logging.info(f"   - Max:  {z.max().item():.4f}")
        
        # 檢查分佈是否合理（預期應該覆蓋較大範圍）
        z_range = z.max().item() - z.min().item()
        logging.info(f"   - Range: {z_range:.4f}")
        
        if z_range < 2 * np.pi:
            logging.warning(f"   ⚠️ Phase 範圍過小 ({z_range:.4f} < 2π)，可能失去高頻覆蓋")
        elif z_range < 10 * np.pi:
            logging.warning(f"   ⚠️ Phase 範圍偏小 ({z_range:.4f} < 10π)，高頻覆蓋可能不足")
        else:
            logging.info(f"   ✅ Phase 範圍正常 ({z_range:.4f} ≥ 10π)")
        
        # 檢查 sin/cos 激活後的分佈
        z_sin = torch.sin(z)
        z_cos = torch.cos(z)
        
        logging.info(f"\n🔄 Fourier 激活後統計:")
        logging.info(f"   sin(z):")
        logging.info(f"     - Mean: {z_sin.mean().item():.4f}")
        logging.info(f"     - Std:  {z_sin.std().item():.4f}")
        logging.info(f"   cos(z):")
        logging.info(f"     - Mean: {z_cos.mean().item():.4f}")
        logging.info(f"     - Std:  {z_cos.std().item():.4f}")
        
        # 理論上 sin/cos 均值應接近 0，標準差應在 0.5-0.7 之間
        if abs(z_sin.mean().item()) > 0.3:
            logging.warning(f"   ⚠️ sin(z) 均值偏離 0 較大，可能存在偏置")
        if z_sin.std().item() < 0.3:
            logging.warning(f"   ⚠️ sin(z) 標準差過小，頻率覆蓋可能不足")


def main():
    parser = argparse.ArgumentParser(description='Fourier Phase 分佈驗證')
    parser.add_argument('--fourier_m', type=int, default=64, help='Fourier 特徵維度')
    parser.add_argument('--fourier_sigma', type=float, default=5.0, help='Fourier 採樣標準差')
    parser.add_argument('--n_samples', type=int, default=10000, help='測試樣本數')
    parser.add_argument('--in_dim', type=int, default=3, help='輸入維度')
    
    args = parser.parse_args()
    setup_logging()
    
    # 創建測試模型
    logging.info("🔧 創建測試模型...")
    model = PINNNet(
        in_dim=args.in_dim,
        out_dim=4,
        width=200,
        depth=8,
        activation='sine',
        use_fourier=True,
        fourier_m=args.fourier_m,
        fourier_sigma=args.fourier_sigma
    )
    
    # 生成測試數據（物理座標）
    logging.info(f"📊 生成 {args.n_samples} 個測試點...")
    
    # JHTDB Channel Flow 域範圍
    x_range = (0.0, 25.13)
    y_range = (-1.0, 1.0)
    z_range = (0.0, 9.42)
    
    x = torch.rand(args.n_samples, 1) * (x_range[1] - x_range[0]) + x_range[0]
    y = torch.rand(args.n_samples, 1) * (y_range[1] - y_range[0]) + y_range[0]
    
    if args.in_dim == 3:
        z = torch.rand(args.n_samples, 1) * (z_range[1] - z_range[0]) + z_range[0]
        coords = torch.cat([x, y, z], dim=1)
        coord_ranges = {'x': x_range, 'y': y_range, 'z': z_range}
    else:
        coords = torch.cat([x, y], dim=1)
        coord_ranges = {'x': x_range, 'y': y_range}
    
    # 驗證 Fourier phase
    validate_fourier_phase(model, coords, coord_ranges)
    
    logging.info("\n" + "=" * 80)
    logging.info("✅ 驗證完成")
    logging.info("=" * 80)


if __name__ == '__main__':
    main()
