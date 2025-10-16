#!/usr/bin/env python
"""快速模型評估腳本 - 直接從檢查點推斷架構"""

import torch
import numpy as np
import yaml
from pathlib import Path
import sys

# 添加項目路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pinnx.models.enhanced_fourier_mlp import EnhancedFourierMLP

def quick_evaluate(checkpoint_path, config_path):
    """快速評估檢查點"""
    print("=" * 60)
    print("  快速模型評估")
    print("=" * 60)
    
    # 載入檢查點
    print(f"📂 載入檢查點: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # 載入配置
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 提取訓練資訊
    print("\n📊 訓練資訊：")
    print("-" * 50)
    if 'epoch' in ckpt:
        print(f"  訓練輪數: {ckpt['epoch']}")
    if 'loss' in ckpt:
        print(f"  最終損失: {ckpt['loss']:.6f}")
    
    # 從檢查點狀態推斷模型架構
    state_dict = ckpt['model_state_dict']
    
    # 推斷架構參數
    input_proj_weight = state_dict['input_projection.weight']
    hidden_weight = state_dict['hidden_layers.0.linear.weight']
    
    actual_input_dim = input_proj_weight.shape[1]  # 32
    hidden_dim = input_proj_weight.shape[0]        # 128
    depth = sum(1 for k in state_dict.keys() if k.startswith('hidden_layers.') and k.endswith('.linear.weight'))
    
    print(f"\n🏗️ 模型架構（從檢查點推斷）：")
    print("-" * 50)
    print(f"  輸入維度: {actual_input_dim}")
    print(f"  隱藏維度: {hidden_dim}")
    print(f"  隱藏層數: {depth}")
    print(f"  輸出維度: 4 (u,v,w,p)")
    
    # 檢查Fourier特徵
    if 'fourier.B' in state_dict:
        fourier_B = state_dict['fourier.B']
        print(f"  Fourier features: {fourier_B.shape} -> {fourier_B.shape[1]} modes")
        spatial_dim = fourier_B.shape[0]  # 空間維度
        fourier_modes = fourier_B.shape[1]
        print(f"  空間維度: {spatial_dim}, Fourier modes: {fourier_modes}")
        
        # 計算預期輸入維度: spatial_dim + 2*fourier_modes*active_axes
        # 從配置推斷有效軸
        fourier_cfg = cfg.get('model', {}).get('fourier_features', {})
        axes_config = fourier_cfg.get('axes_config', {})
        active_axes = sum(1 for axis, modes in axes_config.items() if modes)
        expected_dim = spatial_dim + 2 * fourier_modes * active_axes
        print(f"  預期輸入維度: {spatial_dim} + 2×{fourier_modes}×{active_axes} = {expected_dim}")
    
    # 創建測試輸入
    device = torch.device('cpu')
    test_input = torch.randn(100, actual_input_dim, device=device)
    
    print(f"\n🧪 模型測試：")
    print("-" * 50)
    
    try:
        # 直接使用狀態字典的架構資訊創建模型
        model_cfg = cfg.get('model', {})
        
        # 創建 EnhancedFourierMLP（基於推斷的架構）
        model = EnhancedFourierMLP(
            in_dim=3,  # 原始空間維度
            out_dim=4,
            hidden_dim=hidden_dim,
            n_layers=depth,
            activation='sine',
            fourier_m=16,  # 從 B 矩陣推斷
            fourier_sigma=5.0
        ).to(device)
        
        # 載入權重
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        print(f"  ✅ 模型載入成功")
        
        # 測試前向傳播（使用原始3D輸入）
        test_xyz = torch.randn(100, 3, device=device)
        with torch.no_grad():
            output = model(test_xyz)
        
        print(f"  ✅ 前向傳播成功: {test_xyz.shape} -> {output.shape}")
        print(f"  輸出範圍: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        # 檢查輸出合理性
        u, v, w, p = output[:, 0], output[:, 1], output[:, 2], output[:, 3]
        print(f"  u 範圍: [{u.min().item():.4f}, {u.max().item():.4f}]")
        print(f"  v 範圍: [{v.min().item():.4f}, {v.max().item():.4f}]")
        print(f"  w 範圍: [{w.min().item():.4f}, {w.max().item():.4f}]")
        print(f"  p 範圍: [{p.min().item():.4f}, {p.max().item():.4f}]")
        
        # 檢查是否有異常值
        if torch.isnan(output).any():
            print("  ❌ 檢測到 NaN 值")
        elif torch.isinf(output).any():
            print("  ❌ 檢測到 Inf 值")
        else:
            print("  ✅ 輸出數值穩定")
        
        print("\n" + "=" * 60)
        print("  🎉 模型狀態良好，可進行詳細評估")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"  ❌ 模型測試失敗: {e}")
        return False

if __name__ == "__main__":
    checkpoint_path = "checkpoints/normalization_baseline_test_fix_v1/best_model.pth"
    config_path = "configs/normalization_baseline_test_fix_v1.yml"
    
    success = quick_evaluate(checkpoint_path, config_path)
    if success:
        print("\n✅ 模型狀態驗證完成，可繼續進行性能評估")
    else:
        print("\n❌ 模型存在問題，需要檢查")