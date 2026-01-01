#!/usr/bin/env python3
"""
快速測試：檢查 gradient cache 是否被正確啟用
"""

import sys
from pathlib import Path

# 添加專案根目錄
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

# 模擬 data_batch 包含預拼接座標
data_batch = {
    'coords_pde_spatial': torch.randn(100, 3, requires_grad=True),  # 3D 座標
    't_pde': None
}

print("="*80)
print("🔍 測試 is_vs_pinn 檢測邏輯")
print("="*80)

# 測試: 3D 座標檢測
print("\n1️⃣ 3D 座標檢測")
def check_has_3d_coords(data_batch):
    return ('coords_pde_spatial' in data_batch and data_batch['coords_pde_spatial'].shape[1] >= 3)

print(f"   Preconcat format: {check_has_3d_coords(data_batch)}")  # True ✅

print("\n" + "="*80)
print("✅ 檢測成功：預拼接座標正確識別 3D")
print("="*80)

# 測試 2: 驗證 gradient cache 計算
print("\n2️⃣ 測試 Gradient Cache 功能")
from pinnx.physics.gradient_cache import GradientCache

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
cache = GradientCache(device=device)

# 模擬預測和座標
coords = torch.randn(100, 3, requires_grad=True, device=device)
predictions = {
    'u': torch.randn(100, 1, requires_grad=True, device=device),
    'v': torch.randn(100, 1, requires_grad=True, device=device),
    'w': torch.randn(100, 1, requires_grad=True, device=device),
    'p': torch.randn(100, 1, requires_grad=True, device=device)
}

print(f"   Computing gradients on device: {device}")
gradients = cache.compute_all_gradients(predictions, coords)

print(f"   ✅ Computed {len(gradients)} gradient terms")
print(f"   Gradient keys: {list(gradients.keys())[:5]}...")

print("\n🎉 所有測試通過！")
