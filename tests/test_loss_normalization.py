"""
測試損失歸一化功能

驗證：
1. VSPINNChannelFlow 能正確初始化歸一化參數
2. normalize_loss_dict 在 warmup 階段收集統計
3. normalize_loss_dict 在訓練階段正確歸一化
4. 歸一化不改變損失的相對大小關係
"""

import torch
import sys
sys.path.insert(0, '/Users/latteine/Documents/coding/pinns-mvp')

from pinnx.physics.vs_pinn_channel_flow import VSPINNChannelFlow

def test_normalization_warmup():
    """測試 Warmup 階段"""
    print("\n" + "=" * 80)
    print("測試 1: Warmup 階段（收集統計）")
    print("=" * 80)
    
    physics = VSPINNChannelFlow()
    
    # 模擬 K=500 的損失字典（嚴重失衡）
    loss_dict = {
        'data': torch.tensor(1.51),
        'momentum_y': torch.tensor(31841.96),
        'momentum_x': torch.tensor(479.14),
        'wall_constraint': torch.tensor(102.68),
        'prior': torch.tensor(99.99),
    }
    
    # Warmup epoch 0
    print("\n--- Epoch 0 (Warmup 開始) ---")
    normalized = physics.normalize_loss_dict(loss_dict, epoch=0)
    
    # Warmup 期間應該不進行歸一化
    assert all(
        abs(normalized[k].item() - loss_dict[k].item()) < 1e-6
        for k in loss_dict
    ), "Warmup 期間不應歸一化"
    
    print("✅ Warmup 期間未進行歸一化（預期行為）")
    print(f"Normalizers: {physics.loss_normalizers}")
    
    # 繼續 warmup
    for epoch in range(1, 5):
        physics.normalize_loss_dict(loss_dict, epoch=epoch)
    
    print(f"\n--- Epoch 4 (Warmup 結束) ---")
    print(f"最終 Normalizers: {physics.loss_normalizers}")
    
    return physics


def test_normalization_training(physics):
    """測試訓練階段歸一化"""
    print("\n" + "=" * 80)
    print("測試 2: 訓練階段（歸一化生效）")
    print("=" * 80)
    
    # 模擬損失字典（與 warmup 相同的值）
    loss_dict = {
        'data': torch.tensor(1.51),
        'momentum_y': torch.tensor(31841.96),
        'momentum_x': torch.tensor(479.14),
        'wall_constraint': torch.tensor(102.68),
        'prior': torch.tensor(99.99),
    }
    
    print("\n原始損失:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.2f}")
    
    # Epoch 5 開始歸一化
    normalized = physics.normalize_loss_dict(loss_dict, epoch=5)
    
    print("\n歸一化後損失:")
    for k, v in normalized.items():
        print(f"  {k}: {v.item():.6f}")
    
    # 驗證所有損失都被歸一化到 ~1.0
    for k, v in normalized.items():
        assert 0.99 <= v.item() <= 1.01, f"{k} 未被正確歸一化: {v.item()}"
    
    print("\n✅ 所有損失都被歸一化到 1.0 數量級")
    
    return normalized


def test_normalization_preserves_relative_order(physics):
    """測試歸一化不改變相對大小關係"""
    print("\n" + "=" * 80)
    print("測試 3: 相對大小關係保持不變")
    print("=" * 80)
    
    # 創建兩個不同的損失字典
    loss_dict_1 = {
        'data': torch.tensor(1.51),
        'momentum_y': torch.tensor(31841.96),
    }
    
    loss_dict_2 = {
        'data': torch.tensor(3.02),  # 2 倍
        'momentum_y': torch.tensor(15920.98),  # 0.5 倍
    }
    
    # 歸一化
    norm_1 = physics.normalize_loss_dict(loss_dict_1, epoch=5)
    norm_2 = physics.normalize_loss_dict(loss_dict_2, epoch=5)
    
    # 檢查相對比例
    ratio_original_data = loss_dict_2['data'].item() / loss_dict_1['data'].item()
    ratio_normalized_data = norm_2['data'].item() / norm_1['data'].item()
    
    ratio_original_momentum = loss_dict_2['momentum_y'].item() / loss_dict_1['momentum_y'].item()
    ratio_normalized_momentum = norm_2['momentum_y'].item() / norm_1['momentum_y'].item()
    
    print(f"\nData 損失比例:")
    print(f"  原始: {ratio_original_data:.4f}")
    print(f"  歸一化後: {ratio_normalized_data:.4f}")
    
    print(f"\nMomentum 損失比例:")
    print(f"  原始: {ratio_original_momentum:.4f}")
    print(f"  歸一化後: {ratio_normalized_momentum:.4f}")
    
    # 驗證比例基本保持（允許小誤差）
    assert abs(ratio_original_data - ratio_normalized_data) < 0.01
    assert abs(ratio_original_momentum - ratio_normalized_momentum) < 0.01
    
    print("\n✅ 歸一化保持了損失的相對大小關係")


def test_weighted_loss_balance():
    """測試歸一化後權重的平衡性"""
    print("\n" + "=" * 80)
    print("測試 4: 權重平衡性（歸一化前 vs 後）")
    print("=" * 80)
    
    physics = VSPINNChannelFlow()
    
    # Warmup
    loss_dict = {
        'data': torch.tensor(1.51),
        'momentum_y': torch.tensor(31841.96),
        'momentum_x': torch.tensor(479.14),
    }
    
    for epoch in range(5):
        physics.normalize_loss_dict(loss_dict, epoch=epoch)
    
    # 配置權重（與 vs_pinn_3d_full_training.yml 一致）
    weights = {
        'data': 10.0,
        'momentum_y': 1.0,
        'momentum_x': 1.0,
    }
    
    # 未歸一化的加權損失
    print("\n未歸一化:")
    total_raw = 0.0
    for k in loss_dict:
        weighted = weights[k] * loss_dict[k].item()
        total_raw += weighted
        print(f"  {k}: {loss_dict[k].item():.2f} × {weights[k]:.1f} = {weighted:.2f}")
    print(f"  總損失: {total_raw:.2f}")
    print(f"  主導項: momentum_y ({31841.96:.2f}, {31841.96/total_raw*100:.1f}%)")
    
    # 歸一化後
    normalized = physics.normalize_loss_dict(loss_dict, epoch=5)
    print("\n歸一化後:")
    total_normalized = 0.0
    for k in normalized:
        weighted = weights[k] * normalized[k].item()
        total_normalized += weighted
        print(f"  {k}: {normalized[k].item():.4f} × {weights[k]:.1f} = {weighted:.4f}")
    print(f"  總損失: {total_normalized:.4f}")
    print(f"  Data 佔比: {(weights['data'] * normalized['data'].item()) / total_normalized * 100:.1f}%")
    
    # 驗證平衡性
    data_contribution = (weights['data'] * normalized['data'].item()) / total_normalized
    assert data_contribution > 0.5, "歸一化後 data 損失應該有顯著貢獻"
    
    print("\n✅ 歸一化後各項損失權重更平衡")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("損失歸一化功能測試套件")
    print("=" * 80)
    
    # 測試 1: Warmup
    physics = test_normalization_warmup()
    
    # 測試 2: 訓練階段歸一化
    test_normalization_training(physics)
    
    # 測試 3: 相對大小關係
    test_normalization_preserves_relative_order(physics)
    
    # 測試 4: 權重平衡性
    test_weighted_loss_balance()
    
    print("\n" + "=" * 80)
    print("✅ 所有測試通過！")
    print("=" * 80)
