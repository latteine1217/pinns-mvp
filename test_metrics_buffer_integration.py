"""
測試 MetricsBuffer 整合到 TrainingLoopManager
"""

import torch
import time
from pinnx.train.training_loop_manager import TrainingLoopManager


def test_metrics_buffer_integration():
    """測試 MetricsBuffer 在 TrainingLoopManager 中的整合"""
    print("=== MetricsBuffer 整合測試 ===\n")

    # 創建配置
    config = {
        'logging': {
            'log_interval': 10
        }
    }

    # 測試 1: 使用 MetricsBuffer（優化版）
    print("1️⃣  測試優化版（MetricsBuffer）")
    manager_optimized = TrainingLoopManager(
        config=config,
        wandb_run=None,
        use_metrics_buffer=True
    )

    start = time.perf_counter()
    for epoch in range(100):
        loss_dict = {
            'total_loss': torch.rand(1),
            'val_loss': torch.rand(1)
        }
        manager_optimized.update_history(loss_dict, epoch)
    time_optimized = time.perf_counter() - start

    print(f"   完成 100 次記錄")
    print(f"   耗時: {time_optimized*1000:.3f} ms")
    print(f"   歷史長度: total_loss={len(manager_optimized.history['total_loss'])}")

    # 測試 2: 不使用 MetricsBuffer（傳統版）
    print("\n2️⃣  測試傳統版（無 MetricsBuffer）")
    manager_traditional = TrainingLoopManager(
        config=config,
        wandb_run=None,
        use_metrics_buffer=False
    )

    start = time.perf_counter()
    for epoch in range(100):
        loss_dict = {
            'total_loss': torch.rand(1),
            'val_loss': torch.rand(1)
        }
        manager_traditional.update_history(loss_dict, epoch)
    time_traditional = time.perf_counter() - start

    print(f"   完成 100 次記錄")
    print(f"   耗時: {time_traditional*1000:.3f} ms")
    print(f"   歷史長度: total_loss={len(manager_traditional.history['total_loss'])}")

    # 比較
    print("\n3️⃣  效能比較")
    speedup = time_traditional / time_optimized
    print(f"   優化版: {time_optimized*1000:.3f} ms")
    print(f"   傳統版: {time_traditional*1000:.3f} ms")
    print(f"   加速比: {speedup:.2f}x")

    if speedup > 1.0:
        print(f"   ✅ 優化成功！提升 {(speedup-1)*100:.1f}%")
    else:
        print(f"   ⚠️  未達預期（可能是測試規模太小）")

    # 測試 4: 驗證數據一致性
    print("\n4️⃣  驗證數據一致性")

    # 確保兩種方法記錄的數據點數相同
    assert len(manager_optimized.history['total_loss']) == len(manager_traditional.history['total_loss'])
    assert len(manager_optimized.history['epoch']) == len(manager_traditional.history['epoch'])

    print(f"   ✅ 數據點數一致")
    print(f"   ✅ 所有測試通過！")

if __name__ == "__main__":
    test_metrics_buffer_integration()
