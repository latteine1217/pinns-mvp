"""
指標追蹤完整使用範例

展示如何在訓練流程中整合 Timer、MemoryTracker 和 PhysicsValidator
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, Any

# 從 pinnx.utils 導入工具類
from pinnx.utils.timer import Timer
from pinnx.utils.memory_tracker import MemoryTracker
from pinnx.utils.physics_validator import PhysicsValidator


def example_training_integration():
    """
    範例 1：在訓練流程中整合指標追蹤
    """
    print("\n" + "="*70)
    print("範例 1：訓練流程中的指標追蹤")
    print("="*70)

    # ====================================================================
    # 1. 初始化工具
    # ====================================================================
    # Timer：追蹤訓練時間
    timer = Timer(use_wandb=False)  # 示範模式，不記錄到 WandB

    # MemoryTracker：追蹤 GPU 記憶體
    # 假設我們有一個簡單的模型
    model = nn.Sequential(
        nn.Linear(3, 256),
        nn.Tanh(),
        nn.Linear(256, 256),
        nn.Tanh(),
        nn.Linear(256, 4)  # 輸出 u, v, w, p
    )

    if torch.cuda.is_available():
        model = model.cuda()

    memory_tracker = MemoryTracker(model=model, use_wandb=False)

    # PhysicsValidator：驗證物理約束
    config = {
        'physics': {'nu': 0.01, 'rho': 1.0},
        'dimension': 3,
        # 注意：不設定 boundary_type，因為測試資料不是規則網格
    }
    physics_validator = PhysicsValidator(config=config, use_wandb=False)

    # ====================================================================
    # 2. 訓練循環
    # ====================================================================
    timer.start()

    max_epochs = 10
    target_l2_error = 0.05  # 假設的目標精度

    print(f"\n開始訓練（{max_epochs} epochs）...\n")

    for epoch in range(max_epochs):
        # 模擬訓練一個 epoch
        time.sleep(0.1)  # 模擬計算時間

        # 記錄 epoch 時間
        epoch_time = timer.tick(epoch=epoch, log_to_wandb=False)

        # 記錄記憶體使用
        memory_stats = memory_tracker.log_current(epoch=epoch, log_to_wandb=False)

        # 模擬評估
        l2_error = 0.1 * (1.0 - epoch / max_epochs) + np.random.rand() * 0.01

        # 檢查是否達標
        if l2_error < target_l2_error:
            timer.mark_convergence(
                epoch=epoch,
                metric_value=l2_error,
                log_to_wandb=False
            )

        # 每 5 個 epochs 打印進度
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | "
                  f"Time: {epoch_time:.3f}s | "
                  f"Memory: {memory_stats['current_memory_mb']:.1f}MB | "
                  f"L2 Error: {l2_error:.4f}")

    timer.stop()

    # ====================================================================
    # 3. 訓練後評估
    # ====================================================================
    print("\n" + "-"*70)
    print("訓練完成，執行物理驗證...")
    print("-"*70)

    # 生成測試資料（模擬）
    n_test = 1000
    x = torch.randn(n_test, 1, requires_grad=True)
    y = torch.randn(n_test, 1, requires_grad=True)
    z = torch.randn(n_test, 1, requires_grad=True)

    if torch.cuda.is_available():
        x, y, z = x.cuda(), y.cuda(), z.cuda()

    test_data = {
        'x': x,
        'y': y,
        'z': z,
    }

    # 執行物理驗證
    validation_results = physics_validator.validate(
        model=model,
        test_data=test_data,
        log_to_wandb=False
    )

    # ====================================================================
    # 4. 打印摘要
    # ====================================================================
    print("\n")
    timer.print_summary()
    memory_tracker.print_summary()
    physics_validator.print_summary()

    # ====================================================================
    # 5. 返回完整統計
    # ====================================================================
    summary = {
        'timing': timer.get_summary(),
        'memory': memory_tracker.get_summary(),
        'physics': validation_results,
    }

    return summary


def example_physics_validation_detailed():
    """
    範例 2：詳細的物理驗證
    """
    print("\n" + "="*70)
    print("範例 2：詳細物理驗證")
    print("="*70)

    # 建立簡單的 2D 模型
    model = nn.Sequential(
        nn.Linear(2, 128),
        nn.Tanh(),
        nn.Linear(128, 128),
        nn.Tanh(),
        nn.Linear(128, 3)  # 輸出 u, v, p
    )

    # 2D 驗證器
    validator = PhysicsValidator(
        config={'dimension': 2, 'boundary_type': 'periodic'},
        nu=0.01,
        rho=1.0,
        use_wandb=False
    )

    # 生成測試資料
    n = 100
    x = torch.linspace(0, 2*np.pi, n).reshape(-1, 1).requires_grad_(True)
    y = torch.linspace(0, 2*np.pi, n).reshape(-1, 1).requires_grad_(True)

    X, Y = torch.meshgrid(x.squeeze(), y.squeeze(), indexing='ij')
    x_flat = X.reshape(-1, 1).requires_grad_(True)
    y_flat = Y.reshape(-1, 1).requires_grad_(True)

    # 模型預測
    inputs = torch.cat([x_flat, y_flat], dim=-1)
    outputs = model(inputs)

    u_pred = outputs[:, 0:1].requires_grad_(True)
    v_pred = outputs[:, 1:2].requires_grad_(True)
    p_pred = outputs[:, 2:3].requires_grad_(True)

    # 1. 質量守恆檢查
    print("\n1️⃣  質量守恆檢查（散度）")
    div_results = validator.compute_divergence(
        u=u_pred, v=v_pred, w=None,
        x=x_flat, y=y_flat, z=None
    )

    for key, value in div_results.items():
        print(f"   {key}: {value:.6e}")

    # 2. 動量守恆檢查
    print("\n2️⃣  動量守恆檢查（NS 殘差）")
    momentum_results = validator.compute_momentum_residual(
        u=u_pred, v=v_pred, w=None, p=p_pred,
        x=x_flat, y=y_flat, z=None
    )

    for key, value in momentum_results.items():
        print(f"   {key}: {value:.6e}")

    # 3. 邊界條件檢查
    print("\n3️⃣  邊界條件檢查（週期性）")
    u_reshaped = u_pred.reshape(n, n, 1)
    bc_results = validator.check_boundary_conditions(
        field=u_reshaped,
        bc_type='periodic'
    )

    for key, value in bc_results.items():
        print(f"   {key}: {value:.6e}")

    # 4. 綜合物理違反分數
    print("\n4️⃣  綜合物理違反分數")
    physics_score = validator.compute_physics_violation_score(
        div_results=div_results,
        momentum_results=momentum_results,
        bc_results=bc_results
    )

    print(f"   Physics Violation Score: {physics_score:.6e}")
    print(f"   （分數越低越好，目標 < 0.1）")


def example_memory_profiling():
    """
    範例 3：記憶體分析
    """
    print("\n" + "="*70)
    print("範例 3：記憶體分析")
    print("="*70)

    if not torch.cuda.is_available():
        print("⚠️  CUDA 不可用，跳過記憶體分析範例")
        return

    # 建立不同大小的模型
    model_configs = [
        {'depth': 4, 'width': 128, 'name': 'Small'},
        {'depth': 6, 'width': 256, 'name': 'Medium'},
        {'depth': 8, 'width': 512, 'name': 'Large'},
    ]

    for config in model_configs:
        print(f"\n📊 測試模型：{config['name']} (Depth={config['depth']}, Width={config['width']})")

        # 建立模型
        layers = []
        layers.append(nn.Linear(3, config['width']))
        layers.append(nn.Tanh())

        for _ in range(config['depth'] - 2):
            layers.append(nn.Linear(config['width'], config['width']))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(config['width'], 4))

        model = nn.Sequential(*layers).cuda()

        # 記憶體追蹤
        tracker = MemoryTracker(model=model, use_wandb=False)

        # 測試推理記憶體
        input_tensor = torch.randn(10000, 3).cuda()
        inference_stats = tracker.measure_inference_memory(
            model=model,
            input_tensor=input_tensor,
            num_runs=10
        )

        print(f"   推理記憶體: {inference_stats['inference_memory_mb']:.2f} MB")

        # 估算激活值記憶體
        activation_memory = tracker.estimate_activation_memory(
            batch_size=10000,
            input_dim=3,
            output_dim=4,
            depth=config['depth'],
            width=config['width']
        )

        print(f"   估算激活值記憶體: {activation_memory:.2f} MB")

        # 清理
        del model, tracker
        torch.cuda.empty_cache()


def example_efficiency_metrics():
    """
    範例 4：效率指標計算
    """
    print("\n" + "="*70)
    print("範例 4：效率指標計算")
    print("="*70)

    # 模擬兩個模型的訓練結果
    model_a = {
        'name': 'Vanilla MLP',
        'training_time': 3600,      # 秒
        'peak_memory': 2048,         # MB
        'inference_time': 0.5,       # ms
        'l2_error': 0.15,
    }

    model_b = {
        'name': 'Full (Fourier + GradNorm)',
        'training_time': 1800,
        'peak_memory': 2560,
        'inference_time': 0.8,
        'l2_error': 0.08,
    }

    models = [model_a, model_b]

    print("\n📊 效率對比分析")
    print("-"*70)
    print(f"{'模型':<25} | {'訓練時間':<12} | {'記憶體':<12} | {'推理時間':<12} | {'L2誤差':<10}")
    print("-"*70)

    for model in models:
        # 計算效率分數
        accuracy = 1.0 - model['l2_error']
        time_efficiency = accuracy / model['training_time']
        memory_efficiency = accuracy / model['peak_memory']
        inference_efficiency = accuracy / model['inference_time']

        print(f"{model['name']:<25} | "
              f"{model['training_time']:>10.0f}s | "
              f"{model['peak_memory']:>10.0f}MB | "
              f"{model['inference_time']:>10.2f}ms | "
              f"{model['l2_error']:>9.3f}")

        model['time_efficiency'] = time_efficiency
        model['memory_efficiency'] = memory_efficiency
        model['inference_efficiency'] = inference_efficiency

    print("-"*70)
    print(f"{'效率分數':<25} | {'時間效率':<12} | {'記憶體效率':<12} | {'推理效率':<12}")
    print("-"*70)

    for model in models:
        print(f"{model['name']:<25} | "
              f"{model['time_efficiency']:>12.6e} | "
              f"{model['memory_efficiency']:>12.6e} | "
              f"{model['inference_efficiency']:>12.6e}")

    print("-"*70)

    # 計算相對提升
    time_speedup = model_a['training_time'] / model_b['training_time']
    accuracy_improvement = (model_a['l2_error'] - model_b['l2_error']) / model_a['l2_error'] * 100

    print(f"\n⭐ 效率提升總結：")
    print(f"   訓練時間加速：{time_speedup:.2f}x")
    print(f"   精度提升：{accuracy_improvement:.1f}%")
    print(f"   記憶體成本：+{(model_b['peak_memory'] - model_a['peak_memory']) / model_a['peak_memory'] * 100:.1f}%")


if __name__ == '__main__':
    # 執行所有範例
    print("\n" + "🚀 "*35)
    print("指標追蹤工具完整使用範例")
    print("🚀 "*35)

    # 範例 1：訓練流程整合
    summary = example_training_integration()

    # 範例 2：詳細物理驗證
    example_physics_validation_detailed()

    # 範例 3：記憶體分析
    example_memory_profiling()

    # 範例 4：效率指標
    example_efficiency_metrics()

    print("\n" + "="*70)
    print("✅ 所有範例執行完成！")
    print("="*70)
    print("\n💡 提示：")
    print("   - 在實際訓練中，將 use_wandb=True 以記錄到 WandB")
    print("   - 參考 docs/METRICS_TRACKING_GUIDE.md 了解完整指標定義")
    print("   - 使用 configs/sweeps/sweep_e1_efficiency_analysis.yaml 執行效率分析")
    print("   - 使用 configs/sweeps/sweep_p1_physics_validation.yaml 執行物理驗證")
    print()
