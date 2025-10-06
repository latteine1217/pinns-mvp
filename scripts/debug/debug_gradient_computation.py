#!/usr/bin/env python3
"""
梯度計算診斷腳本
診斷為什麼 compute_derivatives_3d_temporal 返回 None
"""

import torch
import torch.autograd as autograd
import numpy as np
import logging

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_simple_gradient():
    """測試最基本的梯度計算"""
    logger.info("🔍 測試最基本的梯度計算...")
    
    # 創建簡單測試數據
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    f = x[:, 0:1] * x[:, 1:2]  # f = x * y
    
    logger.info(f"  輸入 x: shape={x.shape}, requires_grad={x.requires_grad}")
    logger.info(f"  函數 f: shape={f.shape}, requires_grad={f.requires_grad}")
    
    # 計算梯度
    grad_outputs = torch.ones_like(f)
    grads = autograd.grad(
        outputs=f,
        inputs=x,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True
    )
    
    logger.info(f"  梯度結果: {grads[0]}")
    logger.info(f"  梯度是否為None: {grads[0] is None}")
    
    return grads[0] is not None

def debug_4d_gradient():
    """測試4D座標的梯度計算"""
    logger.info("🔍 測試4D座標梯度計算...")
    
    # 創建4D測試數據 [t, x, y, z]
    coords = torch.tensor([
        [0.0, 1.0, 0.5, 0.0],
        [0.1, 1.5, 0.6, 0.1],
        [0.2, 2.0, 0.7, 0.2],
        [0.3, 2.5, 0.8, 0.3]
    ], requires_grad=True)
    
    # 創建簡單函數 f = x + y (只用空間座標)
    f = coords[:, 1:2] + coords[:, 2:3]  # f = x + y
    
    logger.info(f"  4D座標: shape={coords.shape}, requires_grad={coords.requires_grad}")
    logger.info(f"  函數 f: shape={f.shape}, requires_grad={f.requires_grad}")
    
    # 計算梯度
    grad_outputs = torch.ones_like(f)
    grads = autograd.grad(
        outputs=f,
        inputs=coords,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True
    )
    
    logger.info(f"  梯度結果: {grads[0]}")
    logger.info(f"  梯度是否為None: {grads[0] is None}")
    
    if grads[0] is not None:
        logger.info(f"  ∂f/∂t: {grads[0][:, 0]}")  # 應該是 0
        logger.info(f"  ∂f/∂x: {grads[0][:, 1]}")  # 應該是 1
        logger.info(f"  ∂f/∂y: {grads[0][:, 2]}")  # 應該是 1
        logger.info(f"  ∂f/∂z: {grads[0][:, 3]}")  # 應該是 0
    
    return grads[0] is not None

def debug_neural_network_gradient():
    """測試神經網路輸出的梯度計算"""
    logger.info("🔍 測試神經網路輸出梯度計算...")
    
    # 創建簡單神經網路
    net = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 3)  # 輸出 [u, v, p]
    )
    
    # 4D輸入
    coords = torch.tensor([
        [0.0, 1.0, 0.5, 0.0],
        [0.1, 1.5, 0.6, 0.1],
        [0.2, 2.0, 0.7, 0.2],
        [0.3, 2.5, 0.8, 0.3]
    ], requires_grad=True)
    
    # 神經網路預測
    predictions = net(coords)
    u = predictions[:, 0:1]
    
    logger.info(f"  4D座標: shape={coords.shape}, requires_grad={coords.requires_grad}")
    logger.info(f"  神經網路輸出: shape={predictions.shape}, requires_grad={predictions.requires_grad}")
    logger.info(f"  u分量: shape={u.shape}, requires_grad={u.requires_grad}")
    
    # 計算 u 對座標的梯度
    grad_outputs = torch.ones_like(u)
    grads = autograd.grad(
        outputs=u,
        inputs=coords,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True
    )
    
    logger.info(f"  梯度結果: {grads[0]}")
    logger.info(f"  梯度是否為None: {grads[0] is None}")
    
    if grads[0] is not None:
        logger.info(f"  ∂u/∂t: {grads[0][:, 0]}")
        logger.info(f"  ∂u/∂x: {grads[0][:, 1]}")
        logger.info(f"  ∂u/∂y: {grads[0][:, 2]}")
        logger.info(f"  ∂u/∂z: {grads[0][:, 3]}")
    
    return grads[0] is not None

def test_gradient_computation_issue():
    """模擬物理殘差計算中的梯度問題"""
    logger.info("🔍 模擬物理殘差計算中的梯度問題...")
    
    # 使用與實際程式相同的設定
    coords = torch.tensor([
        [0.0, 1.0, 0.5, 0.0],
        [0.1, 1.5, 0.6, 0.1],
        [0.2, 2.0, 0.7, 0.2],
        [0.3, 2.5, 0.8, 0.3]
    ], requires_grad=True)
    
    # 創建手動設定的速度場
    u = coords[:, 1:2] * 0.5  # u = 0.5 * x
    v = coords[:, 2:3] * 0.1  # v = 0.1 * y
    
    logger.info(f"  座標: shape={coords.shape}, requires_grad={coords.requires_grad}")
    logger.info(f"  u: shape={u.shape}, requires_grad={u.requires_grad}")
    logger.info(f"  v: shape={v.shape}, requires_grad={v.requires_grad}")
    
    # 測試 u 對 x 的導數 (應該是 0.5)
    grad_outputs = torch.ones_like(u)
    u_grads = autograd.grad(
        outputs=u,
        inputs=coords,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True
    )
    
    logger.info(f"  u的梯度: {u_grads[0]}")
    logger.info(f"  u梯度是否為None: {u_grads[0] is None}")
    
    if u_grads[0] is not None:
        logger.info(f"  ∂u/∂x 實際值: {u_grads[0][:, 1]} (預期: 0.5)")
        logger.info(f"  ∂u/∂y 實際值: {u_grads[0][:, 2]} (預期: 0.0)")
    
    return u_grads[0] is not None

def main():
    logger.info("="*60)
    logger.info("=== 梯度計算診斷開始 ===")
    logger.info("="*60)
    
    results = {}
    
    # 測試1: 基本梯度計算
    results['basic'] = debug_simple_gradient()
    logger.info(f"✅ 基本梯度計算: {'通過' if results['basic'] else '失敗'}")
    
    logger.info("")
    
    # 測試2: 4D梯度計算
    results['4d'] = debug_4d_gradient()
    logger.info(f"✅ 4D梯度計算: {'通過' if results['4d'] else '失敗'}")
    
    logger.info("")
    
    # 測試3: 神經網路梯度計算
    results['neural'] = debug_neural_network_gradient()
    logger.info(f"✅ 神經網路梯度計算: {'通過' if results['neural'] else '失敗'}")
    
    logger.info("")
    
    # 測試4: 物理計算模擬
    results['physics'] = test_gradient_computation_issue()
    logger.info(f"✅ 物理計算模擬: {'通過' if results['physics'] else '失敗'}")
    
    logger.info("")
    logger.info("="*60)
    logger.info("=== 梯度計算診斷結果 ===")
    logger.info("="*60)
    
    all_passed = all(results.values())
    logger.info(f"📊 總體狀態: {'✅ 所有測試通過' if all_passed else '❌ 存在問題'}")
    
    for test_name, passed in results.items():
        status = "✅ 通過" if passed else "❌ 失敗"
        logger.info(f"  {test_name}: {status}")
    
    if not all_passed:
        logger.error("❌ 發現梯度計算問題，需要進一步調查")
    else:
        logger.info("✅ 梯度計算基礎功能正常，問題可能在其他地方")

if __name__ == "__main__":
    main()