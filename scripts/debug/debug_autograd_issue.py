#!/usr/bin/env python3
"""
診斷 autograd.grad() 返回 None 的問題
=======================================

專門針對 compute_derivatives_3d_temporal 函數中 autograd.grad() 
返回 None 的問題進行詳細診斷。
"""

import torch
import torch.autograd as autograd
import logging
import sys
import os

# 添加專案路徑
sys.path.insert(0, '/Users/latteine/Documents/coding/pinns-mvp')

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_autograd():
    """測試基本的autograd功能"""
    logger.info("🔧 測試基本autograd功能...")
    
    # 創建簡單的計算圖
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    y = x**2
    
    grad_output = torch.ones_like(y)
    grads = autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=grad_output,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True
    )
    
    logger.info(f"   輸入: {x}")
    logger.info(f"   輸出: {y}")
    logger.info(f"   梯度: {grads[0]}")
    logger.info(f"   梯度是否為None: {grads[0] is None}")
    
    return grads[0] is not None

def test_4d_autograd():
    """測試4D輸入的autograd"""
    logger.info("🔧 測試4D座標autograd...")
    
    # 4D座標 [t, x, y, z]
    coords = torch.tensor([
        [0.0, 1.0, 0.5, 0.2],
        [0.1, 1.5, 0.6, 0.3],
        [0.2, 2.0, 0.7, 0.4],
        [0.3, 2.5, 0.8, 0.5]
    ], requires_grad=True)
    
    # 簡單的函數 f = x * y + t
    f = coords[:, 1:2] * coords[:, 2:3] + coords[:, 0:1]
    
    logger.info(f"   座標形狀: {coords.shape}")
    logger.info(f"   函數值: {f.shape} = {f.squeeze()}")
    
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
    
    logger.info(f"   梯度形狀: {grads[0].shape if grads[0] is not None else 'None'}")
    logger.info(f"   梯度值: {grads[0] if grads[0] is not None else 'None'}")
    logger.info(f"   梯度是否為None: {grads[0] is None}")
    
    return grads[0] is not None

def test_original_derivative_function():
    """測試原始的compute_derivatives_3d_temporal函數"""
    logger.info("🔧 測試原始的compute_derivatives_3d_temporal函數...")
    
    from pinnx.physics.ns_3d_temporal import compute_derivatives_3d_temporal
    
    # 4D座標
    coords = torch.tensor([
        [0.0, 1.0, 0.5, 0.2],
        [0.1, 1.5, 0.6, 0.3]
    ], requires_grad=True)
    
    # 簡單函數
    f = coords[:, 1:2] * coords[:, 2:3]  # x * y
    f.requires_grad_(True)
    
    logger.info(f"   座標: {coords}")
    logger.info(f"   函數: {f}")
    
    # 調用原始函數
    try:
        derivs = compute_derivatives_3d_temporal(f, coords, order=1)
        logger.info(f"   導數結果: {derivs}")
        logger.info(f"   導數形狀: {derivs.shape}")
        logger.info(f"   是否全零: {torch.allclose(derivs, torch.zeros_like(derivs))}")
        return not torch.allclose(derivs, torch.zeros_like(derivs))
    except Exception as e:
        logger.error(f"   錯誤: {e}")
        return False

def test_step_by_step_debug():
    """逐步調試原始函數內部邏輯"""
    logger.info("🔧 逐步調試原始函數內部邏輯...")
    
    # 4D座標
    coords = torch.tensor([
        [0.0, 1.0, 0.5, 0.2],
        [0.1, 1.5, 0.6, 0.3]
    ], requires_grad=True)
    
    # 簡單函數 f = x * y
    f = coords[:, 1:2] * coords[:, 2:3]
    logger.info(f"   函數 f = x*y: {f}")
    logger.info(f"   f.requires_grad: {f.requires_grad}")
    logger.info(f"   coords.requires_grad: {coords.requires_grad}")
    
    # 手動執行autograd.grad
    grad_outputs = torch.ones_like(f)
    logger.info(f"   grad_outputs: {grad_outputs}")
    
    try:
        grads = autograd.grad(
            outputs=f, 
            inputs=coords,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True
        )
        
        logger.info(f"   grads 型別: {type(grads)}")
        logger.info(f"   grads 長度: {len(grads)}")
        logger.info(f"   grads[0]: {grads[0]}")
        logger.info(f"   grads[0] is None: {grads[0] is None}")
        
        if grads[0] is not None:
            logger.info(f"   ✅ 梯度計算成功!")
            logger.info(f"   梯度值: {grads[0]}")
            return True
        else:
            logger.error(f"   ❌ 梯度為None!")
            return False
            
    except Exception as e:
        logger.error(f"   錯誤: {e}")
        return False

def test_compute_graph_connectivity():
    """測試計算圖連接性"""
    logger.info("🔧 測試計算圖連接性...")
    
    # 創建神經網路
    import torch.nn as nn
    
    net = nn.Sequential(
        nn.Linear(4, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 3)  # u, v, p (忽略w)
    )
    
    # 4D座標
    coords = torch.tensor([
        [0.0, 1.0, 0.5, 0.2],
        [0.1, 1.5, 0.6, 0.3]
    ], requires_grad=True)
    
    # 神經網路輸出
    output = net(coords)
    u = output[:, 0:1]
    
    logger.info(f"   座標: {coords}")
    logger.info(f"   網路輸出: {output}")
    logger.info(f"   u分量: {u}")
    logger.info(f"   u.requires_grad: {u.requires_grad}")
    logger.info(f"   coords.requires_grad: {coords.requires_grad}")
    
    # 測試梯度計算
    grad_outputs = torch.ones_like(u)
    try:
        grads = autograd.grad(
            outputs=u,
            inputs=coords,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True
        )
        
        logger.info(f"   神經網路梯度: {grads[0]}")
        logger.info(f"   梯度是否為None: {grads[0] is None}")
        
        if grads[0] is not None:
            logger.info(f"   ✅ 神經網路梯度計算成功!")
            return True
        else:
            logger.error(f"   ❌ 神經網路梯度為None!")
            return False
            
    except Exception as e:
        logger.error(f"   錯誤: {e}")
        return False

def main():
    """主函數"""
    logger.info("🔍 開始autograd問題診斷")
    logger.info("=" * 60)
    
    results = {}
    
    # 1. 基本autograd測試
    results['basic_autograd'] = test_basic_autograd()
    logger.info("")
    
    # 2. 4D autograd測試
    results['4d_autograd'] = test_4d_autograd()
    logger.info("")
    
    # 3. 原始函數測試
    results['original_function'] = test_original_derivative_function()
    logger.info("")
    
    # 4. 逐步調試
    results['step_by_step'] = test_step_by_step_debug()
    logger.info("")
    
    # 5. 計算圖連接性測試
    results['compute_graph'] = test_compute_graph_connectivity()
    logger.info("")
    
    # 總結
    logger.info("=" * 60)
    logger.info("📋 診斷總結:")
    for test_name, result in results.items():
        status = "✅ 通過" if result else "❌ 失敗"
        logger.info(f"   {test_name}: {status}")
    
    # 識別問題
    if all(results.values()):
        logger.info("🎉 所有測試通過，autograd功能正常")
    else:
        failed_tests = [name for name, result in results.items() if not result]
        logger.error(f"💥 以下測試失敗: {failed_tests}")
        
        if not results['basic_autograd']:
            logger.error("   基本autograd功能有問題!")
        elif not results['4d_autograd']:
            logger.error("   4D座標autograd有問題!")
        elif not results['original_function']:
            logger.error("   原始compute_derivatives_3d_temporal函數有問題!")
        elif not results['step_by_step']:
            logger.error("   內部邏輯有問題!")
        elif not results['compute_graph']:
            logger.error("   神經網路計算圖有問題!")

if __name__ == "__main__":
    main()