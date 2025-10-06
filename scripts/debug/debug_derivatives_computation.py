#!/usr/bin/env python3
"""
深度診斷微分計算問題
分析 compute_derivatives_3d_temporal 函數中的張量形狀和零值問題
"""

import sys
from pathlib import Path
import torch
import torch.autograd as autograd
import logging

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_autograd():
    """測試基礎的 PyTorch 自動微分功能"""
    logger.info("=== 測試基礎自動微分功能 ===")
    
    # 創建測試函數 f = x^2 + y^2
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
    
    coords = torch.stack([x, y], dim=1)  # [3, 2]
    logger.info(f"輸入座標: shape={coords.shape}")
    
    # 簡單函數
    f = torch.sum(coords**2, dim=1, keepdim=True)  # [3, 1]
    logger.info(f"函數值: shape={f.shape}, values={f.squeeze()}")
    
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
    
    if grads[0] is not None:
        grad_result = grads[0]
        logger.info(f"梯度結果: shape={grad_result.shape}")
        logger.info(f"梯度值: {grad_result}")
        logger.info(f"✅ 基礎自動微分正常")
        return True
    else:
        logger.error("❌ 基礎自動微分失敗")
        return False

def test_current_derivative_function():
    """測試當前的 compute_derivatives_3d_temporal 函數"""
    logger.info("=== 測試當前微分函數 ===")
    
    from pinnx.physics.ns_3d_temporal import compute_derivatives_3d_temporal
    
    # 創建4D測試點
    batch_size = 4
    t = torch.tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True).unsqueeze(1)  # [4, 1]
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True).unsqueeze(1)  # [4, 1]
    y = torch.tensor([0.5, 1.0, 1.5, 2.0], requires_grad=True).unsqueeze(1)  # [4, 1]
    z = torch.tensor([0.2, 0.4, 0.6, 0.8], requires_grad=True).unsqueeze(1)  # [4, 1]
    
    coords = torch.cat([t, x, y, z], dim=1)  # [4, 4] = [t, x, y, z]
    logger.info(f"座標: shape={coords.shape}")
    logger.info(f"座標內容:\n{coords}")
    
    # 簡單測試函數 f = x^2 + y^2 + z^2 + t^2
    f = torch.sum(coords**2, dim=1, keepdim=True)  # [4, 1]
    logger.info(f"測試函數: shape={f.shape}")
    logger.info(f"函數值: {f.squeeze()}")
    
    # 檢查是否需要梯度
    logger.info(f"座標需要梯度: {coords.requires_grad}")
    logger.info(f"函數需要梯度: {f.requires_grad}")
    
    # 測試一階偏微分 - 全部分量
    try:
        first_derivs_all = compute_derivatives_3d_temporal(f, coords, order=1, component=None)
        logger.info(f"一階偏微分 (全部): shape={first_derivs_all.shape}")
        logger.info(f"一階偏微分值:\n{first_derivs_all}")
        
        # 檢查是否為零
        if torch.allclose(first_derivs_all, torch.zeros_like(first_derivs_all)):
            logger.warning("⚠️ 一階偏微分為零！")
        else:
            logger.info("✅ 一階偏微分非零")
    except Exception as e:
        logger.error(f"❌ 一階偏微分計算失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 測試一階偏微分 - 單個分量
    try:
        for i in range(4):
            component_name = ['t', 'x', 'y', 'z'][i]
            first_deriv_i = compute_derivatives_3d_temporal(f, coords, order=1, component=i)
            logger.info(f"∂f/∂{component_name}: shape={first_deriv_i.shape}, 值={first_deriv_i.squeeze()}")
    except Exception as e:
        logger.error(f"❌ 單分量偏微分計算失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_simple_derivative_implementation():
    """測試簡化的微分實現"""
    logger.info("=== 測試簡化微分實現 ===")
    
    def simple_grad(f, x, component=None):
        """簡化的梯度計算"""
        grad_outputs = torch.ones_like(f)
        grads = autograd.grad(
            outputs=f.sum() if f.shape[1] > 1 else f,  # 確保標量輸出
            inputs=x,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True
        )
        
        if grads[0] is None:
            return torch.zeros_like(x)
        
        grad_result = grads[0]
        
        if component is not None:
            return grad_result[:, component:component+1]
        return grad_result
    
    # 創建測試數據
    batch_size = 4
    coords = torch.tensor([
        [0.1, 1.0, 0.5, 0.2],
        [0.2, 2.0, 1.0, 0.4],
        [0.3, 3.0, 1.5, 0.6],
        [0.4, 4.0, 2.0, 0.8]
    ], requires_grad=True)  # [4, 4]
    
    # 測試函數 f = x^2 + y^2 (只使用x,y分量)
    f = coords[:, 1:2]**2 + coords[:, 2:3]**2  # [4, 1]
    
    logger.info(f"座標: shape={coords.shape}")
    logger.info(f"函數: shape={f.shape}, 值={f.squeeze()}")
    
    # 測試簡化梯度計算
    try:
        grad_all = simple_grad(f, coords)
        logger.info(f"簡化梯度 (全部): shape={grad_all.shape}")
        logger.info(f"簡化梯度值:\n{grad_all}")
        
        # 分量測試
        for i in range(4):
            grad_i = simple_grad(f, coords, component=i)
            component_name = ['t', 'x', 'y', 'z'][i]
            logger.info(f"∂f/∂{component_name}: shape={grad_i.shape}, 值={grad_i.squeeze()}")
            
        logger.info("✅ 簡化微分實現正常")
        return True
        
    except Exception as e:
        logger.error(f"❌ 簡化微分實現失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_neural_network_gradients():
    """測試神經網路輸出的微分"""
    logger.info("=== 測試神經網路梯度 ===")
    
    import torch.nn as nn
    
    # 創建簡單神經網路
    model = nn.Sequential(
        nn.Linear(4, 32),
        nn.Tanh(),
        nn.Linear(32, 1)
    )
    
    # 測試數據
    coords = torch.tensor([
        [0.1, 1.0, 0.5, 0.2],
        [0.2, 2.0, 1.0, 0.4],
        [0.3, 3.0, 1.5, 0.6],
        [0.4, 4.0, 2.0, 0.8]
    ], requires_grad=True)
    
    logger.info(f"輸入座標: shape={coords.shape}, requires_grad={coords.requires_grad}")
    
    # 模型預測
    output = model(coords)
    logger.info(f"模型輸出: shape={output.shape}, 值={output.squeeze()}")
    
    # 計算梯度
    try:
        grad_outputs = torch.ones_like(output)
        grads = autograd.grad(
            outputs=output,
            inputs=coords,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True
        )
        
        if grads[0] is not None:
            grad_result = grads[0]
            logger.info(f"神經網路梯度: shape={grad_result.shape}")
            logger.info(f"梯度值:\n{grad_result}")
            
            # 檢查是否為零
            grad_norm = torch.norm(grad_result)
            logger.info(f"梯度範數: {grad_norm.item()}")
            
            if grad_norm < 1e-10:
                logger.warning("⚠️ 神經網路梯度接近零")
            else:
                logger.info("✅ 神經網路梯度正常")
                
            return True
        else:
            logger.error("❌ 神經網路梯度為 None")
            return False
            
    except Exception as e:
        logger.error(f"❌ 神經網路梯度計算失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主診斷函數"""
    logger.info("🔍 開始微分計算深度診斷")
    
    results = {}
    
    try:
        # 1. 基礎自動微分測試
        logger.info("\n" + "="*60)
        results['basic_autograd'] = test_basic_autograd()
        
        # 2. 當前微分函數測試
        logger.info("\n" + "="*60)
        results['current_function'] = test_current_derivative_function()
        
        # 3. 簡化微分實現測試
        logger.info("\n" + "="*60)
        results['simplified_implementation'] = test_simple_derivative_implementation()
        
        # 4. 神經網路梯度測試
        logger.info("\n" + "="*60)
        results['neural_network_gradients'] = test_neural_network_gradients()
        
        # 總結
        logger.info("\n" + "="*60)
        logger.info("📋 微分計算診斷總結:")
        
        for test_name, success in results.items():
            status = "✅ 通過" if success else "❌ 失敗"
            logger.info(f"   {test_name}: {status}")
            
        return results
        
    except Exception as e:
        logger.error(f"❌ 診斷失敗: {e}")
        import traceback
        traceback.print_exc()
        return results

if __name__ == "__main__":
    results = main()