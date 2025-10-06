#!/usr/bin/env python3
"""
直接測試 compute_derivatives_3d_temporal 函數
=======================================
繞過導入問題，直接在此腳本中定義和測試函數
"""

import torch
import torch.autograd as autograd
import logging
from typing import Optional

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_derivatives_3d_temporal(f: torch.Tensor, x: torch.Tensor, 
                                  order: int = 1, component: Optional[int] = None) -> torch.Tensor:
    """
    計算3D時間依賴函數的偏微分
    
    Args:
        f: 待微分的標量場 [batch_size, 1]
        x: 座標變數 [batch_size, 4] = [t, x, y, z] 
        order: 微分階數 (1 或 2)
        component: 指定微分變數 (0=t, 1=x, 2=y, 3=z)，None表示全部
        
    Returns:
        偏微分結果 [batch_size, 4] (一階) 或指定分量 [batch_size, 1]
    """
    # 確保計算圖連接
    if not f.requires_grad:
        f.requires_grad_(True)
    if not x.requires_grad:
        x.requires_grad_(True)
        
    # 計算一階偏微分
    grad_outputs = torch.ones_like(f)
    
    logger.info(f"準備計算梯度:")
    logger.info(f"  f.shape: {f.shape}, f.requires_grad: {f.requires_grad}")
    logger.info(f"  x.shape: {x.shape}, x.requires_grad: {x.requires_grad}")
    logger.info(f"  grad_outputs.shape: {grad_outputs.shape}")
    
    try:
        grads = autograd.grad(
            outputs=f, 
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True
        )
        
        logger.info(f"autograd.grad() 返回結果:")
        logger.info(f"  grads類型: {type(grads)}")
        logger.info(f"  grads長度: {len(grads)}")
        logger.info(f"  grads[0]: {grads[0]}")
        logger.info(f"  grads[0] is None: {grads[0] is None}")
        
    except Exception as e:
        logger.error(f"autograd.grad() 拋出異常: {e}")
        return torch.zeros_like(x)
    
    first_derivs = grads[0]
    if first_derivs is None:
        logger.error("❌ first_derivs 為 None，返回零張量")
        return torch.zeros_like(x)
    
    logger.info(f"✅ 一階導數計算成功: {first_derivs}")
    
    if order == 1:
        if component is not None:
            return first_derivs[:, component:component+1]
        return first_derivs
    
    # 如果需要二階導數...
    elif order == 2:
        # 這裡省略二階導數的代碼以專注於一階導數問題
        raise NotImplementedError("二階導數暫時跳過")
    
    else:
        raise ValueError(f"不支援的微分階數: {order}")

def test_simple_function():
    """測試簡單函數的微分"""
    logger.info("🔧 測試簡單函數 f = x*y 的微分...")
    
    # 4D座標 [t, x, y, z]
    coords = torch.tensor([
        [0.0, 1.0, 0.5, 0.2],
        [0.1, 1.5, 0.6, 0.3]
    ], requires_grad=True)
    
    # 簡單函數 f = x * y
    f = coords[:, 1:2] * coords[:, 2:3]
    
    logger.info(f"輸入座標: {coords}")
    logger.info(f"函數值: {f}")
    
    # 調用微分函數
    derivs = compute_derivatives_3d_temporal(f, coords, order=1)
    
    logger.info(f"微分結果: {derivs}")
    logger.info(f"微分形狀: {derivs.shape}")
    logger.info(f"是否全零: {torch.allclose(derivs, torch.zeros_like(derivs))}")
    
    # 手動驗證
    logger.info("🔍 手動驗證:")
    expected_df_dt = torch.zeros_like(f)  # f不依賴於t
    expected_df_dx = coords[:, 2:3]       # ∂(x*y)/∂x = y
    expected_df_dy = coords[:, 1:2]       # ∂(x*y)/∂y = x
    expected_df_dz = torch.zeros_like(f)  # f不依賴於z
    
    logger.info(f"  預期 ∂f/∂t = {expected_df_dt.squeeze()}")
    logger.info(f"  預期 ∂f/∂x = {expected_df_dx.squeeze()}")
    logger.info(f"  預期 ∂f/∂y = {expected_df_dy.squeeze()}")
    logger.info(f"  預期 ∂f/∂z = {expected_df_dz.squeeze()}")
    
    if derivs is not None and derivs.shape[1] == 4:
        logger.info(f"  實際 ∂f/∂t = {derivs[:, 0]}")
        logger.info(f"  實際 ∂f/∂x = {derivs[:, 1]}")
        logger.info(f"  實際 ∂f/∂y = {derivs[:, 2]}")
        logger.info(f"  實際 ∂f/∂z = {derivs[:, 3]}")
        
        # 檢查是否匹配
        matches = [
            torch.allclose(derivs[:, 0], expected_df_dt.squeeze()),
            torch.allclose(derivs[:, 1], expected_df_dx.squeeze()),
            torch.allclose(derivs[:, 2], expected_df_dy.squeeze()),
            torch.allclose(derivs[:, 3], expected_df_dz.squeeze())
        ]
        logger.info(f"  匹配結果: [dt={matches[0]}, dx={matches[1]}, dy={matches[2]}, dz={matches[3]}]")
        
        return all(matches)
    else:
        logger.error("❌ 微分結果格式錯誤")
        return False

def test_neural_network_output():
    """測試神經網路輸出的微分"""
    logger.info("🔧 測試神經網路輸出的微分...")
    
    import torch.nn as nn
    
    # 創建簡單神經網路
    net = nn.Sequential(
        nn.Linear(4, 32),
        nn.Tanh(),
        nn.Linear(32, 32),
        nn.Tanh(),
        nn.Linear(32, 3)  # u, v, p (忽略w)
    )
    
    # 4D座標
    coords = torch.tensor([
        [0.0, 1.0, 0.5, 0.2],
        [0.1, 1.5, 0.6, 0.3]
    ], requires_grad=True)
    
    # 神經網路輸出
    output = net(coords)
    u = output[:, 0:1]  # 選擇u分量
    
    logger.info(f"座標: {coords}")
    logger.info(f"神經網路輸出: {output}")
    logger.info(f"u分量: {u}")
    
    # 調用微分函數
    derivs = compute_derivatives_3d_temporal(u, coords, order=1)
    
    logger.info(f"神經網路微分結果: {derivs}")
    logger.info(f"微分形狀: {derivs.shape}")
    logger.info(f"是否全零: {torch.allclose(derivs, torch.zeros_like(derivs))}")
    
    return not torch.allclose(derivs, torch.zeros_like(derivs))

def main():
    """主函數"""
    logger.info("🔍 開始微分函數直接測試")
    logger.info("=" * 60)
    
    results = {}
    
    # 1. 測試簡單函數
    results['simple_function'] = test_simple_function()
    logger.info("")
    
    # 2. 測試神經網路輸出
    results['neural_network'] = test_neural_network_output()
    logger.info("")
    
    # 總結
    logger.info("=" * 60)
    logger.info("📋 測試總結:")
    for test_name, result in results.items():
        status = "✅ 通過" if result else "❌ 失敗"
        logger.info(f"   {test_name}: {status}")
    
    if all(results.values()):
        logger.info("🎉 所有測試通過，微分函數正常工作")
    else:
        failed_tests = [name for name, result in results.items() if not result]
        logger.error(f"💥 以下測試失敗: {failed_tests}")

if __name__ == "__main__":
    main()