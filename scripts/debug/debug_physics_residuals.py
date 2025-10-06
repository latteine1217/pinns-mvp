#!/usr/bin/env python3
"""
物理殘差計算流程深度診斷
專門分析 NSEquations3DTemporal.residual() 中為什麼殘差全為零
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import logging

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_physics_residual_step_by_step():
    """逐步診斷物理殘差計算過程"""
    logger.info("=== 逐步診斷物理殘差計算 ===")
    
    try:
        from pinnx.physics.ns_3d_temporal import NSEquations3DTemporal, compute_derivatives_3d_temporal
        
        # 創建簡單的測試案例
        Re = 1000.0
        viscosity = 1.0 / Re
        physics = NSEquations3DTemporal(viscosity=viscosity)
        
        # 創建測試數據
        batch_size = 4
        
        # 3D座標 [x, y, z]
        coords_3d = torch.tensor([
            [1.0, 0.5, 0.25],
            [2.0, 0.8, 0.30], 
            [1.5, 0.6, 0.35],
            [1.8, 0.7, 0.40]
        ], requires_grad=True)
        
        # 3D速度 [u, v, w]
        velocity_3d = torch.tensor([
            [0.5, 0.1, 0.0],
            [0.6, 0.2, 0.0],
            [0.7, 0.15, 0.0],
            [0.8, 0.25, 0.0]
        ], requires_grad=True)
        
        # 壓力 [p]
        pressure = torch.tensor([
            [0.1],
            [0.2],
            [0.15],
            [0.18]
        ], requires_grad=True)
        
        # 時間 [t]
        time = torch.tensor([
            [0.0],
            [0.0],
            [0.0],
            [0.0]
        ], requires_grad=True)
        
        logger.info(f"測試數據準備完成:")
        logger.info(f"  座標3D: shape={coords_3d.shape}, requires_grad={coords_3d.requires_grad}")
        logger.info(f"  速度3D: shape={velocity_3d.shape}, requires_grad={velocity_3d.requires_grad}")
        logger.info(f"  壓力: shape={pressure.shape}, requires_grad={pressure.requires_grad}")
        logger.info(f"  時間: shape={time.shape}, requires_grad={time.requires_grad}")
        
        # 調用residual方法
        logger.info("📞 調用 physics.residual() 方法...")
        residuals = physics.residual(coords_3d, velocity_3d, pressure, time=time)
        
        logger.info(f"🔍 殘差計算結果:")
        for name, residual in residuals.items():
            if residual is not None:
                logger.info(f"  {name}: shape={residual.shape}")
                logger.info(f"    值: {residual.squeeze()}")
                logger.info(f"    範數: {torch.norm(residual).item():.6f}")
                logger.info(f"    均值: {torch.mean(torch.abs(residual)).item():.6f}")
            else:
                logger.error(f"  {name}: None!")
        
        # 檢查問題根源
        if all(torch.allclose(residual, torch.zeros_like(residual)) for residual in residuals.values() if residual is not None):
            logger.error("❌ 所有殘差都為零！進入深度診斷...")
            
            # 深度診斷：檢查內部計算過程
            logger.info("🔬 深度診斷內部計算過程...")
            
            # 重新組織為4D格式 (residual方法內部執行的操作)
            coords_4d = torch.cat([time, coords_3d], dim=1)  # [t, x, y, z]
            predictions = torch.cat([velocity_3d, pressure], dim=1)  # [u, v, w, p]
            
            logger.info(f"📐 內部4D格式:")
            logger.info(f"  座標4D: shape={coords_4d.shape}")
            logger.info(f"  預測4D: shape={predictions.shape}")
            
            # 手動計算動量殘差
            logger.info("🧮 手動計算動量方程殘差...")
            u, v, w, p = predictions[:, 0:1], predictions[:, 1:2], predictions[:, 2:3], predictions[:, 3:4]
            
            logger.info(f"分解後的變數:")
            logger.info(f"  u: shape={u.shape}, 值={u.squeeze()}")
            logger.info(f"  v: shape={v.shape}, 值={v.squeeze()}")
            logger.info(f"  w: shape={w.shape}, 值={w.squeeze()}")
            logger.info(f"  p: shape={p.shape}, 值={p.squeeze()}")
            
            # 計算時間導數
            logger.info("⏰ 計算時間導數項...")
            u_t = compute_derivatives_3d_temporal(u, coords_4d, order=1, component=0)
            logger.info(f"  ∂u/∂t: shape={u_t.shape}, 值={u_t.squeeze()}")
            
            # 計算空間一階導數
            logger.info("📍 計算空間一階導數...")
            u_x = compute_derivatives_3d_temporal(u, coords_4d, order=1, component=1)
            u_y = compute_derivatives_3d_temporal(u, coords_4d, order=1, component=2)
            logger.info(f"  ∂u/∂x: shape={u_x.shape}, 值={u_x.squeeze()}")
            logger.info(f"  ∂u/∂y: shape={u_y.shape}, 值={u_y.squeeze()}")
            
            # 計算對流項
            logger.info("🌊 計算對流項...")
            conv_u = u * u_x + v * u_y + w * compute_derivatives_3d_temporal(u, coords_4d, order=1, component=3)
            logger.info(f"  對流項: shape={conv_u.shape}, 值={conv_u.squeeze()}")
            
            # 計算壓力項
            logger.info("💨 計算壓力項...")
            p_x = compute_derivatives_3d_temporal(p, coords_4d, order=1, component=1)
            logger.info(f"  ∂p/∂x: shape={p_x.shape}, 值={p_x.squeeze()}")
            
            return False
        else:
            logger.info("✅ 殘差計算正常，存在非零值")
            return True
            
    except Exception as e:
        logger.error(f"❌ 物理殘差計算失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_neural_network_output():
    """使用神經網路輸出測試物理殘差"""
    logger.info("=== 使用神經網路輸出測試 ===")
    
    try:
        from pinnx.physics.ns_3d_temporal import NSEquations3DTemporal
        
        # 創建物理引擎
        Re = 1000.0
        viscosity = 1.0 / Re
        physics = NSEquations3DTemporal(viscosity=viscosity)
        
        # 創建4D模型：(t,x,y,z) -> (u,v,p)
        model = nn.Sequential(
            nn.Linear(4, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3)  # 輸出 u, v, p
        )
        
        # 創建測試點
        batch_size = 4
        t = torch.tensor([[0.0], [0.0], [0.0], [0.0]], requires_grad=True)
        x = torch.tensor([[1.0], [1.5], [2.0], [2.5]], requires_grad=True)
        y = torch.tensor([[0.5], [0.6], [0.7], [0.8]], requires_grad=True)
        z = torch.tensor([[0.25], [0.30], [0.35], [0.40]], requires_grad=True)
        
        # 組合4D輸入 [t, x, y, z]
        inputs_4d = torch.cat([t, x, y, z], dim=1)
        logger.info(f"4D輸入: shape={inputs_4d.shape}")
        
        # 神經網路預測
        outputs = model(inputs_4d)  # [u, v, p]
        logger.info(f"神經網路輸出: shape={outputs.shape}")
        
        # 分解輸出
        u_pred = outputs[:, 0:1]
        v_pred = outputs[:, 1:2]
        p_pred = outputs[:, 2:3]
        
        logger.info(f"分解後的預測:")
        logger.info(f"  u: 範圍=[{u_pred.min():.3f}, {u_pred.max():.3f}]")
        logger.info(f"  v: 範圍=[{v_pred.min():.3f}, {v_pred.max():.3f}]")
        logger.info(f"  p: 範圍=[{p_pred.min():.3f}, {p_pred.max():.3f}]")
        
        # 準備物理計算所需的輸入
        coords_3d = torch.cat([x, y, z], dim=1)  # [x, y, z]
        w_pred = torch.zeros_like(u_pred)  # w=0 假設
        velocity_3d = torch.cat([u_pred, v_pred, w_pred], dim=1)  # [u, v, w]
        
        logger.info("🔬 計算神經網路輸出的物理殘差...")
        
        # 計算物理殘差
        residuals = physics.residual(coords_3d, velocity_3d, p_pred, time=t)
        
        logger.info(f"📊 神經網路輸出的物理殘差:")
        total_residual_norm = 0.0
        for name, residual in residuals.items():
            if residual is not None:
                residual_norm = torch.norm(residual).item()
                total_residual_norm += residual_norm
                logger.info(f"  {name}: norm={residual_norm:.6f}, shape={residual.shape}")
            else:
                logger.error(f"  {name}: None!")
        
        logger.info(f"總殘差範數: {total_residual_norm:.6f}")
        
        if total_residual_norm < 1e-10:
            logger.error("❌ 神經網路輸出的物理殘差也為零！")
            return False
        else:
            logger.info("✅ 神經網路輸出的物理殘差正常")
            return True
            
    except Exception as e:
        logger.error(f"❌ 神經網路測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_manual_residual_calculation():
    """手動實現簡化的殘差計算來對比"""
    logger.info("=== 手動殘差計算對比 ===")
    
    try:
        from pinnx.physics.ns_3d_temporal import compute_derivatives_3d_temporal
        
        # 創建測試數據
        batch_size = 4
        coords_4d = torch.tensor([
            [0.0, 1.0, 0.5, 0.25],
            [0.0, 1.5, 0.6, 0.30],
            [0.0, 2.0, 0.7, 0.35],
            [0.0, 2.5, 0.8, 0.40]
        ], requires_grad=True)  # [t, x, y, z]
        
        # 簡單的速度場：u = x, v = y, w = 0, p = 0.1
        u = coords_4d[:, 1:2]  # u = x
        v = coords_4d[:, 2:3]  # v = y  
        w = torch.zeros_like(u)  # w = 0
        p = torch.full_like(u, 0.1)  # p = 0.1
        
        logger.info(f"手動設定的場:")
        logger.info(f"  u = x: {u.squeeze()}")
        logger.info(f"  v = y: {v.squeeze()}")
        logger.info(f"  w = 0: {w.squeeze()}")
        logger.info(f"  p = 0.1: {p.squeeze()}")
        
        # 計算導數
        logger.info("🧮 手動計算導數...")
        
        # 時間導數 (應該為0，因為場不隨時間變化)
        u_t = compute_derivatives_3d_temporal(u, coords_4d, order=1, component=0)
        v_t = compute_derivatives_3d_temporal(v, coords_4d, order=1, component=0)
        
        # 空間導數
        u_x = compute_derivatives_3d_temporal(u, coords_4d, order=1, component=1)  # ∂u/∂x = 1
        u_y = compute_derivatives_3d_temporal(u, coords_4d, order=1, component=2)  # ∂u/∂y = 0
        v_x = compute_derivatives_3d_temporal(v, coords_4d, order=1, component=1)  # ∂v/∂x = 0
        v_y = compute_derivatives_3d_temporal(v, coords_4d, order=1, component=2)  # ∂v/∂y = 1
        
        logger.info(f"時間導數:")
        logger.info(f"  ∂u/∂t: {u_t.squeeze()}")
        logger.info(f"  ∂v/∂t: {v_t.squeeze()}")
        
        logger.info(f"空間導數:")
        logger.info(f"  ∂u/∂x: {u_x.squeeze()}")
        logger.info(f"  ∂u/∂y: {u_y.squeeze()}")
        logger.info(f"  ∂v/∂x: {v_x.squeeze()}")
        logger.info(f"  ∂v/∂y: {v_y.squeeze()}")
        
        # 計算對流項
        conv_u = u * u_x + v * u_y  # u*∂u/∂x + v*∂u/∂y
        conv_v = u * v_x + v * v_y  # u*∂v/∂x + v*∂v/∂y
        
        logger.info(f"對流項:")
        logger.info(f"  u*∂u/∂x + v*∂u/∂y: {conv_u.squeeze()}")
        logger.info(f"  u*∂v/∂x + v*∂v/∂y: {conv_v.squeeze()}")
        
        # 壓力項
        p_x = compute_derivatives_3d_temporal(p, coords_4d, order=1, component=1)  # ∂p/∂x = 0 (常數壓力)
        p_y = compute_derivatives_3d_temporal(p, coords_4d, order=1, component=2)  # ∂p/∂y = 0
        
        logger.info(f"壓力導數:")
        logger.info(f"  ∂p/∂x: {p_x.squeeze()}")
        logger.info(f"  ∂p/∂y: {p_y.squeeze()}")
        
        # 簡化的動量方程殘差 (忽略黏性項)
        # ∂u/∂t + u∂u/∂x + v∂u/∂y = -∂p/∂x
        residual_u_simple = u_t + conv_u + p_x
        # ∂v/∂t + u∂v/∂x + v∂v/∂y = -∂p/∂y  
        residual_v_simple = v_t + conv_v + p_y
        
        logger.info(f"簡化動量殘差:")
        logger.info(f"  u方向: {residual_u_simple.squeeze()}")
        logger.info(f"  v方向: {residual_v_simple.squeeze()}")
        
        # 連續方程: ∂u/∂x + ∂v/∂y + ∂w/∂z = 0
        w_z = compute_derivatives_3d_temporal(w, coords_4d, order=1, component=3)
        continuity = u_x + v_y + w_z
        
        logger.info(f"連續方程殘差: {continuity.squeeze()}")
        
        # 檢查是否有非零殘差
        total_norm = torch.norm(residual_u_simple) + torch.norm(residual_v_simple) + torch.norm(continuity)
        logger.info(f"總殘差範數: {total_norm.item():.6f}")
        
        if total_norm > 1e-6:
            logger.info("✅ 手動計算產生了非零殘差")
            return True
        else:
            logger.warning("⚠️ 手動計算的殘差也接近零")
            return False
            
    except Exception as e:
        logger.error(f"❌ 手動殘差計算失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主診斷函數"""
    logger.info("🔍 開始物理殘差計算深度診斷")
    
    results = {}
    
    try:
        # 1. 逐步診斷
        logger.info("\n" + "="*60)
        results['step_by_step'] = test_physics_residual_step_by_step()
        
        # 2. 神經網路輸出測試
        logger.info("\n" + "="*60)
        results['neural_network'] = test_with_neural_network_output()
        
        # 3. 手動計算對比
        logger.info("\n" + "="*60)
        results['manual_calculation'] = test_manual_residual_calculation()
        
        # 總結
        logger.info("\n" + "="*60)
        logger.info("📋 物理殘差診斷總結:")
        
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