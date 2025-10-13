#!/usr/bin/env python3
"""
NS 連續性方程驗證腳本
Validate Navier-Stokes Continuity Equation

診斷並修復質量守恆計算問題
"""

import sys
from pathlib import Path
import torch
import numpy as np
import logging

sys.path.append(str(Path(__file__).parent.parent.parent))

from pinnx.losses.residuals import divergence, compute_gradients

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_conservation_2d(u: torch.Tensor, v: torch.Tensor, coords: torch.Tensor) -> dict:
    """
    2D 質量守恆驗證 (∂u/∂x + ∂v/∂y = 0)
    
    Args:
        u: x方向速度 [N]
        v: y方向速度 [N]
        coords: 座標 [N, 2] 格式為 (x, y)
    
    Returns:
        結果字典包含誤差與診斷信息
    """
    logger.info(f"Input shapes: u={u.shape}, v={v.shape}, coords={coords.shape}")
    
    # 確保 requires_grad
    if not coords.requires_grad:
        coords = coords.requires_grad_(True)
    if not u.requires_grad:
        u = u.requires_grad_(True)
    if not v.requires_grad:
        v = v.requires_grad_(True)
    
    results = {}
    
    try:
        # 方法 1: 使用 divergence 函數
        velocity = torch.stack([u, v], dim=-1)  # [N, 2]
        div = divergence(velocity, coords)
        
        results['method'] = 'divergence_function'
        results['divergence_values'] = div.detach().cpu().numpy()
        results['mean_abs_div'] = torch.mean(torch.abs(div)).item()
        results['rms_div'] = torch.sqrt(torch.mean(div**2)).item()
        results['max_abs_div'] = torch.max(torch.abs(div)).item()
        
        logger.info(f"✅ Method 1 (divergence function): RMS={results['rms_div']:.2e}")
        
    except Exception as e:
        logger.error(f"❌ Method 1 failed: {e}")
        results['method'] = 'failed'
        results['error'] = str(e)
    
    try:
        # 方法 2: 手動計算梯度（正確索引）
        u_grad = compute_gradients(u, coords, order=1)  # [N, 2]
        v_grad = compute_gradients(v, coords, order=1)  # [N, 2]
        
        # 連續性方程：∂u/∂x + ∂v/∂y
        # coords = [x, y] → u_grad[:, 0] = ∂u/∂x, u_grad[:, 1] = ∂u/∂y
        dudx = u_grad[:, 0]
        dvdy = v_grad[:, 1]
        div_manual = dudx + dvdy
        
        results['method_2'] = 'manual_gradient'
        results['div_manual_rms'] = torch.sqrt(torch.mean(div_manual**2)).item()
        results['div_manual_mean'] = div_manual.mean().item()
        
        logger.info(f"✅ Method 2 (manual gradient): RMS={results['div_manual_rms']:.2e}")
        
        # 檢查兩種方法是否一致
        if 'rms_div' in results:
            diff = abs(results['rms_div'] - results['div_manual_rms'])
            logger.info(f"Method 1 vs Method 2 difference: {diff:.2e}")
        
    except Exception as e:
        logger.error(f"❌ Method 2 failed: {e}")
        results['method_2_error'] = str(e)
    
    return results


def validate_conservation_3d(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, 
                            coords: torch.Tensor) -> dict:
    """
    3D 質量守恆驗證 (∂u/∂x + ∂v/∂y + ∂w/∂z = 0)
    
    Args:
        u, v, w: 三個方向速度 [N]
        coords: 座標 [N, 3] 格式為 (x, y, z) 或 (x, y, t)
    """
    logger.info(f"Input shapes: u={u.shape}, v={v.shape}, w={w.shape}, coords={coords.shape}")
    
    if not coords.requires_grad:
        coords = coords.requires_grad_(True)
    
    results = {}
    
    try:
        velocity = torch.stack([u, v, w], dim=-1)  # [N, 3]
        div = divergence(velocity, coords)
        
        results['method'] = 'divergence_function_3d'
        results['rms_div'] = torch.sqrt(torch.mean(div**2)).item()
        results['mean_div'] = div.mean().item()
        
        logger.info(f"✅ 3D divergence: RMS={results['rms_div']:.2e}")
        
    except Exception as e:
        logger.error(f"❌ 3D divergence failed: {e}")
        results['error'] = str(e)
    
    return results


def test_with_analytical_field():
    """使用解析解測試"""
    logger.info("\n" + "="*60)
    logger.info("測試 1: 解析解驗證 (無散場)")
    logger.info("="*60)
    
    # 創建無散速度場：u = sin(x)cos(y), v = -cos(x)sin(y)
    # ∂u/∂x + ∂v/∂y = cos(x)cos(y) - cos(x)cos(y) = 0 ✅
    
    x = torch.linspace(0, 2*np.pi, 50)
    y = torch.linspace(0, 2*np.pi, 50)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    coords = torch.stack([X.flatten(), Y.flatten()], dim=-1).requires_grad_(True)
    
    u = torch.sin(coords[:, 0]) * torch.cos(coords[:, 1])
    v = -torch.cos(coords[:, 0]) * torch.sin(coords[:, 1])
    
    u.requires_grad_(True)
    v.requires_grad_(True)
    
    results = validate_conservation_2d(u, v, coords)
    
    # 理論上應該 ≈ 0
    logger.info(f"\n理論值: div = 0")
    logger.info(f"計算值: RMS div = {results.get('rms_div', 'N/A'):.2e}")
    
    if 'rms_div' in results and results['rms_div'] < 1e-6:
        logger.info("✅ 解析解測試通過")
    else:
        logger.warning("⚠️ 解析解測試未達預期精度")
    
    return results


def test_with_model_predictions(checkpoint_path: str):
    """使用模型預測測試"""
    logger.info("\n" + "="*60)
    logger.info("測試 2: 模型預測驗證")
    logger.info("="*60)
    
    # 載入模型和數據
    from scripts.train import load_checkpoint, create_model, get_device
    
    device = get_device('auto')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 假設有配置
    if 'config' not in checkpoint:
        logger.error("Checkpoint 缺少 config，無法繼續測試")
        return None
    
    config = checkpoint['config']
    model = create_model(config, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 創建測試網格
    x = torch.linspace(0, 1, 32)
    y = torch.linspace(-1, 1, 32)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    coords = torch.stack([X.flatten(), Y.flatten()], dim=-1).to(device).requires_grad_(True)
    
    with torch.no_grad():
        predictions = model(coords)
    
    u = predictions[:, 0].requires_grad_(True)
    v = predictions[:, 1].requires_grad_(True)
    
    results = validate_conservation_2d(u, v, coords)
    
    logger.info(f"\n模型預測質量守恆誤差: {results.get('rms_div', 'N/A'):.2e}")
    
    # 判斷是否達標
    threshold = 1e-3
    if 'rms_div' in results:
        if results['rms_div'] < threshold:
            logger.info(f"✅ 質量守恆達標 (< {threshold})")
        else:
            logger.warning(f"❌ 質量守恆未達標 (> {threshold})")
    
    return results


def diagnose_metrics_bug():
    """診斷 metrics.py 中的 bug 並測試修復後的版本"""
    logger.info("\n" + "="*60)
    logger.info("測試 3: 診斷並測試修復後的 conservation_error")
    logger.info("="*60)
    
    # 導入修復後的函數
    from pinnx.evals.metrics import conservation_error
    
    # 測試 2D 無散場
    logger.info("\n[3.1] 測試 2D 座標 [x, y]")
    x = torch.linspace(0, 2*np.pi, 50)
    y = torch.linspace(0, 2*np.pi, 50)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    coords_2d = torch.stack([X.flatten(), Y.flatten()], dim=-1).requires_grad_(True)
    
    # 無散場：u = sin(x)cos(y), v = -cos(x)sin(y)
    u = torch.sin(coords_2d[:, 0]) * torch.cos(coords_2d[:, 1])
    v = -torch.cos(coords_2d[:, 0]) * torch.sin(coords_2d[:, 1])
    
    try:
        error_2d = conservation_error(u, v, coords_2d)
        logger.info(f"✅ 2D conservation_error = {error_2d:.2e}")
        
        if error_2d < 1e-6:
            logger.info("   ✅ 2D 測試通過 (誤差 < 1e-6)")
        else:
            logger.warning(f"   ⚠️ 2D 誤差偏高: {error_2d:.2e}")
    except Exception as e:
        logger.error(f"   ❌ 2D 測試失敗: {e}")
    
    # 測試 3D 座標格式
    logger.info("\n[3.2] 測試 3D 座標 [t, x, y]")
    t = torch.zeros(coords_2d.shape[0], 1)
    coords_3d = torch.cat([t, coords_2d], dim=1).requires_grad_(True)
    
    try:
        error_3d = conservation_error(u, v, coords_3d)
        logger.info(f"✅ 3D conservation_error = {error_3d:.2e}")
        
        if error_3d < 1e-6:
            logger.info("   ✅ 3D 測試通過 (誤差 < 1e-6)")
        else:
            logger.warning(f"   ⚠️ 3D 誤差偏高: {error_3d:.2e}")
    except Exception as e:
        logger.error(f"   ❌ 3D 測試失敗: {e}")


def main():
    """主程式"""
    logger.info("="*60)
    logger.info("NS 連續性方程驗證工具")
    logger.info("="*60)
    
    # 測試 1: 解析解
    test_with_analytical_field()
    
    # 測試 3: 診斷 bug
    diagnose_metrics_bug()
    
    # 測試 2: 模型預測（如果提供檢查點）
    checkpoint_path = "checkpoints/pinnx_channel_flow_re1000_fix6_k50_phase4b_latest.pth"
    if Path(checkpoint_path).exists():
        test_with_model_predictions(checkpoint_path)
    else:
        logger.warning(f"⚠️ Checkpoint 不存在，跳過模型測試: {checkpoint_path}")
    
    logger.info("\n" + "="*60)
    logger.info("診斷完成")
    logger.info("="*60)


if __name__ == "__main__":
    main()
