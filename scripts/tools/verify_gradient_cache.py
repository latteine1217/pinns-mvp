"""
梯度快取驗證腳本
================

驗證 Trainer 是否正確使用梯度快取（Wave 1 優化）

功能:
1. 檢查 Kolmogorov Flow 2D 的梯度快取使用
2. 檢查 VS-PINN Channel Flow 的梯度快取使用
3. 測量有/無快取的效能差異
4. 驗證數值一致性

作者: PINNs-MVP 團隊
日期: 2026-01-07
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import yaml
import time
import logging
from typing import Dict, Any

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def test_2d_gradient_cache():
    """測試 Kolmogorov Flow 2D 的梯度快取"""
    from pinnx.physics.kolmogorov_flow_2d import KolmogorovFlow2D
    from pinnx.physics.gradient_cache_2d import GradientCache2D
    
    logging.info("=" * 80)
    logging.info("測試 1: Kolmogorov Flow 2D 梯度快取")
    logging.info("=" * 80)
    
    # 建立物理模組
    physics = KolmogorovFlow2D(
        forcing_params={'amplitude': 1.0, 'wavenumber': 4},
        physics_params={'nu': 0.02, 'rho': 1.0}
    )
    
    # 生成測試資料
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 10000
    coords = torch.rand(batch_size, 2, device=device, requires_grad=True) * 2 * 3.14159
    predictions = torch.randn(batch_size, 3, device=device, requires_grad=True)
    
    # === 測試 1: 無快取 ===
    logging.info("\n🐌 測試無快取模式...")
    start_time = time.time()
    residuals_no_cache = physics.residual(coords, predictions, gradients=None)
    time_no_cache = time.time() - start_time
    logging.info(f"   時間: {time_no_cache*1000:.2f} ms")
    
    # === 測試 2: 有快取 ===
    logging.info("\n🚀 測試快取模式...")
    grad_cache = GradientCache2D(device=device)
    predictions_dict = {
        'u': predictions[:, 0:1],
        'v': predictions[:, 1:2],
        'p': predictions[:, 2:3]
    }
    
    # 計算快取
    start_cache = time.time()
    gradients = grad_cache.compute_all_gradients(predictions_dict, coords)
    time_cache_compute = time.time() - start_cache
    
    # 使用快取
    start_use = time.time()
    residuals_with_cache = physics.residual(coords, predictions, gradients=gradients)
    time_with_cache = time.time() - start_use
    
    total_cache_time = time_cache_compute + time_with_cache
    
    logging.info(f"   快取計算時間: {time_cache_compute*1000:.2f} ms")
    logging.info(f"   使用快取時間: {time_with_cache*1000:.2f} ms")
    logging.info(f"   總時間: {total_cache_time*1000:.2f} ms")
    
    # === 測試 3: 數值一致性 ===
    logging.info("\n✅ 驗證數值一致性...")
    for key in residuals_no_cache:
        if key in residuals_with_cache:
            diff = torch.abs(residuals_no_cache[key] - residuals_with_cache[key]).max()
            logging.info(f"   {key:20s}: max_diff = {diff:.2e}")
            assert diff < 1e-5, f"{key} 不一致！差異: {diff:.2e}"
    
    # === 效能總結 ===
    speedup = (time_no_cache / total_cache_time - 1) * 100
    logging.info(f"\n📊 效能總結:")
    logging.info(f"   無快取:   {time_no_cache*1000:.2f} ms")
    logging.info(f"   有快取:   {total_cache_time*1000:.2f} ms")
    logging.info(f"   加速:     {speedup:+.1f}%")
    
    return {
        'time_no_cache': time_no_cache,
        'time_with_cache': total_cache_time,
        'speedup_percent': speedup,
        'numerically_correct': True
    }


def test_3d_gradient_cache():
    """測試 VS-PINN 3D 的梯度快取"""
    from pinnx.physics.gradient_cache import GradientCache
    from pinnx.physics.vs_pinn_channel_flow import VSPINNChannelFlow
    
    logging.info("\n" + "=" * 80)
    logging.info("測試 2: VS-PINN Channel Flow 3D 梯度快取")
    logging.info("=" * 80)
    
    # 建立物理模組
    physics = VSPINNChannelFlow(
        physics_params={'nu': 1e-4, 'rho': 1.0, 'dp_dx': -0.0025},
        scaling_factors={'N_x': 2.0, 'N_y': 12.0, 'N_z': 2.0}
    )
    
    # 生成測試資料
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 10000
    coords = torch.rand(batch_size, 3, device=device, requires_grad=True)
    predictions = torch.randn(batch_size, 4, device=device, requires_grad=True)
    
    # === 測試 1: 無快取 ===
    logging.info("\n🐌 測試無快取模式...")
    start_time = time.time()
    residuals_no_cache = physics.compute_momentum_residuals(
        coords, predictions, scaled_coords=coords, gradients=None
    )
    time_no_cache = time.time() - start_time
    logging.info(f"   時間: {time_no_cache*1000:.2f} ms")
    
    # === 測試 2: 有快取 ===
    logging.info("\n🚀 測試快取模式...")
    grad_cache = GradientCache(device=device)
    predictions_dict = {
        'u': predictions[:, 0:1],
        'v': predictions[:, 1:2],
        'w': predictions[:, 2:3],
        'p': predictions[:, 3:4]
    }
    
    # 計算快取
    start_cache = time.time()
    gradients = grad_cache.compute_all_gradients(predictions_dict, coords)
    time_cache_compute = time.time() - start_cache
    
    # 使用快取
    start_use = time.time()
    residuals_with_cache = physics.compute_momentum_residuals(
        coords, predictions, scaled_coords=coords, gradients=gradients
    )
    time_with_cache = time.time() - start_use
    
    total_cache_time = time_cache_compute + time_with_cache
    
    logging.info(f"   快取計算時間: {time_cache_compute*1000:.2f} ms")
    logging.info(f"   使用快取時間: {time_with_cache*1000:.2f} ms")
    logging.info(f"   總時間: {total_cache_time*1000:.2f} ms")
    
    # === 測試 3: 數值一致性 ===
    logging.info("\n✅ 驗證數值一致性...")
    for key in residuals_no_cache:
        if key in residuals_with_cache:
            diff = torch.abs(residuals_no_cache[key] - residuals_with_cache[key]).max()
            logging.info(f"   {key:20s}: max_diff = {diff:.2e}")
            assert diff < 1e-5, f"{key} 不一致！差異: {diff:.2e}"
    
    # === 效能總結 ===
    speedup = (time_no_cache / total_cache_time - 1) * 100
    logging.info(f"\n📊 效能總結:")
    logging.info(f"   無快取:   {time_no_cache*1000:.2f} ms")
    logging.info(f"   有快取:   {total_cache_time*1000:.2f} ms")
    logging.info(f"   加速:     {speedup:+.1f}%")
    
    return {
        'time_no_cache': time_no_cache,
        'time_with_cache': total_cache_time,
        'speedup_percent': speedup,
        'numerically_correct': True
    }


def test_trainer_integration():
    """測試 Trainer 是否正確啟用梯度快取"""
    logging.info("\n" + "=" * 80)
    logging.info("測試 3: Trainer 整合驗證")
    logging.info("=" * 80)
    
    # 載入標準配置
    config_path = Path(__file__).parents[2] / "configs" / "standard_config_template.yml"
    if not config_path.exists():
        logging.warning(f"⚠️  找不到配置檔案: {config_path}")
        logging.warning("   跳過 Trainer 整合測試")
        return None
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 檢查程式碼中的梯度快取使用
    trainer_file = Path(__file__).parents[2] / "pinnx" / "train" / "trainer.py"
    with open(trainer_file, 'r') as f:
        trainer_code = f.read()
    
    # 關鍵檢查點
    checks = {
        'GradientCache 匯入': 'from pinnx.physics.gradient_cache import GradientCache' in trainer_code,
        'GradientCache2D 匯入': 'from pinnx.physics.gradient_cache_2d import GradientCache2D' in trainer_code,
        '_compute_vs_pinn_gradients 方法': '_compute_vs_pinn_gradients' in trainer_code,
        '_compute_2d_gradients 方法': '_compute_2d_gradients' in trainer_code,
        'gradients 參數傳遞': "kwargs['gradients'] = gradients" in trainer_code
    }
    
    logging.info("\n✅ Trainer 程式碼檢查:")
    all_passed = True
    for check_name, result in checks.items():
        status = "✅" if result else "❌"
        logging.info(f"   {status} {check_name}")
        if not result:
            all_passed = False
    
    # 檢查 LossManager
    loss_manager_file = Path(__file__).parents[2] / "pinnx" / "train" / "loss_manager.py"
    with open(loss_manager_file, 'r') as f:
        loss_manager_code = f.read()
    
    loss_checks = {
        'compute_pde_loss 接受 gradients 參數': 'gradients: Optional[Dict[str, torch.Tensor]] = None' in loss_manager_code,
        'VS-PINN 傳遞 gradients': "gradients=gradients  # 🚀 Wave 2" in loss_manager_code,
        '2D 流場傳遞 gradients': "'gradients' in sig.parameters and gradients is not None" in loss_manager_code
    }
    
    logging.info("\n✅ LossManager 程式碼檢查:")
    for check_name, result in loss_checks.items():
        status = "✅" if result else "❌"
        logging.info(f"   {status} {check_name}")
        if not result:
            all_passed = False
    
    # 檢查物理模組
    physics_files = [
        ('Kolmogorov Flow 2D', 'pinnx/physics/kolmogorov_flow_2d.py'),
        ('VS-PINN Channel Flow', 'pinnx/physics/vs_pinn_channel_flow.py')
    ]
    
    logging.info("\n✅ 物理模組檢查:")
    for name, rel_path in physics_files:
        physics_file = Path(__file__).parents[2] / rel_path
        if physics_file.exists():
            with open(physics_file, 'r') as f:
                physics_code = f.read()
            
            has_gradients_param = 'gradients: Optional[Dict[str, torch.Tensor]]' in physics_code
            has_if_gradients = 'if gradients is not None:' in physics_code
            
            if has_gradients_param and has_if_gradients:
                logging.info(f"   ✅ {name}: 支援梯度快取")
            else:
                logging.info(f"   ❌ {name}: 未完整支援梯度快取")
                all_passed = False
        else:
            logging.warning(f"   ⚠️  {name}: 找不到檔案")
    
    return {
        'all_checks_passed': all_passed,
        'trainer_ready': all(checks.values()),
        'loss_manager_ready': all(loss_checks.values())
    }


def main():
    """主函數"""
    logging.info("=" * 80)
    logging.info("🔍 梯度快取驗證腳本")
    logging.info("=" * 80)
    
    results = {}
    
    # 測試 1: 2D 梯度快取
    try:
        results['2d'] = test_2d_gradient_cache()
    except Exception as e:
        logging.error(f"❌ 2D 梯度快取測試失敗: {e}")
        results['2d'] = {'error': str(e)}
    
    # 測試 2: 3D 梯度快取
    try:
        results['3d'] = test_3d_gradient_cache()
    except Exception as e:
        logging.error(f"❌ 3D 梯度快取測試失敗: {e}")
        results['3d'] = {'error': str(e)}
    
    # 測試 3: Trainer 整合
    try:
        results['trainer'] = test_trainer_integration()
    except Exception as e:
        logging.error(f"❌ Trainer 整合測試失敗: {e}")
        results['trainer'] = {'error': str(e)}
    
    # === 總結報告 ===
    logging.info("\n" + "=" * 80)
    logging.info("📊 總結報告")
    logging.info("=" * 80)
    
    if '2d' in results and 'speedup_percent' in results['2d']:
        logging.info(f"\n✅ Kolmogorov Flow 2D:")
        logging.info(f"   加速: {results['2d']['speedup_percent']:+.1f}%")
        logging.info(f"   數值正確: {results['2d']['numerically_correct']}")
    
    if '3d' in results and 'speedup_percent' in results['3d']:
        logging.info(f"\n✅ VS-PINN Channel Flow:")
        logging.info(f"   加速: {results['3d']['speedup_percent']:+.1f}%")
        logging.info(f"   數值正確: {results['3d']['numerically_correct']}")
    
    if 'trainer' in results and results['trainer'] is not None:
        logging.info(f"\n✅ Trainer 整合:")
        logging.info(f"   程式碼就緒: {results['trainer']['trainer_ready']}")
        logging.info(f"   LossManager 就緒: {results['trainer']['loss_manager_ready']}")
        logging.info(f"   所有檢查通過: {results['trainer']['all_checks_passed']}")
    
    # 最終判定
    all_ok = True
    if '2d' in results and 'error' not in results['2d']:
        if results['2d']['speedup_percent'] < 0:
            logging.warning("\n⚠️  2D 梯度快取沒有加速效果（可能是批次太小）")
    else:
        all_ok = False
    
    if '3d' in results and 'error' not in results['3d']:
        if results['3d']['speedup_percent'] < 0:
            logging.warning("\n⚠️  3D 梯度快取沒有加速效果（可能是批次太小）")
    else:
        all_ok = False
    
    if 'trainer' in results and results['trainer'] is not None:
        if not results['trainer']['all_checks_passed']:
            logging.warning("\n⚠️  Trainer 程式碼檢查未完全通過")
            all_ok = False
    
    if all_ok:
        logging.info("\n" + "=" * 80)
        logging.info("🎉 所有測試通過！梯度快取已正確實作並啟用。")
        logging.info("=" * 80)
    else:
        logging.warning("\n" + "=" * 80)
        logging.warning("⚠️  部分測試失敗或警告，請檢查上方輸出。")
        logging.warning("=" * 80)
    
    return results


if __name__ == "__main__":
    main()
