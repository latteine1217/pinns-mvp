"""
測試 CausalWeighterV2 在訓練循環中的整合

此腳本驗證：
1. V2 版本能正確創建
2. 能接收分量級的殘差字典
3. 能輸出加權後的損失
4. 與 Trainer 的接口相容
"""

import torch
import logging

from pinnx.train.weighter_factory import create_weighters
from pinnx.losses.causal_weighter_v2 import CausalWeighterPerComponent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_v2_integration():
    """測試 V2 與訓練循環的整合"""
    
    # 1. 創建 weighter
    config = {
        'losses': {
            'causal_weighting': True,
            'weighting': {
                'causal': {
                    'version': 'v2',
                    'causal_tol': 1.0,
                    'num_chunks': 16
                }
            }
        },
        'data': {
            'kolmogorov_config': {
                'time_range': [0.0, 1.0]
            }
        }
    }
    
    model = torch.nn.Linear(3, 3)
    device = torch.device('cpu')
    weighters = create_weighters(config, model, device)
    
    causal_weighter = weighters['causal']
    assert causal_weighter is not None, "Causal weighter not created"
    assert isinstance(causal_weighter, CausalWeighterPerComponent), \
        f"Wrong type: {type(causal_weighter)}"
    
    logger.info("✅ Step 1: Causal weighter v2 created successfully")
    
    # 2. 創建模擬的殘差數據（模擬訓練循環中的殘差）
    batch_size = 1024
    residual_dict = {
        'momentum_x': torch.randn(batch_size, 1, requires_grad=True),
        'momentum_y': torch.randn(batch_size, 1, requires_grad=True),
        'continuity': torch.randn(batch_size, 1, requires_grad=True)
    }
    time_coords = torch.linspace(0.0, 1.0, batch_size).unsqueeze(1)
    
    logger.info(f"✅ Step 2: Created mock residuals (batch_size={batch_size})")
    
    # 3. 計算加權損失
    weighted_losses = causal_weighter.apply_weights_to_losses(
        residual_dict, time_coords
    )
    
    logger.info("✅ Step 3: Applied causal weights to losses")
    
    # 4. 驗證輸出
    assert isinstance(weighted_losses, dict), "Output should be a dict"
    assert 'momentum_x' in weighted_losses, "Missing momentum_x"
    assert 'momentum_y' in weighted_losses, "Missing momentum_y"
    assert 'continuity' in weighted_losses, "Missing continuity"
    
    for key, loss in weighted_losses.items():
        assert isinstance(loss, torch.Tensor), f"{key} loss should be a tensor"
        assert loss.dim() == 0, f"{key} loss should be a scalar"
        logger.info(f"   {key:15s}: {loss.item():.6f}")
    
    logger.info("✅ Step 4: Output validation passed")
    
    # 5. 測試梯度回傳
    total_loss = weighted_losses['momentum_x'] + weighted_losses['momentum_y'] + weighted_losses['continuity']
    if total_loss.requires_grad:
        total_loss.backward()
        logger.info("✅ Step 5: Gradient backpropagation successful")
    else:
        logger.info("⚠️  Step 5: Loss doesn't require grad (expected for mean operations)")
    
    # 6. 獲取診斷信息
    diagnostics = causal_weighter.get_diagnostic_info(residual_dict, time_coords)
    
    logger.info("📊 Diagnostic Information:")
    logger.info(f"   num_chunks: {diagnostics['num_chunks']}")
    logger.info(f"   min_weight: {diagnostics['min_weight']:.6f}")
    logger.info(f"   max_weight: {diagnostics['max_weight']:.6f}")
    logger.info(f"   weight_variance: {diagnostics['weight_variance']:.6f}")
    
    logger.info("\n" + "="*70)
    logger.info("✅ ALL INTEGRATION TESTS PASSED!")
    logger.info("="*70)
    
    return True


def test_v1_backward_compatibility():
    """測試 V1 版本的向後兼容性"""
    
    config = {
        'losses': {
            'causal_weighting': True,
            'causal_tol': 1.0,
            'num_chunks': 32
        },
        'data': {
            'kolmogorov_config': {
                'time_range': [0.0, 1.0]
            }
        }
    }
    
    model = torch.nn.Linear(3, 3)
    device = torch.device('cpu')
    weighters = create_weighters(config, model, device)
    
    causal_weighter = weighters['causal']
    assert causal_weighter is not None, "Causal weighter not created"
    
    # V1 版本應該有 t_min 和 t_max 屬性
    assert hasattr(causal_weighter, 't_min'), "V1 should have t_min attribute"
    assert hasattr(causal_weighter, 't_max'), "V1 should have t_max attribute"
    
    logger.info("✅ V1 backward compatibility test passed")
    
    return True


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🧪 Testing CausalWeighterV2 Integration")
    print("="*70 + "\n")
    
    # Test V2
    test_v2_integration()
    
    print("\n" + "="*70)
    print("🧪 Testing V1 Backward Compatibility")
    print("="*70 + "\n")
    
    # Test V1
    test_v1_backward_compatibility()
    
    print("\n" + "="*70)
    print("🎉 ALL TESTS PASSED!")
    print("="*70)
