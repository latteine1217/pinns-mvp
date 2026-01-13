"""
測試 .item() 優化是否正確工作

驗證：
1. loss_manager.combine_losses() 返回 tensor 格式的損失值
2. TrainingLoopManager._convert_tensors_to_float() 正確轉換
3. Trainer._to_scalar() 正確轉換單個值
"""

import torch
import sys
from pathlib import Path

# 添加專案路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_combine_losses_returns_tensors():
    """測試 combine_losses 返回 tensors"""
    from pinnx.train.loss_manager import LossManager
    from pinnx.physics.kolmogorov_flow_2d import KolmogorovFlow2D
    
    print("=" * 80)
    print("測試 1: loss_manager.combine_losses() 返回 tensor 格式")
    print("=" * 80)
    
    # 創建簡單的 physics
    device = torch.device('cpu')
    physics = KolmogorovFlow2D(
        amplitude=1.0,
        wavenumber=4.0,
        nu=0.1,
        rho=1.0,
        device=device
    )
    
    # 創建 LossManager
    config = {
        'losses': {
            'data': 1.0,
            'momentum_x': 0.1,
            'momentum_y': 0.1,
            'continuity': 0.1,
            'periodic_x': 1.0,
            'periodic_y': 1.0,
        }
    }
    loss_manager = LossManager(physics, config, device)
    
    # 創建假的損失字典
    loss_dict = {
        'data_loss': torch.tensor(0.5, device=device),
        'momentum_x_loss': torch.tensor(0.1, device=device),
        'momentum_y_loss': torch.tensor(0.1, device=device),
        'momentum_z_loss': torch.tensor(0.0, device=device),
        'continuity_loss': torch.tensor(0.05, device=device),
        'u_loss': torch.tensor(0.2, device=device),
        'v_loss': torch.tensor(0.3, device=device),
        'w_loss': torch.tensor(0.0, device=device),
        'pressure_loss': torch.tensor(0.0, device=device),
        'periodic_x_loss': torch.tensor(0.02, device=device),
        'periodic_y_loss': torch.tensor(0.03, device=device),
    }
    
    # 調用 combine_losses
    total_loss, result_dict = loss_manager.combine_losses(loss_dict)
    
    # 驗證返回值類型
    print(f"\n✅ total_loss 類型: {type(total_loss)}")
    assert isinstance(total_loss, torch.Tensor), "total_loss 應該是 tensor"
    
    print(f"✅ result_dict['total_loss'] 類型: {type(result_dict['total_loss'])}")
    assert isinstance(result_dict['total_loss'], torch.Tensor), "result_dict['total_loss'] 應該是 tensor"
    
    print(f"✅ result_dict['data_loss'] 類型: {type(result_dict['data_loss'])}")
    assert isinstance(result_dict['data_loss'], torch.Tensor), "result_dict['data_loss'] 應該是 tensor"
    
    print("\n✅ 測試通過：combine_losses() 返回 tensor 格式\n")
    return result_dict


def test_convert_tensors_to_float(result_dict):
    """測試 TrainingLoopManager._convert_tensors_to_float()"""
    from pinnx.train.training_loop_manager import TrainingLoopManager
    
    print("=" * 80)
    print("測試 2: TrainingLoopManager._convert_tensors_to_float()")
    print("=" * 80)
    
    # 轉換
    converted = TrainingLoopManager._convert_tensors_to_float(result_dict)
    
    # 驗證類型
    print(f"\n✅ 轉換後 total_loss 類型: {type(converted['total_loss'])}")
    assert isinstance(converted['total_loss'], float), "轉換後應該是 float"
    
    print(f"✅ 轉換後 data_loss 類型: {type(converted['data_loss'])}")
    assert isinstance(converted['data_loss'], float), "轉換後應該是 float"
    
    # 驗證值正確性
    original_value = result_dict['total_loss'].item()
    converted_value = converted['total_loss']
    print(f"\n✅ 值正確性檢查:")
    print(f"   原始值（使用 .item()）: {original_value:.6f}")
    print(f"   轉換值: {converted_value:.6f}")
    assert abs(original_value - converted_value) < 1e-6, "值應該相同"
    
    print("\n✅ 測試通過：_convert_tensors_to_float() 正確工作\n")


def test_trainer_to_scalar():
    """測試 Trainer._to_scalar()"""
    from pinnx.train.trainer import Trainer
    
    print("=" * 80)
    print("測試 3: Trainer._to_scalar()")
    print("=" * 80)
    
    # 測試 tensor 輸入
    tensor_value = torch.tensor(3.14159, device='cpu')
    scalar = Trainer._to_scalar(tensor_value)
    print(f"\n✅ Tensor 輸入: {tensor_value} → {scalar} (type: {type(scalar)})")
    assert isinstance(scalar, float), "應該轉換為 float"
    assert abs(scalar - 3.14159) < 1e-5, "值應該正確"
    
    # 測試 float 輸入
    float_value = 2.71828
    scalar = Trainer._to_scalar(float_value)
    print(f"✅ Float 輸入: {float_value} → {scalar} (type: {type(scalar)})")
    assert scalar == float_value, "float 應該保持不變"
    
    # 測試 int 輸入
    int_value = 42
    scalar = Trainer._to_scalar(int_value)
    print(f"✅ Int 輸入: {int_value} → {scalar} (type: {type(scalar)})")
    assert scalar == int_value, "int 應該保持不變"
    
    print("\n✅ 測試通過：_to_scalar() 正確工作\n")


def test_performance_comparison():
    """測試效能差異"""
    import time
    
    print("=" * 80)
    print("測試 4: 效能比較")
    print("=" * 80)
    
    # 創建大量 tensors
    num_tensors = 100
    tensor_dict = {f'loss_{i}': torch.randn(1) for i in range(num_tensors)}
    
    # 方法 1：逐個 .item()
    start = time.time()
    result1 = {key: val.item() for key, val in tensor_dict.items()}
    time1 = time.time() - start
    
    # 方法 2：使用 _convert_tensors_to_float
    from pinnx.train.training_loop_manager import TrainingLoopManager
    start = time.time()
    result2 = TrainingLoopManager._convert_tensors_to_float(tensor_dict)
    time2 = time.time() - start
    
    print(f"\n逐個 .item(): {time1 * 1000:.3f} ms")
    print(f"批次轉換: {time2 * 1000:.3f} ms")
    print(f"加速比: {time1 / time2:.2f}×")
    
    # 驗證結果相同
    for key in result1:
        assert abs(result1[key] - result2[key]) < 1e-6, "結果應該相同"
    
    print("\n✅ 測試通過：效能測試完成\n")


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("🚀 開始測試 .item() 優化")
    print("=" * 80 + "\n")
    
    try:
        # 測試 1
        result_dict = test_combine_losses_returns_tensors()
        
        # 測試 2
        test_convert_tensors_to_float(result_dict)
        
        # 測試 3
        test_trainer_to_scalar()
        
        # 測試 4
        test_performance_comparison()
        
        print("=" * 80)
        print("✅ 所有測試通過！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 測試失敗：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
