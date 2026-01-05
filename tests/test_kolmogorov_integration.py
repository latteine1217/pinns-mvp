"""端到端測試：驗證 Kolmogorov 標準化在訓練管線中的集成."""

import math
import sys
import torch
import yaml
from pathlib import Path

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pinnx.utils.normalization_helpers import create_input_normalizer


def test_kolmogorov_normalizer_integration():
    """測試 Kolmogorov 標準化器集成."""
    
    # 1. 模擬配置（與實際配置文件一致）
    config = {
        'data': {
            'kolmogorov_config': {
                'time_range': [0.0, 50.0]
            }
        },
        'model': {
            'scaling': {
                'input_norm': 'kolmogorov',
                'input_norm_range': [0.0, 1.0]
            }
        }
    }
    
    # 2. 模擬訓練數據（3D 坐標）
    N = 100
    t = torch.linspace(0, 50, N).unsqueeze(1)
    x = torch.linspace(0, 2*math.pi, N).unsqueeze(1)
    y = torch.linspace(0, 2*math.pi, N).unsqueeze(1)
    
    # 創建多組數據（模擬 PDE、sensor、BC、IC 等）
    training_data = {
        'coords_pde_spatial': torch.cat([x, y], dim=1),
        't_pde': t,
        'coords_sensors_spatial': torch.cat([x[:50], y[:50]], dim=1),
        't_sensors': t[:50],
    }
    
    device = torch.device('cpu')
    is_vs_pinn = False
    
    # 3. 創建標準化器
    print("="*60)
    print("測試 Kolmogorov 標準化器集成")
    print("="*60)
    
    normalizer = create_input_normalizer(
        config=config,
        training_data=training_data,
        is_vs_pinn=is_vs_pinn,
        device=device
    )
    
    # 4. 驗證標準化器類型
    from pinnx.utils.normalization import KolmogorovInputTransform
    assert isinstance(normalizer, KolmogorovInputTransform), \
        f"預期 KolmogorovInputTransform，實際: {type(normalizer)}"
    print("✅ 標準化器類型正確: KolmogorovInputTransform")
    
    # 5. 驗證 t_max 配置
    assert normalizer.t_max == 50.0, f"預期 t_max=50.0，實際: {normalizer.t_max}"
    print(f"✅ t_max 正確: {normalizer.t_max}")
    
    # 6. 測試標準化效果
    coords = torch.cat([t[:10], x[:10], y[:10]], dim=1)
    coords_norm = normalizer.transform(coords)
    
    # 驗證時間維度
    assert coords_norm[:, 0].min().item() >= 0.0
    assert coords_norm[:, 0].max().item() <= 1.0
    print(f"✅ 時間維度標準化: [{coords_norm[:, 0].min():.4f}, {coords_norm[:, 0].max():.4f}]")
    
    # 驗證空間維度保持不變
    assert torch.allclose(coords_norm[:, 1], x[:10].squeeze(), rtol=1e-5)
    assert torch.allclose(coords_norm[:, 2], y[:10].squeeze(), rtol=1e-5)
    print(f"✅ 空間維度保持不變:")
    print(f"   X: [{coords_norm[:, 1].min():.4f}, {coords_norm[:, 1].max():.4f}]")
    print(f"   Y: [{coords_norm[:, 2].min():.4f}, {coords_norm[:, 2].max():.4f}]")
    
    print("="*60)
    print("✅ 所有測試通過！Kolmogorov 標準化器已正確集成")
    print("="*60)
    
    return normalizer


def test_config_file_parsing():
    """測試從實際配置文件解析."""
    
    config_path = Path(__file__).parent.parent / 'configs' / 'experiments' / 'time_window_kolmogorov.yml'
    
    if not config_path.exists():
        print(f"⚠️  配置文件不存在: {config_path}")
        return
    
    print("\n" + "="*60)
    print("測試配置文件解析")
    print("="*60)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 驗證配置結構
    assert 'model' in config, "缺少 'model' 配置"
    assert 'scaling' in config['model'], "缺少 'model.scaling' 配置"
    assert 'input_norm' in config['model']['scaling'], "缺少 'model.scaling.input_norm' 配置"
    
    input_norm = config['model']['scaling']['input_norm']
    assert input_norm == 'kolmogorov', f"預期 input_norm='kolmogorov'，實際: {input_norm}"
    
    print(f"✅ 配置文件解析成功")
    print(f"   input_norm: {input_norm}")
    print(f"   time_range: {config['data']['kolmogorov_config']['time_range']}")
    
    print("="*60)


def test_backward_compatibility():
    """測試向後相容性：其他標準化類型仍可用."""
    
    print("\n" + "="*60)
    print("測試向後相容性")
    print("="*60)
    
    # 測試 standard 標準化
    config_standard = {
        'data': {},
        'model': {
            'scaling': {
                'input_norm': 'standard',
                'input_norm_range': [-1.0, 1.0]
            }
        }
    }
    
    training_data = {
        'coords_pde_spatial': torch.randn(100, 2),
        't_pde': torch.randn(100, 1),
    }
    
    normalizer = create_input_normalizer(
        config=config_standard,
        training_data=training_data,
        is_vs_pinn=False,
        device=torch.device('cpu')
    )
    
    from pinnx.utils.normalization import InputTransform
    assert isinstance(normalizer, InputTransform), \
        f"Standard 標準化應返回 InputTransform，實際: {type(normalizer)}"
    
    print("✅ Standard 標準化仍可用")
    
    # 測試 minmax 標準化
    config_minmax = {
        'data': {},
        'model': {
            'scaling': {
                'input_norm': 'minmax',
                'input_norm_range': [0.0, 1.0]
            }
        }
    }
    
    normalizer = create_input_normalizer(
        config=config_minmax,
        training_data=training_data,
        is_vs_pinn=False,
        device=torch.device('cpu')
    )
    
    assert isinstance(normalizer, InputTransform), \
        f"MinMax 標準化應返回 InputTransform，實際: {type(normalizer)}"
    
    print("✅ MinMax 標準化仍可用")
    print("="*60)


if __name__ == '__main__':
    print("\n" + "🚀 Kolmogorov 標準化集成測試套件")
    print("="*60 + "\n")
    
    try:
        # 1. 基礎集成測試
        test_kolmogorov_normalizer_integration()
        
        # 2. 配置文件解析測試
        test_config_file_parsing()
        
        # 3. 向後相容性測試
        test_backward_compatibility()
        
        print("\n" + "="*60)
        print("🎉 所有集成測試通過！")
        print("="*60)
        
        print("\n📝 總結:")
        print("  ✅ KolmogorovInputTransform 正確創建")
        print("  ✅ t_max 從配置中正確提取")
        print("  ✅ 標準化效果符合 JAX-PI：時間 [0,1]，空間 [0,2π]")
        print("  ✅ 配置文件解析正常")
        print("  ✅ 向後相容性保持（standard/minmax 仍可用）")
        print("\n✨ 系統已準備好使用 Kolmogorov 標準化進行訓練！")
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
