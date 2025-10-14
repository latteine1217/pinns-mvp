#!/usr/bin/env python3
"""
配置解析驗證腳本

測試 normalize_config_structure() 是否正確處理：
1. fourier_features.type = "disabled" → use_fourier = False
2. fourier_features.type = "standard" → use_fourier = True
"""

import sys
from pathlib import Path

# 添加專案根目錄到路徑
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pinnx.train.config_loader import load_config


def test_fourier_disabled_config():
    """測試 Fourier 禁用配置"""
    config_path = PROJECT_ROOT / "configs" / "test_rans_phase6b_no_fourier_2k.yml"
    
    print("=" * 80)
    print("🧪 測試配置: test_rans_phase6b_no_fourier_2k.yml")
    print("=" * 80)
    
    try:
        config = load_config(str(config_path))
        model_cfg = config['model']
        
        print(f"✅ 配置載入成功")
        print(f"\n📋 Fourier 配置解析結果:")
        print(f"   use_fourier:       {model_cfg.get('use_fourier')}")
        print(f"   fourier_m:         {model_cfg.get('fourier_m')}")
        print(f"   fourier_sigma:     {model_cfg.get('fourier_sigma')}")
        print(f"   fourier_trainable: {model_cfg.get('fourier_trainable')}")
        
        # 驗證期望值
        assert model_cfg['use_fourier'] == False, "❌ use_fourier 應為 False"
        assert model_cfg['fourier_m'] == 0, "❌ fourier_m 應為 0"
        assert model_cfg['fourier_sigma'] == 0.0, "❌ fourier_sigma 應為 0.0"
        
        print(f"\n✅ 所有驗證通過！Fourier Features 已正確禁用")
        return True
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fourier_enabled_config():
    """測試 Fourier 啟用配置（baseline）"""
    config_path = PROJECT_ROOT / "configs" / "test_rans_phase6b_extended_2k.yml"
    
    print("\n" + "=" * 80)
    print("🧪 測試配置: test_rans_phase6b_extended_2k.yml")
    print("=" * 80)
    
    try:
        config = load_config(str(config_path))
        model_cfg = config['model']
        
        print(f"✅ 配置載入成功")
        print(f"\n📋 Fourier 配置解析結果:")
        print(f"   use_fourier:       {model_cfg.get('use_fourier')}")
        print(f"   fourier_m:         {model_cfg.get('fourier_m')}")
        print(f"   fourier_sigma:     {model_cfg.get('fourier_sigma')}")
        print(f"   fourier_trainable: {model_cfg.get('fourier_trainable')}")
        
        # 驗證期望值
        assert model_cfg['use_fourier'] == True, "❌ use_fourier 應為 True"
        assert model_cfg['fourier_m'] > 0, "❌ fourier_m 應 > 0"
        
        print(f"\n✅ 所有驗證通過！Fourier Features 已正確啟用")
        return True
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🔍 開始配置解析驗證...\n")
    
    result1 = test_fourier_disabled_config()
    result2 = test_fourier_enabled_config()
    
    print("\n" + "=" * 80)
    print("📊 測試總結")
    print("=" * 80)
    print(f"   Fourier 禁用配置: {'✅ PASS' if result1 else '❌ FAIL'}")
    print(f"   Fourier 啟用配置: {'✅ PASS' if result2 else '❌ FAIL'}")
    print(f"\n   總體結果: {'✅ ALL TESTS PASSED' if (result1 and result2) else '❌ SOME TESTS FAILED'}")
    print("=" * 80)
    
    sys.exit(0 if (result1 and result2) else 1)
