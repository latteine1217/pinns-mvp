"""
測試損失歸一化配置讀取修正
================================

驗證 Bug Fix: normalize_losses 從配置正確讀取，而非硬編碼為 True

問題描述:
- pinnx/physics/kolmogorov_flow_2d.py line 131
- pinnx/physics/vs_pinn_channel_flow.py line 207
- 原本: self.normalize_losses = True (硬編碼，忽略配置)
- 修正: self.normalize_losses = (loss_config or {}).get('normalize_losses', True)

測試目標:
1. 配置設為 True 時，physics.normalize_losses = True
2. 配置設為 False 時，physics.normalize_losses = False
3. 無配置時，默認值 True
4. 驗證 Kolmogorov Flow 2D 和 Channel Flow 均已修正

作者: PINNs-MVP Team
日期: 2025-12-19
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pytest


class TestNormalizeLossesConfigKolmogorov:
    """測試 Kolmogorov Flow 2D 的 normalize_losses 配置讀取"""

    def test_normalize_true_explicit(self):
        """測試明確設置 normalize_losses: true"""
        from pinnx.physics.kolmogorov_flow_2d import KolmogorovFlow2D
        
        physics = KolmogorovFlow2D(
            domain_bounds={'x': (0, 2*3.14159), 'y': (0, 2*3.14159)},
            loss_config={'normalize_losses': True}
        )
        
        assert physics.normalize_losses == True, \
            "配置 normalize_losses: true 時應為 True"

    def test_normalize_false_explicit(self):
        """測試明確設置 normalize_losses: false（修正前會失敗）"""
        from pinnx.physics.kolmogorov_flow_2d import KolmogorovFlow2D
        
        physics = KolmogorovFlow2D(
            domain_bounds={'x': (0, 2*3.14159), 'y': (0, 2*3.14159)},
            loss_config={'normalize_losses': False}
        )
        
        assert physics.normalize_losses == False, \
            "❌ Bug: 配置 normalize_losses: false 時應為 False，但實際為 True（硬編碼）"

    def test_normalize_default(self):
        """測試無配置時默認值為 True"""
        from pinnx.physics.kolmogorov_flow_2d import KolmogorovFlow2D
        
        physics = KolmogorovFlow2D(
            domain_bounds={'x': (0, 2*3.14159), 'y': (0, 2*3.14159)},
            loss_config={}
        )
        
        assert physics.normalize_losses == True, \
            "無配置時，normalize_losses 應默認為 True"

    def test_normalize_no_loss_config(self):
        """測試 loss_config=None 時默認值為 True"""
        from pinnx.physics.kolmogorov_flow_2d import KolmogorovFlow2D
        
        physics = KolmogorovFlow2D(
            domain_bounds={'x': (0, 2*3.14159), 'y': (0, 2*3.14159)},
            loss_config=None
        )
        
        assert physics.normalize_losses == True, \
            "loss_config=None 時，normalize_losses 應默認為 True"


class TestNormalizeLossesConfigChannelFlow:
    """測試 VS-PINN Channel Flow 的 normalize_losses 配置讀取"""

    def _create_minimal_channel_flow(self, loss_config):
        """創建最小化 Channel Flow 實例（避免初始化所有複雜組件）"""
        from pinnx.physics.vs_pinn_channel_flow import VSPINNChannelFlow
        
        # 最小配置
        return VSPINNChannelFlow(
            domain_bounds={
                'x': (0.0, 8*3.14159),
                'y': (-1.0, 1.0),
                'z': (0.0, 3*3.14159)
            },
            enable_rans=False,
            loss_config=loss_config
        )

    def test_normalize_true_explicit(self):
        """測試明確設置 normalize_losses: true"""
        physics = self._create_minimal_channel_flow(
            loss_config={'normalize_losses': True}
        )
        
        assert physics.normalize_losses == True, \
            "配置 normalize_losses: true 時應為 True"

    def test_normalize_false_explicit(self):
        """測試明確設置 normalize_losses: false（修正前會失敗）"""
        physics = self._create_minimal_channel_flow(
            loss_config={'normalize_losses': False}
        )
        
        assert physics.normalize_losses == False, \
            "❌ Bug: 配置 normalize_losses: false 時應為 False，但實際為 True（硬編碼）"

    def test_normalize_default(self):
        """測試無配置時默認值為 True"""
        physics = self._create_minimal_channel_flow(loss_config={})
        
        assert physics.normalize_losses == True, \
            "無配置時，normalize_losses 應默認為 True"

    def test_normalize_no_loss_config(self):
        """測試 loss_config=None 時默認值為 True"""
        physics = self._create_minimal_channel_flow(loss_config=None)
        
        assert physics.normalize_losses == True, \
            "loss_config=None 時，normalize_losses 應默認為 True"


class TestWarmupEpochsConfig:
    """驗證 warmup_epochs 仍正確讀取（對比檢查）"""

    def test_warmup_kolmogorov(self):
        """驗證 Kolmogorov Flow 的 warmup_epochs 配置讀取正常"""
        from pinnx.physics.kolmogorov_flow_2d import KolmogorovFlow2D
        
        physics = KolmogorovFlow2D(
            domain_bounds={'x': (0, 2*3.14159), 'y': (0, 2*3.14159)},
            loss_config={'warmup_epochs': 10}
        )
        
        assert physics.warmup_epochs == 10, \
            "warmup_epochs 應正確讀取配置值"

    def test_warmup_channel_flow(self):
        """驗證 Channel Flow 的 warmup_epochs 配置讀取正常"""
        from pinnx.physics.vs_pinn_channel_flow import VSPINNChannelFlow
        
        physics = VSPINNChannelFlow(
            domain_bounds={
                'x': (0.0, 8*3.14159),
                'y': (-1.0, 1.0),
                'z': (0.0, 3*3.14159)
            },
            enable_rans=False,
            loss_config={'warmup_epochs': 15}
        )
        
        assert physics.warmup_epochs == 15, \
            "warmup_epochs 應正確讀取配置值"


def run_tests():
    """運行所有測試並生成報告"""
    import subprocess
    result = subprocess.run(
        ['pytest', __file__, '-v', '--tb=short'],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    print(result.stderr)
    return result.returncode


if __name__ == '__main__':
    print("=" * 80)
    print("損失歸一化配置讀取修正驗證")
    print("=" * 80)
    print("\n📋 測試範圍:")
    print("  ✅ Kolmogorov Flow 2D: normalize_losses 配置讀取")
    print("  ✅ VS-PINN Channel Flow: normalize_losses 配置讀取")
    print("  ✅ 默認值驗證 (True)")
    print("  ✅ warmup_epochs 對比測試（確保修正未破壞其他功能）")
    print("\n" + "=" * 80)
    
    exit_code = run_tests()
    
    print("\n" + "=" * 80)
    if exit_code == 0:
        print("✅ 所有測試通過！normalize_losses 配置讀取修正成功")
    else:
        print("❌ 測試失敗，請檢查修正是否正確應用")
    print("=" * 80)
    
    sys.exit(exit_code)
