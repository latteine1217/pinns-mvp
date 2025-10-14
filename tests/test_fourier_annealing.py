"""
FourierAnnealingScheduler 單元測試

測試覆蓋：
1. 階段配置驗證
2. 訓練進度計算與階段切換
3. 全局與每軸配置
4. 與 AxisSelectiveFourierFeatures 集成
5. 預設策略工廠
"""

import pytest
import torch
from pinnx.train.fourier_annealing import (
    FourierAnnealingScheduler,
    AnnealingStage,
    create_default_annealing,
    create_channel_flow_annealing
)
from pinnx.models.axis_selective_fourier import AxisSelectiveFourierFeatures


class TestAnnealingStage:
    """AnnealingStage 數據類測試"""
    
    def test_valid_stage(self):
        """測試合法階段配置"""
        stage = AnnealingStage(
            end_ratio=0.5,
            frequencies=[1, 2, 4],
            description="測試階段"
        )
        assert stage.end_ratio == 0.5
        assert stage.frequencies == [1, 2, 4]
        assert stage.description == "測試階段"
    
    def test_empty_frequencies(self):
        """測試空頻率列表（允許）"""
        stage = AnnealingStage(end_ratio=1.0, frequencies=[])
        assert stage.frequencies == []
    
    def test_invalid_end_ratio(self):
        """測試無效的 end_ratio"""
        # 超出範圍
        with pytest.raises(ValueError, match="end_ratio 必須在"):
            AnnealingStage(end_ratio=0.0, frequencies=[1])
        
        with pytest.raises(ValueError, match="end_ratio 必須在"):
            AnnealingStage(end_ratio=1.5, frequencies=[1])
    
    def test_invalid_frequencies(self):
        """測試無效頻率值"""
        # 負數
        with pytest.raises(ValueError, match="frequencies 必須全為正整數"):
            AnnealingStage(end_ratio=1.0, frequencies=[1, -2])
        
        # 零
        with pytest.raises(ValueError, match="frequencies 必須全為正整數"):
            AnnealingStage(end_ratio=1.0, frequencies=[0, 1])


class TestFourierAnnealingScheduler:
    """FourierAnnealingScheduler 核心功能測試"""
    
    def test_basic_initialization(self):
        """測試基本初始化"""
        stages = [
            AnnealingStage(0.5, [1, 2]),
            AnnealingStage(1.0, [1, 2, 4])
        ]
        scheduler = FourierAnnealingScheduler(stages, axes_names=['x', 'y'])
        
        assert scheduler.current_stage_idx == 0
        assert scheduler.current_progress == 0.0
        assert len(scheduler.stages) == 2
    
    def test_stage_validation_empty(self):
        """測試空階段列表"""
        with pytest.raises(ValueError, match="stages 不能為空"):
            FourierAnnealingScheduler([])
    
    def test_stage_validation_last_ratio(self):
        """測試最後階段 end_ratio 必須為 1.0"""
        stages = [
            AnnealingStage(0.5, [1, 2]),
            AnnealingStage(0.8, [1, 2, 4])  # 錯誤：不是 1.0
        ]
        with pytest.raises(ValueError, match="最後階段的 end_ratio 必須為 1.0"):
            FourierAnnealingScheduler(stages)
    
    def test_stage_validation_monotonic(self):
        """測試階段單調性"""
        stages = [
            AnnealingStage(0.8, [1, 2, 4]),  # 順序錯誤
            AnnealingStage(0.3, [1, 2]),
            AnnealingStage(1.0, [1, 2, 4, 8])
        ]
        with pytest.raises(ValueError, match="end_ratio 必須單調遞增"):
            FourierAnnealingScheduler(stages)
    
    def test_progress_calculation(self):
        """測試訓練進度計算"""
        stages = [
            AnnealingStage(0.3, [1, 2]),
            AnnealingStage(0.7, [1, 2, 4]),
            AnnealingStage(1.0, [1, 2, 4, 8])
        ]
        scheduler = FourierAnnealingScheduler(stages, axes_names=['x'])
        
        # 測試不同 epoch 的進度
        test_cases = [
            (0, 100, 0.01),    # 第 1 個 epoch：1%
            (49, 100, 0.50),   # 第 50 個 epoch：50%
            (99, 100, 1.00),   # 第 100 個 epoch：100%
        ]
        
        for epoch, total, expected_progress in test_cases:
            scheduler.step(epoch, total)
            assert abs(scheduler.current_progress - expected_progress) < 1e-6
    
    def test_stage_switching(self):
        """測試階段切換邏輯"""
        stages = [
            AnnealingStage(0.3, [1, 2]),
            AnnealingStage(0.6, [1, 2, 4]),
            AnnealingStage(1.0, [1, 2, 4, 8])
        ]
        scheduler = FourierAnnealingScheduler(stages, axes_names=['x'])
        total_epochs = 1000
        
        # 階段 1 (0-30%)
        config = scheduler.step(100, total_epochs)  # 10%
        assert config['x'] == [1, 2]
        
        config = scheduler.step(299, total_epochs)  # 30%
        assert config['x'] == [1, 2]
        
        # 階段 2 (30-60%)
        config = scheduler.step(300, total_epochs)  # 30.1%
        assert config['x'] == [1, 2, 4]
        
        config = scheduler.step(500, total_epochs)  # 50%
        assert config['x'] == [1, 2, 4]
        
        # 階段 3 (60-100%)
        config = scheduler.step(600, total_epochs)  # 60.1%
        assert config['x'] == [1, 2, 4, 8]
        
        config = scheduler.step(999, total_epochs)  # 100%
        assert config['x'] == [1, 2, 4, 8]
    
    def test_global_configuration(self):
        """測試全局配置（所有軸相同）"""
        stages = [
            AnnealingStage(0.5, [1, 2]),
            AnnealingStage(1.0, [1, 2, 4])
        ]
        scheduler = FourierAnnealingScheduler(stages, axes_names=['x', 'y', 'z'])
        
        config = scheduler.step(0, 100)  # 1% 進度
        
        assert config['x'] == [1, 2]
        assert config['y'] == [1, 2]
        assert config['z'] == [1, 2]
    
    def test_per_axis_configuration(self):
        """測試每軸專門配置"""
        global_stages = [
            AnnealingStage(0.5, [1, 2]),
            AnnealingStage(1.0, [1, 2, 4])
        ]
        
        # y 軸專門配置（始終空列表）
        per_axis_stages = {
            'y': [AnnealingStage(1.0, [], "y 軸無 Fourier")]
        }
        
        scheduler = FourierAnnealingScheduler(
            global_stages, 
            axes_names=['x', 'y', 'z'],
            per_axis_stages=per_axis_stages
        )
        
        # 早期階段
        config = scheduler.step(0, 100)  # 1%
        assert config['x'] == [1, 2]  # 使用全局
        assert config['y'] == []      # 使用專門配置
        assert config['z'] == [1, 2]  # 使用全局
        
        # 後期階段
        config = scheduler.step(99, 100)  # 100%
        assert config['x'] == [1, 2, 4]
        assert config['y'] == []  # y 軸始終為空
        assert config['z'] == [1, 2, 4]
    
    def test_get_info(self):
        """測試獲取調度器狀態資訊"""
        stages = [
            AnnealingStage(0.3, [1, 2], "低頻"),
            AnnealingStage(1.0, [1, 2, 4], "中頻")
        ]
        scheduler = FourierAnnealingScheduler(stages, axes_names=['x'])
        
        scheduler.step(0, 100)  # 1% 進度
        info = scheduler.get_info()
        
        assert 'progress' in info
        assert 'stage_index' in info
        assert 'stage_description' in info
        assert 'active_frequencies' in info
        assert info['stage_description'] == "低頻"
        assert info['total_stages'] == 2


class TestIntegrationWithFourierFeatures:
    """與 AxisSelectiveFourierFeatures 集成測試"""
    
    def test_update_fourier_module(self):
        """測試更新實際 Fourier 模組"""
        from pinnx.models.axis_selective_fourier import FourierFeatureFactory
        
        # 使用工廠創建模組
        axes_config = {'x': [1, 2, 4, 8], 'y': [], 'z': [1, 2, 4, 8]}
        config = {'type': 'axis_selective', 'axes_config': axes_config}
        fourier = FourierFeatureFactory.create(config=config, in_dim=3)
        
        original_out_dim = fourier.out_dim
        
        # 全局階段 + y 軸專用配置（保持為空）
        stages = [
            AnnealingStage(0.5, [1, 2]),
            AnnealingStage(1.0, [1, 2, 4, 8])
        ]
        per_axis_stages = {'y': [AnnealingStage(1.0, [])]}  # y 軸始終為空
        scheduler = FourierAnnealingScheduler(stages, per_axis_stages=per_axis_stages, axes_names=['x', 'y', 'z'])
        
        # 早期階段：減少頻率
        scheduler.update_fourier_features(fourier, current_epoch=0, total_epochs=100)
        assert fourier.out_dim < original_out_dim  # 維度應減少
        assert fourier.axes_config['x'] == [1, 2]
        assert fourier.axes_config['y'] == []  # y 軸保持空列表
        assert fourier.axes_config['z'] == [1, 2]
        
        # 測試前向傳播仍正常
        x = torch.randn(10, 3)
        output = fourier(x)
        assert output.shape[0] == 10
    
    def test_update_without_method_raises(self):
        """測試不支持更新的模組會拋出錯誤"""
        import torch.nn as nn
        
        # 創建一個沒有 set_active_frequencies 的模組
        dummy_module = nn.Linear(3, 10)
        
        stages = [AnnealingStage(1.0, [1, 2])]
        scheduler = FourierAnnealingScheduler(stages)
        
        with pytest.raises(AttributeError, match="沒有 set_active_frequencies"):
            scheduler.update_fourier_features(dummy_module, 0, 100)


class TestDefaultStrategies:
    """預設策略工廠測試"""
    
    def test_conservative_strategy(self):
        """測試保守策略"""
        stages = create_default_annealing('conservative')
        
        assert len(stages) == 3
        assert stages[0].frequencies == [1, 2]
        assert stages[1].frequencies == [1, 2, 4]
        assert stages[2].frequencies == [1, 2, 4, 8]
    
    def test_aggressive_strategy(self):
        """測試激進策略"""
        stages = create_default_annealing('aggressive')
        
        assert len(stages) == 2
        assert stages[0].frequencies == [1, 2, 4]
        assert stages[1].frequencies == [1, 2, 4, 8]  # 修正：實際實現到 K=8
    
    def test_fine_strategy(self):
        """測試精細策略"""
        stages = create_default_annealing('fine')
        
        assert len(stages) == 4
        assert stages[-1].end_ratio == 1.0
    
    def test_channel_flow_annealing(self):
        """測試通道流專用退火配置"""
        annealing_config = create_channel_flow_annealing()
        
        # 檢查返回的數據結構（直接返回每軸配置）
        assert 'x' in annealing_config
        assert 'y' in annealing_config
        assert 'z' in annealing_config
        
        # 檢查 x 軸（流向）
        x_stages = annealing_config['x']
        assert len(x_stages) == 3
        assert x_stages[-1].end_ratio == 1.0
        assert x_stages[-1].frequencies == [1, 2, 4, 8]
        
        # 檢查 y 軸（壁法向）：應始終為空
        y_stages = annealing_config['y']
        assert len(y_stages) == 1
        assert y_stages[0].frequencies == []
        
        # 檢查 z 軸（展向）
        z_stages = annealing_config['z']
        assert len(z_stages) == 2
        assert z_stages[-1].frequencies == [1, 2, 4]
    
    def test_invalid_strategy(self):
        """測試無效策略名稱"""
        with pytest.raises(ValueError, match="未知策略"):
            create_default_annealing('invalid_strategy')


class TestEdgeCases:
    """邊界條件測試"""
    
    def test_single_stage(self):
        """測試單階段配置（無退火）"""
        stages = [AnnealingStage(1.0, [1, 2, 4, 8], "全頻段")]
        scheduler = FourierAnnealingScheduler(stages, axes_names=['x'])
        
        # 任何進度都應返回相同配置
        config_early = scheduler.step(0, 1000)
        config_late = scheduler.step(999, 1000)
        
        assert config_early['x'] == [1, 2, 4, 8]
        assert config_late['x'] == [1, 2, 4, 8]
    
    def test_very_fine_stages(self):
        """測試極細粒度階段（10 階段）"""
        stages = [
            AnnealingStage(i * 0.1 + 0.1, [1, 2] + list(range(1, i+2)))
            for i in range(10)
        ]
        stages[-1].end_ratio = 1.0  # 確保最後為 1.0
        
        scheduler = FourierAnnealingScheduler(stages, axes_names=['x'])
        
        # 驗證階段切換正常
        for epoch in range(0, 1000, 100):
            config = scheduler.step(epoch, 1000)
            assert 'x' in config
    
    def test_axes_without_names(self):
        """測試未提供 axes_names 的情況"""
        stages = [
            AnnealingStage(0.5, [1, 2]),
            AnnealingStage(1.0, [1, 2, 4])
        ]
        scheduler = FourierAnnealingScheduler(stages)
        
        config = scheduler.step(0, 100)
        # 應返回 'default' 鍵
        assert 'default' in config
        assert config['default'] == [1, 2]


# ========== 測試運行配置 ==========

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
