"""
Fourier Features 頻率退火調度器

實現階段式頻率解鎖策略，避免訓練初期高頻特徵導致不穩定：
- Stage 1 (0-30%): 僅使用最低頻 {1, 2}
- Stage 2 (30-60%): 解鎖中頻 {4}
- Stage 3 (60-100%): 解鎖高頻 {8, 16}（視需要）

設計原則：
1. 逐步增加頻譜複雜度，避免震盪
2. 支持自定義階段配置
3. 與 AxisSelectiveFourierFeatures 無縫集成
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AnnealingStage:
    """
    單個退火階段配置
    
    Attributes:
        end_ratio: 階段結束時的訓練進度比例（0.0-1.0）
        frequencies: 該階段允許的頻率列表（循環數）
        description: 階段描述（用於日誌）
    """
    end_ratio: float
    frequencies: List[int]
    description: str = ""
    
    def __post_init__(self):
        if not 0.0 < self.end_ratio <= 1.0:
            raise ValueError(f"end_ratio 必須在 (0, 1] 範圍內，當前: {self.end_ratio}")
        # 允許空頻率列表（表示該軸不使用 Fourier）
        if not all(isinstance(k, int) and k > 0 for k in self.frequencies):
            raise ValueError(f"frequencies 必須全為正整數，當前: {self.frequencies}")


class FourierAnnealingScheduler:
    """
    Fourier Features 頻率退火調度器
    
    根據訓練進度動態調整啟用的頻率範圍，實現以下策略：
    1. 訓練初期：僅保留低頻（平滑先驗）
    2. 訓練中期：逐步解鎖中頻
    3. 訓練後期：啟用全部頻率（捕捉細節）
    
    Example:
        >>> stages = [
        ...     AnnealingStage(0.3, [1, 2], "低頻預熱"),
        ...     AnnealingStage(0.6, [1, 2, 4], "中頻解鎖"),
        ...     AnnealingStage(1.0, [1, 2, 4, 8], "全頻段")
        ... ]
        >>> scheduler = FourierAnnealingScheduler(stages)
        >>> active_freqs = scheduler.step(epoch=100, total_epochs=1000)  # 10% 進度
        >>> print(active_freqs)  # [1, 2]
    """
    
    def __init__(
        self,
        stages: List[AnnealingStage],
        axes_names: Optional[List[str]] = None,
        per_axis_stages: Optional[Dict[str, List[AnnealingStage]]] = None
    ):
        """
        Args:
            stages: 全局退火階段列表（按 end_ratio 排序）
            axes_names: 軸名稱列表（如 ['x', 'y', 'z']），用於驗證
            per_axis_stages: 每個軸的專門階段配置（可選，覆蓋全局）
                例：{'x': [stage1, stage2], 'y': [stage_y1]}
        """
        # 驗證階段順序
        self._validate_stages(stages)
        self.stages = sorted(stages, key=lambda s: s.end_ratio)
        
        # 處理每軸配置
        self.axes_names = axes_names or []
        self.per_axis_stages = per_axis_stages or {}
        
        # 驗證每軸階段
        for axis, axis_stages in self.per_axis_stages.items():
            self._validate_stages(axis_stages)
            self.per_axis_stages[axis] = sorted(axis_stages, key=lambda s: s.end_ratio)
        
        # 當前狀態
        self.current_stage_idx = 0
        self.current_progress = 0.0
    
    def _validate_stages(self, stages: List[AnnealingStage]):
        """驗證階段配置合法性"""
        if not stages:
            raise ValueError("stages 不能為空")
        
        if stages[-1].end_ratio != 1.0:
            raise ValueError(f"最後階段的 end_ratio 必須為 1.0，當前: {stages[-1].end_ratio}")
        
        # 檢查單調性
        ratios = [s.end_ratio for s in stages]
        if ratios != sorted(ratios):
            raise ValueError(f"階段的 end_ratio 必須單調遞增，當前: {ratios}")
    
    def step(
        self, 
        current_epoch: int, 
        total_epochs: int
    ) -> Dict[str, List[int]]:
        """
        根據當前訓練進度更新啟用頻率
        
        Args:
            current_epoch: 當前 epoch 數（從 0 開始）
            total_epochs: 總 epoch 數
        
        Returns:
            啟用頻率配置字典 {'x': [1, 2, 4], 'y': [], 'z': [1, 2]}
            - 如果沒有 per_axis_stages，返回全局配置應用於所有軸
        """
        # 計算訓練進度
        self.current_progress = (current_epoch + 1) / total_epochs
        
        # 找到當前階段
        current_stage = self._find_current_stage(self.stages, self.current_progress)
        self.current_stage_idx = self.stages.index(current_stage)
        
        # 構建返回配置
        if not self.per_axis_stages:
            # 全局配置：所有軸使用相同頻率
            global_freqs = current_stage.frequencies
            return {axis: global_freqs for axis in self.axes_names} if self.axes_names else {'default': global_freqs}
        else:
            # 每軸配置
            result = {}
            for axis in self.axes_names:
                if axis in self.per_axis_stages:
                    axis_stage = self._find_current_stage(
                        self.per_axis_stages[axis], 
                        self.current_progress
                    )
                    result[axis] = axis_stage.frequencies
                else:
                    # 未指定的軸使用全局配置
                    result[axis] = current_stage.frequencies
            return result
    
    def _find_current_stage(
        self, 
        stages: List[AnnealingStage], 
        progress: float
    ) -> AnnealingStage:
        """根據進度找到對應階段"""
        for stage in stages:
            if progress <= stage.end_ratio:
                return stage
        return stages[-1]  # 容錯：返回最後階段
    
    def get_info(self) -> Dict:
        """
        獲取當前調度器狀態資訊
        
        Returns:
            包含當前階段、進度、頻率等資訊的字典
        """
        current_stage = self.stages[self.current_stage_idx]
        return {
            'progress': self.current_progress,
            'stage_index': self.current_stage_idx,
            'stage_description': current_stage.description,
            'active_frequencies': current_stage.frequencies,
            'total_stages': len(self.stages)
        }
    
    def update_fourier_features(
        self, 
        fourier_module: nn.Module,
        current_epoch: int,
        total_epochs: int
    ):
        """
        直接更新 AxisSelectiveFourierFeatures 模組的頻率配置
        
        Args:
            fourier_module: AxisSelectiveFourierFeatures 實例
            current_epoch: 當前 epoch
            total_epochs: 總 epoch 數
        
        Note:
            這會修改模組的 axes_config 並重新構建 Fourier 矩陣
        """
        # 獲取新頻率配置
        new_config = self.step(current_epoch, total_epochs)
        
        # 檢查模組是否有 set_active_frequencies 方法
        if not hasattr(fourier_module, 'set_active_frequencies'):
            raise AttributeError(
                f"{type(fourier_module).__name__} 沒有 set_active_frequencies() 方法。"
                f"請確保使用 AxisSelectiveFourierFeatures 類。"
            )
        
        # 更新模組
        fourier_module.set_active_frequencies(new_config)


# ========== 預設配置工廠 ==========

def create_default_annealing(strategy: str = 'conservative') -> List[AnnealingStage]:
    """
    創建預設退火配置
    
    Args:
        strategy: 策略名稱
            - 'conservative': 保守策略（緩慢解鎖，3 階段）
            - 'aggressive': 激進策略（快速解鎖，2 階段）
            - 'fine': 精細策略（4 階段）
            - 'custom': 從配置文件讀取（返回空列表，由 Trainer 處理）
    
    Returns:
        階段配置列表（custom 策略返回空列表）
    """
    if strategy == 'conservative':
        return [
            AnnealingStage(0.3, [1, 2], "低頻預熱（K=1,2）"),
            AnnealingStage(0.6, [1, 2, 4], "中頻解鎖（K=4）"),
            AnnealingStage(1.0, [1, 2, 4, 8], "全頻段（K=8）")
        ]
    elif strategy == 'aggressive':
        return [
            AnnealingStage(0.4, [1, 2, 4], "低中頻（K≤4）"),
            AnnealingStage(1.0, [1, 2, 4, 8], "全頻段（K=8）")
        ]
    elif strategy == 'fine':
        return [
            AnnealingStage(0.2, [1, 2], "極低頻"),
            AnnealingStage(0.4, [1, 2, 4], "低中頻"),
            AnnealingStage(0.7, [1, 2, 4, 8], "中高頻"),
            AnnealingStage(1.0, [1, 2, 4, 8, 16], "全頻段")
        ]
    elif strategy == 'custom':
        # 🔧 custom 策略：由 Trainer 從配置文件的 fourier_annealing.stages 讀取
        # Factory 階段返回空列表，避免重複定義
        return []
    else:
        raise ValueError(f"未知策略: {strategy}，可選: conservative, aggressive, fine, custom")


def create_channel_flow_annealing() -> Dict[str, List[AnnealingStage]]:
    """
    創建通道流專用退火配置
    
    Returns:
        每軸配置字典 {'x': [...], 'y': [...], 'z': [...]}
    """
    # x 軸（流向）：週期，使用完整退火
    x_stages = [
        AnnealingStage(0.3, [1, 2], "流向低頻"),
        AnnealingStage(0.6, [1, 2, 4], "流向中頻"),
        AnnealingStage(1.0, [1, 2, 4, 8], "流向全頻段")
    ]
    
    # y 軸（壁法向）：非週期，極少或不用 Fourier
    y_stages = [
        AnnealingStage(1.0, [], "壁法向無 Fourier")  # 始終為空
    ]
    
    # z 軸（展向）：週期，中等頻率即可
    z_stages = [
        AnnealingStage(0.4, [1, 2], "展向低頻"),
        AnnealingStage(1.0, [1, 2, 4], "展向中頻")
    ]
    
    return {'x': x_stages, 'y': y_stages, 'z': z_stages}


# ========== 工具函數 ==========

def visualize_annealing_schedule(
    scheduler: FourierAnnealingScheduler,
    total_epochs: int,
    num_checkpoints: int = 10
):
    """
    可視化退火時程表（用於調試）
    
    Args:
        scheduler: 調度器實例
        total_epochs: 總 epoch 數
        num_checkpoints: 檢查點數量
    """
    print("=" * 60)
    print(f"Fourier Annealing Schedule ({total_epochs} epochs)")
    print("=" * 60)
    
    checkpoints = [int(total_epochs * i / num_checkpoints) for i in range(num_checkpoints + 1)]
    
    for epoch in checkpoints:
        config = scheduler.step(epoch, total_epochs)
        info = scheduler.get_info()
        
        print(f"\nEpoch {epoch:5d} ({info['progress']*100:5.1f}%)")
        print(f"  Stage: {info['stage_description']}")
        if scheduler.axes_names:
            for axis in scheduler.axes_names:
                print(f"    {axis}: {config[axis]}")
        else:
            print(f"  Frequencies: {info['active_frequencies']}")


# ========== 測試案例 ==========

if __name__ == "__main__":
    print("=== 測試 FourierAnnealingScheduler ===\n")
    
    # 測試 1: 全局配置
    print("【測試 1】全局配置（conservative 策略）")
    stages = create_default_annealing('conservative')
    scheduler = FourierAnnealingScheduler(stages, axes_names=['x', 'y', 'z'])
    visualize_annealing_schedule(scheduler, total_epochs=1000, num_checkpoints=5)
    
    print("\n" + "=" * 60 + "\n")
    
    # 測試 2: 通道流配置
    print("【測試 2】通道流專用配置（每軸獨立）")
    per_axis = create_channel_flow_annealing()
    global_stages = create_default_annealing('conservative')
    scheduler2 = FourierAnnealingScheduler(
        global_stages, 
        axes_names=['x', 'y', 'z'],
        per_axis_stages=per_axis
    )
    visualize_annealing_schedule(scheduler2, total_epochs=2000, num_checkpoints=6)
    
    print("\n=== 所有測試完成 ===")
