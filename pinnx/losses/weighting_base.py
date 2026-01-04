"""
權重調整器抽象基類

定義所有 Loss Weighter 的統一接口，確保一致性和可擴展性。
遵循 P1-2 設計模式：純多態、明確契約。

設計哲學：
1. 統一簽名：所有 weighter 使用相同的 update_weights 接口
2. Context 字典：通過 context 傳遞所有可選參數，避免簽名膨脹
3. 類型安全：明確輸入輸出類型，便於靜態檢查
4. 可測試性：每個實現可獨立測試，無隱式依賴
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch


class LossWeighter(ABC):
    """
    損失權重調整器抽象基類

    所有動態權重調整策略的統一接口。

    職責：
    - 根據訓練狀態動態調整各損失項的權重
    - 返回標準化的權重字典 Dict[str, float]
    - 支援可選的上下文參數（通過 context 字典）

    使用示例：
    ```python
    weighter = GradNormWeighter(model, loss_names=['data', 'pde'])

    context = {
        'step': 1000,
        'total_loss': total_loss_tensor,
        'x_train': training_data
    }

    weights = weighter.update_weights(losses, context)
    # weights = {'data': 1.2, 'pde': 0.8}
    ```
    """

    @abstractmethod
    def update_weights(
        self,
        losses: Dict[str, torch.Tensor],
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        更新並返回損失權重

        Args:
            losses: 損失項字典，鍵為損失名稱，值為未加權的損失張量
                   例如：{'data': tensor(0.5), 'pde': tensor(1.2)}

            context: 上下文參數字典，包含所有可選參數：
                - step (int): 當前訓練步數（適用於 NTK、Adaptive）
                - total_steps (int): 總訓練步數（適用於 Adaptive）
                - total_loss (torch.Tensor): 總損失（適用於 GradNorm）
                - x_train (torch.Tensor): 訓練數據（適用於 NTK）
                - time_coords (torch.Tensor): 時間坐標（適用於 Causal）
                - pde_losses (torch.Tensor): PDE 損失（適用於 Causal）
                - epoch (int): 當前訓練輪次
                - model (nn.Module): 模型引用（如果需要）

        Returns:
            Dict[str, float]: 更新後的權重字典
                鍵必須與 losses 中的鍵對應
                值為正浮點數，表示該損失項的相對權重

        Raises:
            ValueError: 如果必要的 context 參數缺失
            RuntimeError: 如果權重計算失敗

        Note:
            - 權重應該是正數（>0）
            - 權重的絕對值由各實現決定，LossManager 會進一步縮放
            - 如果某個損失項不需要調整，返回 1.0
        """
        pass

    def get_weights(self) -> Dict[str, float]:
        """
        獲取當前權重（不更新）

        Returns:
            當前緩存的權重字典

        Note:
            子類應該實現此方法以返回最近一次 update_weights 的結果
            如果從未調用過 update_weights，應返回初始權重
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_weights() method"
        )

    def reset_weights(self) -> None:
        """
        重置權重到初始狀態

        用於訓練重啟或課程學習階段切換。

        Note:
            子類應該實現此方法以重置內部狀態
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement reset_weights() method"
        )


class PointWeighter(ABC):
    """
    點權重調整器抽象基類（可選）

    用於計算每個採樣點的權重（如 Causal Weighting）。
    與 LossWeighter 不同，PointWeighter 返回的是張量而非字典。

    使用場景：
    - 時間因果權重（早期時間點權重更高）
    - 區域自適應權重（邊界層 vs 自由流）
    - 誤差驅動採樣權重

    Note:
        PointWeighter 通常不與 LossWeighter 混用，
        而是在採樣或損失計算時直接應用。
    """

    @abstractmethod
    def compute_weights(
        self,
        residuals: torch.Tensor,
        coords: torch.Tensor,
        context: Dict[str, Any]
    ) -> torch.Tensor:
        """
        計算每個點的權重

        Args:
            residuals: 每個點的殘差或損失 [N, 1]
            coords: 點的坐標 [N, dim]
            context: 額外上下文參數

        Returns:
            torch.Tensor: 點權重 [N, 1]，與 residuals 形狀相同
        """
        pass
