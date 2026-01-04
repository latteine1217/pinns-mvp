"""
Validation 管理模組

提供統一的驗證接口，將驗證邏輯從 Trainer 中解耦。

設計模式：
- Strategy Pattern: 可插拔的驗證策略
- Manager Pattern: 集中管理多種驗證策略
- Dependency Injection: Trainer 接收 ValidationManager 實例

設計哲學：
1. Single Responsibility: ValidationManager 只負責驗證協調
2. Open/Closed: 易於添加新的驗證策略，無需修改現有代碼
3. Simplicity: 清晰的接口，易於理解和擴展
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import logging


class ValidationStrategy(ABC):
    """
    驗證策略抽象基類

    職責：
    - 定義統一的驗證接口
    - 子類實現具體的驗證邏輯

    設計原則：
    - 接口統一：所有驗證策略使用相同的 validate 接口
    - 獨立性：每個策略可獨立測試
    - 可組合性：多個策略可組合使用
    """

    @abstractmethod
    def validate(
        self,
        model: nn.Module,
        device: torch.device,
        **kwargs
    ) -> Dict[str, float]:
        """
        執行驗證並返回 metrics

        Args:
            model: PINN 模型
            device: 計算設備
            **kwargs: 額外參數（由具體策略定義）

        Returns:
            驗證指標字典 (e.g., {'mse': 0.01, 'relative_l2': 0.05})

        Raises:
            RuntimeError: 驗證執行失敗
        """
        pass


class DataBasedValidation(ValidationStrategy):
    """
    基於數據的驗證策略

    特性：
    - 與 ground truth 數據對比
    - 計算 MSE, RMSE, MAE, Relative L2 等指標
    - 支持部分變量驗證（預測維度 != 目標維度）

    使用示例：
    ```python
    validation_data = {
        'coords': torch.randn(1000, 2),  # [N, d]
        'targets': torch.randn(1000, 4)  # [N, n_vars]
    }

    strategy = DataBasedValidation(
        validation_data=validation_data,
        metrics=['mse', 'relative_l2', 'rmse'],
        data_normalizer=normalizer
    )

    results = strategy.validate(model, device)
    # {'mse': 0.01, 'relative_l2': 0.05, 'rmse': 0.1}
    ```
    """

    def __init__(
        self,
        validation_data: Dict[str, torch.Tensor],
        metrics: List[str] = ['mse', 'relative_l2'],
        data_normalizer: Optional[Any] = None,
        model_input_preprocessor: Optional[callable] = None,
        variable_order_inference: Optional[callable] = None
    ):
        """
        初始化 DataBasedValidation

        Args:
            validation_data: 驗證數據字典，包含 'coords' 和 'targets'
            metrics: 要計算的指標列表
            data_normalizer: 數據標準化器（用於反標準化模型輸出）
            model_input_preprocessor: 模型輸入預處理函數
            variable_order_inference: 變量順序推斷函數
        """
        self.validation_data = validation_data
        self.metrics = metrics
        self.data_normalizer = data_normalizer
        self.model_input_preprocessor = model_input_preprocessor
        self.variable_order_inference = variable_order_inference

    def validate(
        self,
        model: nn.Module,
        device: torch.device,
        **kwargs
    ) -> Dict[str, float]:
        """執行基於數據的驗證"""
        # 1. 檢查數據可用性
        if not self._is_data_available():
            logging.debug("驗證數據不可用，跳過驗證")
            return {}

        # 2. 準備數據
        coords, targets = self._prepare_data(device)

        # 3. 執行推理
        predictions = self._run_inference(model, coords, device)

        # 4. 計算指標
        return self._compute_metrics(predictions, targets)

    def _is_data_available(self) -> bool:
        """檢查驗證數據是否可用"""
        if self.validation_data is None:
            return False
        if self.validation_data.get('size', 0) == 0:
            return False

        coords = self.validation_data.get('coords')
        targets = self.validation_data.get('targets')

        if coords is None or targets is None:
            return False
        if coords.numel() == 0 or targets.numel() == 0:
            return False

        return True

    def _prepare_data(self, device: torch.device):
        """準備驗證數據"""
        coords = self.validation_data['coords'].to(device)
        targets = self.validation_data['targets'].to(device)
        return coords, targets

    def _run_inference(
        self,
        model: nn.Module,
        coords: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """執行模型推理"""
        # 保存訓練狀態
        training_mode = model.training
        model.eval()

        try:
            with torch.no_grad():
                # 預處理坐標（如果提供）
                if self.model_input_preprocessor is not None:
                    coords_for_model = self.model_input_preprocessor(coords)
                else:
                    coords_for_model = coords

                # 模型預測（標準化空間）
                preds_norm = model(coords_for_model)

                # 反標準化為物理量（如果提供）
                if self.data_normalizer is not None:
                    # 推斷變量順序
                    if self.variable_order_inference is not None:
                        var_order = self.variable_order_inference(preds_norm.shape[1])
                    else:
                        var_order = None

                    preds_phys_raw = self.data_normalizer.denormalize_batch(
                        preds_norm, var_order=var_order
                    )
                    preds_phys = preds_phys_raw if isinstance(preds_phys_raw, torch.Tensor) \
                        else torch.tensor(preds_phys_raw, device=device)
                else:
                    preds_phys = preds_norm

                return preds_phys
        finally:
            # 恢復訓練狀態
            if training_mode:
                model.train()

    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """計算驗證指標"""
        # 處理維度不匹配
        n_pred = predictions.shape[1]
        n_targets = targets.shape[1]
        n_common = min(n_pred, n_targets)

        if n_pred != n_targets:
            logging.debug(
                f"[Validation] 輸出維度不匹配 (pred={n_pred}, target={n_targets})；"
                f"比較前 {n_common} 個分量。"
            )

        preds_final = predictions[:, :n_common]
        targets_final = targets[:, :n_common]

        # 計算指標
        results = {}
        for metric_name in self.metrics:
            results[metric_name] = self._compute_single_metric(
                metric_name, preds_final, targets_final
            )

        return results

    def _compute_single_metric(
        self,
        metric_name: str,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """計算單個指標"""
        diff = predictions - targets

        if metric_name == 'mse':
            return torch.mean(diff ** 2).item()

        elif metric_name == 'rmse':
            return torch.sqrt(torch.mean(diff ** 2)).item()

        elif metric_name == 'mae':
            return torch.mean(torch.abs(diff)).item()

        elif metric_name == 'relative_l2':
            # Relative L2: ||pred - target|| / ||target||
            from pinnx.evals.metrics import relative_L2
            return relative_L2(predictions, targets).mean().item()

        else:
            raise ValueError(f"Unknown metric: {metric_name}")


class PhysicsBasedValidation(ValidationStrategy):
    """
    基於物理的驗證策略

    特性：
    - 檢查物理約束滿足情況（連續性方程、邊界條件等）
    - 使用 PhysicsValidator 執行驗證
    - 可配置驗證項目

    使用示例：
    ```python
    strategy = PhysicsBasedValidation(
        physics_validator=validator,
        validation_coords=torch.randn(1000, 2),
        checks=['continuity', 'boundary', 'momentum']
    )

    results = strategy.validate(model, device)
    # {'physics_continuity': 0.001, 'physics_boundary': 0.002}
    ```
    """

    def __init__(
        self,
        physics_validator: Any,
        validation_coords: Optional[torch.Tensor] = None,
        checks: Optional[List[str]] = None
    ):
        """
        初始化 PhysicsBasedValidation

        Args:
            physics_validator: PhysicsValidator 實例
            validation_coords: 驗證坐標（可選，如果為 None 則從 validator 獲取）
            checks: 檢查項目列表（可選，如果為 None 則執行所有檢查）
        """
        self.physics_validator = physics_validator
        self.validation_coords = validation_coords
        self.checks = checks or []

    def validate(
        self,
        model: nn.Module,
        device: torch.device,
        **kwargs
    ) -> Dict[str, float]:
        """執行基於物理的驗證"""
        try:
            # 準備驗證數據
            validation_data = kwargs.get('validation_data')
            if validation_data is None:
                logging.debug("物理驗證數據不可用，跳過驗證")
                return {}

            coords = validation_data.get('coords')
            if coords is None or coords.numel() == 0:
                return {}

            coords = coords.to(device)

            # 提取坐標分量
            test_data = self._prepare_coords(coords, kwargs.get('is_3d', False))

            # 執行驗證
            physics_results = self.physics_validator.validate(
                model=model,
                test_data=test_data,
                log_to_wandb=kwargs.get('log_to_wandb', False)
            )

            return physics_results if physics_results is not None else {}

        except Exception as e:
            logging.warning(f"⚠️  物理驗證執行失敗: {e}")
            return {}

    def _prepare_coords(
        self,
        coords: torch.Tensor,
        is_3d: bool
    ) -> Dict[str, torch.Tensor]:
        """準備坐標數據"""
        test_data = {}

        if is_3d or coords.shape[-1] == 3:
            # 3D: x, y, z
            test_data['x'] = coords[:, 0:1].requires_grad_(True)
            test_data['y'] = coords[:, 1:2].requires_grad_(True)
            test_data['z'] = coords[:, 2:3].requires_grad_(True)
        else:
            # 2D: x, y
            test_data['x'] = coords[:, 0:1].requires_grad_(True)
            test_data['y'] = coords[:, 1:2].requires_grad_(True)

        return test_data


class ValidationManager:
    """
    驗證管理器

    職責：
    - 管理多種驗證策略
    - 協調驗證執行
    - 匯總驗證結果

    設計原則：
    - Composition over Inheritance（組合優於繼承）
    - Strategy Pattern（策略模式）
    - 支持動態添加策略

    使用示例：
    ```python
    # 創建驗證管理器
    manager = ValidationManager()

    # 添加數據驗證策略
    manager.add_strategy(DataBasedValidation(
        validation_data=test_data,
        metrics=['mse', 'relative_l2']
    ))

    # 添加物理驗證策略
    manager.add_strategy(PhysicsBasedValidation(
        physics_validator=validator
    ))

    # 執行所有驗證
    results = manager.validate(model, device)
    # {'mse': 0.01, 'relative_l2': 0.05, 'physics_continuity': 0.001}
    ```
    """

    def __init__(
        self,
        strategies: Optional[List[ValidationStrategy]] = None
    ):
        """
        初始化 ValidationManager

        Args:
            strategies: 驗證策略列表（可選）
        """
        self.strategies = strategies or []

    def add_strategy(self, strategy: ValidationStrategy):
        """
        添加驗證策略

        Args:
            strategy: ValidationStrategy 實例
        """
        self.strategies.append(strategy)

    def validate(
        self,
        model: nn.Module,
        device: torch.device,
        **kwargs
    ) -> Dict[str, float]:
        """
        執行所有驗證策略並匯總結果

        Args:
            model: PINN 模型
            device: 計算設備
            **kwargs: 傳遞給各策略的額外參數

        Returns:
            匯總的驗證指標字典
        """
        all_results = {}

        for strategy in self.strategies:
            try:
                results = strategy.validate(model, device, **kwargs)
                all_results.update(results)
            except Exception as e:
                logging.warning(
                    f"⚠️  驗證策略 {strategy.__class__.__name__} 執行失敗: {e}"
                )

        return all_results

    def has_strategies(self) -> bool:
        """檢查是否有配置任何驗證策略"""
        return len(self.strategies) > 0
