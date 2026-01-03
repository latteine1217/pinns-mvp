"""
Registry Pattern 實現 - 優化器與調度器工廠

目標：
1. 消除 if-elif 條件分支鏈（Good Taste 原則）
2. 實現 Open-Closed 原則（新增類型只需 @register）
3. 提供統一的錯誤處理與 fallback 機制
"""

import logging
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================================
# Scheduler Factory (Registry Pattern)
# ============================================================================

class SchedulerFactory:
    """
    基於 Registry Pattern 的學習率調度器工廠

    優點：
    - 無條件分支（Good Taste）
    - 新增類型只需 @register 裝飾器
    - 集中的錯誤處理
    """

    def __init__(self):
        self._registry: Dict[str, Callable] = {}

    def register(self, name: str):
        """
        註冊裝飾器

        用法：
            @_scheduler_factory.register('cosine')
            def _create_cosine(optimizer, config):
                return CosineAnnealingLR(...)
        """
        def decorator(func: Callable):
            self._registry[name.lower()] = func
            return func
        return decorator

    def create(self, optimizer, config):
        """
        創建學習率調度器（無條件分支！）

        Args:
            optimizer: PyTorch 優化器
            config: 調度器配置（字典或字串）

        Returns:
            調度器實例或 None
        """
        # 標準化配置格式
        if isinstance(config, str):
            scheduler_type = config.lower()
            config = {'type': scheduler_type, 'max_epochs': 1000}  # 字串轉字典，提供預設值
        elif isinstance(config, dict):
            scheduler_type = config.get('type', 'none')
            scheduler_type = str(scheduler_type).lower()
        else:
            logging.warning(f"⚠️ 無效的調度器配置類型: {type(config)}，使用固定學習率")
            return None

        # 特殊處理：none / constant 類型
        if scheduler_type in {'none', 'constant'}:
            logging.info("未配置學習率調度器，使用固定學習率")
            return None

        # Registry 查找（無條件分支！）
        factory_func = self._registry.get(scheduler_type)

        if factory_func:
            try:
                return factory_func(optimizer, config)
            except Exception as e:
                logging.error(f"創建調度器 '{scheduler_type}' 失敗: {e}")
                logging.info("回退到固定學習率")
                return None

        # 未知類型
        logging.warning(f"未知調度器類型: '{scheduler_type}'，使用固定學習率")
        return None


# 全局單例
_scheduler_factory = SchedulerFactory()


# ============================================================================
# Scheduler Registrations (使用裝飾器註冊)
# ============================================================================

@_scheduler_factory.register('warmup_cosine')
def _create_warmup_cosine(optimizer, config):
    """Warmup + Cosine Annealing 調度器"""
    from pinnx.train.schedulers import WarmupCosineScheduler

    warmup_epochs = config.get('warmup_epochs', 100)
    max_epochs = config.get('max_epochs', 1000)
    base_lr = optimizer.param_groups[0]['lr']  # 從優化器獲取 base_lr
    min_lr = config.get('eta_min', 0.0)  # eta_min 對應 min_lr

    logging.info(
        f"✅ 使用 Warmup Cosine 調度器 "
        f"(warmup={warmup_epochs}, max={max_epochs}, base_lr={base_lr}, min_lr={min_lr})"
    )

    return WarmupCosineScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        base_lr=base_lr,
        min_lr=min_lr
    )


@_scheduler_factory.register('cosine')
def _create_cosine(optimizer, config):
    """Cosine Annealing 調度器"""
    T_max = config.get('max_epochs', 1000)
    eta_min = config.get('eta_min', 0.0)

    logging.info(f"✅ 使用 Cosine Annealing 調度器 (T_max={T_max}, eta_min={eta_min})")

    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=T_max,
        eta_min=eta_min
    )


@_scheduler_factory.register('step')
def _create_step(optimizer, config):
    """Step 調度器（每隔固定步數降低學習率）"""
    step_size = config.get('step_size', 1000)
    gamma = config.get('gamma', 0.5)

    logging.info(f"✅ 使用 Step 調度器 (step_size={step_size}, gamma={gamma})")

    return optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma
    )


@_scheduler_factory.register('exponential')
def _create_exponential(optimizer, config):
    """Exponential 調度器（指數衰減）"""
    gamma = config.get('gamma', 0.95)

    logging.info(f"✅ 使用 Exponential 調度器 (gamma={gamma})")

    return optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=gamma
    )


@_scheduler_factory.register('multistep')
def _create_multistep(optimizer, config):
    """MultiStep 調度器（多階段降低學習率）"""
    milestones = config.get('milestones', [1000, 2000, 3000])
    gamma = config.get('gamma', 0.5)

    logging.info(f"✅ 使用 MultiStep 調度器 (milestones={milestones}, gamma={gamma})")

    return optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=gamma
    )


@_scheduler_factory.register('reduce_on_plateau')
def _create_reduce_on_plateau(optimizer, config):
    """ReduceLROnPlateau 調度器（基於指標自適應降低學習率）"""
    mode = config.get('mode', 'min')
    factor = config.get('factor', 0.5)
    patience = config.get('patience', 10)
    threshold = config.get('threshold', 1e-4)

    logging.info(
        f"✅ 使用 ReduceLROnPlateau 調度器 "
        f"(mode={mode}, factor={factor}, patience={patience})"
    )

    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        threshold=threshold
    )


# ============================================================================
# Optimizer Factory (Registry Pattern)
# ============================================================================

class OptimizerFactory:
    """
    基於 Registry Pattern 的優化器工廠

    優點：
    - 無條件分支（Good Taste）
    - 自動 fallback 到 Adam（容錯機制）
    - 統一的參數處理
    """

    def __init__(self):
        self._registry: Dict[str, Callable] = {}

    def register(self, name: str):
        """註冊裝飾器"""
        def decorator(func: Callable):
            self._registry[name.lower()] = func
            return func
        return decorator

    def create(self, model: nn.Module, config) -> optim.Optimizer:
        """
        創建優化器（無條件分支 + 自動 fallback）

        Args:
            model: PyTorch 模型
            config: 優化器配置（字典或字串）

        Returns:
            優化器實例（失敗時自動回退到 Adam）
        """
        # 標準化配置格式
        if isinstance(config, str):
            optimizer_type = config.lower()
            config = {'type': optimizer_type, 'lr': 1e-3}  # 預設 lr
        elif isinstance(config, dict):
            optimizer_type = config.get('type', 'adam')
            optimizer_type = str(optimizer_type).lower()
        else:
            logging.warning(f"⚠️ 無效的優化器配置類型: {type(config)}，回退到 Adam")
            return _create_adam_fallback(model, {'lr': 1e-3})

        # Registry 查找（無條件分支！）
        factory_func = self._registry.get(optimizer_type)

        if factory_func:
            try:
                return factory_func(model, config)
            except Exception as e:
                logging.warning(f"⚠️ 創建優化器 '{optimizer_type}' 失敗: {e}")
                logging.info("回退到 Adam 優化器")

        # Fallback: Adam
        return _create_adam_fallback(model, config)


# 全局單例
_optimizer_factory = OptimizerFactory()


# ============================================================================
# Optimizer Registrations
# ============================================================================

@_optimizer_factory.register('adam')
def _create_adam(model, config):
    """Adam 優化器"""
    lr = config.get('lr', 1e-3)
    betas = tuple(config.get('betas', (0.9, 0.999)))
    weight_decay = config.get('weight_decay', 0.0)

    logging.info(f"✅ 使用 Adam 優化器（lr={lr}, wd={weight_decay}）")

    return torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=betas,
        weight_decay=weight_decay
    )


@_optimizer_factory.register('adamw')
def _create_adamw(model, config):
    """AdamW 優化器（解耦權重衰減）"""
    lr = config.get('lr', 1e-3)
    betas = tuple(config.get('betas', (0.9, 0.999)))
    weight_decay = config.get('weight_decay', 0.01)

    logging.info(f"✅ 使用 AdamW 優化器（lr={lr}, wd={weight_decay}）")

    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        weight_decay=weight_decay
    )


@_optimizer_factory.register('lbfgs')
def _create_lbfgs(model, config):
    """L-BFGS 優化器（二階優化器）"""
    lr = config.get('lr', 1.0)
    max_iter = config.get('max_iter', 20)
    history_size = config.get('history_size', 100)

    logging.info(
        f"✅ 使用 L-BFGS 優化器 "
        f"(lr={lr}, max_iter={max_iter}, history_size={history_size})"
    )

    return torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=max_iter,
        history_size=history_size
    )


@_optimizer_factory.register('soap')
def _create_soap(model, config):
    """SOAP 優化器（Shampoo + Adam 混合優化器）"""
    try:
        from pinnx.optim.soap import SOAP
    except ImportError:
        logging.error("SOAP 優化器未安裝，請檢查 pinnx.optim.soap 模組")
        raise

    lr = config.get('lr', 1e-3)
    betas = tuple(config.get('betas', (0.95, 0.95)))
    shampoo_beta = config.get('shampoo_beta', -1)
    weight_decay = config.get('weight_decay', 0.01)

    logging.info(
        f"✅ 使用 SOAP 優化器 "
        f"(lr={lr}, betas={betas}, shampoo_beta={shampoo_beta}, wd={weight_decay})"
    )

    return SOAP(
        model.parameters(),
        lr=lr,
        betas=betas,
        shampoo_beta=shampoo_beta,
        weight_decay=weight_decay
    )


@_optimizer_factory.register('sgd')
def _create_sgd(model, config):
    """SGD 優化器（隨機梯度下降）"""
    lr = config.get('lr', 1e-2)
    momentum = config.get('momentum', 0.9)
    weight_decay = config.get('weight_decay', 0.0)
    nesterov = config.get('nesterov', False)

    logging.info(
        f"✅ 使用 SGD 優化器 "
        f"(lr={lr}, momentum={momentum}, nesterov={nesterov}, wd={weight_decay})"
    )

    return torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov
    )


def _create_adam_fallback(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """
    Fallback 優化器：Adam（當其他優化器失敗時使用）

    Args:
        model: PyTorch 模型
        config: 配置字典

    Returns:
        Adam 優化器
    """
    lr = config.get('lr', 1e-3)
    weight_decay = config.get('weight_decay', 0.0)

    logging.warning(
        f"⚠️ 使用 Adam 作為 fallback 優化器（lr={lr}, wd={weight_decay}）"
    )

    return torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )


# ============================================================================
# Public API (便捷函數)
# ============================================================================

def create_optimizer(
    model: nn.Module,
    config: Dict[str, Any]
) -> optim.Optimizer:
    """
    便捷函數：創建優化器

    Args:
        model: PyTorch 模型
        config: 優化器配置字典
            - type: 優化器類型 ('adam', 'adamw', 'lbfgs', 'soap', 'sgd')
            - lr: 學習率
            - weight_decay: 權重衰減
            - ... 其他參數

    Returns:
        優化器實例

    Examples:
        >>> optimizer = create_optimizer(model, {'type': 'adam', 'lr': 1e-3})
        >>> optimizer = create_optimizer(model, {'type': 'soap', 'lr': 1e-3, 'shampoo_beta': 0.9})
    """
    return _optimizer_factory.create(model, config)


def create_scheduler(
    optimizer,
    config
) -> Optional[Any]:
    """
    便捷函數：創建學習率調度器

    Args:
        optimizer: PyTorch 優化器
        config: 調度器配置（字典或字串）
            - type: 調度器類型 ('cosine', 'step', 'warmup_cosine', etc.)
            - ... 其他參數

    Returns:
        調度器實例或 None

    Examples:
        >>> scheduler = create_scheduler(optimizer, {'type': 'cosine', 'max_epochs': 1000})
        >>> scheduler = create_scheduler(optimizer, {'type': 'step', 'step_size': 500, 'gamma': 0.5})
        >>> scheduler = create_scheduler(optimizer, 'none')  # 返回 None
    """
    return _scheduler_factory.create(optimizer, config)


# ============================================================================
# Registry 查詢工具（調試用）
# ============================================================================

def list_available_optimizers():
    """列出所有已註冊的優化器類型"""
    return sorted(_optimizer_factory._registry.keys())


def list_available_schedulers():
    """列出所有已註冊的調度器類型"""
    return sorted(_scheduler_factory._registry.keys())


if __name__ == "__main__":
    # 測試註冊的類型
    print("Available Optimizers:", list_available_optimizers())
    print("Available Schedulers:", list_available_schedulers())
