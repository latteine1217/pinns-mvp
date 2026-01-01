"""
訓練循環輔助模組
提供訓練過程中的自適應機制與監控功能
"""

import logging
from typing import Dict, Any, Optional, Tuple, Callable
import torch
import torch.nn as nn
import numpy as np

from pinnx.train.adaptive_collocation import AdaptiveCollocationSampler

logger = logging.getLogger(__name__)


class TrainingLoopManager:
    """
    訓練循環管理器
    
    職責：
    1. 管理自適應殘差點採樣器
    2. 協調動態權重調整
    3. 監控訓練進度與診斷
    4. 處理訓練數據更新（殘差點、監督點追加等）
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 完整訓練配置
        """
        self.config = config
        
        # 自適應殘差點採樣器（如果啟用）
        adaptive_cfg = config.get('sampling', {}).get('adaptive_collocation', {})
        if adaptive_cfg.get('enabled', False):
            self.adaptive_sampler = AdaptiveCollocationSampler(adaptive_cfg)
            logger.info("✅ 自適應殘差點採樣器已啟用")
        else:
            self.adaptive_sampler = None
            logger.info("⚠️  自適應殘差點採樣器已禁用")
        
        # 新點權重衰減配置
        new_point_cfg = adaptive_cfg.get('new_point_weighting', {})
        self.new_point_weight_enabled = new_point_cfg.get('enabled', True)
        self.new_point_initial_weight = new_point_cfg.get('initial_weight', 2.5)
        self.new_point_decay_epochs = new_point_cfg.get('decay_epochs', 500)
        self.new_point_final_weight = new_point_cfg.get('final_weight', 1.0)
        
        # 內部狀態
        self.current_pde_points = None
        self.new_point_mask = None  # 標記哪些點是新加入的
        self.last_resample_epoch = 0
        self.resample_count = 0
        
        # 監控統計
        self.epoch_stats = []
    
    def setup_initial_points(self, pde_points: torch.Tensor):
        """
        設置初始殘差點（訓練開始時調用一次）
        
        Args:
            pde_points: 初始 PDE 殘差點 [N_pde, dim]
        """
        self.current_pde_points = pde_points.clone()
        self.new_point_mask = torch.zeros(len(pde_points), dtype=torch.bool)
        logger.info(f"✅ 初始化殘差點: {len(pde_points)} 個點")
    
    def should_resample_collocation_points(self,
                                          epoch: int,
                                          current_loss: float,
                                          residuals: Optional[torch.Tensor] = None) -> bool:
        """
        判斷是否應該重新採樣殘差點
        
        Args:
            epoch: 當前 epoch
            current_loss: 當前總損失
            residuals: 當前 PDE 殘差 [N_pde, n_equations]
            
        Returns:
            是否觸發重採樣
        """
        if self.adaptive_sampler is None:
            return False
        
        return self.adaptive_sampler.should_trigger(epoch, current_loss, residuals)
    
    def resample_collocation_points(self,
                                   model: nn.Module,
                                   physics_module: Any,
                                   domain_bounds: Dict[str, Tuple[float, float]],
                                   epoch: int,
                                   device: str = 'cpu') -> Tuple[torch.Tensor, Dict]:
        """
        執行殘差點重採樣
        
        Args:
            model: PINNs 模型
            physics_module: 物理方程模組（用於計算殘差）
            domain_bounds: 域邊界
            epoch: 當前 epoch
            device: 計算設備
            
        Returns:
            (new_pde_points, metrics)
        """
        if self.adaptive_sampler is None:
            raise RuntimeError("自適應採樣器未啟用")
        
        if self.current_pde_points is None:
            raise RuntimeError("尚未初始化殘差點，請先調用 setup_initial_points()")
        
        logger.info("=" * 80)
        logger.info(f"🔄 觸發殘差點重採樣 (Epoch {epoch})")
        logger.info("=" * 80)
        
        # 定義殘差計算函數（閉包）
        def residual_fn(points: torch.Tensor) -> torch.Tensor:
            """
            計算給定點的 PDE 殘差
            
            Args:
                points: 輸入點 [N, dim]
                
            Returns:
                residuals: PDE 殘差 [N, n_equations]
            """
            with torch.no_grad():
                points = points.to(device)
                points.requires_grad_(True)
                
                # 模型預測
                u_pred = model(points)
                
                # 根據維度判斷是 2D 還是 3D
                if points.shape[1] == 2:  # 2D (x, y)
                    coords = points
                    velocity = u_pred[:, :2]  # u, v
                    pressure = u_pred[:, 2:3]  # p
                elif points.shape[1] == 3:  # 3D (x, y, z)
                    coords = points
                    velocity = u_pred[:, :3]  # u, v, w
                    pressure = u_pred[:, 3:4]  # p
                else:
                    raise ValueError(f"不支援的點維度: {points.shape[1]}")
                
                # 計算 PDE 殘差
                residual_dict = physics_module.residual(coords, velocity, pressure)
                
                # 組合所有殘差項到單一張量
                # 2D: [momentum_x, momentum_y, continuity]
                # 3D: [momentum_x, momentum_y, momentum_z, continuity]
                residual_list = []
                for key in sorted(residual_dict.keys()):
                    residual_list.append(residual_dict[key])
                
                residuals = torch.stack(residual_list, dim=1)  # [N, n_equations]
                
                return residuals
        
        # 執行重採樣
        new_points, metrics = self.adaptive_sampler.resample_collocation_points(
            current_points=self.current_pde_points,
            domain_bounds=domain_bounds,
            residual_fn=residual_fn,
            n_keep=None,  # 自動計算
            device=device
        )
        
        # 更新內部狀態
        n_old = len(self.current_pde_points)
        n_new = len(new_points)
        n_replaced = metrics['n_replaced']
        
        # 創建新點掩碼（標記新加入的點）
        new_mask = torch.zeros(n_new, dtype=torch.bool)
        new_mask[-n_replaced:] = True  # 最後 n_replaced 個點是新的
        
        self.current_pde_points = new_points
        self.new_point_mask = new_mask
        self.last_resample_epoch = epoch
        self.resample_count += 1
        
        logger.info(f"✅ 殘差點重採樣完成:")
        logger.info(f"   舊點數: {n_old}")
        logger.info(f"   新點數: {n_new}")
        logger.info(f"   保留: {metrics['n_kept']}, 替換: {n_replaced}")
        logger.info(f"   SVD 秩: {metrics.get('svd_rank', 'N/A')}")
        logger.info(f"   能量保留: {metrics.get('svd_energy_ratio', 'N/A'):.4f}" if 'svd_energy_ratio' in metrics else "")
        logger.info("=" * 80)
        
        return new_points, metrics
    
    def get_point_weights(self, epoch: int) -> Optional[torch.Tensor]:
        """
        獲取當前殘差點的權重（新點權重衰減）
        
        Args:
            epoch: 當前 epoch
            
        Returns:
            weights: 點權重 [N_pde,]，如果未啟用則返回 None
        """
        if not self.new_point_weight_enabled or self.new_point_mask is None:
            return None
        
        # 計算衰減因子（線性衰減）
        epochs_since_resample = epoch - self.last_resample_epoch
        if epochs_since_resample >= self.new_point_decay_epochs:
            # 完全衰減到 final_weight
            decay_factor = 0.0
        else:
            # 線性衰減
            decay_factor = 1.0 - (epochs_since_resample / self.new_point_decay_epochs)
        
        # 計算權重
        weights = torch.ones(len(self.new_point_mask))
        
        # 新點權重 = final + (initial - final) * decay_factor
        new_point_weight = (
            self.new_point_final_weight +
            (self.new_point_initial_weight - self.new_point_final_weight) * decay_factor
        )
        
        weights[self.new_point_mask] = new_point_weight
        
        if epochs_since_resample == 0 or epochs_since_resample % 100 == 0:
            logger.debug(f"新點權重: {new_point_weight:.3f} "
                        f"(衰減進度: {(1-decay_factor)*100:.1f}%)")
        
        return weights
    
    def update_training_batch(self, 
                             data_batch: Dict[str, torch.Tensor],
                             epoch: int) -> Dict[str, torch.Tensor]:
        """
        更新訓練批次（在每個 epoch 開始時調用）
        
        更新內容：
        1. 如果有新的殘差點，更新 coords_pde_spatial（以及 t_pde）
        2. 計算並附加點權重（如果啟用）
        
        Args:
            data_batch: 原始訓練批次
            epoch: 當前 epoch
            
        Returns:
            updated_batch: 更新後的訓練批次
        """
        updated_batch = data_batch.copy()
        
        # 如果有當前殘差點，更新到批次中
        if self.current_pde_points is not None:
            if 'coords_pde_spatial' not in updated_batch:
                raise KeyError("data_batch 缺少必要鍵: coords_pde_spatial")
            spatial_dim = updated_batch['coords_pde_spatial'].shape[1]
            if self.current_pde_points.shape[1] < spatial_dim:
                raise ValueError("current_pde_points 維度不足以更新 coords_pde_spatial")
            updated_batch['coords_pde_spatial'] = self.current_pde_points[:, :spatial_dim]
            if 't_pde' in updated_batch and updated_batch['t_pde'] is not None:
                updated_batch['t_pde'] = self.current_pde_points[:, spatial_dim:]
        
        # 附加點權重
        point_weights = self.get_point_weights(epoch)
        if point_weights is not None:
            updated_batch['pde_point_weights'] = point_weights
        
        return updated_batch
    
    def collect_epoch_stats(self, epoch: int, loss_dict: Dict[str, float]):
        """
        收集每個 epoch 的統計信息
        
        Args:
            epoch: 當前 epoch
            loss_dict: 損失字典
        """
        stats = {
            'epoch': epoch,
            'total_loss': loss_dict.get('total_loss', 0.0),
            'n_pde_points': len(self.current_pde_points) if self.current_pde_points is not None else 0,
            'n_new_points': self.new_point_mask.sum().item() if self.new_point_mask is not None else 0,
            'resample_count': self.resample_count,
        }
        
        self.epoch_stats.append(stats)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        獲取訓練循環管理器的摘要信息
        
        Returns:
            summary: 摘要字典
        """
        return {
            'adaptive_sampler_enabled': self.adaptive_sampler is not None,
            'resample_count': self.resample_count,
            'last_resample_epoch': self.last_resample_epoch,
            'current_n_pde_points': len(self.current_pde_points) if self.current_pde_points is not None else 0,
            'n_new_points': self.new_point_mask.sum().item() if self.new_point_mask is not None else 0,
            'adaptive_sampler_stats': self.adaptive_sampler.get_statistics() if self.adaptive_sampler else {},
        }


def create_training_loop_manager(config: Dict[str, Any]) -> TrainingLoopManager:
    """
    創建訓練循環管理器的便捷函數
    
    Args:
        config: 完整訓練配置
        
    Returns:
        manager: 訓練循環管理器實例
    """
    return TrainingLoopManager(config)


def apply_point_weights_to_loss(loss: torch.Tensor,
                                point_weights: Optional[torch.Tensor]) -> torch.Tensor:
    """
    應用點權重到損失（加權平均）
    
    Args:
        loss: 原始損失 [N_pde,] 或標量
        point_weights: 點權重 [N_pde,]
        
    Returns:
        weighted_loss: 加權損失（標量）
    """
    if point_weights is None:
        # 無權重，直接平均
        return torch.mean(loss)
    
    # 確保形狀匹配
    if loss.dim() == 0:
        # 損失已經是標量，無法應用點權重
        logger.warning("損失已是標量，無法應用點權重")
        return loss
    
    if loss.shape[0] != point_weights.shape[0]:
        logger.warning(f"損失形狀 {loss.shape} 與權重形狀 {point_weights.shape} 不匹配，回退到均勻權重")
        return torch.mean(loss)
    
    # 加權平均
    weighted_loss = torch.sum(loss * point_weights) / torch.sum(point_weights)
    
    return weighted_loss


# ========================================
# 監督點追加邏輯（預留接口）
# ========================================

class SupervisedPointAugmentor:
    """
    監督點追加器
    
    功能：
    - 在訓練過程中動態追加監督點
    - 支援固定週期與手動指定 Epoch 觸發
    - 目前提供均勻採樣追加策略（之後可擴展為殘差導向）
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('enabled', False)
        self.start_epoch = int(config.get('start_epoch', 0))
        self.interval = max(1, int(config.get('interval', 100)))
        self.max_rounds = config.get('max_rounds')
        self.manual_epochs = set(config.get('epochs', []))
        self.strategy = config.get('strategy', 'uniform')
        self.jitter = float(config.get('jitter', 0.0))
        self.default_dim = config.get('dim')
        
        self.last_augment_epoch: Optional[int] = None
        self._pending_epoch: Optional[int] = None
        self.augment_count = 0
        self.bounds = self._parse_bounds(config.get('bounds'))
        
        if self.enabled:
            logger.info(
                "✅ 監督點追加器啟用 | strategy=%s start_epoch=%d interval=%d max_rounds=%s",
                self.strategy,
                self.start_epoch,
                self.interval,
                str(self.max_rounds) if self.max_rounds is not None else "∞"
            )
    
    def _parse_bounds(self, bounds_cfg: Optional[Any]) -> list:
        """解析邊界配置為 [(min, max), ...]"""
        bounds: list = []
        if bounds_cfg is None:
            return bounds
        
        if isinstance(bounds_cfg, dict):
            # 按常見順序讀取
            for key in ['x', 'y', 'z', 't']:
                if key in bounds_cfg:
                    rng = bounds_cfg[key]
                    if isinstance(rng, (list, tuple)) and len(rng) == 2:
                        bounds.append((float(rng[0]), float(rng[1])))
        elif isinstance(bounds_cfg, (list, tuple)):
            for rng in bounds_cfg:
                if isinstance(rng, (list, tuple)) and len(rng) == 2:
                    bounds.append((float(rng[0]), float(rng[1])))
        
        return bounds
    
    def _resolve_bounds(self, dim: int) -> list:
        """根據需求維度取得邊界（不足時使用 [-1, 1] 補齊）"""
        if dim <= 0:
            raise ValueError("維度需大於 0 才能生成監督點")
        
        if not self.bounds:
            return [(-1.0, 1.0)] * dim
        
        if len(self.bounds) >= dim:
            return self.bounds[:dim]
        
        # 邊界不足，使用最後一個或預設區間補齊
        padding = [self.bounds[-1]] if self.bounds else [(-1.0, 1.0)]
        while len(self.bounds) + len(padding) < dim:
            padding.append(padding[-1])
        
        return self.bounds + padding[: dim - len(self.bounds)]
    
    def _can_augment(self) -> bool:
        if not self.enabled:
            return False
        if self.max_rounds is not None and self.augment_count >= self.max_rounds:
            return False
        return True
    
    def should_augment(self, epoch: int) -> bool:
        """判斷是否應該追加監督點"""
        if not self._can_augment():
            return False
        if epoch < self.start_epoch:
            return False
        if epoch in self.manual_epochs:
            self._pending_epoch = epoch
            return True
        
        if self.last_augment_epoch is None:
            due_epoch = self.start_epoch
        else:
            due_epoch = self.last_augment_epoch + self.interval
        
        if epoch >= due_epoch:
            self._pending_epoch = epoch
            return True
        
        return False
    
    def augment_supervised_points(self, 
                                  current_points: torch.Tensor,
                                  model: nn.Module,
                                  n_augment: int) -> torch.Tensor:
        """
        追加監督點
        
        Args:
            current_points: 當前監督點
            model: PINNs 模型（保留接口，未使用）
            n_augment: 追加數量
            
        Returns:
            augmented_points: 追加後的監督點
        """
        if not self._can_augment() or n_augment <= 0:
            return current_points
        
        dim = current_points.shape[1] if current_points.numel() > 0 else self.default_dim
        if dim is None:
            raise ValueError("無法推斷監督點維度，請在配置中設定 dim 或提供初始點")
        
        bounds = self._resolve_bounds(dim)
        device = current_points.device
        dtype = current_points.dtype
        
        samples = []
        for low, high in bounds:
            rand = torch.rand(n_augment, 1, device=device, dtype=dtype)
            samples.append(rand * (high - low) + low)
        
        new_points = torch.cat(samples, dim=1)
        
        if self.jitter > 0.0 and current_points.numel() > 0:
            noise = torch.randn_like(new_points) * self.jitter
            new_points = new_points + noise
            for dim_idx, (low, high) in enumerate(bounds):
                new_points[:, dim_idx].clamp_(min=low, max=high)
        
        augmented = torch.cat([current_points, new_points], dim=0)
        
        self.augment_count += 1
        if self._pending_epoch is not None:
            self.last_augment_epoch = self._pending_epoch
            self._pending_epoch = None
        
        logger.info("📈 已追加 %d 個監督點 (總數: %d)", n_augment, augmented.shape[0])
        
        return augmented


if __name__ == "__main__":
    # 單元測試
    print("🧪 測試訓練循環管理器...")
    
    # 模擬配置
    config = {
        'sampling': {
            'adaptive_collocation': {
                'enabled': True,
                'trigger': {
                    'method': 'epoch_interval',
                    'epoch_interval': 1000,
                },
                'resampling_strategy': 'incremental_replace',
                'incremental_replace': {
                    'keep_ratio': 0.7,
                    'replace_ratio': 0.3,
                },
                'residual_qr': {
                    'enabled': True,
                    'candidate_pool_size': 1000,
                },
                'new_point_weighting': {
                    'enabled': True,
                    'initial_weight': 2.5,
                    'decay_epochs': 500,
                    'final_weight': 1.0,
                }
            }
        }
    }
    
    manager = TrainingLoopManager(config)
    print(f"✅ 管理器創建成功")
    print(f"   自適應採樣器: {'啟用' if manager.adaptive_sampler else '禁用'}")
    
    # 測試初始化點
    initial_points = torch.rand(100, 2)
    manager.setup_initial_points(initial_points)
    print(f"✅ 初始化 {len(initial_points)} 個點")
    
    # 測試觸發檢查
    for epoch in range(0, 3000, 500):
        triggered = manager.should_resample_collocation_points(
            epoch, current_loss=0.1 * np.exp(-epoch/1000)
        )
        if triggered:
            print(f"✅ Epoch {epoch}: 觸發重採樣")
    
    # 測試點權重
    print("\n測試新點權重衰減...")
    manager.last_resample_epoch = 1000
    manager.new_point_mask = torch.zeros(100, dtype=torch.bool)
    manager.new_point_mask[-30:] = True  # 最後 30 個是新點
    
    for epoch in [1000, 1250, 1500, 1750, 2000]:
        weights = manager.get_point_weights(epoch)
        if weights is not None:
            print(f"  Epoch {epoch}: 新點權重={weights[manager.new_point_mask].mean():.3f}, "
                  f"舊點權重={weights[~manager.new_point_mask].mean():.3f}")
    
    # 測試摘要
    print("\n訓練循環管理器摘要:")
    summary = manager.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n✅ 訓練循環管理器測試完成！")
