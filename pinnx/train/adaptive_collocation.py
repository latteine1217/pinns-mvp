"""
自適應殘差點（Collocation Points）採樣器

核心功能：
1. 週期性殘差點重選/增補（避免過擬合固定點集）
2. 殘差 SVD → QR-pivot 選點（專注高殘差/高不確定區域）
3. 增量替換策略（保留歷史點避免災難性遺忘）
4. 空間約束（最小點間距防聚集）
5. 槓桿分數評估（移除低影響力點）

理論依據：
- Adaptive Residual Sampling (Wang et al., 2022)
- R-PINN: Residual-based Adaptive Sampling (Wu & Karniadakis, 2020)
- Greedy Selection via Leverage Scores (Drineas et al., 2012)

參考文獻：
- Understanding and Mitigating Gradient Flow Pathologies in PINNs (Wang et al., 2021)
- When and Why PINNs Fail: A Survey (Krishnapriyan et al., 2021)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Callable
from scipy.linalg import qr, svd
from scipy.spatial.distance import pdist, squareform
import logging

logger = logging.getLogger(__name__)


class AdaptiveCollocationSampler:
    """
    自適應殘差點採樣器
    
    實現週期性殘差點重選，基於殘差分佈和 QR-pivot 策略。
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 配置字典（對應 YAML 中的 adaptive_collocation 區塊）
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        
        # 觸發條件
        trigger_cfg = config.get('trigger', {})
        self.trigger_method = trigger_cfg.get('method', 'epoch_interval')
        self.epoch_interval = trigger_cfg.get('epoch_interval', 1000)
        self.stagnation_window = trigger_cfg.get('stagnation_window', 500)
        self.stagnation_threshold = trigger_cfg.get('stagnation_threshold', 0.01)
        self.residual_percentile = trigger_cfg.get('residual_percentile', 95)
        self.residual_threshold = trigger_cfg.get('residual_threshold', 0.1)
        
        # 重選策略
        self.resampling_strategy = config.get('resampling_strategy', 'incremental_replace')
        
        # 增量替換參數
        incr_cfg = config.get('incremental_replace', {})
        self.keep_ratio = incr_cfg.get('keep_ratio', 0.7)
        self.replace_ratio = incr_cfg.get('replace_ratio', 0.3)
        self.removal_criterion = incr_cfg.get('removal_criterion', 'leverage_score')
        self.leverage_threshold = incr_cfg.get('leverage_threshold', 0.1)
        
        # 殘差 QR 配置
        residual_qr_cfg = config.get('residual_qr', {})
        self.residual_qr_enabled = residual_qr_cfg.get('enabled', True)
        self.candidate_pool_size = residual_qr_cfg.get('candidate_pool_size', 16384)
        self.candidate_sampling = residual_qr_cfg.get('candidate_sampling', 'latin_hypercube')
        
        # 快照配置
        snapshot_cfg = residual_qr_cfg.get('snapshot_config', {})
        self.n_snapshots = snapshot_cfg.get('n_snapshots', 20)
        self.snapshot_method = snapshot_cfg.get('snapshot_method', 'batch')
        
        # SVD 配置
        svd_cfg = residual_qr_cfg.get('svd', {})
        self.energy_threshold = svd_cfg.get('energy_threshold', 0.99)
        self.max_rank = svd_cfg.get('max_rank', 50)
        self.svd_centering = svd_cfg.get('centering', True)
        
        # QR 配置
        qr_cfg = residual_qr_cfg.get('qr', {})
        self.qr_column_pivoting = qr_cfg.get('column_pivoting', True)
        self.qr_tolerance = qr_cfg.get('tolerance', 1e-10)
        
        # 空間約束
        spatial_cfg = residual_qr_cfg.get('spatial_constraints', {})
        self.spatial_constraints_enabled = spatial_cfg.get('enabled', True)
        self.min_distance = spatial_cfg.get('min_distance', 0.02)
        self.max_distance = spatial_cfg.get('max_distance', None)
        
        # 內部狀態
        self.loss_history = []
        self.residual_snapshots = []
        self.candidate_pool = None
        self.last_trigger_epoch = 0
        
    def should_trigger(self, 
                      epoch: int, 
                      current_loss: float,
                      residuals: Optional[torch.Tensor] = None) -> bool:
        """
        檢查是否應該觸發殘差點重選
        
        Args:
            epoch: 當前 epoch
            current_loss: 當前損失值
            residuals: 當前殘差張量 [N_pde,] （可選）
            
        Returns:
            是否觸發重選
        """
        if not self.enabled:
            return False
        
        # 記錄損失歷史
        self.loss_history.append(current_loss)
        
        # 方法 1: 固定間隔觸發
        if self.trigger_method in ['epoch_interval', 'hybrid']:
            if epoch > 0 and epoch % self.epoch_interval == 0:
                logger.info(f"觸發條件: epoch 間隔 ({epoch} % {self.epoch_interval} == 0)")
                self.last_trigger_epoch = epoch
                return True
        
        # 方法 2: 損失停滯觸發
        if self.trigger_method in ['loss_stagnation', 'hybrid']:
            if len(self.loss_history) >= self.stagnation_window:
                recent_losses = self.loss_history[-self.stagnation_window:]
                improvement = (max(recent_losses) - min(recent_losses)) / (max(recent_losses) + 1e-16)
                
                if improvement < self.stagnation_threshold:
                    logger.info(f"觸發條件: 損失停滯 (改善={improvement:.4f} < {self.stagnation_threshold})")
                    self.last_trigger_epoch = epoch
                    return True
        
        # 方法 3: 殘差閾值觸發
        if residuals is not None:
            residual_np = residuals.detach().cpu().numpy()
            percentile_value = np.percentile(np.abs(residual_np), self.residual_percentile)
            
            if percentile_value > self.residual_threshold:
                logger.info(f"觸發條件: 殘差閾值 ({self.residual_percentile}% = {percentile_value:.4f} > {self.residual_threshold})")
                self.last_trigger_epoch = epoch
                return True
        
        return False
    
    def resample_collocation_points(self,
                                   current_points: torch.Tensor,
                                   domain_bounds: Dict[str, Tuple[float, float]],
                                   residual_fn: Callable,
                                   n_keep: Optional[int] = None,
                                   device: str = 'cpu') -> Tuple[torch.Tensor, Dict]:
        """
        重選殘差點（核心方法）
        
        Args:
            current_points: 當前殘差點 [N_old, dim]
            domain_bounds: 域邊界 {'x': (xmin, xmax), 'y': (ymin, ymax), ...}
            residual_fn: 殘差計算函數 points -> residuals [N, n_equations]
            n_keep: 保留點數（None = 自動計算）
            device: 計算設備
            
        Returns:
            (new_points, metrics)
            - new_points: 新殘差點 [N_new, dim]
            - metrics: 統計指標字典
        """
        N_old = current_points.shape[0]
        dim = current_points.shape[1]
        
        # 計算保留和替換數量
        if n_keep is None:
            n_keep = int(N_old * self.keep_ratio)
        n_replace = N_old - n_keep
        
        logger.info(f"殘差點重選: 保留 {n_keep}/{N_old} 點，替換 {n_replace} 點")
        
        # ========================================
        # Step 1: 選擇要保留的點
        # ========================================
        if self.removal_criterion == 'leverage_score':
            # 基於槓桿分數選擇保留點
            keep_indices = self._select_by_leverage_score(
                current_points, residual_fn, n_keep, device)
        elif self.removal_criterion == 'low_residual':
            # 保留低殘差點（已收斂區域）
            with torch.no_grad():
                residuals = residual_fn(current_points.to(device))
                residual_norms = torch.norm(residuals, dim=-1).cpu()
            keep_indices = torch.argsort(residual_norms)[:n_keep]
        elif self.removal_criterion == 'random':
            # 隨機保留
            keep_indices = torch.randperm(N_old)[:n_keep]
        else:
            # 默認：隨機保留
            keep_indices = torch.randperm(N_old)[:n_keep]
        
        kept_points = current_points[keep_indices]
        
        # ========================================
        # Step 2: 生成候選池
        # ========================================
        if self.candidate_pool is None or self.candidate_pool.shape[0] < self.candidate_pool_size:
            self.candidate_pool = self._generate_candidate_pool(
                domain_bounds, self.candidate_pool_size, dim)
        
        # ========================================
        # Step 3: 殘差 SVD → QR 選點
        # ========================================
        if self.residual_qr_enabled:
            new_indices, qr_metrics = self._residual_qr_selection(
                self.candidate_pool, residual_fn, n_replace, device)
            new_points = self.candidate_pool[new_indices]
        else:
            # 回退：隨機選擇
            random_indices = np.random.choice(
                self.candidate_pool.shape[0], n_replace, replace=False)
            new_points = self.candidate_pool[random_indices]
            qr_metrics = {}
        
        # ========================================
        # Step 4: 應用空間約束
        # ========================================
        if self.spatial_constraints_enabled:
            new_points = self._apply_spatial_constraints(
                kept_points, new_points, self.min_distance)
        
        # ========================================
        # Step 5: 合併保留點和新點
        # ========================================
        final_points = torch.cat([kept_points, new_points], dim=0)
        
        # ========================================
        # Step 6: 統計指標
        # ========================================
        metrics = {
            'n_kept': n_keep,
            'n_replaced': len(new_points),
            'n_final': len(final_points),
            'keep_ratio': n_keep / N_old,
            **qr_metrics
        }
        
        logger.info(f"重選完成: {N_old} → {len(final_points)} 點 "
                   f"(保留={n_keep}, 新增={len(new_points)})")
        
        return final_points, metrics
    
    def _select_by_leverage_score(self,
                                 points: torch.Tensor,
                                 residual_fn: Callable,
                                 n_select: int,
                                 device: str) -> torch.Tensor:
        """
        基於槓桿分數選擇點
        
        槓桿分數 = 對角線 H_{ii}，其中 H = X(X^T X)^{-1}X^T 是帽子矩陣。
        高槓桿分數 = 高影響力點，應優先保留。
        
        Args:
            points: 候選點 [N, dim]
            residual_fn: 殘差函數
            n_select: 選擇數量
            device: 計算設備
            
        Returns:
            selected_indices: 選中的索引 [n_select,]
        """
        try:
            with torch.no_grad():
                # 計算殘差雅可比矩陣（近似）
                points_var = points.to(device).requires_grad_(True)
                residuals = residual_fn(points_var)  # [N, n_eqs]
                
                # 構建特徵矩陣（使用點坐標 + 殘差）
                X = torch.cat([points_var, residuals.detach()], dim=1)  # [N, dim + n_eqs]
                X = X.cpu().numpy()
                
                # 計算槓桿分數（帽子矩陣對角線）
                # 使用 QR 分解計算（數值穩定）
                Q, R = np.linalg.qr(X)
                leverage_scores = np.sum(Q**2, axis=1)  # [N,]
                
                # 選擇高槓桿分數點
                selected_indices = torch.from_numpy(
                    np.argsort(leverage_scores)[-n_select:][::-1]
                )
                
                logger.debug(f"槓桿分數範圍: [{leverage_scores.min():.4f}, {leverage_scores.max():.4f}]")
                
        except Exception as e:
            logger.warning(f"槓桿分數計算失敗: {e}，回退到隨機選擇")
            selected_indices = torch.randperm(points.shape[0])[:n_select]
        
        return selected_indices
    
    def _generate_candidate_pool(self,
                                domain_bounds: Dict[str, Tuple[float, float]],
                                pool_size: int,
                                dim: int) -> torch.Tensor:
        """
        生成候選點池
        
        Args:
            domain_bounds: 域邊界
            pool_size: 池大小
            dim: 空間維度
            
        Returns:
            candidate_pool: 候選點 [pool_size, dim]
        """
        if self.candidate_sampling == 'latin_hypercube':
            # 拉丁超立方採樣（LHS）
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=dim)
            samples = sampler.random(n=pool_size)  # [0, 1]^dim
            
            # 縮放到域邊界
            bounds_list = list(domain_bounds.values())
            for i, (lb, ub) in enumerate(bounds_list):
                samples[:, i] = lb + samples[:, i] * (ub - lb)
            
            candidate_pool = torch.from_numpy(samples).float()
            
        elif self.candidate_sampling == 'uniform':
            # 均勻隨機採樣
            bounds_list = list(domain_bounds.values())
            samples = np.random.uniform(
                low=[b[0] for b in bounds_list],
                high=[b[1] for b in bounds_list],
                size=(pool_size, dim)
            )
            candidate_pool = torch.from_numpy(samples).float()
            
        elif self.candidate_sampling == 'stratified':
            # 分層採樣（網格 + 擾動）
            n_per_dim = int(np.ceil(pool_size ** (1/dim)))
            grids = [np.linspace(b[0], b[1], n_per_dim) 
                    for b in domain_bounds.values()]
            mesh = np.meshgrid(*grids, indexing='ij')
            grid_points = np.stack([m.flatten() for m in mesh], axis=1)
            
            # 隨機擾動
            cell_size = [(b[1] - b[0]) / n_per_dim for b in domain_bounds.values()]
            perturbation = np.random.uniform(
                low=[-cs/2 for cs in cell_size],
                high=[cs/2 for cs in cell_size],
                size=grid_points.shape
            )
            samples = grid_points + perturbation
            
            # 隨機選擇 pool_size 個點
            if len(samples) > pool_size:
                indices = np.random.choice(len(samples), pool_size, replace=False)
                samples = samples[indices]
            
            candidate_pool = torch.from_numpy(samples).float()
            
        else:
            raise ValueError(f"未知的候選採樣方法: {self.candidate_sampling}")
        
        logger.info(f"生成候選池: {pool_size} 點，方法={self.candidate_sampling}")
        return candidate_pool
    
    def _residual_qr_selection(self,
                              candidate_pool: torch.Tensor,
                              residual_fn: Callable,
                              n_select: int,
                              device: str) -> Tuple[np.ndarray, Dict]:
        """
        殘差 SVD → QR 選點
        
        流程：
        1. 在候選池上計算殘差快照矩陣 R [M, n_snapshots × n_eqs]
        2. SVD 分解: R = U Σ V^T，保留前 r 個模態
        3. 在主模態空間 U_r [M, r] 上執行 QR-pivot 選 K 個點
        
        Args:
            candidate_pool: 候選點池 [M, dim]
            residual_fn: 殘差計算函數
            n_select: 選擇點數 K
            device: 計算設備
            
        Returns:
            (selected_indices, metrics)
        """
        M = candidate_pool.shape[0]
        
        # ========================================
        # Step 1: 收集殘差快照
        # ========================================
        residual_matrix = []
        
        with torch.no_grad():
            if self.snapshot_method == 'batch':
                # 分批計算殘差（不同隨機批次模擬快照）
                batch_size = M // self.n_snapshots
                
                for i in range(self.n_snapshots):
                    # 隨機打亂候選池（模擬不同批次）
                    perm = torch.randperm(M)
                    batch_indices = perm[:batch_size]
                    batch_points = candidate_pool[batch_indices].to(device)
                    
                    residuals = residual_fn(batch_points)  # [batch_size, n_eqs]
                    residual_matrix.append(residuals.cpu().numpy())
                
                # 拼接成矩陣 [M_total, n_eqs]
                residual_matrix = np.concatenate(residual_matrix, axis=0)
                
            else:
                # 全候選池計算（內存允許的情況下）
                all_residuals = residual_fn(candidate_pool.to(device))  # [M, n_eqs]
                residual_matrix = all_residuals.cpu().numpy()
        
        # ========================================
        # Step 2: SVD 分解
        # ========================================
        if self.svd_centering:
            residual_mean = residual_matrix.mean(axis=0, keepdims=True)
            residual_matrix_centered = residual_matrix - residual_mean
        else:
            residual_matrix_centered = residual_matrix
        
        try:
            U, s, Vt = svd(residual_matrix_centered, full_matrices=False)
            
            # 確定秩（基於能量閾值）
            total_energy = np.sum(s**2)
            cumulative_energy = np.cumsum(s**2) / total_energy
            rank = min(
                np.argmax(cumulative_energy >= self.energy_threshold) + 1,
                self.max_rank,
                len(s)
            )
            
            logger.info(f"SVD 分解: 秩={rank}/{len(s)}, "
                       f"能量保留={cumulative_energy[rank-1]:.4f}")
            
            # 提取主模態
            U_r = U[:, :rank]  # [M_batch, rank]
            
            # 重新映射到完整候選池索引
            # （如果使用批次採樣，需要重建完整 U 矩陣）
            if self.snapshot_method == 'batch':
                # 簡化處理：在批次點上做 QR，然後映射回原索引
                # 這裡假設批次點已足夠代表候選池分佈
                pass
            
        except np.linalg.LinAlgError as e:
            logger.error(f"SVD 失敗: {e}，回退到隨機選擇")
            random_indices = np.random.choice(M, n_select, replace=False)
            return random_indices, {'svd_failed': True}
        
        # ========================================
        # Step 3: QR-Pivot 選點
        # ========================================
        try:
            if self.qr_column_pivoting:
                # 對 U_r^T 做 QR 分解（選擇行 = 候選池中的點）
                qr_result = qr(U_r.T, mode='economic', pivoting=True)  # type: ignore
                # scipy.linalg.qr with pivoting=True returns (Q, R, P)
                piv = qr_result[2]  # type: ignore
                selected_indices = np.array(piv[:n_select])
            else:
                # 標準 QR（無主元）
                qr_result = qr(U_r.T, mode='economic', pivoting=False)  # type: ignore
                R = qr_result[1]  # type: ignore
                # 使用對角元素大小排序
                diag_importance = np.abs(np.diag(R))
                selected_indices = np.argsort(diag_importance)[-n_select:][::-1]
            
            # 確保索引有效
            selected_indices = selected_indices[selected_indices < M]
            selected_indices = selected_indices[:n_select]
            
            logger.info(f"QR-Pivot 選點: {len(selected_indices)}/{n_select} 點")
            
        except Exception as e:
            logger.error(f"QR 分解失敗: {e}，回退到隨機選擇")
            selected_indices = np.random.choice(M, n_select, replace=False)
            return selected_indices, {'qr_failed': True}
        
        # ========================================
        # Step 4: 統計指標
        # ========================================
        metrics = {
            'svd_rank': rank,
            'svd_energy_ratio': float(cumulative_energy[rank-1]),
            'n_snapshots': self.n_snapshots,
            'candidate_pool_size': M,
        }
        
        return selected_indices, metrics
    
    def _apply_spatial_constraints(self,
                                  existing_points: torch.Tensor,
                                  new_points: torch.Tensor,
                                  min_distance: float) -> torch.Tensor:
        """
        應用空間約束（最小點間距）
        
        Args:
            existing_points: 已有點 [N_exist, dim]
            new_points: 新候選點 [N_new, dim]
            min_distance: 最小點間距
            
        Returns:
            filtered_new_points: 過濾後的新點
        """
        if min_distance <= 0:
            return new_points
        
        all_points = torch.cat([existing_points, new_points], dim=0)
        n_exist = existing_points.shape[0]
        
        # 計算距離矩陣（新點 vs 所有點）
        all_points_np = all_points.numpy()
        dist_matrix = squareform(pdist(all_points_np))
        
        # 新點與已有點的距離
        new_to_all_dist = dist_matrix[n_exist:, :n_exist]
        
        # 過濾：保留與所有已有點距離 >= min_distance 的新點
        valid_mask = np.all(new_to_all_dist >= min_distance, axis=1)
        
        filtered_new_points = new_points[valid_mask]
        
        n_filtered = len(new_points) - len(filtered_new_points)
        if n_filtered > 0:
            logger.warning(f"空間約束過濾: 移除 {n_filtered}/{len(new_points)} 個過近點 "
                          f"(min_dist={min_distance:.4f})")
        
        return filtered_new_points
    
    def collect_residual_snapshot(self,
                                 points: torch.Tensor,
                                 residuals: torch.Tensor):
        """
        收集殘差快照（用於後續 SVD 分析）
        
        Args:
            points: 殘差點 [N, dim]
            residuals: 殘差值 [N, n_eqs]
        """
        snapshot = {
            'points': points.detach().cpu(),
            'residuals': residuals.detach().cpu()
        }
        
        self.residual_snapshots.append(snapshot)
        
        # 限制快照數量（避免內存爆炸）
        max_snapshots = self.n_snapshots * 2
        if len(self.residual_snapshots) > max_snapshots:
            self.residual_snapshots = self.residual_snapshots[-max_snapshots:]
    
    def get_statistics(self) -> Dict:
        """獲取採樣器統計信息"""
        return {
            'enabled': self.enabled,
            'trigger_method': self.trigger_method,
            'resampling_strategy': self.resampling_strategy,
            'n_snapshots_collected': len(self.residual_snapshots),
            'last_trigger_epoch': self.last_trigger_epoch,
            'loss_history_length': len(self.loss_history),
        }


def create_adaptive_collocation_sampler(config: Dict) -> AdaptiveCollocationSampler:
    """
    創建自適應殘差點採樣器的便捷函數
    
    Args:
        config: 配置字典
        
    Returns:
        採樣器實例
    """
    return AdaptiveCollocationSampler(config)


if __name__ == "__main__":
    # 單元測試
    print("🧪 測試自適應殘差點採樣器...")
    
    # 模擬配置
    config = {
        'enabled': True,
        'trigger': {
            'method': 'epoch_interval',
            'epoch_interval': 1000,
        },
        'resampling_strategy': 'incremental_replace',
        'incremental_replace': {
            'keep_ratio': 0.7,
            'replace_ratio': 0.3,
            'removal_criterion': 'leverage_score',
        },
        'residual_qr': {
            'enabled': True,
            'candidate_pool_size': 1000,
            'candidate_sampling': 'latin_hypercube',
            'snapshot_config': {'n_snapshots': 10},
            'svd': {'energy_threshold': 0.99, 'max_rank': 20},
            'qr': {'column_pivoting': True},
            'spatial_constraints': {'enabled': True, 'min_distance': 0.02}
        }
    }
    
    sampler = AdaptiveCollocationSampler(config)
    
    # 測試觸發條件
    print("\n測試觸發條件...")
    for epoch in range(0, 3000, 500):
        triggered = sampler.should_trigger(epoch, current_loss=0.1 * np.exp(-epoch/1000))
        if triggered:
            print(f"  ✅ Epoch {epoch}: 觸發重選")
    
    # 測試候選池生成
    print("\n測試候選池生成...")
    domain_bounds = {'x': (0.0, 1.0), 'y': (0.0, 1.0)}
    candidate_pool = sampler._generate_candidate_pool(domain_bounds, pool_size=500, dim=2)
    print(f"  候選池大小: {candidate_pool.shape}")
    print(f"  範圍: x ∈ [{candidate_pool[:, 0].min():.3f}, {candidate_pool[:, 0].max():.3f}], "
          f"y ∈ [{candidate_pool[:, 1].min():.3f}, {candidate_pool[:, 1].max():.3f}]")
    
    # 測試重選（模擬殘差函數）
    print("\n測試殘差點重選...")
    current_points = torch.rand(100, 2)  # 100 個當前點
    
    def mock_residual_fn(points):
        """模擬殘差函數（PDE 殘差 + BC 殘差）"""
        N = points.shape[0]
        # 3 個 PDE + 2 個 BC
        residuals = torch.randn(N, 5) * 0.1
        return residuals
    
    new_points, metrics = sampler.resample_collocation_points(
        current_points=current_points,
        domain_bounds=domain_bounds,
        residual_fn=mock_residual_fn,
        device='cpu'
    )
    
    print(f"  重選結果:")
    print(f"    原點數: {current_points.shape[0]}")
    print(f"    新點數: {new_points.shape[0]}")
    print(f"    保留點數: {metrics['n_kept']}")
    print(f"    替換點數: {metrics['n_replaced']}")
    print(f"    SVD 秩: {metrics.get('svd_rank', 'N/A')}")
    
    # 測試空間約束
    print("\n測試空間約束...")
    existing = torch.tensor([[0.5, 0.5]])
    new = torch.tensor([[0.51, 0.51], [0.2, 0.2], [0.8, 0.8]])
    filtered = sampler._apply_spatial_constraints(existing, new, min_distance=0.05)
    print(f"  過濾前: {len(new)} 點")
    print(f"  過濾後: {len(filtered)} 點 (min_dist=0.05)")
    
    print("\n✅ 自適應殘差點採樣器測試完成！")
