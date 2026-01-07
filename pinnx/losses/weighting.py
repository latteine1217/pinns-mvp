"""
動態權重平衡模組

實現多種自適應權重策略，用於平衡 PINNs 訓練中的多項損失函數。
這是實現穩定高效訓練的關鍵組件，特別適用於逆問題與高 Reynolds 數流場。

主要功能：
- GradNorm 梯度範數平衡
- 時間因果權重 (Causal Weighting)
- 自適應權重調度
- 多項損失函數動態平衡

核心算法：
1. GradNorm: 通過梯度範數平衡不同損失項（JaxPI 風格）
2. Causal Weights: 時間序列訓練中的因果約束
3. Adaptive Scheduling: 訓練過程中的自適應調整
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import numpy as np
import math
from collections import defaultdict

from pinnx.losses.weighting_base import LossWeighter, PointWeighter

_EPS = 1e-12


class GradNormWeighter(LossWeighter):
    """
    GradNorm 動態權重平衡器（對齊 JaxPI 實作）

    基於梯度範數的自適應權重調整，確保不同損失項對模型參數的影響平衡。
    特別適用於 PINNs 中物理殘差、資料一致性、邊界條件等多項損失的平衡。
    
    核心公式（JaxPI 簡化版本）:
        w_i = Ḡ / (G_i + ε·Ḡ)
    
    其中:
        - Ḡ = mean(G_i): 平均梯度範數
        - G_i = ||∇_θ L_i||: 第 i 個損失的梯度範數
        - ε = 1e-5: 數值穩定性參數
    
    與論文差異:
        - 簡化版本: 不使用 alpha 參數和相對損失（alpha=0 假設）
        - 直接計算: 無需二階梯度，計算效率更高
        - JaxPI 驗證: 已在多個 PINNs 任務上驗證有效
    
    參考:
        - 論文: Chen et al., "GradNorm" (ICML 2018)
        - 實作: JaxPI models.py:208-224
        - 分析: context/technical_reviews/JAXPI_GRADNORM_ANALYSIS_2026-01-06.md
    """
    
    def __init__(self,
                 model: nn.Module,
                 loss_names: List[str],
                 update_frequency: int = 100,
                 initial_weights: Optional[Dict[str, float]] = None,
                 target_ratios: Optional[List[float]] = None,
                 device: Optional[str] = None,
                 min_weight: float = 0.1,
                 max_weight: float = 10.0,
                 momentum: float = 0.95):
        """
        Args:
            model: PINN 模型
            loss_names: 損失項名稱列表 ['data', 'residual', 'boundary', 'prior']
            update_frequency: 權重更新頻率（每多少步更新一次）
                            推薦值: 100 (JaxPI 每步更新，但 PINNs 可用低頻減少開銷)
            initial_weights: 初始權重字典
            target_ratios: 目標分佈比例（可選，用於不同損失項的相對重要性）
            device: 計算設備 (None 為自動檢測)
            min_weight: 權重下界比例（相對於 initial_weights，默認 0.1 = 10%）
            max_weight: 權重上界比例（相對於 initial_weights，默認 10.0 = 1000%）
            momentum: EMA 平滑係數（推薦值: 0.95，JaxPI 默認值）
        
        Note:
            權重裁剪使用**相對範圍**:
            - 絕對範圍: [initial_weight * min_weight, initial_weight * max_weight]
            - 例如: initial_weight=100, min=0.1, max=10 → 裁剪範圍 [10, 1000]
        """
        self.model = model
        self.loss_names = loss_names
        self.update_frequency = update_frequency
        self.target_ratios = target_ratios
        self.min_weight_ratio = float(min_weight)
        self.max_weight_ratio = float(max_weight)
        self.momentum = float(momentum)
        self.eps = _EPS
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            if isinstance(device, torch.device):
                self.device = device
            elif isinstance(device, str):
                self.device = torch.device(device)
            else:
                raise TypeError(f"Unsupported device type: {type(device)}")
        
        if initial_weights is None:
            initial_weights = {name: 1.0 for name in loss_names}
        
        self.initial_weight_values = {
            name: float(initial_weights.get(name, 1.0)) for name in loss_names
        }
        self.initial_weight_sum = float(sum(self.initial_weight_values.values()))
        
        # 🔧 FIX: 為每個損失項計算絕對裁剪邊界（相對於 initial_weight）
        self.min_weight_abs = {}
        self.max_weight_abs = {}
        for name in loss_names:
            initial = self.initial_weight_values[name]
            self.min_weight_abs[name] = initial * self.min_weight_ratio
            self.max_weight_abs[name] = initial * self.max_weight_ratio
        
        self.weights = {}
        for name in loss_names:
            base_weight = torch.tensor(
                self.initial_weight_values[name],
                device=self.device,
                dtype=torch.float32,
                requires_grad=False
            )
            # 使用相對裁剪範圍
            clamped_weight = torch.clamp(
                base_weight,
                min=self.min_weight_abs[name],
                max=self.max_weight_abs[name]
            )
            self.weights[name] = clamped_weight
        
        if target_ratios is not None:
            if len(target_ratios) != len(loss_names):
                raise ValueError(
                    "target_ratios length must match loss_names length "
                    f"({len(target_ratios)} != {len(loss_names)})"
                )
            ratios = torch.as_tensor(
                target_ratios, dtype=torch.float32, device=self.device
            )
            ratios = torch.clamp(ratios, min=self.eps)
            normalized = (ratios / ratios.mean()).cpu().tolist()
            self.target_distribution = {
                name: float(normalized[idx]) for idx, name in enumerate(loss_names)
            }
        else:
            self.target_distribution = {name: 1.0 for name in loss_names}
        
        self.gradient_history = defaultdict(list)
        self.step_count = 0
        
    def compute_gradients(self, losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """計算每個損失項對模型參數的梯度範數（JaxPI 風格）- 批次化版本
        
        關鍵：計算**未加權**損失的梯度範數，即 ||∇_θ L_i||，而非 ||∇_θ (w_i × L_i)||
        
        優化：使用單次參數遍歷 + 分別backward，相比原方法減少重複的參數列表創建
        
        Returns:
            Dict[str, torch.Tensor]: 每個損失項的梯度範數（Tensor 類型）
        """
        gradients = {}
        
        # Step 1: 分離有效損失和無效損失
        valid_losses = {}
        for name, loss in losses.items():
            if name not in self.loss_names:
                continue
            
            if not loss.requires_grad or abs(float(loss.detach())) < self.eps:
                gradients[name] = torch.tensor(self.eps, device=self.device)
            else:
                valid_losses[name] = loss
        
        # Step 2: 如果沒有有效損失，直接返回
        if len(valid_losses) == 0:
            return gradients
        
        # Step 3: 獲取模型參數列表（只獲取一次，避免重複）
        params = list(self.model.parameters())
        
        # Step 4: 對每個損失分別計算梯度（但優化了參數訪問）
        # 注意：PyTorch 的 autograd.grad 設計不支持真正的批次計算
        # 但我們可以優化其他部分（如參數緩存、減少 detach 調用）
        for name, loss in valid_losses.items():
            try:
                # 計算單個損失的梯度
                grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=params,  # 使用緩存的參數列表
                    grad_outputs=torch.ones_like(loss),
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True
                )
                
                # 計算梯度範數（優化版）
                grad_norm_sq = torch.tensor(0.0, device=self.device)
                for grad in grads:
                    if grad is not None:
                        # 使用 .pow(2) 替代 ** 2，並合並 detach + sum
                        grad_norm_sq += grad.detach().pow(2).sum()
                
                gradients[name] = torch.sqrt(grad_norm_sq + self.eps)
                
            except Exception as e:
                gradients[name] = torch.tensor(self.eps, device=self.device)
                print(f"Warning: Gradient computation failed for {name}: {e}")
        
        return gradients
    
    def update_weights(
        self,
        losses: Dict[str, torch.Tensor],
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        更新 GradNorm 動態權重（對齊 JaxPI 實作）
        
        核心公式（JaxPI 風格）:
            w_i = Ḡ / (G_i + ε·Ḡ)
        
        其中:
            - Ḡ = mean(G_i): 平均梯度範數
            - G_i = ||∇_θ L_i||: 第 i 個損失的梯度範數
            - ε = 1e-5: 數值穩定性參數
        
        參考: JaxPI models.py:208-224
        論文: Chen et al., "GradNorm" (ICML 2018) - 簡化版本（alpha=0）

        Args:
            losses: 損失項字典
            context: 上下文參數，可包含：
                - step (int): 當前步數（可選，用於同步）

        Returns:
            Dict[str, float]: 更新後的權重字典
        """
        self.step_count += 1
        
        # 低頻更新機制（減少計算開銷）
        if self.step_count % self.update_frequency != 0:
            return self.get_weights()
        
        # 1. 計算每個損失的梯度範數
        gradients = self.compute_gradients(losses)
        
        usable_gradients = [
            gradients[name] for name in self.loss_names if name in gradients
        ]
        if len(usable_gradients) < 2:
            return self.get_weights()
        
        # 2. 計算平均梯度範數
        grad_values = torch.stack(usable_gradients)
        mean_grad = grad_values.mean()
        
        # 3. JaxPI 風格權重計算
        for name in self.loss_names:
            if name not in gradients:
                continue
            
            # 當前梯度範數
            grad_norm = gradients[name]
            
            # JaxPI 公式: w_i = mean_grad / (grad_norm + eps * mean_grad)
            eps_adjusted = self.eps * mean_grad
            new_weight = mean_grad / (grad_norm + eps_adjusted)
            
            # 考慮 target_distribution（不同損失項的相對重要性）
            if hasattr(self, 'target_distribution') and name in self.target_distribution:
                distribution_scale = self.target_distribution[name]
                new_weight = new_weight * distribution_scale
            
            # 應用 EMA 平滑（對齊 JaxPI）
            # JaxPI: w_new = w_old * momentum + w_computed * (1 - momentum)
            # 注意：第一次更新時 (step_count == update_frequency) 不使用 EMA
            if self.momentum > 0 and self.step_count > self.update_frequency:
                old_weight = self.weights[name]
                new_weight = old_weight * self.momentum + new_weight * (1 - self.momentum)
            
            # 裁剪權重範圍（防止極端值）
            # 🔧 FIX: 使用相對裁剪範圍（針對每個損失項）
            self.weights[name] = torch.clamp(
                new_weight.detach(),
                min=self.min_weight_abs[name],
                max=self.max_weight_abs[name]
            )
            
            # 記錄梯度歷史（用於診斷）
            self.gradient_history[name].append(grad_norm.item())
            if len(self.gradient_history[name]) > 100:
                self.gradient_history[name].pop(0)
        
        return self.get_weights()
    
    def get_weights(self) -> Dict[str, float]:
        return {name: weight.item() for name, weight in self.weights.items()}
    
    def reset_weights(self):
        """重置權重到初始值"""
        for name in self.loss_names:
            # 🔧 FIX: 使用相對裁剪範圍
            self.weights[name] = torch.clamp(
                torch.tensor(
                    self.initial_weight_values.get(name, 1.0),
                    device=self.device,
                    dtype=torch.float32
                ),
                min=self.min_weight_abs[name],
                max=self.max_weight_abs[name]
            )
        self.step_count = 0
        self.gradient_history.clear()
    
    # 🗑️ _normalize_weights() 方法已移除（已棄用，從未被調用）
    # 原因：JaxPI-style GradNorm 不使用權重正規化
    # 歷史：該方法定義於早期版本但從未在 update_weights() 中調用
    # 如需權重正規化功能，請參考 Git 歷史或聯繫開發團隊


class CausalWeighter(PointWeighter):
    """
    時間因果權重器 (Causal Training)

    基於 Wang et al. (2022) "Respecting Causality is all you need for training Physics-Informed Neural Networks"
    遵循 PointWeighter 接口，計算每個時間點的因果權重。

    核心機制（對齊 JaxPI）：
    gamma_k = exp(-tol * sum_{i<k} mean(L_i))
    其中 L_i 是第 i 個時間分塊的平均殘差平方。
    
    JaxPI 參數對應：
    - causal_tol (JaxPI) = epsilon (本實現)
    - num_chunks (JaxPI) = num_chunks (本實現)

    優化版本（v2.0）：
    - 預計算因果矩陣（避免每次調用重新生成）
    - 支援多設備（CPU/CUDA/MPS）
    - 可選 chunk-level 返回（性能優化）
    """
    
    def __init__(
        self,
        causal_tol: float = 1.0,  # 對齊 JaxPI 命名
        num_chunks: int = 32,      # 對齊 JaxPI 默認值
        t_min: float = 0.0,
        t_max: float = 1.0,
        device: str = 'cpu',
    ):
        """
        Args:
            causal_tol: 因果容差參數（對齊 JaxPI），控制權重衰減速度。值越大，對早期時間的強制性越強。
                       JaxPI 默認值: 1.0
            num_chunks: 時間分塊數量（對齊 JaxPI）。
                       JaxPI 默認值: 32 (Burgers), 16 (NS unsteady cylinder)
            t_min: 時間域下界
            t_max: 時間域上界
            device: 計算設備 ('cpu', 'cuda', 'mps')
        """
        self.causal_tol = float(causal_tol)  # 對齊 JaxPI 命名
        self.epsilon = self.causal_tol       # 保持向後兼容
        self.num_chunks = int(num_chunks)
        self.t_min = t_min
        self.t_max = t_max
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # 預計算因果矩陣（JAXpi 風格優化）
        self._precompute_causal_matrix()
        
    
    def _precompute_causal_matrix(self):
        """
        預計算因果矩陣 M (strict lower triangular)
        
        M[i, j] = 1 if j < i else 0
        
        這樣 M @ chunk_means 就是每個 chunk 之前所有 chunk 的累積損失
        """
        self.causal_matrix = torch.tril(
            torch.ones((self.num_chunks, self.num_chunks), device=self.device),
            diagonal=-1
        )
    
    def to(self, device):
        """將因果矩陣移動到指定設備"""
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.causal_matrix = self.causal_matrix.to(self.device)
        return self
    
    def compute_weights(
        self,
        residuals: torch.Tensor,
        coords: torch.Tensor,
        context: Dict[str, Any]
    ) -> torch.Tensor:
        """
        計算每個採樣點的因果權重（符合 PointWeighter 接口）

        Args:
            residuals: 每點的殘差或損失值 [N, 1] 或 [N]
            coords: 時間坐標 [N, 1] 或 [N]
            context: 上下文參數，可包含：
                - return_pointwise (bool): 是否返回點級權重（默認 True）

        Returns:
            torch.Tensor: 點權重 [N, 1]
                若 context['return_pointwise']=False，返回 chunk 權重的重複版本
        """
        return_pointwise = context.get('return_pointwise', True)
        result = self._compute_weights_impl(residuals, coords, return_pointwise)
        # 確保符合接口：始終返回 Tensor
        if isinstance(result, tuple):
            # 若內部返回 tuple，將 chunk weights 轉為 point weights
            chunk_weights, _ = result
            # 簡單重複策略：返回第一個 chunk 的權重
            return chunk_weights[0].unsqueeze(0).expand(residuals.shape[0], 1)
        return result

    def _compute_weights_impl(
        self,
        pde_losses: torch.Tensor,
        time_coords: torch.Tensor,
        return_pointwise: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        計算每個採樣點的因果權重（內部實現）

        Args:
            pde_losses: 每點的 PDE 損失值 [N, 1] 或 [N]
            time_coords: 對應的時間坐標 [N, 1] 或 [N]
            return_pointwise: 若 True 返回 point-level 權重 [N, 1]；
                             若 False 返回 chunk-level (chunk_weights, chunk_means)

        Returns:
            若 return_pointwise=True: torch.Tensor [N, 1] 權重張量
            若 return_pointwise=False: Tuple[torch.Tensor, torch.Tensor] (chunk_weights, chunk_means)
        """
        # 形狀驗證
        if pde_losses.numel() != time_coords.numel():
            raise ValueError(
                f"Shape mismatch in CausalWeighter: pde_losses {pde_losses.shape}, "
                f"time_coords {time_coords.shape}"
            )

        # 確保輸入是展平的
        loss_vals = pde_losses.detach().flatten()  # [N], 阻斷梯度
        t_vals = time_coords.detach().flatten()       # [N]
        
        device = loss_vals.device
        
        # 如果設備不匹配，移動因果矩陣
        if self.causal_matrix.device != device:
            self.causal_matrix = self.causal_matrix.to(device)
        
        # 1. 依時間排序
        t_sorted, sort_idx = torch.sort(t_vals)
        loss_sorted = loss_vals[sort_idx]
        n_points = loss_sorted.numel()
        if n_points == 0:
            return loss_vals.new_zeros((0, 1))
        
        # 2. 動態調整 num_chunks（如果採樣點太少）
        num_chunks = max(1, min(self.num_chunks, n_points))
        chunk_size = n_points // num_chunks
        if chunk_size == 0:
            chunk_size = 1
            num_chunks = n_points
        
        usable = chunk_size * num_chunks
        loss_main = loss_sorted[:usable].view(num_chunks, chunk_size)
        
        # 3. 計算每個 chunk 的平均損失
        chunk_means = torch.mean(loss_main, dim=1)  # [num_chunks]
        
        # 4. 使用預計算的因果矩陣計算 chunk 權重
        # 如果 num_chunks 與預計算的不同（動態調整情況），使用切片
        if num_chunks < self.num_chunks:
            causal_matrix_used = self.causal_matrix[:num_chunks, :num_chunks]
        else:
            causal_matrix_used = self.causal_matrix
        
        chunk_weights = torch.exp(-self.epsilon * (causal_matrix_used @ chunk_means))
        
        # 5. 若要求 chunk-level 結果，直接返回
        if not return_pointwise:
            return (chunk_weights, chunk_means)
        
        # 6. 映射回每個採樣點
        weights_sorted = torch.empty_like(loss_sorted)
        weights_sorted[:usable] = chunk_weights.repeat_interleave(chunk_size)
        if usable < n_points:
            weights_sorted[usable:] = chunk_weights[-1]
        
        # 7. 還原到原本順序
        weights = torch.empty_like(weights_sorted)
        weights[sort_idx] = weights_sorted
        
        return weights.unsqueeze(1) # [N, 1]
    
    # 舊接口（time_window / compute_causal_weights / temporal_decay）已移除


class AdaptiveWeightScheduler(LossWeighter):
    """
    自適應權重調度器

    基於訓練階段（warmup / main / refinement）動態調整損失權重。
    遵循統一的 LossWeighter 接口。
    """

    def __init__(self,
                 loss_names: List[str],
                 phases: Optional[Dict[str, Dict]] = None,
                 adaptation_method: str = 'exponential',
                 adaptation_rate: float = 0.1):
        self.loss_names = loss_names
        self.adaptation_method = adaptation_method
        self.adaptation_rate = adaptation_rate

        if phases is None:
            phases = {
                'warmup': {'duration_ratio': 0.1, 'weight_ratios': {name: 1.0 for name in loss_names}},
                'main': {'duration_ratio': 0.7, 'weight_ratios': {name: 1.0 for name in loss_names}},
                'refinement': {'duration_ratio': 0.2, 'weight_ratios': {name: 1.0 for name in loss_names}}
            }
        self.phases = phases
        self.current_phase = 'warmup'
        self._cached_weights = {name: 1.0 for name in loss_names}
        
    def get_current_phase(self, current_step: int, total_steps: int) -> str:
        progress = current_step / total_steps
        warmup_end = self.phases['warmup']['duration_ratio']
        main_end = warmup_end + self.phases['main']['duration_ratio']
        
        if progress <= warmup_end: return 'warmup'
        elif progress <= main_end: return 'main'
        else: return 'refinement'
    
    def get_phase_weights(self, current_step: int, total_steps: int) -> Dict[str, float]:
        phase = self.get_current_phase(current_step, total_steps)
        self.current_phase = phase
        phase_config = self.phases[phase]
        weight_ratios = phase_config['weight_ratios']
        weights = {}
        for name in self.loss_names:
            weights[name] = weight_ratios.get(name, 1.0)
        return weights

    def update_weights(
        self,
        losses: Dict[str, torch.Tensor],
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        根據訓練階段更新權重

        Args:
            losses: 損失項字典（當前實現未使用，預留擴展）
            context: 上下文參數，需包含：
                - step (int): 當前訓練步數
                - total_steps (int): 總訓練步數

        Returns:
            Dict[str, float]: 當前階段的權重字典
        """
        step = context.get('step', 0)
        total_steps = context.get('total_steps', 100000)

        weights = self.get_phase_weights(step, total_steps)
        self._cached_weights = weights  # 緩存最新權重
        return weights

    def get_weights(self) -> Dict[str, float]:
        """獲取當前緩存的權重"""
        return self._cached_weights.copy()

    def reset_weights(self) -> None:
        """重置到初始狀態"""
        self.current_phase = 'warmup'
        self._cached_weights = {name: 1.0 for name in self.loss_names}


class MultiWeightManager:
    """多權重策略管理器"""
    
    def __init__(self,
                 objectives_or_model = None,
                 loss_names: Optional[List[str]] = None,
                 strategies: Optional[List[str]] = None,
                 strategy_weights: Optional[Dict[str, float]] = None,
                 method: str = 'weighted_sum',
                 preference_weights: Optional[List[float]] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.preference_weights = preference_weights
        self.eps = _EPS
        if strategies is None:
            strategies = self.config.get('strategies', ['gradnorm', 'adaptive'])
        
        # 確保 strategies 不為 None
        self.strategies: List[str] = strategies if strategies is not None else ['gradnorm', 'adaptive']
            
        if isinstance(objectives_or_model, list):
            self.objectives = objectives_or_model
            self.loss_names = objectives_or_model
            self.model = None
        else:
            self.model = objectives_or_model
            self.loss_names = loss_names or ['data', 'residual']
            self.objectives = self.loss_names
            
        self.weighters = {}
        
        if 'gradnorm' in self.strategies and self.model is not None:
            self.weighters['gradnorm'] = GradNormWeighter(self.model, self.loss_names)
        if 'causal' in self.strategies:
            self.weighters['causal'] = CausalWeighter()
        if 'adaptive' in self.strategies:
            self.weighters['adaptive'] = AdaptiveWeightScheduler(self.loss_names)
            
    def update_weights(self, losses, current_step=0, total_steps=100000, **kwargs):
        # 簡化的更新邏輯
        if self.model is None:
            objective_values = losses or {}
            if isinstance(objective_values, dict):
                weights = {}
                prefs = self.config.get('preference_weights')
                if prefs is None and hasattr(self, 'preference_weights'):
                    prefs = self.preference_weights
                if prefs is None:
                    prefs = [1.0] * len(self.objectives)
                for idx, name in enumerate(self.objectives):
                    value = objective_values.get(name, 0.0)
                    if torch.is_tensor(value):
                        value = float(value.detach().item())
                    weight = max(float(value), self.eps) * float(prefs[idx])
                    weights[name] = weight
                return weights
            return {name: 1.0 for name in self.loss_names}
        
        final_weights = {name: 1.0 for name in self.loss_names}
        if 'gradnorm' in self.weighters:
            # 使用統一的 LossWeighter 接口
            context = {
                'step': current_step,
                'total_steps': total_steps,
                **kwargs  # 傳遞額外的上下文參數
            }
            grad_weights = self.weighters['gradnorm'].update_weights(losses, context)
            for k, v in grad_weights.items():
                final_weights[k] = v

        return final_weights
    
    def get_weights(self):
        return {name: 1.0 for name in self.loss_names}


def create_weight_manager(model: nn.Module, loss_names: List[str], config: Optional[Dict[str, Any]] = None) -> MultiWeightManager:
    return MultiWeightManager(model, loss_names, config=config)
