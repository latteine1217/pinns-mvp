"""
動態權重平衡模組

實現多種自適應權重策略，用於平衡 PINNs 訓練中的多項損失函數。
這是實現穩定高效訓練的關鍵組件，特別適用於逆問題與高 Reynolds 數流場。

主要功能：
- GradNorm 梯度範數平衡
- NTK 權重策略 
- 時間因果權重 (Causal Weighting)
- 自適應權重調度
- 多項損失函數動態平衡

核心算法：
1. GradNorm: 通過梯度範數平衡不同損失項
2. Causal Weights: 時間序列訓練中的因果約束
3. NTK Weighting: 基於神經正切核的權重策略
4. Adaptive Scheduling: 訓練過程中的自適應調整
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import numpy as np
import math
from collections import defaultdict

_EPS = 1e-12


class GradNormWeighter:
    """
    GradNorm 動態權重平衡器
    
    基於梯度範數的自適應權重調整，確保不同損失項對模型參數的影響平衡。
    特別適用於 PINNs 中物理殘差、資料一致性、邊界條件等多項損失的平衡。
    
    參考: GradNorm: Gradient Normalization for Adaptive Loss Balancing (ICML 2018)
    """
    
    def __init__(self,
                 model: nn.Module,
                 loss_names: List[str],
                 alpha: float = 1.5,
                 update_frequency: int = 1000,
                 initial_weights: Optional[Dict[str, float]] = None,
                 target_gradient_ratio: float = 1.0,
                 target_ratios: Optional[List[float]] = None,
                 device: Optional[str] = None,
                 min_weight: float = 0.1,
                 max_weight: float = 10.0,
                 max_ratio: float = 50.0):
        """
        Args:
            model: PINN 模型
            loss_names: 損失項名稱列表 ['data', 'residual', 'boundary', 'prior']
            alpha: 梯度平衡的更新率 (1.5 為推薦值，配合低頻更新提升響應強度)
            update_frequency: 權重更新頻率 (每多少步更新一次)
            initial_weights: 初始權重字典
            target_gradient_ratio: 目標梯度比例
            target_ratios: 目標比例列表（可選，用於測試相容性）
            device: 計算設備 (None 為自動檢測)
        """
        self.model = model
        self.loss_names = loss_names
        self.alpha = alpha
        self.update_frequency = update_frequency
        self.target_gradient_ratio = target_gradient_ratio
        self.target_ratios = target_ratios
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.max_ratio = float(max(1.0, max_ratio))
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
        self.weights = {}
        for name in loss_names:
            base_weight = torch.tensor(
                self.initial_weight_values[name],
                device=self.device,
                dtype=torch.float32,
                requires_grad=False
            )
            clamped_weight = torch.clamp(base_weight, self.min_weight, self.max_weight)
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
        self.initial_losses = None
        
    def compute_gradients(self, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """計算每個損失項對模型參數的梯度範數"""
        gradients = {}
        for name, loss in losses.items():
            if name not in self.loss_names:
                continue
            
            if not loss.requires_grad or abs(float(loss.detach())) < self.eps:
                gradients[name] = torch.tensor(self.eps, device=self.device)
                continue
                
            try:
                weight_tensor = self.weights.get(name, None)
                if isinstance(weight_tensor, torch.Tensor):
                    weight_tensor = weight_tensor.detach()
                elif weight_tensor is None:
                    weight_tensor = torch.tensor(1.0, device=self.device)
                else:
                    weight_tensor = torch.tensor(
                        float(weight_tensor), device=self.device
                    )
                
                weighted_loss = loss * weight_tensor
                
                grads = torch.autograd.grad(
                    outputs=weighted_loss,
                    inputs=list(self.model.parameters()),
                    grad_outputs=torch.ones_like(weighted_loss),
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True
                )
                
                grad_norm = torch.tensor(0.0, device=self.device)
                for grad in grads:
                    if grad is not None:
                        grad_norm += (grad.detach() ** 2).sum()
                
                gradients[name] = torch.sqrt(grad_norm + self.eps)
                
            except Exception as e:
                gradients[name] = torch.tensor(self.eps, device=self.device)
                print(f"Warning: Gradient computation failed for {name}: {e}")
            
        return gradients
    
    def update_weights(self, 
                      losses: Dict[str, torch.Tensor], 
                      total_loss: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """更新動態權重"""
        self.step_count += 1
        
        if self.initial_losses is None:
            self.initial_losses = {name: loss.detach().item() 
                                 for name, loss in losses.items() 
                                 if name in self.loss_names}
        
        if self.step_count % self.update_frequency != 0:
            return self.get_weights()
        
        gradients = self.compute_gradients(losses)
        
        usable_gradients = [
            gradients[name] for name in self.loss_names if name in gradients
        ]
        if len(usable_gradients) < 2:
            return self.get_weights()
        
        grad_values = torch.stack(usable_gradients)
        total_grad = grad_values.sum()
        avg_grad = grad_values.mean()
        
        relative_losses = {}
        for name in self.loss_names:
            if name in losses and name in self.initial_losses:
                current_loss = losses[name].detach().item()
                initial_loss = self.initial_losses[name]
                relative_losses[name] = current_loss / (initial_loss + self.eps)
        
        if relative_losses:
            avg_relative_loss = float(np.mean(list(relative_losses.values())))
            if not np.isfinite(avg_relative_loss) or avg_relative_loss < self.eps:
                avg_relative_loss = 1.0
        else:
            avg_relative_loss = 1.0
        
        for name in self.loss_names:
            if name not in gradients:
                continue
                
            distribution_scale = self.target_distribution.get(name, 1.0)
            target_grad = avg_grad * distribution_scale * self.target_gradient_ratio
            target_grad = torch.clamp(target_grad, min=self.eps)
            
            current_grad = gradients[name]
            gradient_ratio = current_grad / (target_grad + self.eps)
            gradient_ratio = torch.clamp(gradient_ratio, min=self.eps)
            
            if name in relative_losses:
                loss_ratio = relative_losses[name] / (avg_relative_loss + self.eps)
                loss_ratio = max(loss_ratio, self.eps)
                adjustment_factor = gradient_ratio * loss_ratio
            else:
                adjustment_factor = gradient_ratio
            
            weight_adjustment = torch.clamp(
                adjustment_factor.pow(-self.alpha), 0.5, 2.0
            )
            new_weight = torch.clamp(
                self.weights[name] * weight_adjustment,
                self.min_weight,
                self.max_weight
            )
            
            self.weights[name] = new_weight.detach()
            
            self.gradient_history[name].append(current_grad.item())
            if len(self.gradient_history[name]) > 100:
                self.gradient_history[name].pop(0)
        
        self._normalize_weights()
        
        return self.get_weights()
    
    def get_weights(self) -> Dict[str, float]:
        return {name: weight.item() for name, weight in self.weights.items()}
    
    def reset_weights(self):
        for name in self.loss_names:
            self.weights[name] = torch.clamp(
                torch.tensor(
                    self.initial_weight_values.get(name, 1.0),
                    device=self.device,
                    dtype=torch.float32
                ),
                min=self.min_weight,
                max=self.max_weight
            )
        self._normalize_weights()
        self.step_count = 0
        self.initial_losses = None
        self.gradient_history.clear()

    def _normalize_weights(self) -> None:
        if not self.loss_names:
            return
        
        target_sum = torch.tensor(
            max(self.initial_weight_sum, self.eps),
            device=self.device,
            dtype=torch.float32
        )
        
        for _ in range(3):
            weights_tensor = torch.stack([self.weights[name] for name in self.loss_names])
            total = weights_tensor.sum()
            if not torch.isfinite(total) or total.abs() <= self.eps:
                break
            scale = target_sum / total
            updated = []
            for name in self.loss_names:
                scaled = self.weights[name] * scale
                updated_weight = torch.clamp(scaled, self.min_weight, self.max_weight)
                self.weights[name] = updated_weight
                updated.append(updated_weight)
            new_total = torch.stack(updated).sum()
            if torch.abs(new_total - target_sum) / target_sum < 1e-6:
                break
        
        weights_tensor = torch.stack([self.weights[name] for name in self.loss_names])
        max_w = torch.max(weights_tensor)
        min_w = torch.clamp(torch.min(weights_tensor), min=self.min_weight)
        ratio = max_w / (min_w + self.eps)
        if ratio > self.max_ratio:
            geometric_mean = torch.exp(torch.log(weights_tensor + self.eps).mean())
            span = math.sqrt(self.max_ratio)
            lower = torch.tensor(
                max(self.min_weight, geometric_mean / span),
                device=self.device,
                dtype=torch.float32
            )
            upper = torch.tensor(
                min(self.max_weight, geometric_mean * span),
                device=self.device,
                dtype=torch.float32
            )
            for name in self.loss_names:
                self.weights[name] = torch.clamp(self.weights[name], lower, upper)
            
            for _ in range(3):
                weights_tensor = torch.stack([self.weights[name] for name in self.loss_names])
                total = weights_tensor.sum()
                if not torch.isfinite(total) or total.abs() <= self.eps:
                    break
                scale = target_sum / total
                updated = []
                for name in self.loss_names:
                    scaled = self.weights[name] * scale
                    updated_weight = torch.clamp(scaled, self.min_weight, self.max_weight)
                    self.weights[name] = updated_weight
                    updated.append(updated_weight)
                new_total = torch.stack(updated).sum()
                if torch.abs(new_total - target_sum) / target_sum < 1e-6:
                    break


class CausalWeighter:
    """
    時間因果權重器 (Causal Training)
    
    基於 Wang et al. (2022) "Respecting Causality is all you need for training Physics-Informed Neural Networks"
    
    核心機制：
    w_k = exp(-epsilon * sum_{i<k} mean(L_i))
    其中 L_i 是第 i 個時間分塊的平均殘差平方。
    """
    
    def __init__(self, 
                 epsilon: float = 1.0, 
                 n_time_bins: int = None,
                 num_chunks: int = 10,
                 t_min: float = 0.0,
                 t_max: float = 1.0,
                 causality_strength: float = None, # 兼容舊接口
                 time_window_size: int = None,
                 time_window: int = None,
                 decay_rate: float = None,
                 temporal_decay: float = None):
        """
        Args:
            epsilon: 因果容差參數，控制權重衰減速度。值越大，對早期時間的強制性越強。
            n_time_bins: 舊接口，對應 num_chunks（保留相容性）。
            num_chunks: 時間分塊數量（jaxpi 設定）。
            t_min: 時間域下界
            t_max: 時間域上界
        """
        self.epsilon = epsilon if causality_strength is None else causality_strength
        self.num_chunks = int(n_time_bins) if n_time_bins is not None else int(num_chunks)
        self.t_min = t_min
        self.t_max = t_max
        
        # 兼容舊接口屬性
        self.causality_strength = self.epsilon
        self.time_window_size = time_window_size or time_window or 10
        self.decay_rate = decay_rate or temporal_decay or 0.1

    def compute_weights(self, 
                       pde_losses: torch.Tensor, 
                       time_coords: torch.Tensor) -> torch.Tensor:
        """
        計算每個採樣點的因果權重
        
        Args:
            pde_losses: 每點的 PDE 損失值 [N, 1] 或 [N] (通常為殘差平方和)
            time_coords: 對應的時間坐標 [N, 1] 或 [N]
            
        Returns:
            weights: 與輸入形狀相同的權重張量 [N, 1]
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
        
        # 1. 依時間排序（jaxpi：只依時間排序，數值大小不影響權重）
        t_sorted, sort_idx = torch.sort(t_vals)
        loss_sorted = loss_vals[sort_idx]
        n_points = loss_sorted.numel()
        if n_points == 0:
            return loss_vals.new_zeros((0, 1))
        
        num_chunks = max(1, min(self.num_chunks, n_points))
        chunk_size = n_points // num_chunks
        if chunk_size == 0:
            chunk_size = 1
            num_chunks = n_points
        
        usable = chunk_size * num_chunks
        loss_main = loss_sorted[:usable].view(num_chunks, chunk_size)
        
        # 2. 計算每個 chunk 的平均損失
        chunk_means = torch.mean(loss_main, dim=1)  # [num_chunks]
        
        # 3. 建立因果矩陣 M (strict lower triangular)
        causal_matrix = torch.tril(
            torch.ones((num_chunks, num_chunks), device=device),
            diagonal=-1
        )
        
        # 4. 計算 chunk 權重
        chunk_weights = torch.exp(-self.epsilon * (causal_matrix @ chunk_means))
        
        # 5. 映射回每個採樣點
        weights_sorted = torch.empty_like(loss_sorted)
        weights_sorted[:usable] = chunk_weights.repeat_interleave(chunk_size)
        if usable < n_points:
            weights_sorted[usable:] = chunk_weights[-1]
        
        # 6. 還原到原本順序
        weights = torch.empty_like(weights_sorted)
        weights[sort_idx] = weights_sorted
        
        return weights.unsqueeze(1) # [N, 1]

    # 兼容舊接口
    def compute_causal_weights(self, time_losses, time_points=None):
        if not time_losses:
            return []
        loss_vals = torch.stack([loss.detach() if torch.is_tensor(loss) else torch.tensor(loss)
                                 for loss in time_losses])
        if time_points is None:
            time_points = torch.arange(len(time_losses), device=loss_vals.device).float()
        else:
            time_points = torch.as_tensor(time_points, device=loss_vals.device).float()
        weights = self.compute_weights(loss_vals, time_points)
        return [float(w.item()) for w in weights.squeeze(1)]
    
    def apply_temporal_decay(self, weights, current_epoch):
        return weights


class NTKWeighter:
    """
    神經正切核 (NTK) 權重器
    """
    
    def __init__(self,
                 model: nn.Module,
                 loss_names: List[str] = None,
                 sample_size: int = 100,
                 update_frequency: int = 2000,
                 update_freq: Optional[int] = None,
                 reg_param: float = 1e-6):
        self.model = model
        self.loss_names = loss_names or ['data', 'pde']
        self.sample_size = sample_size
        self.update_frequency = update_freq if update_freq is not None else update_frequency
        self.reg_param = reg_param
        self.step_count = 0
        self.ntk_weights = {name: 1.0 for name in self.loss_names}
        
    def compute_ntk_eigenvalues(self, inputs: torch.Tensor, loss_type: str) -> torch.Tensor:
        batch_size = min(self.sample_size, inputs.shape[0])
        sample_inputs = inputs[:batch_size]
        
        outputs = self.model(sample_inputs)
        jacobians = []
        
        for i in range(outputs.shape[1]):
            grads = torch.autograd.grad(
                outputs[:, i].sum(),
                self.model.parameters(),
                retain_graph=True,
                create_graph=False
            )
            jacobian = torch.cat([g.view(-1) for g in grads])
            jacobians.append(jacobian)
        
        J = torch.stack(jacobians, dim=0)
        K = J @ J.T
        eigenvals = torch.linalg.eigvals(K).real
        
        return eigenvals
    
    def update_ntk_weights(self, data_inputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.step_count += 1
        if self.step_count % self.update_frequency != 0:
            return self.ntk_weights.copy()
        # 這裡可以加入實際的 NTK 計算邏輯
        return self.ntk_weights.copy()
    
    def update_weights(self, losses: Dict[str, torch.Tensor], x_train: torch.Tensor, step: int = 0) -> Dict[str, float]:
        data_inputs = {}
        for name in losses.keys():
            if name in self.loss_names:
                data_inputs[name] = x_train
        self.step_count = step
        return self.update_ntk_weights(data_inputs)


class AdaptiveWeightScheduler:
    """自適應權重調度器"""
    
    def __init__(self,
                 loss_names: List[str],
                 phases: Dict[str, Dict] = None,
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

    def update_weights(self, losses: Dict[str, torch.Tensor], step: int = 0, total_steps: int = 100000) -> Dict[str, float]:
        """回傳當前 phase 對應的權重配置（簡化版）"""
        _ = losses  # losses 預留未來擴展，目前不影響 phase-based 權重
        return self.get_phase_weights(step, total_steps)


class MultiWeightManager:
    """多權重策略管理器"""
    
    def __init__(self,
                 objectives_or_model = None,
                 loss_names: List[str] = None,
                 strategies: List[str] = ['gradnorm', 'adaptive'],
                 strategy_weights: Optional[Dict[str, float]] = None,
                 method: str = 'weighted_sum',
                 preference_weights: List[float] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.preference_weights = preference_weights
        self.eps = _EPS
        if strategies is None:
            strategies = self.config.get('strategies', ['gradnorm', 'adaptive'])
            
        if isinstance(objectives_or_model, list):
            self.objectives = objectives_or_model
            self.loss_names = objectives_or_model
            self.model = None
        else:
            self.model = objectives_or_model
            self.loss_names = loss_names or ['data', 'residual']
            self.objectives = self.loss_names
            
        self.strategies = strategies
        self.weighters = {}
        
        if 'gradnorm' in strategies and self.model is not None:
            self.weighters['gradnorm'] = GradNormWeighter(self.model, self.loss_names)
        if 'causal' in strategies:
            self.weighters['causal'] = CausalWeighter()
        if 'ntk' in strategies and self.model is not None:
            self.weighters['ntk'] = NTKWeighter(self.model, self.loss_names)
        if 'adaptive' in strategies:
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
            grad_weights = self.weighters['gradnorm'].update_weights(losses)
            for k, v in grad_weights.items():
                final_weights[k] = v
        
        return final_weights
    
    def get_weights(self):
        return {name: 1.0 for name in self.loss_names}


def create_weight_manager(model: nn.Module, loss_names: List[str], config: Optional[Dict[str, Any]] = None) -> MultiWeightManager:
    return MultiWeightManager(model, loss_names, config=config)
