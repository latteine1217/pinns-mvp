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
                 alpha: float = 0.12,
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
            alpha: 梯度平衡的更新率 (0.12 為論文建議值)
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
        self.target_ratios = target_ratios  # 存儲目標比例（用於相容性）
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.max_ratio = float(max(1.0, max_ratio))
        self.eps = _EPS
        
        # 自動檢測設備
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            if isinstance(device, torch.device):
                self.device = device
            elif isinstance(device, str):
                self.device = torch.device(device)
            else:
                raise TypeError(f"Unsupported device type: {type(device)}")
        
        # 初始化權重
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
        
        # 預先計算目標分佈
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
        
        # 記錄梯度歷史用於穩定化
        self.gradient_history = defaultdict(list)
        self.step_count = 0
        self.initial_losses = None
        
    def compute_gradients(self, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        計算每個損失項對模型參數的梯度範數
        
        Args:
            losses: 損失項字典
            
        Returns:
            每個損失項的梯度範數
        """
        gradients = {}
        
        for name, loss in losses.items():
            if name not in self.loss_names:
                continue
            
            # 檢查損失是否需要梯度和是否為非零
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
                
                # 計算該損失項對模型參數的梯度
                grads = torch.autograd.grad(
                    outputs=weighted_loss,
                    inputs=list(self.model.parameters()),
                    grad_outputs=torch.ones_like(weighted_loss),
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True
                )
                
                # 計算梯度範數
                grad_norm = torch.tensor(0.0, device=self.device)
                for grad in grads:
                    if grad is not None:
                        grad_norm += (grad.detach() ** 2).sum()
                
                gradients[name] = torch.sqrt(grad_norm + self.eps)
                
            except Exception as e:
                # 如果梯度計算失敗，使用小值
                gradients[name] = torch.tensor(self.eps, device=self.device)
                print(f"Warning: Gradient computation failed for {name}: {e}")
            
        return gradients
    
    def update_weights(self, 
                      losses: Dict[str, torch.Tensor], 
                      total_loss: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        更新動態權重（與測試相容的接口）
        
        Args:
            losses: 當前損失項字典
            total_loss: 總損失（可選，用於觸發梯度計算）
            
        Returns:
            更新後的權重字典
        """
        self.step_count += 1
        
        # 記錄初始損失值用於相對比較
        if self.initial_losses is None:
            self.initial_losses = {name: loss.detach().item() 
                                 for name, loss in losses.items() 
                                 if name in self.loss_names}
        
        # 檢查是否需要更新權重
        if self.step_count % self.update_frequency != 0:
            return self.get_weights()
        
        # 計算梯度範數
        gradients = self.compute_gradients(losses)
        
        usable_gradients = [
            gradients[name] for name in self.loss_names if name in gradients
        ]
        if len(usable_gradients) < 2:  # 至少需要兩個損失項才進行平衡
            return self.get_weights()
        
        grad_values = torch.stack(usable_gradients)
        total_grad = grad_values.sum()
        avg_grad = grad_values.mean()
        
        # 計算相對損失率 (相對於初始值的變化)
        relative_losses = {}
        for name in self.loss_names:
            if name in losses and name in self.initial_losses:
                current_loss = losses[name].detach().item()
                initial_loss = self.initial_losses[name]
                relative_losses[name] = current_loss / (initial_loss + self.eps)
        
        # 計算平均相對損失率
        if relative_losses:
            avg_relative_loss = float(np.mean(list(relative_losses.values())))
            if not np.isfinite(avg_relative_loss) or avg_relative_loss < self.eps:
                avg_relative_loss = 1.0
        else:
            avg_relative_loss = 1.0
        
        # 更新權重
        for name in self.loss_names:
            if name not in gradients:
                continue
                
            # 計算目標梯度範數
            distribution_scale = self.target_distribution.get(name, 1.0)
            target_grad = avg_grad * distribution_scale * self.target_gradient_ratio
            target_grad = torch.clamp(target_grad, min=self.eps)
            
            # 計算當前梯度與目標的比值
            current_grad = gradients[name]
            gradient_ratio = current_grad / (target_grad + self.eps)
            gradient_ratio = torch.clamp(gradient_ratio, min=self.eps)
            
            # 考慮相對損失率的影響
            if name in relative_losses:
                loss_ratio = relative_losses[name] / (avg_relative_loss + self.eps)
                loss_ratio = max(loss_ratio, self.eps)
                adjustment_factor = gradient_ratio * loss_ratio
            else:
                adjustment_factor = gradient_ratio
            
            # 使用指數移動平均更新權重，但限制調整幅度
            weight_adjustment = torch.clamp(
                adjustment_factor.pow(-self.alpha), 0.5, 2.0
            )
            new_weight = torch.clamp(
                self.weights[name] * weight_adjustment,
                self.min_weight,
                self.max_weight
            )
            
            self.weights[name] = new_weight.detach()
            
            # 記錄梯度歷史
            self.gradient_history[name].append(current_grad.item())
            if len(self.gradient_history[name]) > 100:  # 保持歷史長度
                self.gradient_history[name].pop(0)
        
        self._normalize_weights()
        
        return self.get_weights()
    
    def get_weights(self) -> Dict[str, float]:
        """獲取當前權重"""
        return {name: weight.item() for name, weight in self.weights.items()}
    
    def reset_weights(self):
        """重置權重為初始值"""
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
        """Normalize weights to keep total constant and ratios bounded."""
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
        
        # Enforce ratio constraint
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
            
            # Re-normalize after clamping (iterate to respect bounds)
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
    時間因果權重器
    
    在非定常問題中，確保模型優先擬合早期時間的解，然後逐步推進到後期。
    這有助於避免時間序列訓練中的梯度消失和累積誤差問題。
    """
    
    def __init__(self,
                 causality_strength: float = 1.0,
                 time_window_size: int = 10,
                 decay_rate: float = 0.1,
                 time_window: int = None,
                 temporal_decay: float = None):
        """
        Args:
            causality_strength: 因果約束強度
            time_window_size: 時間窗口大小
            decay_rate: 權重衰減率
            time_window: 時間窗口（別名，用於測試相容性）
            temporal_decay: 時間衰減參數（別名，用於測試相容性）
        """
        self.causality_strength = causality_strength
        # 優先使用測試期待的參數名稱
        self.time_window_size = time_window if time_window is not None else time_window_size
        self.decay_rate = temporal_decay if temporal_decay is not None else decay_rate
        self.accumulated_errors = []
        
    def compute_causal_weights(self, 
                             time_losses: List[torch.Tensor],
                             time_points: Optional[List[float]] = None) -> List[float]:
        """
        計算時間因果權重
        
        Args:
            time_losses: 按時間順序的損失列表
            time_points: 對應的時間點（可選）
            
        Returns:
            時間因果權重列表
        """
        if len(time_losses) <= 1:
            return [1.0] * len(time_losses)
        
        # 計算累積誤差
        accumulated_error = 0.0
        weights = []
        
        for i, loss in enumerate(time_losses):
            # 當前時間步的基礎權重
            base_weight = 1.0
            
            # 根據累積誤差調整權重
            if i == 0:
                # 第一個時間步總是全權重
                weight = base_weight
            else:
                # 基於前面累積誤差計算權重
                causal_factor = math.exp(-self.causality_strength * accumulated_error)
                weight = base_weight * causal_factor
            
            weights.append(weight)
            
            # 更新累積誤差（使用 detach 避免梯度計算）
            accumulated_error += loss.detach().item()
        
        # 正規化權重
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight * len(weights) for w in weights]
        
        return weights
    
    def apply_temporal_decay(self, 
                           weights: List[float], 
                           current_epoch: int) -> List[float]:
        """
        應用時間衰減策略
        
        隨著訓練進行，逐漸減少因果約束強度，允許模型更好地擬合後期時間
        """
        decay_factor = math.exp(-self.decay_rate * current_epoch / 1000.0)
        causality_factor = self.causality_strength * decay_factor
        
        # 重新計算衰減後的權重
        adjusted_weights = []
        for i, w in enumerate(weights):
            if i == 0:
                adjusted_weights.append(w)
            else:
                # 後期時間步受到較少的因果約束
                causal_adjustment = 1.0 + (causality_factor - 1.0) * math.exp(-i * 0.1)
                adjusted_weights.append(w * causal_adjustment)
        
        return adjusted_weights
    
class NTKWeighter:
    """
    神經正切核 (NTK) 權重器
    
    基於神經正切核理論的權重策略，通過分析不同損失項對應的核函數
    來動態調整權重，確保訓練過程中的理論收斂性。
    """
    
    def __init__(self,
                 model: nn.Module,
                 loss_names: List[str] = None,
                 sample_size: int = 100,
                 update_frequency: int = 2000,
                 update_freq: Optional[int] = None,
                 reg_param: float = 1e-6):
        """
        Args:
            model: PINN 模型
            loss_names: 損失項名稱列表
            sample_size: NTK 計算的採樣點數量
            update_frequency: 權重更新頻率
            update_freq: 權重更新頻率（別名，用於相容性）
            reg_param: 正則化參數
        """
        self.model = model
        self.loss_names = loss_names or ['data', 'pde']  # 預設損失名稱
        self.sample_size = sample_size
        # 優先使用 update_freq 參數（測試相容性）
        self.update_frequency = update_freq if update_freq is not None else update_frequency
        self.reg_param = reg_param
        self.step_count = 0
        self.ntk_weights = {name: 1.0 for name in self.loss_names}
        
    def compute_ntk_eigenvalues(self, 
                              inputs: torch.Tensor,
                              loss_type: str) -> torch.Tensor:
        """
        計算 NTK 特徵值
        
        Args:
            inputs: 輸入採樣點
            loss_type: 損失類型
            
        Returns:
            NTK 矩陣的特徵值
        """
        # 簡化的 NTK 估計（完整實現需要更複雜的核計算）
        batch_size = min(self.sample_size, inputs.shape[0])
        sample_inputs = inputs[:batch_size]
        
        # 計算雅可比矩陣
        outputs = self.model(sample_inputs)
        jacobians = []
        
        for i in range(outputs.shape[1]):  # 對每個輸出維度
            grads = torch.autograd.grad(
                outputs[:, i].sum(),
                self.model.parameters(),
                retain_graph=True,
                create_graph=False
            )
            jacobian = torch.cat([g.view(-1) for g in grads])
            jacobians.append(jacobian)
        
        J = torch.stack(jacobians, dim=0)  # [output_dim, param_dim]
        
        # 計算 NTK 矩陣 K = J @ J^T
        K = J @ J.T
        
        # 計算特徵值
        eigenvals = torch.linalg.eigvals(K).real
        
        return eigenvals
    
    def update_ntk_weights(self, 
                         data_inputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        基於 NTK 分析更新權重
        
        Args:
            data_inputs: 不同損失項對應的輸入數據
            
        Returns:
            更新後的 NTK 權重
        """
        self.step_count += 1
        
        if self.step_count % self.update_frequency != 0:
            return self.ntk_weights.copy()
    
    def update_weights(self, 
                      losses: Dict[str, torch.Tensor],
                      x_train: torch.Tensor,
                      step: int = 0) -> Dict[str, float]:
        """
        更新權重（測試相容性方法）
        
        Args:
            losses: 損失字典
            x_train: 訓練輸入數據
            step: 當前步數
            
        Returns:
            更新後的權重字典
        """
        # 構建 data_inputs 格式
        data_inputs = {}
        for name in losses.keys():
            if name in self.loss_names:
                data_inputs[name] = x_train
        
        # 設置步數並調用原方法
        self.step_count = step
        return self.update_ntk_weights(data_inputs)
        
        ntk_metrics = {}
        
        for loss_name, inputs in data_inputs.items():
            if loss_name not in self.loss_names:
                continue
                
            try:
                eigenvals = self.compute_ntk_eigenvalues(inputs, loss_name)
                
                # 計算 NTK 度量（條件數的倒數作為權重指標）
                max_eigenval = torch.max(eigenvals)
                min_eigenval = torch.max(eigenvals[eigenvals > 1e-12])
                condition_number = max_eigenval / (min_eigenval + 1e-12)
                
                # 條件數越小，權重越大（更好的收斂性）
                ntk_metrics[loss_name] = 1.0 / (condition_number + 1.0)
                
            except Exception as e:
                # 如果計算失敗，保持當前權重
                ntk_metrics[loss_name] = self.ntk_weights[loss_name]
        
        # 正規化權重
        if ntk_metrics:
            total_metric = sum(ntk_metrics.values())
            for name in self.loss_names:
                if name in ntk_metrics:
                    self.ntk_weights[name] = ntk_metrics[name] / (total_metric + 1e-12)
        
        return self.ntk_weights.copy()


class AdaptiveWeightScheduler:
    """
    自適應權重調度器
    
    結合多種權重策略，根據訓練階段動態調整權重組合。
    支援訓練初期、中期、後期的不同權重策略。
    """
    
    def __init__(self,
                 loss_names: List[str],
                 phases: Dict[str, Dict] = None,
                 adaptation_method: str = 'exponential',
                 adaptation_rate: float = 0.1):
        """
        Args:
            loss_names: 損失項名稱列表
            phases: 訓練階段配置字典
            adaptation_method: 適應方法 ('exponential', 'linear', 'cosine')
            adaptation_rate: 適應率
        """
        self.loss_names = loss_names
        self.adaptation_method = adaptation_method
        self.adaptation_rate = adaptation_rate
        
        # 預設三階段訓練策略
        if phases is None:
            phases = {
                'warmup': {
                    'duration_ratio': 0.1,      # 前 10% 的訓練步數
                    'primary_losses': ['data'],  # 主要關注資料擬合
                    'weight_ratios': {'data': 2.0, 'residual': 0.5, 'boundary': 1.0}
                },
                'main': {
                    'duration_ratio': 0.7,      # 中間 70% 的訓練步數
                    'primary_losses': ['residual', 'boundary'],  # 主要關注物理一致性
                    'weight_ratios': {'data': 1.0, 'residual': 2.0, 'boundary': 1.5}
                },
                'refinement': {
                    'duration_ratio': 0.2,      # 最後 20% 的訓練步數
                    'primary_losses': ['data', 'residual'],  # 平衡擬合
                    'weight_ratios': {'data': 1.5, 'residual': 1.5, 'boundary': 1.0}
                }
            }
        
        self.phases = phases
        self.current_phase = 'warmup'
        
    def get_current_phase(self, current_step: int, total_steps: int) -> str:
        """確定當前訓練階段"""
        progress = current_step / total_steps
        
        warmup_end = self.phases['warmup']['duration_ratio']
        main_end = warmup_end + self.phases['main']['duration_ratio']
        
        if progress <= warmup_end:
            return 'warmup'
        elif progress <= main_end:
            return 'main'
        else:
            return 'refinement'
    
    def get_phase_weights(self, 
                         current_step: int, 
                         total_steps: int) -> Dict[str, float]:
        """
        獲取當前階段的權重配置
        
        Args:
            current_step: 當前訓練步數
            total_steps: 總訓練步數
            
        Returns:
            階段權重字典
        """
        phase = self.get_current_phase(current_step, total_steps)
        self.current_phase = phase
        
        phase_config = self.phases[phase]
        weight_ratios = phase_config['weight_ratios']
        
        # 生成標準化權重
        weights = {}
        for name in self.loss_names:
            weights[name] = weight_ratios.get(name, 1.0)
        
        return weights
    
    def update_weights(self, 
                      losses: Dict[str, torch.Tensor], 
                      step: int,
                      total_steps: int = 10000) -> Dict[str, float]:
        """
        根據當前訓練步數和損失值動態更新權重
        
        Args:
            losses: 當前損失值字典
            step: 當前訓練步數  
            total_steps: 總訓練步數
            
        Returns:
            更新後的權重字典
        """
        # 獲取當前階段的基礎權重
        base_weights = self.get_phase_weights(step, total_steps)
        
        # 根據損失大小動態調整（可選的自適應調整）
        if self.adaptation_method == 'exponential':
            # 損失大的項獲得更高權重
            loss_magnitudes = {}
            for name, loss in losses.items():
                if name in self.loss_names:
                    loss_magnitudes[name] = float(loss.detach())
            
            # 正規化並應用到基礎權重
            if loss_magnitudes:
                max_loss = max(loss_magnitudes.values()) + 1e-12
                adaptive_weights = {}
                for name in self.loss_names:
                    if name in loss_magnitudes:
                        # 損失越大，權重適應性調整越強
                        loss_ratio = loss_magnitudes[name] / max_loss
                        adaptation_factor = 1.0 + self.adaptation_rate * loss_ratio
                        adaptive_weights[name] = base_weights[name] * adaptation_factor
                    else:
                        adaptive_weights[name] = base_weights[name]
                
                return adaptive_weights
        
        # 如果無法計算適應性權重，返回基礎權重
        return base_weights
    
    def get_weights(self) -> Dict[str, float]:
        """
        獲取當前權重（預設均等權重）
        
        Returns:
            當前權重字典
        """
        return {name: 1.0 for name in self.loss_names}
    
    def combine_weights(self,
                       phase_weights: Dict[str, float],
                       dynamic_weights: Dict[str, float],
                       combination_ratio: float = 0.7) -> Dict[str, float]:
        """
        組合階段權重與動態權重
        
        Args:
            phase_weights: 階段權重
            dynamic_weights: 動態權重（如 GradNorm）
            combination_ratio: 組合比例（階段權重的比重）
            
        Returns:
            組合後的最終權重
        """
        final_weights = {}
        
        for name in self.loss_names:
            phase_w = phase_weights.get(name, 1.0)
            dynamic_w = dynamic_weights.get(name, 1.0)
            
            # 加權組合
            final_w = combination_ratio * phase_w + (1 - combination_ratio) * dynamic_w
            final_weights[name] = final_w
        
        return final_weights


class MultiWeightManager:
    """
    多權重策略管理器
    
    整合 GradNorm、Causal、NTK 等多種權重策略，提供統一的權重管理接口。
    """
    
    def __init__(self,
                 objectives_or_model = None,  # 兼容測試期待的第一個參數
                 loss_names: List[str] = None,
                 strategies: List[str] = ['gradnorm', 'adaptive'],
                 strategy_weights: Optional[Dict[str, float]] = None,
                 method: str = 'weighted_sum',
                 preference_weights: List[float] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Args:
            objectives_or_model: 目標列表（測試模式）或 PINN 模型（正常模式）
            loss_names: 損失項名稱列表  
            strategies: 權重策略列表
            strategy_weights: 策略權重字典
            method: 多目標方法 ('weighted_sum', 'pareto')
            preference_weights: 偏好權重列表
            config: 策略配置
        """
        self.config = config or {}
        if strategies is None:
            strategies = self.config.get('strategies', ['gradnorm', 'adaptive'])
        # 判斷是測試模式還是正常模式
        if isinstance(objectives_or_model, list):
            # 測試模式：objectives_or_model 是目標列表
            self.objectives = objectives_or_model
            self.loss_names = objectives_or_model
            self.model = None
            self.method = self.config.get('method', method)
            base_pref = preference_weights or self.config.get(
                'preference_weights',
                [1.0/len(objectives_or_model)] * len(objectives_or_model)
            )
        else:
            # 正常模式：objectives_or_model 是模型
            self.model = objectives_or_model
            self.loss_names = loss_names or ['data', 'residual']
            self.objectives = self.loss_names
            self.method = self.config.get('method', method if method else 'weighted_sum')
            base_pref = preference_weights or self.config.get(
                'preference_weights',
                [1.0/len(self.loss_names)] * len(self.loss_names)
            )
        self.strategies = strategies

        # 正規化偏好權重長度
        self.preference_weights = [float(w) for w in base_pref]
        if len(self.preference_weights) < len(self.objectives):
            missing = len(self.objectives) - len(self.preference_weights)
            fill_value = 1.0 / max(1, len(self.objectives))
            self.preference_weights.extend([fill_value] * missing)
        elif len(self.preference_weights) > len(self.objectives):
            self.preference_weights = self.preference_weights[:len(self.objectives)]

        if strategy_weights is None:
            strategy_weights = self.config.get('strategy_weights')
        if strategy_weights is None:
            strategy_weights = {strategy: 1.0/len(strategies) for strategy in strategies}
        self.strategy_weights = strategy_weights
        
        # 初始化各權重器
        self.weighters = {}
        
        if 'gradnorm' in strategies:
            if self.model is not None:
                gradnorm_kwargs = self._build_gradnorm_kwargs()
                self.weighters['gradnorm'] = GradNormWeighter(
                    self.model,
                    self.loss_names,
                    **gradnorm_kwargs
                )
            else:
                import logging
                logging.warning("GradNorm requires model reference, skipping in test mode")
                self.weighters['gradnorm'] = None
            
        if 'causal' in strategies:
            self.weighters['causal'] = CausalWeighter()
            
        if 'ntk' in strategies:
            if self.model is not None:
                self.weighters['ntk'] = NTKWeighter(self.model, self.loss_names)
            else:
                import logging
                logging.warning("NTK weighting requires model reference, skipping in test mode")
                self.weighters['ntk'] = None
            
        if 'adaptive' in strategies:
            if self.loss_names is not None:
                self.weighters['adaptive'] = AdaptiveWeightScheduler(self.loss_names)
            else:
                import logging
                logging.warning("Adaptive weighting requires loss_names, skipping in test mode")
                self.weighters['adaptive'] = None
    
    def _build_gradnorm_kwargs(self) -> Dict[str, Any]:
        """提取 GradNorm 初始化參數"""
        valid_keys = {
            'alpha',
            'update_frequency',
            'initial_weights',
            'target_gradient_ratio',
            'target_ratios',
            'device',
            'min_weight',
            'max_weight'
        }
        gradnorm_kwargs: Dict[str, Any] = {}
        
        gradnorm_cfg = self.config.get('gradnorm', {})
        if isinstance(gradnorm_cfg, dict):
            for key in valid_keys:
                if key in gradnorm_cfg:
                    gradnorm_kwargs[key] = gradnorm_cfg[key]
            # 支援簡寫名稱
            if 'update_freq' in gradnorm_cfg and 'update_frequency' not in gradnorm_kwargs:
                gradnorm_kwargs['update_frequency'] = gradnorm_cfg['update_freq']
        
        return gradnorm_kwargs
    
    def _update_objective_mode(self, losses: Optional[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """處理純多目標權重更新（無模型參與的情況）"""
        if not self.objectives:
            return {}

        eps = 1e-12
        weights = {}

        for idx, name in enumerate(self.objectives):
            pref = self.preference_weights[idx]
            value = losses.get(name) if losses else None
            if isinstance(value, torch.Tensor):
                magnitude = float(value.detach().abs().item())
            elif value is not None:
                magnitude = float(abs(value))
            else:
                magnitude = 1.0

            if self.method == 'pareto':
                weight = pref * (magnitude + eps)
            else:
                weight = pref / (magnitude + eps)
            weights[name] = max(weight, eps)

        total_weight = sum(weights.values()) or 1.0
        for name in weights:
            weights[name] /= total_weight

        return weights

    def update_weights(self,
                      losses: Dict[str, torch.Tensor],
                      current_step: int = 0,
                      total_steps: int = 100000,
                      time_losses: Optional[List[torch.Tensor]] = None,
                      data_inputs: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
        """
        更新綜合權重

        Args:
            losses: 當前損失字典
            current_step: 當前步數
            total_steps: 總步數
            time_losses: 時間序列損失（用於 causal）
            data_inputs: 輸入數據（用於 NTK）
            
        Returns:
            最終權重字典
        """
        if self.model is None:
            return self._update_objective_mode(losses)

        strategy_results = {}
        
        # 計算各策略的權重
        if 'gradnorm' in self.weighters and self.weighters['gradnorm'] is not None:
            strategy_results['gradnorm'] = self.weighters['gradnorm'].update_weights(losses)
        
        if 'adaptive' in self.weighters and self.weighters['adaptive'] is not None:
            strategy_results['adaptive'] = self.weighters['adaptive'].get_phase_weights(
                current_step, total_steps)
        
        if 'causal' in self.weighters and time_losses:
            causal_weights = self.weighters['causal'].compute_causal_weights(time_losses)
            # 將 causal 權重轉換為字典格式（假設按順序對應）
            if len(causal_weights) == len(self.loss_names):
                strategy_results['causal'] = dict(zip(self.loss_names, causal_weights))
        
        if 'ntk' in self.weighters and data_inputs:
            strategy_results['ntk'] = self.weighters['ntk'].update_ntk_weights(data_inputs)
        
        # 組合所有策略的結果
        final_weights = {}
        
        for name in self.loss_names:
            combined_weight = 0.0
            total_strategy_weight = 0.0
            
            for strategy, results in strategy_results.items():
                if strategy in self.strategy_weights and name in results:
                    strategy_w = self.strategy_weights[strategy]
                    result_w = results[name]
                    combined_weight += strategy_w * result_w
                    total_strategy_weight += strategy_w
            
            # 正規化
            if total_strategy_weight > 0:
                final_weights[name] = combined_weight / total_strategy_weight
            else:
                final_weights[name] = 1.0
        
        return final_weights
    
    def get_weights(self) -> Dict[str, float]:
        """獲取當前權重（無更新）"""
        weights = {}
        for name in self.loss_names:
            weights[name] = 1.0  # 預設權重
            
        # 獲取最新的 gradnorm 權重
        if 'gradnorm' in self.weighters:
            gradnorm_weights = self.weighters['gradnorm'].get_weights()
            for name in self.loss_names:
                if name in gradnorm_weights:
                    weights[name] = gradnorm_weights[name]
        
        return weights


# 便捷函數
def create_weight_manager(model: nn.Module,
                         loss_names: List[str],
                         config: Optional[Dict[str, Any]] = None) -> MultiWeightManager:
    """
    創建權重管理器的便捷函數
    
    Args:
        model: PINN 模型
        loss_names: 損失項名稱列表
        config: 配置字典
        
    Returns:
        配置好的權重管理器
    """
    if config is None:
        config = {
            'strategies': ['gradnorm', 'adaptive'],
            'gradnorm': {
                'alpha': 0.12,
                'update_frequency': 1000
            },
            'adaptive_phases': None  # 使用預設階段
        }
    
    strategies = config.get('strategies', ['gradnorm', 'adaptive'])
    strategy_weights = config.get('strategy_weights')
    method = config.get('method', 'weighted_sum')
    preference_weights = config.get('preference_weights')
    
    return MultiWeightManager(
        model=model,
        loss_names=loss_names,
        strategies=strategies,
        strategy_weights=strategy_weights,
        method=method,
        preference_weights=preference_weights,
        config=config
    )


if __name__ == "__main__":
    # 測試程式碼
    print("🧪 測試動態權重模組...")
    
    # 創建簡單測試模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 3)
        def forward(self, x):
            return self.linear(x)
    
    model = TestModel()
    loss_names = ['data', 'residual', 'boundary', 'prior']
    
    # 測試 GradNorm
    print("測試 GradNorm...")
    gradnorm = GradNormWeighter(model, loss_names)
    
    # 模擬損失
    test_losses = {
        'data': torch.tensor(2.0, requires_grad=True),
        'residual': torch.tensor(0.5, requires_grad=True),
        'boundary': torch.tensor(1.0, requires_grad=True),
        'prior': torch.tensor(0.8, requires_grad=True)
    }
    
    weights = gradnorm.update_weights(test_losses)
    print(f"GradNorm 權重: {weights}")
    
    # 測試 Causal Weighter
    print("\n測試 Causal Weighter...")
    causal = CausalWeighter()
    time_losses = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(1.5)]
    causal_weights = causal.compute_causal_weights(time_losses)
    print(f"Causal 權重: {causal_weights}")
    
    # 測試多權重管理器
    print("\n測試 MultiWeightManager...")
    manager = create_weight_manager(model, loss_names)
    final_weights = manager.update_weights(test_losses, current_step=1000)
    print(f"最終權重: {final_weights}")
    
    print("✅ 動態權重模組測試完成！")
