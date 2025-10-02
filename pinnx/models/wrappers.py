"""
PINN 模型包裝器模組

提供各種 PINN 模型包裝功能，支援：
- VS-PINN 變數尺度化整合
- 多變數物理場輸出 (u,v,p,源項等)
- 物理約束與邊界條件強制
- 時間因果性與保守律
- 模型組合與集成架構

主要用於將基礎神經網路包裝成完整的 PINN 求解器。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import numpy as np

from .fourier_mlp import PINNNet, MultiScalePINNNet
from ..physics.scaling import VSScaler, StandardScaler, MinMaxScaler


class MultiHeadWrapper(nn.Module):
    """
    多頭輸出包裝器
    
    將單一神經網路包裝成多個輸出頭，每個頭負責預測不同的物理變量。
    例如：速度場、壓力場、源項等。
    """
    
    def __init__(self, 
                 base_model: nn.Module,
                 head_configs: List[Dict[str, any]]):
        """
        Args:
            base_model: 基礎神經網路模型
            head_configs: 輸出頭配置列表，每個配置包含:
                - name: 變數名稱
                - dim: 輸出維度
                - activation: 激活函數 ('tanh', 'relu', 'sigmoid', None)
        """
        super().__init__()
        
        self.base_model = base_model
        self.head_configs = head_configs
        
        # 計算基礎模型的輸出維度
        if hasattr(base_model, 'out_dim'):
            base_out_dim = base_model.out_dim
        else:
            # 嘗試推斷輸出維度
            with torch.no_grad():
                test_input = torch.randn(1, 2)  # 假設2D輸入
                test_output = base_model(test_input)
                base_out_dim = test_output.shape[-1]
        
        # 建立輸出頭
        self.heads = nn.ModuleDict()
        total_out_dim = sum(config['dim'] for config in head_configs)
        
        # 如果基礎模型輸出維度與總需求不符，需要調整
        if base_out_dim != total_out_dim:
            # 添加一個線性層來調整維度
            self.projection = nn.Linear(base_out_dim, total_out_dim)
        else:
            self.projection = None
        
        # 為每個頭建立激活函數
        for config in head_configs:
            name = config['name']
            activation = config.get('activation', None)
            
            if activation == 'tanh':
                self.heads[name] = nn.Tanh()
            elif activation == 'relu':
                self.heads[name] = nn.ReLU()
            elif activation == 'sigmoid':
                self.heads[name] = nn.Sigmoid()
            else:
                self.heads[name] = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向傳播，返回字典格式的輸出
        
        Args:
            x: 輸入張量 [batch_size, in_dim]
            
        Returns:
            字典，鍵為變數名稱，值為對應的預測張量
        """
        # 基礎模型前向傳播
        base_output = self.base_model(x)
        
        # 如果需要，調整輸出維度
        if self.projection is not None:
            base_output = self.projection(base_output)
        
        # 分解到各個輸出頭
        outputs = {}
        start_idx = 0
        
        for config in self.head_configs:
            name = config['name']
            dim = config['dim']
            
            # 提取對應的輸出片段
            head_output = base_output[:, start_idx:start_idx + dim]
            
            # 應用激活函數
            outputs[name] = self.heads[name](head_output)
            
            start_idx += dim
        
        return outputs
    
    def get_variable_names(self) -> List[str]:
        """返回所有變數名稱"""
        return [config['name'] for config in self.head_configs]
    
    def get_total_output_dim(self) -> int:
        """返回總輸出維度"""
        return sum(config['dim'] for config in self.head_configs)


class ScaledPINNWrapper(nn.Module):
    """
    整合 VS-PINN 尺度化的 PINN 包裝器
    
    將神經網路與變數尺度器組合，提供完整的輸入標準化、
    輸出反標準化，以及尺度化梯度計算功能。
    """
    
    def __init__(self, 
                 base_model: nn.Module,
                 input_scaler: Optional[Union[VSScaler, StandardScaler, MinMaxScaler]] = None,
                 output_scaler: Optional[Union[VSScaler, StandardScaler, MinMaxScaler]] = None,
                 variable_names: Optional[List[str]] = None):
        """
        Args:
            base_model: 基礎神經網路模型
            input_scaler: 輸入變數尺度器 (t,x,y)
            output_scaler: 輸出變數尺度器 (u,v,p,S)
            variable_names: 輸出變數名稱列表
        """
        super().__init__()
        
        self.base_model = base_model
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        
        # 動態調整變數名稱以匹配模型輸出維度
        if hasattr(base_model, 'out_dim'):
            if variable_names is None:
                # 根據輸出維度生成預設變數名稱
                if base_model.out_dim == 1:
                    self.variable_names = ['u']
                elif base_model.out_dim == 2:
                    self.variable_names = ['u', 'v']
                elif base_model.out_dim == 3:
                    self.variable_names = ['u', 'v', 'p']
                elif base_model.out_dim == 4:
                    self.variable_names = ['u', 'v', 'p', 'S']
                else:
                    # 生成通用變數名稱
                    self.variable_names = [f'var_{i}' for i in range(base_model.out_dim)]
            else:
                # 檢查用戶提供的變數名稱
                if len(variable_names) != base_model.out_dim:
                    print(f"警告: 變數名稱數量 ({len(variable_names)}) 與模型輸出維度 ({base_model.out_dim}) 不符，將自動調整")
                    if len(variable_names) > base_model.out_dim:
                        self.variable_names = variable_names[:base_model.out_dim]
                    else:
                        self.variable_names = variable_names + [f'var_{i}' for i in range(len(variable_names), base_model.out_dim)]
                else:
                    self.variable_names = variable_names
        else:
            self.variable_names = variable_names or ['u', 'v', 'p', 'S']
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播：輸入標準化 -> 網路預測 -> 輸出反標準化
        """
        # 輸入標準化
        if self.input_scaler is not None:
            x_scaled = self.input_scaler.encode(x)
        else:
            x_scaled = x
        
        # 網路預測
        y_scaled = self.base_model(x_scaled)
        
        # 輸出反標準化
        if self.output_scaler is not None:
            y = self.output_scaler.decode(y_scaled)
        else:
            y = y_scaled
        
        return y
    
    def predict_dict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        返回字典格式的預測結果，便於索引各物理變數
        """
        y = self.forward(x)
        result = {}
        for i, name in enumerate(self.variable_names):
            result[name] = y[..., i]
        return result
    
    def compute_gradients(self, x: torch.Tensor, 
                         var_name: str, 
                         wrt_vars: List[str] = ['x', 'y']) -> Dict[str, torch.Tensor]:
        """
        計算指定變數對座標的梯度（考慮尺度化影響）
        
        Args:
            x: 輸入座標，需要 requires_grad=True
            var_name: 求梯度的變數名稱
            wrt_vars: 對哪些座標求梯度
        
        Returns:
            梯度字典，例如 {'dx': du_dx, 'dy': du_dy}
        """
        if var_name not in self.variable_names:
            raise ValueError(f"變數 '{var_name}' 不在模型輸出中: {self.variable_names}")
        
        var_idx = self.variable_names.index(var_name)
        y = self.forward(x)
        var = y[..., var_idx]
        
        # 計算梯度
        gradients = torch.autograd.grad(
            var.sum(), x, 
            create_graph=True, retain_graph=True
        )[0]
        
        # 解析座標名稱到索引的映射
        coord_map = {'t': 0, 'x': 1, 'y': 2, 'z': 3}  # 支援時空座標
        
        result = {}
        for wrt_var in wrt_vars:
            if wrt_var in coord_map:
                coord_idx = coord_map[wrt_var]
                if coord_idx < gradients.shape[-1]:
                    result[f'd{wrt_var}'] = gradients[..., coord_idx]
        
        return result
    
    def get_scalers_info(self) -> Dict:
        """返回尺度器資訊，用於除錯與分析"""
        info = {}
        if self.input_scaler is not None:
            info['input_scaler'] = {
                'type': type(self.input_scaler).__name__,
                'params': self.input_scaler.get_params() if hasattr(self.input_scaler, 'get_params') else None
            }
        if self.output_scaler is not None:
            info['output_scaler'] = {
                'type': type(self.output_scaler).__name__,
                'params': self.output_scaler.get_params() if hasattr(self.output_scaler, 'get_params') else None
            }
        return info


class ResidualWrapper(nn.Module):
    """在基礎模型輸出上疊加殘差分支以增強表達能力"""

    def __init__(self,
                 base_model: nn.Module,
                 residual_config: Optional[Dict[str, Any]] = None):
        super().__init__()

        config = residual_config or {}
        self.base_model = base_model
        self.skip_connections = config.get('skip_connections', [])
        self.residual_scale = float(config.get('residual_scale', 1.0))

        try:
            first_param = next(base_model.parameters())
            self.model_device = first_param.device
        except StopIteration:
            self.model_device = torch.device('cpu')

        # 判定輸入與輸出維度
        self.input_dim = config.get('input_dim', getattr(base_model, 'in_dim', None))
        if self.input_dim is None:
            self.input_dim = getattr(base_model, 'input_dim', None)
        if self.input_dim is None:
            raise ValueError("ResidualWrapper 需要透過 base_model 屬性或 residual_config['input_dim'] 指定輸入維度")

        self.output_dim = config.get('output_dim', getattr(base_model, 'out_dim', None))
        if self.output_dim is None:
            self.output_dim = getattr(base_model, 'output_dim', None)
        if self.output_dim is None:
            with torch.no_grad():
                probe = torch.zeros(1, self.input_dim, device=self.model_device)
                self.output_dim = base_model(probe).shape[-1]

        branch_count = max(1, len(self.skip_connections) or 1)
        self.residual_branches = nn.ModuleList()
        for _ in range(branch_count):
            layer = nn.Linear(self.input_dim, self.output_dim)
            nn.init.xavier_normal_(layer.weight, gain=0.1)
            nn.init.zeros_(layer.bias)
            self.residual_branches.append(layer)

        activation_name = config.get('activation', 'identity')
        self.activation = self._resolve_activation(activation_name)

    @staticmethod
    def _resolve_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
        name = (name or 'identity').lower()
        if name == 'tanh':
            return torch.tanh
        if name == 'relu':
            return F.relu
        if name == 'sigmoid':
            return torch.sigmoid
        return lambda x: x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_model(x)

        residual = None
        for branch in self.residual_branches:
            branch_out = branch(x)
            residual = branch_out if residual is None else residual + branch_out

        if residual is None:
            residual = torch.zeros_like(base_output)

        residual = self.activation(residual)
        return base_output + self.residual_scale * residual


class PhysicsConstrainedWrapper(nn.Module):
    """
    物理約束 PINN 包裝器
    
    在神經網路輸出上強制施加物理約束，例如：
    - 不可壓縮條件 (∇·u = 0)
    - 邊界條件 (無滑移、對稱等)
    - 守恆定律 (質量、動量守恆)
    """
    
    def __init__(self, 
                 base_wrapper: ScaledPINNWrapper,
                 constraints: List[str] = None):
        """
        Args:
            base_wrapper: 基礎 PINN 包裝器
            constraints: 約束類型列表，例如 ['incompressible', 'no_slip']
        """
        super().__init__()
        
        self.base_wrapper = base_wrapper
        self.constraints = constraints or []
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向傳播，應用物理約束"""
        y = self.base_wrapper(x)
        
        # 應用約束
        if 'incompressible' in self.constraints:
            y = self._apply_incompressible_constraint(x, y)
        
        if 'no_slip' in self.constraints:
            y = self._apply_no_slip_constraint(x, y)
        
        return y
    
    def _apply_incompressible_constraint(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        強制不可壓縮約束：調整壓力場使 ∇·u = 0
        這是一個簡化實現，實際可能需要更複雜的投影方法
        """
        # 這裡可以實現投影到無散度空間的方法
        # 暫時返回原始輸出
        return y
    
    def _apply_no_slip_constraint(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        強制無滑移邊界條件：在邊界上速度為零
        需要根據具體的邊界定義來實現
        """
        # 這裡可以實現邊界條件的硬約束
        # 暫時返回原始輸出  
        return y


class EnsemblePINNWrapper(nn.Module):
    """
    集成 PINN 包裝器：管理多個 PINN 模型的集成預測與不確定性量化
    """
    
    def __init__(self, 
                 models: List[nn.Module],
                 weights: Optional[torch.Tensor] = None,
                 uncertainty_method: str = 'variance'):
        """
        Args:
            models: PINN 模型列表
            weights: 模型權重，預設為等權重
            uncertainty_method: 不確定性量化方法 ('variance', 'std', 'entropy')
        """
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        
        if weights is None:
            self.register_buffer('weights', torch.ones(self.num_models) / self.num_models)
        else:
            self.register_buffer('weights', weights)
        
        self.uncertainty_method = uncertainty_method
    
    def forward(self, x: torch.Tensor, mode: str = 'mean') -> torch.Tensor:
        """
        前向傳播，支援不同的集成模式
        
        Args:
            x: 輸入張量
            mode: 集成模式
                - 'mean': 返回集成平均 (預設)
                - 'stats': 返回統計字典 {'mean': mean, 'std': std, 'var': var}
                - 'all': 返回所有模型預測
        
        Returns:
            根據模式返回不同格式的結果
        """
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # 堆疊 [num_models, batch_size, out_dim]
        stacked = torch.stack(predictions, dim=0)
        
        if mode == 'mean':
            # 加權平均
            weighted_mean = torch.einsum('m,mbo->bo', self.weights, stacked)
            return weighted_mean
            
        elif mode == 'stats':
            # 返回統計字典
            mean = torch.einsum('m,mbo->bo', self.weights, stacked)
            var = torch.var(stacked, dim=0)
            std = torch.std(stacked, dim=0)
            min_vals = torch.min(stacked, dim=0)[0]  # 新增
            max_vals = torch.max(stacked, dim=0)[0]  # 新增
            
            return {
                'mean': mean,
                'var': var,
                'std': std,
                'uncertainty': std,  # 別名
                'min': min_vals,     # 新增
                'max': max_vals      # 新增
            }
            
        elif mode == 'all':
            # 返回所有模型的預測
            return stacked
            
        else:
            raise ValueError(f"不支援的模式: {mode}")
        
        # 舊的實作保持不變（用於向後相容）
        # 加權平均
        weighted_mean = torch.einsum('m,mbo->bo', self.weights, stacked)
        return weighted_mean
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回集成預測與不確定性估計
        
        Returns:
            mean: 集成平均 [batch_size, out_dim]
            uncertainty: 不確定性估計 [batch_size, out_dim]
        """
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # 堆疊 [num_models, batch_size, out_dim]
        stacked = torch.stack(predictions, dim=0)
        
        # 加權平均
        mean = torch.einsum('m,mbo->bo', self.weights, stacked)
        
        # 計算不確定性
        if self.uncertainty_method == 'variance':
            # 方差作為不確定性指標
            variance = torch.var(stacked, dim=0)
            uncertainty = variance
        elif self.uncertainty_method == 'std':
            # 標準差作為不確定性指標
            std = torch.std(stacked, dim=0)
            uncertainty = std
        else:
            raise ValueError(f"不支援的不確定性方法: {self.uncertainty_method}")
        
        return mean, uncertainty
    
    def predict_all_models(self, x: torch.Tensor) -> torch.Tensor:
        """返回所有模型的預測結果 [num_models, batch_size, out_dim]"""
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        return torch.stack(predictions, dim=0)


class AdaptivePINNWrapper(nn.Module):
    """
    自適應 PINN 包裝器：根據訓練進程動態調整網路行為
    
    支援功能：
    - 動態激活函數選擇
    - 自適應 Fourier 頻率
    - 殘差自適應權重
    """
    
    def __init__(self, 
                 base_wrapper: ScaledPINNWrapper,
                 adaptation_schedule: Dict = None):
        super().__init__()
        
        self.base_wrapper = base_wrapper
        self.adaptation_schedule = adaptation_schedule or {}
        self.training_step = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向傳播，應用自適應策略"""
        # 更新訓練步數
        if self.training:
            self.training_step += 1
        
        # 應用自適應策略
        self._apply_adaptations()
        
        return self.base_wrapper(x)
    
    def _apply_adaptations(self):
        """根據訓練進程應用自適應策略"""
        # 這裡可以實現各種自適應策略
        # 例如：動態調整 Fourier 頻率、學習率、權重等
        pass


# 為測試相容性提供別名（保留向後相容性）
# MultiHeadWrapper 已經是正確的類別，不需要別名
ResidualWrapper = ScaledPINNWrapper   # 殘差包裝器使用 ScaledPINNWrapper
EnsembleWrapper = EnsemblePINNWrapper  # 集成模型別名

# 便捷建構函數
def create_scaled_pinn(model_config: Dict, 
                      scaler_config: Dict = None,
                      variable_names: List[str] = None) -> ScaledPINNWrapper:
    """
    建立整合尺度化的 PINN 模型
    
    Args:
        model_config: 模型配置字典
        scaler_config: 尺度器配置字典
        variable_names: 輸出變數名稱
    
    Returns:
        完整的 ScaledPINNWrapper
    """
    from .fourier_mlp import create_pinn_model
    
    # 建立基礎模型
    base_model = create_pinn_model(model_config)
    
    # 建立尺度器
    input_scaler = None
    output_scaler = None
    
    if scaler_config:
        if 'input' in scaler_config:
            input_cfg = scaler_config['input']
            if input_cfg['type'] == 'standard':
                input_scaler = StandardScaler(
                    input_cfg.get('mean', 0.0),
                    input_cfg.get('std', 1.0),
                    learnable=input_cfg.get('learnable', False)
                )
            elif input_cfg['type'] == 'vs':
                input_scaler = VSScaler(
                    input_cfg.get('mu_in'),
                    input_cfg.get('std_in'),
                    input_cfg.get('mu_out'),
                    input_cfg.get('std_out'),
                    learnable=input_cfg.get('learnable', True)
                )
        
        if 'output' in scaler_config:
            output_cfg = scaler_config['output']
            if output_cfg['type'] == 'standard':
                output_scaler = StandardScaler(
                    output_cfg.get('mean', 0.0),
                    output_cfg.get('std', 1.0),
                    learnable=output_cfg.get('learnable', False)
                )
            elif output_cfg['type'] == 'vs':
                output_scaler = VSScaler(
                    output_cfg.get('mu_in'),
                    output_cfg.get('std_in'),
                    output_cfg.get('mu_out'),
                    output_cfg.get('std_out'),
                    learnable=output_cfg.get('learnable', True)
                )
    
    return ScaledPINNWrapper(
        base_model=base_model,
        input_scaler=input_scaler,
        output_scaler=output_scaler,
        variable_names=variable_names
    )


def create_ensemble_pinn(model_configs: List[Dict],
                        scaler_config: Dict = None,
                        variable_names: List[str] = None,
                        weights: Optional[torch.Tensor] = None) -> EnsemblePINNWrapper:
    """
    建立集成 PINN 模型
    
    Args:
        model_configs: 多個模型配置字典列表
        scaler_config: 共用的尺度器配置
        variable_names: 輸出變數名稱
        weights: 模型權重
    
    Returns:
        EnsemblePINNWrapper
    """
    models = []
    for config in model_configs:
        model = create_scaled_pinn(config, scaler_config, variable_names)
        models.append(model)
    
    return EnsemblePINNWrapper(models=models, weights=weights)


# 向後相容別名
EnsembleWrapper = EnsemblePINNWrapper


if __name__ == "__main__":
    # 測試程式碼
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    print("=== PINN 包裝器測試 ===")
    
    # 建立基礎模型
    from fourier_mlp import PINNNet
    base_model = PINNNet(in_dim=3, out_dim=4, width=64, depth=2)
    
    # 測試 ScaledPINNWrapper
    print("\n--- ScaledPINNWrapper 測試 ---")
    wrapper = ScaledPINNWrapper(
        base_model=base_model,
        variable_names=['u', 'v', 'p', 'S']
    )
    
    x = torch.randn(50, 3, requires_grad=True)
    y = wrapper(x)
    print(f"輸入形狀: {x.shape}")
    print(f"輸出形狀: {y.shape}")
    
    # 測試字典預測
    pred_dict = wrapper.predict_dict(x)
    print(f"字典預測變數: {list(pred_dict.keys())}")
    print(f"u 變數形狀: {pred_dict['u'].shape}")
    
    # 測試梯度計算
    gradients = wrapper.compute_gradients(x, 'u', ['x', 'y'])
    print(f"梯度計算: {list(gradients.keys())}")
    
    # 測試集成模型
    print("\n--- EnsemblePINNWrapper 測試 ---")
    models = [
        ScaledPINNWrapper(PINNNet(in_dim=3, out_dim=4, width=32, depth=2)),
        ScaledPINNWrapper(PINNNet(in_dim=3, out_dim=4, width=32, depth=2)),
        ScaledPINNWrapper(PINNNet(in_dim=3, out_dim=4, width=32, depth=2))
    ]
    
    ensemble = EnsemblePINNWrapper(models)
    
    with torch.no_grad():
        mean_pred = ensemble(x)
        mean, uncertainty = ensemble.predict_with_uncertainty(x)
    
    print(f"集成平均預測形狀: {mean_pred.shape}")
    print(f"不確定性估計形狀: {uncertainty.shape}")
    print(f"平均不確定性: {uncertainty.mean(0)}")
    
    print("\n✅ 包裝器測試通過！")
