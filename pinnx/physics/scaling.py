"""
VS-PINN 變數尺度化模組
====================

實現Variable Scaling Physics-Informed Neural Networks的尺度化功能：
1. 可學習的輸入/輸出變數尺度化
2. 標準化、最大最小值標準化等傳統方法
3. 從資料自動推斷尺度參數
4. 梯度反標準化 (用於物理定律計算)
"""

import torch
import torch.nn as nn
from typing import Union, Tuple, Optional, Dict, Any
import numpy as np
from abc import ABC, abstractmethod

class BaseScaler(ABC, nn.Module):
    """尺度化器基類"""
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def fit(self, data: torch.Tensor) -> 'BaseScaler':
        """根據資料擬合尺度參數"""
        pass
    
    @abstractmethod
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """將資料轉換到標準化空間"""
        pass
    
    @abstractmethod  
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """將標準化資料轉換回原始空間"""
        pass
    
    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        """擬合並轉換資料"""
        return self.fit(data).transform(data)

class StandardScaler(BaseScaler):
    """
    標準化尺度器: x_norm = (x - μ) / σ
    使用均值和標準差進行標準化
    """
    
    def __init__(self, learnable: bool = False):
        super().__init__()
        self.learnable = learnable
        self.fitted = False
        
    def fit(self, data: torch.Tensor) -> 'StandardScaler':
        """
        根據資料計算均值和標準差
        
        Args:
            data: 輸入資料 [batch_size, feature_dim]
        """
        with torch.no_grad():
            mean = torch.mean(data, dim=0, keepdim=True)
            std = torch.std(data, dim=0, keepdim=True)
            
            # 避免除零錯誤
            std = torch.where(std < 1e-8, torch.ones_like(std), std)
        
        # 註冊參數 (可學習或固定)
        self.register_parameter(
            'mean', 
            nn.Parameter(mean, requires_grad=self.learnable)
        )
        self.register_parameter(
            'std',
            nn.Parameter(std, requires_grad=self.learnable)  
        )
        
        self.fitted = True
        return self
    
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """標準化轉換"""
        if not self.fitted:
            raise RuntimeError("尺度器尚未擬合，請先調用 fit() 方法")
        
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """逆標準化轉換"""
        if not self.fitted:
            raise RuntimeError("尺度器尚未擬合，請先調用 fit() 方法")
        
        return data * self.std + self.mean

class MinMaxScaler(BaseScaler):
    """
    最大最小值尺度器: x_norm = (x - min) / (max - min)
    將資料縮放到 [0, 1] 範圍
    """
    
    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0), 
                 learnable: bool = False):
        super().__init__()
        self.feature_range = feature_range
        self.learnable = learnable  
        self.fitted = False
        
    def fit(self, data: torch.Tensor) -> 'MinMaxScaler':
        """根據資料計算最大最小值"""
        with torch.no_grad():
            data_min = torch.min(data, dim=0, keepdim=True)[0]
            data_max = torch.max(data, dim=0, keepdim=True)[0]
            
            # 避免除零錯誤
            data_range = data_max - data_min
            data_range = torch.where(
                data_range < 1e-8, 
                torch.ones_like(data_range), 
                data_range
            )
        
        self.register_parameter(
            'data_min',
            nn.Parameter(data_min, requires_grad=self.learnable)
        )
        self.register_parameter(
            'data_range', 
            nn.Parameter(data_range, requires_grad=self.learnable)
        )
        
        self.fitted = True
        return self
        
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """最大最小值標準化"""
        if not self.fitted:
            raise RuntimeError("尺度器尚未擬合")
            
        # 標準化到 [0, 1]
        normalized = (data - self.data_min) / self.data_range
        
        # 縮放到指定範圍
        scale = self.feature_range[1] - self.feature_range[0]
        return normalized * scale + self.feature_range[0]
    
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """逆最大最小值標準化"""
        if not self.fitted:
            raise RuntimeError("尺度器尚未擬合")
            
        # 從指定範圍縮放回 [0, 1]
        scale = self.feature_range[1] - self.feature_range[0]
        normalized = (data - self.feature_range[0]) / scale
        
        # 逆標準化到原始範圍
        return normalized * self.data_range + self.data_min

class VSScaler(BaseScaler):
    """
    VS-PINN 變數尺度器
    可學習的輸入輸出尺度化，支援獨立的均值和方差學習
    """
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int, 
                 learnable: bool = True,
                 init_method: str = "standard"):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learnable = learnable
        self.init_method = init_method
        self.fitted = False
        
        # 初始化可學習參數
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """初始化尺度參數"""
        # 輸入尺度參數
        if self.init_method == "standard":
            input_mean = torch.zeros(1, self.input_dim)
            input_std = torch.ones(1, self.input_dim)
        elif self.init_method == "uniform":
            input_mean = torch.zeros(1, self.input_dim)
            input_std = torch.ones(1, self.input_dim)
        else:
            raise ValueError(f"不支援的初始化方法: {self.init_method}")
        
        self.register_parameter(
            'input_mean',
            nn.Parameter(input_mean, requires_grad=self.learnable)
        )
        self.register_parameter(
            'input_std', 
            nn.Parameter(input_std, requires_grad=self.learnable)
        )
        
        # 輸出尺度參數  
        output_mean = torch.zeros(1, self.output_dim)
        output_std = torch.ones(1, self.output_dim)
        
        self.register_parameter(
            'output_mean',
            nn.Parameter(output_mean, requires_grad=self.learnable)
        )
        self.register_parameter(
            'output_std',
            nn.Parameter(output_std, requires_grad=self.learnable)
        )
    
    def fit(self, input_data: torch.Tensor, 
            output_data: Optional[torch.Tensor] = None) -> 'VSScaler':
        """
        根據資料擬合尺度參數
        
        Args:
            input_data: 輸入資料 [batch_size, input_dim]
            output_data: 輸出資料 [batch_size, output_dim] (可選)
        """
        with torch.no_grad():
            # 擬合輸入尺度
            input_mean = torch.mean(input_data, dim=0, keepdim=True)
            input_std = torch.std(input_data, dim=0, keepdim=True)
            input_std = torch.where(
                input_std < 1e-8, 
                torch.ones_like(input_std), 
                input_std
            )
            
            self.input_mean.data.copy_(input_mean)
            self.input_std.data.copy_(input_std)
            
            # 擬合輸出尺度 (如果提供輸出資料)
            if output_data is not None:
                output_mean = torch.mean(output_data, dim=0, keepdim=True)
                output_std = torch.std(output_data, dim=0, keepdim=True) 
                output_std = torch.where(
                    output_std < 1e-8,
                    torch.ones_like(output_std),
                    output_std
                )
                
                self.output_mean.data.copy_(output_mean)
                self.output_std.data.copy_(output_std)
        
        self.fitted = True
        return self
    
    def transform_input(self, data: torch.Tensor) -> torch.Tensor:
        """標準化輸入資料"""
        return (data - self.input_mean) / self.input_std
    
    def transform_output(self, data: torch.Tensor) -> torch.Tensor:
        """標準化輸出資料"""  
        return (data - self.output_mean) / self.output_std
    
    def inverse_transform_input(self, data: torch.Tensor) -> torch.Tensor:
        """逆標準化輸入資料"""
        return data * self.input_std + self.input_mean
    
    def inverse_transform_output(self, data: torch.Tensor) -> torch.Tensor:
        """逆標準化輸出資料"""
        return data * self.output_std + self.output_mean
    
    # 為了兼容BaseScaler介面
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        return self.transform_input(data)
    
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        return self.inverse_transform_input(data)
    
    def get_scale_info(self) -> Dict[str, torch.Tensor]:
        """獲取當前尺度參數資訊"""
        return {
            'input_mean': self.input_mean.data.clone(),
            'input_std': self.input_std.data.clone(),
            'output_mean': self.output_mean.data.clone(), 
            'output_std': self.output_std.data.clone()
        }

def create_scaler_from_data(input_data: torch.Tensor,
                          output_data: Optional[torch.Tensor] = None,
                          scaler_type: str = "standard",
                          learnable: bool = False,
                          **kwargs) -> BaseScaler:
    """
    根據資料自動創建並擬合尺度器
    
    Args:
        input_data: 輸入資料
        output_data: 輸出資料 (可選)
        scaler_type: 尺度器類型 ("standard", "minmax", "vs")
        learnable: 是否可學習
        **kwargs: 額外參數
        
    Returns:
        已擬合的尺度器
    """
    if scaler_type == "standard":
        scaler = StandardScaler(learnable=learnable)
        scaler.fit(input_data)
        
    elif scaler_type == "minmax":
        feature_range = kwargs.get('feature_range', (0.0, 1.0))
        scaler = MinMaxScaler(feature_range=feature_range, learnable=learnable)
        scaler.fit(input_data)
        
    elif scaler_type == "vs":
        input_dim = input_data.shape[1]
        output_dim = output_data.shape[1] if output_data is not None else 1
        init_method = kwargs.get('init_method', "standard")
        
        scaler = VSScaler(
            input_dim=input_dim,
            output_dim=output_dim,
            learnable=learnable,
            init_method=init_method
        )
        scaler.fit(input_data, output_data)
        
    else:
        raise ValueError(f"不支援的尺度器類型: {scaler_type}")
    
    return scaler

def denormalize_gradients(gradients: torch.Tensor,
                         scaler: BaseScaler,
                         input_coords: torch.Tensor,
                         output_vars: torch.Tensor) -> torch.Tensor:
    """
    將標準化空間的梯度轉換回物理空間
    用於物理定律計算中的梯度校正
    
    Args:
        gradients: 標準化空間的梯度
        scaler: 使用的尺度器
        input_coords: 輸入座標
        output_vars: 輸出變數
        
    Returns:
        物理空間的梯度
    """
    if isinstance(scaler, VSScaler):
        # VS-PINN 梯度變換
        # ∂f_phys/∂x_phys = (∂f_norm/∂x_norm) * (σ_f/σ_x)
        input_scale = scaler.input_std
        output_scale = scaler.output_std
        
        scale_factor = output_scale / input_scale
        physical_gradients = gradients * scale_factor
        
    elif isinstance(scaler, StandardScaler):
        # 標準尺度器梯度變換
        scale_factor = scaler.std  # 只考慮標準差縮放
        physical_gradients = gradients / scale_factor
        
    elif isinstance(scaler, MinMaxScaler):
        # 最大最小值尺度器梯度變換
        scale_factor = scaler.data_range
        physical_gradients = gradients * scale_factor
        
    else:
        # 如果不是已知的尺度器，返回原始梯度
        physical_gradients = gradients
    
    return physical_gradients

# 高級尺度化工具
class AdaptiveScaler(VSScaler):
    """
    自適應尺度器
    在訓練過程中動態調整尺度參數
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 adaptation_rate: float = 0.01):
        super().__init__(input_dim, output_dim, learnable=True)
        self.adaptation_rate = adaptation_rate
        self.update_count = 0
    
    def adaptive_update(self, input_batch: torch.Tensor,
                       output_batch: torch.Tensor):
        """根據當前批次自適應更新尺度參數"""
        with torch.no_grad():
            # 計算當前批次統計
            batch_input_mean = torch.mean(input_batch, dim=0, keepdim=True)
            batch_input_std = torch.std(input_batch, dim=0, keepdim=True)
            
            batch_output_mean = torch.mean(output_batch, dim=0, keepdim=True)
            batch_output_std = torch.std(output_batch, dim=0, keepdim=True)
            
            # 指數移動平均更新
            alpha = self.adaptation_rate
            
            self.input_mean.data = (1-alpha) * self.input_mean.data + alpha * batch_input_mean
            self.input_std.data = (1-alpha) * self.input_std.data + alpha * batch_input_std
            
            self.output_mean.data = (1-alpha) * self.output_mean.data + alpha * batch_output_mean
            self.output_std.data = (1-alpha) * self.output_std.data + alpha * batch_output_std
            
            self.update_count += 1

def analyze_scaling_sensitivity(model: nn.Module,
                              scaler: BaseScaler,
                              test_data: torch.Tensor,
                              scale_factors: np.ndarray = None) -> Dict[str, Any]:
    """
    分析模型對尺度變化的敏感性
    用於優化尺度參數
    
    Args:
        model: 神經網路模型
        scaler: 尺度器
        test_data: 測試資料
        scale_factors: 尺度因子範圍
        
    Returns:
        敏感性分析結果
    """
    if scale_factors is None:
        scale_factors = np.logspace(-1, 1, 11)  # 0.1 到 10
    
    results = {
        'scale_factors': scale_factors,
        'losses': [],
        'gradients': []
    }
    
    original_state = scaler.state_dict()
    
    for factor in scale_factors:
        # 修改尺度參數
        if isinstance(scaler, VSScaler):
            scaler.input_std.data *= factor
            scaler.output_std.data *= factor
        
        # 評估模型性能
        with torch.no_grad():
            scaled_input = scaler.transform_input(test_data)
            output = model(scaled_input)
            scaled_output = scaler.inverse_transform_output(output)
            
            # 計算某種損失 (這裡需要根據實際情況定義)
            loss = torch.mean(scaled_output**2)  # 簡單的L2損失示例
            results['losses'].append(loss.item())
        
        # 恢復原始狀態
        scaler.load_state_dict(original_state)
    
    return results