"""
Fourier 特徵 MLP 網路架構模組

實現支援 Fourier 特徵編碼、可變深度/寬度、多輸出的神經網路架構，
專為 PINNs 設計，支援 VS-PINN 變數尺度化與自動微分。

核心特色：
- Fourier Random Features 提升高頻擬合能力  
- 可配置的網路深度與寬度
- 支援多種激活函數 (tanh, swish, gelu)
- 針對 PINNs 自動微分優化的權重初始化
- 梯度友善的網路設計
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Union, List, Callable
import numpy as np


class FourierFeatures(nn.Module):
    """
    Fourier Random Features 編碼層
    
    將輸入座標 (t,x,y) 透過隨機 Fourier 特徵映射到高維空間，
    提升神經網路對高頻函數的擬合能力。
    
    Args:
        in_dim: 輸入維度 (例如 3 代表 t,x,y)
        m: Fourier 特徵數量 (輸出維度為 2m)
        sigma: Fourier 頻率尺度參數
        trainable: 是否讓 Fourier 係數可訓練
    """
    
    def __init__(self, in_dim: int, m: int = 32, sigma: float = 5.0, trainable: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.m = m
        self.sigma = sigma
        self.out_dim = 2 * m
        
        # 隨機 Fourier 係數矩陣 B ~ N(0, sigma^2)
        B = torch.randn(in_dim, m) * sigma
        
        if trainable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, in_dim] 輸入座標
        Returns:
            [batch_size, 2*m] Fourier 特徵
        """
        # z = 2π * x @ B
        z = 2.0 * math.pi * x @ self.B
        # 回傳 [cos(z), sin(z)]
        return torch.cat([torch.cos(z), torch.sin(z)], dim=-1)
    
    def extra_repr(self) -> str:
        return f'in_dim={self.in_dim}, m={self.m}, sigma={self.sigma:.2f}, out_dim={self.out_dim}'


class DenseLayer(nn.Module):
    """
    密集連接層，支援多種激活函數與殘差連接
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 activation: str = 'tanh', use_residual: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.use_residual = use_residual and (in_features == out_features)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # 激活函數選擇
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'swish':
            self.activation = nn.SiLU()  # SiLU = Swish
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"不支援的激活函數: {activation}")
        
        # Xavier/Glorot 權重初始化，針對 tanh 優化
        if activation == 'tanh':
            nn.init.xavier_normal_(self.linear.weight)
        else:
            nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = self.activation(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        if self.use_residual:
            out = out + x
            
        return out


class PINNNet(nn.Module):
    """
    基於 Fourier 特徵的 PINN 神經網路
    
    網路架構：
    輸入 -> Fourier 特徵 -> 多層 Dense -> 輸出
    
    專為 PINNs 設計的特性：
    - 支援自動微分的權重初始化
    - 可變網路深度與寬度
    - Fourier 特徵提升高頻擬合
    - 多輸出支援 (u,v,p,source等)
    """
    
    def __init__(self,
                 in_dim: int = 3,           # 輸入維度 (t,x,y)
                 out_dim: int = 4,          # 輸出維度 (u,v,p,S)
                 width: int = 256,          # 隱藏層寬度
                 depth: int = 5,            # 網路深度
                 fourier_m: int = 32,       # Fourier 特徵數
                 fourier_sigma: float = 5.0, # Fourier 頻率尺度
                 activation: str = 'tanh',   # 激活函數
                 use_fourier: bool = True,   # 是否使用 Fourier 特徵
                 trainable_fourier: bool = False, # Fourier 係數是否可訓練
                 use_residual: bool = False, # 是否使用殘差連接
                 dropout: float = 0.0):     # Dropout 比率
        
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_fourier = use_fourier
        
        # Fourier 特徵編碼
        if use_fourier:
            self.fourier = FourierFeatures(in_dim, fourier_m, fourier_sigma, trainable_fourier)
            input_features = self.fourier.out_dim
        else:
            self.fourier = None
            input_features = in_dim
        
        # 隱藏層
        layers = []
        current_dim = input_features
        
        for i in range(depth):
            layers.append(DenseLayer(
                current_dim, width, 
                activation=activation,
                use_residual=use_residual and i > 0,  # 第一層不用殘差
                dropout=dropout
            ))
            current_dim = width
        
        self.hidden_layers = nn.ModuleList(layers)
        
        # 輸出層 (線性，無激活函數)
        self.output_layer = nn.Linear(current_dim, out_dim)
        
        # 輸出層特殊初始化：較小的權重，有助於訓練穩定
        nn.init.xavier_normal_(self.output_layer.weight, gain=0.1)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            x: [batch_size, in_dim] 輸入座標
        Returns:
            [batch_size, out_dim] 網路輸出
        """
        # Fourier 特徵編碼
        if self.use_fourier:
            h = self.fourier(x)
        else:
            h = x
        
        # 隱藏層前向傳播
        for layer in self.hidden_layers:
            h = layer(h)
        
        # 輸出層
        output = self.output_layer(h)
        
        return output
    
    def get_num_params(self) -> int:
        """返回模型參數總數"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def extra_repr(self) -> str:
        fourier_info = f", fourier_m={self.fourier.m}" if self.use_fourier else ""
        return (f'in_dim={self.in_dim}, out_dim={self.out_dim}, '
                f'width={self.hidden_layers[0].linear.out_features}, '
                f'depth={len(self.hidden_layers)}{fourier_info}, '
                f'params={self.get_num_params():,}')


class MultiScalePINNNet(nn.Module):
    """
    多尺度 PINN 網路：使用不同 Fourier 頻率的子網路組合
    
    適用於包含多個特徵尺度的問題（例如湍流中的大尺度結構與小尺度渦漩）
    """
    
    def __init__(self,
                 in_dim: int = 3,
                 out_dim: int = 4,
                 width: int = 128,
                 depth: int = 4,
                 num_scales: int = 3,
                 sigma_min: float = 1.0,
                 sigma_max: float = 10.0,
                 fourier_m: int = 16,
                 activation: str = 'tanh'):
        
        super().__init__()
        
        self.num_scales = num_scales
        self.out_dim = out_dim
        
        # 生成多個不同頻率尺度的子網路
        sigmas = np.logspace(np.log10(sigma_min), np.log10(sigma_max), num_scales)
        
        self.subnets = nn.ModuleList()
        for sigma in sigmas:
            subnet = PINNNet(
                in_dim=in_dim,
                out_dim=out_dim,
                width=width,
                depth=depth,
                fourier_m=fourier_m,
                fourier_sigma=float(sigma),
                activation=activation,
                use_fourier=True
            )
            self.subnets.append(subnet)
        
        # 尺度權重（可學習）
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        多尺度前向傳播：各子網路輸出的加權組合
        """
        outputs = []
        for subnet in self.subnets:
            outputs.append(subnet(x))
        
        # 堆疊 [num_scales, batch_size, out_dim]
        stacked = torch.stack(outputs, dim=0)
        
        # 加權平均：[batch_size, out_dim]
        weights = F.softmax(self.scale_weights, dim=0)
        weighted_output = torch.einsum('s,sbo->bo', weights, stacked)
        
        return weighted_output
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# 便捷函數：根據配置建立模型
def create_pinn_model(config: dict) -> nn.Module:
    """
    根據配置字典建立 PINN 模型
    
    Args:
        config: 模型配置字典，包含網路參數
    Returns:
        建立好的 PINN 模型
    """
    model_type = config.get('type', 'standard')
    
    if model_type == 'standard':
        return PINNNet(
            in_dim=config.get('in_dim', 3),
            out_dim=config.get('out_dim', 4),
            width=config.get('width', 256),
            depth=config.get('depth', 5),
            fourier_m=config.get('fourier_m', 32),
            fourier_sigma=config.get('fourier_sigma', 5.0),
            activation=config.get('activation', 'tanh'),
            use_fourier=config.get('use_fourier', True),
            trainable_fourier=config.get('trainable_fourier', False),
            use_residual=config.get('use_residual', False),
            dropout=config.get('dropout', 0.0)
        )
    
    elif model_type == 'multiscale':
        return MultiScalePINNNet(
            in_dim=config.get('in_dim', 3),
            out_dim=config.get('out_dim', 4),
            width=config.get('width', 128),
            depth=config.get('depth', 4),
            num_scales=config.get('num_scales', 3),
            sigma_min=config.get('sigma_min', 1.0),
            sigma_max=config.get('sigma_max', 10.0),
            fourier_m=config.get('fourier_m', 16),
            activation=config.get('activation', 'tanh')
        )
    
    else:
        raise ValueError(f"不支援的模型類型: {model_type}")


# 模型工廠函數
def fourier_pinn(in_dim: int = 3, out_dim: int = 4, **kwargs) -> PINNNet:
    """快速建立標準 Fourier PINN 模型"""
    return PINNNet(in_dim=in_dim, out_dim=out_dim, **kwargs)


def multiscale_pinn(in_dim: int = 3, out_dim: int = 4, **kwargs) -> MultiScalePINNNet:
    """快速建立多尺度 PINN 模型"""
    return MultiScalePINNNet(in_dim=in_dim, out_dim=out_dim, **kwargs)


if __name__ == "__main__":
    # 測試程式碼
    import torch
    
    print("=== PINNNet 測試 ===")
    
    # 建立模型
    model = PINNNet(in_dim=3, out_dim=4, width=128, depth=3, fourier_m=16)
    print(f"模型架構: {model}")
    print(f"參數總數: {model.get_num_params():,}")
    
    # 測試前向傳播
    x = torch.randn(100, 3)  # [batch, (t,x,y)]
    with torch.no_grad():
        y = model(x)
    print(f"輸入形狀: {x.shape}")
    print(f"輸出形狀: {y.shape}")
    
    # 測試梯度計算（PINN 核心）
    x.requires_grad_(True)
    y = model(x)
    u = y[:, 0]  # 第一個輸出變數
    
    # 計算 ∂u/∂x
    du_dx = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, 1]
    print(f"梯度計算成功: ∂u/∂x 形狀 = {du_dx.shape}")
    
    print("\n=== MultiScalePINNNet 測試 ===")
    ms_model = MultiScalePINNNet(in_dim=3, out_dim=4, num_scales=2)
    print(f"多尺度模型參數: {ms_model.get_num_params():,}")
    
    with torch.no_grad():
        y_ms = ms_model(x)
    print(f"多尺度輸出形狀: {y_ms.shape}")
    
    print("\n✅ 所有測試通過！")