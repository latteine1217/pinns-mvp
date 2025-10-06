"""
增強版 Fourier MLP 網路架構模組
專為 Task-014 PINNs 重建精度優化設計

基於根因分析的關鍵改進：
1. 增加網絡容量：深度4→8層，寬度128→256
2. 改進激活函數：tanh → swish (更好的梯度傳播)
3. 添加殘差連接：改善深度網絡訓練
4. 增強正則化：防止過擬合
5. 優化權重初始化：針對深度網絡

目標：將重建誤差從>100%降至<30%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Union, List, Callable
import numpy as np


class EnhancedFourierFeatures(nn.Module):
    """
    增強版 Fourier Random Features 編碼層
    - 支援多尺度頻率
    - 可學習的頻率權重
    - 改善的初始化策略
    """
    
    def __init__(self, in_dim: int, m: int = 64, sigma: float = 5.0, 
                 multiscale: bool = True, trainable: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.m = m
        self.sigma = sigma
        self.multiscale = multiscale
        self.out_dim = 2 * m
        
        if multiscale:
            # 多尺度頻率：低頻到高頻
            sigmas = torch.logspace(-1, math.log10(sigma), m // 4).repeat(4)[:m]
            B = torch.randn(in_dim, m) * sigmas.unsqueeze(0)
        else:
            B = torch.randn(in_dim, m) * sigma
        
        if trainable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = 2.0 * math.pi * x @ self.B
        return torch.cat([torch.cos(z), torch.sin(z)], dim=-1)


class EnhancedDenseLayer(nn.Module):
    """
    增強版密集連接層
    - 支援殘差連接
    - 層歸一化
    - 改進的正則化
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 activation: str = 'swish', use_residual: bool = True,
                 dropout: float = 0.1, use_layer_norm: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.use_residual = use_residual and (in_features == out_features)
        self.use_layer_norm = use_layer_norm
        
        # 層歸一化
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(out_features)
        
        # 激活函數
        if activation == 'swish':
            self.activation = nn.SiLU()  # Swish activation
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"不支援的激活函數: {activation}")
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # 針對swish的權重初始化
        if activation == 'swish':
            nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        else:
            nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.use_residual else None
        
        out = self.linear(x)
        
        if self.use_layer_norm:
            out = self.layer_norm(out)
        
        out = self.activation(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        if residual is not None:
            out = out + residual
            
        return out


class EnhancedPINNNet(nn.Module):
    """
    增強版 PINN 神經網路 - 針對Task-014優化
    
    關鍵改進：
    1. 更深的網絡：8層隱藏層
    2. 更寬的網絡：256神經元
    3. Swish激活函數：更好的梯度傳播
    4. 殘差連接：改善深度網絡訓練
    5. 層歸一化：穩定訓練
    6. 多尺度Fourier特徵：捕捉不同尺度的特徵
    """
    
    def __init__(self,
                 in_dim: int = 2,           # 輸入維度 (x,y)
                 out_dim: int = 3,          # 輸出維度 (u,v,p)
                 width: int = 256,          # 隱藏層寬度 (增加)
                 depth: int = 8,            # 網路深度 (增加)
                 fourier_m: int = 64,       # Fourier 特徵數 (增加)
                 fourier_sigma: float = 5.0,
                 activation: str = 'swish', # 改用swish
                 use_fourier: bool = True,
                 use_residual: bool = True, # 啟用殘差連接
                 use_layer_norm: bool = True, # 啟用層歸一化
                 dropout: float = 0.1):     # 適度的dropout
        
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_fourier = use_fourier
        
        # 增強版Fourier特徵編碼
        if use_fourier:
            self.fourier = EnhancedFourierFeatures(
                in_dim, fourier_m, fourier_sigma, 
                multiscale=True, trainable=False
            )
            input_features = self.fourier.out_dim
        else:
            self.fourier = None
            input_features = in_dim
        
        # 輸入投影層
        self.input_projection = nn.Linear(input_features, width)
        nn.init.xavier_normal_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        
        # 深度隱藏層
        layers = []
        for i in range(depth):
            layers.append(EnhancedDenseLayer(
                width, width,
                activation=activation,
                use_residual=use_residual,
                dropout=dropout,
                use_layer_norm=use_layer_norm
            ))
        
        self.hidden_layers = nn.ModuleList(layers)
        
        # 輸出層 (線性，無激活函數)
        self.output_layer = nn.Linear(width, out_dim)
        
        # 輸出層特殊初始化：更小的權重，增強穩定性
        nn.init.xavier_normal_(self.output_layer.weight, gain=0.01)
        nn.init.zeros_(self.output_layer.bias)
        
        print(f"✅ 增強版PINNNet建立:")
        print(f"   深度: {depth} 層, 寬度: {width}")
        print(f"   參數總數: {self.get_num_params():,}")
        print(f"   激活函數: {activation}")
        print(f"   Fourier特徵: {fourier_m if use_fourier else 'None'}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fourier 特徵編碼
        if self.use_fourier:
            h = self.fourier(x)
        else:
            h = x
        
        # 輸入投影
        h = self.input_projection(h)
        h = F.silu(h)  # Swish activation for input projection
        
        # 深度隱藏層前向傳播
        for layer in self.hidden_layers:
            h = layer(h)
        
        # 輸出層
        output = self.output_layer(h)
        
        return output
    
    def get_num_params(self) -> int:
        """返回模型參數總數"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_summary(self) -> dict:
        """返回模型結構摘要"""
        return {
            'input_dim': self.in_dim,
            'output_dim': self.out_dim,
            'depth': len(self.hidden_layers),
            'width': self.hidden_layers[0].linear.out_features if self.hidden_layers else 0,
            'total_params': self.get_num_params(),
            'fourier_features': self.fourier.m if self.use_fourier else 0,
            'use_residual': self.hidden_layers[0].use_residual if self.hidden_layers else False,
            'use_layer_norm': self.hidden_layers[0].use_layer_norm if self.hidden_layers else False
        }


def create_enhanced_pinn(config: dict = None) -> EnhancedPINNNet:
    """
    創建增強版PINN模型的便捷函數
    
    Args:
        config: 配置字典，如果為None則使用優化的預設配置
    
    Returns:
        EnhancedPINNNet實例
    """
    if config is None:
        # Task-014優化的預設配置
        config = {
            'in_dim': 2,
            'out_dim': 3,
            'width': 256,      # 增加寬度
            'depth': 8,        # 增加深度
            'fourier_m': 64,   # 增加Fourier特徵
            'fourier_sigma': 5.0,
            'activation': 'swish',  # 使用swish激活
            'use_residual': True,   # 啟用殘差連接
            'use_layer_norm': True, # 啟用層歸一化
            'dropout': 0.1          # 適度dropout
        }
    
    return EnhancedPINNNet(**config)


if __name__ == "__main__":
    # 測試增強版模型
    print("=== 增強版 PINNNet 測試 ===")
    
    # 建立模型
    model = create_enhanced_pinn()
    summary = model.get_model_summary()
    
    print("模型摘要:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # 測試前向傳播
    x = torch.randn(100, 2)  # [batch, (x,y)]
    with torch.no_grad():
        y = model(x)
    print(f"\n前向傳播測試:")
    print(f"  輸入形狀: {x.shape}")
    print(f"  輸出形狀: {y.shape}")
    
    # 測試梯度計算
    x.requires_grad_(True)
    y = model(x)
    u = y[:, 0]
    
    du_dx = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, 0]
    print(f"  梯度計算: ∂u/∂x 形狀 = {du_dx.shape}")
    
    print("\n✅ 增強版模型測試通過！")
    print(f"參數量提升: 58,243 → {model.get_num_params():,} (+{model.get_num_params()/58243:.1f}x)")