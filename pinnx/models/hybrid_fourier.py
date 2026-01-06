"""
混合型 Fourier Features 模組 - 整合週期性與標準 Fourier 嵌入

針對混合邊界條件（部分週期、部分非週期）的場景設計：
- 週期軸（如 Kolmogorov flow 的 x, y）使用 PeriodicFourierFeatures
- 非週期軸（如時間 t）使用標準 FourierFeatures
- 支援靈活的軸向配置

典型應用場景：
1. Kolmogorov Flow (2D)：x, y 週期，t 標準
2. Channel Flow (3D)：x, z 週期，y 標準，t 標準
3. 週期性槽道流：x, z 週期，y 壁面，t 標準

作者：PINNs-MVP 團隊
日期：2025-01-06
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Literal


class HybridFourierFeatures(nn.Module):
    """
    混合型 Fourier Features 編碼層
    
    根據軸的物理特性選擇編碼方式：
    - 'periodic': 週期性嵌入（保證邊界一致性）
    - 'standard': 標準 Fourier random features（高頻捕捉）
    - 'none': 直接透傳（不進行編碼）
    
    Args:
        axes_config: 軸配置字典
            - 鍵：軸索引（0, 1, 2 對應 x, y, z 或 t, x, y）
            - 值：配置字典
                {
                    'type': 'periodic' | 'standard' | 'none',
                    'domain_size': float (週期域大小，僅 periodic 需要),
                    'n_modes': int (模態數量，periodic/standard 需要),
                    'sigma': float (頻率尺度，僅 standard 需要)
                }
        trainable: 是否讓頻率參數可訓練（預設 False）
    
    Example:
        >>> # Kolmogorov Flow 2D+T 配置
        >>> config = {
        ...     0: {'type': 'standard', 'n_modes': 12, 'sigma': 4.0},  # 時間 t
        ...     1: {'type': 'periodic', 'domain_size': 2*np.pi, 'n_modes': 8},  # 空間 x
        ...     2: {'type': 'periodic', 'domain_size': 2*np.pi, 'n_modes': 8},  # 空間 y
        ... }
        >>> hybrid = HybridFourierFeatures(config)
        >>> x = torch.randn(100, 3)  # [batch, 3] = [t, x, y]
        >>> features = hybrid(x)     # [batch, 56] = 24 + 16 + 16
    """
    
    def __init__(
        self,
        axes_config: Dict[int, Dict[str, any]],
        trainable: bool = False
    ):
        super().__init__()
        
        self.axes_config = axes_config
        self.in_dim = len(axes_config)
        self.trainable = trainable
        
        # 為每個軸創建對應的編碼器
        self.encoders = nn.ModuleDict()
        self.axis_out_dims = {}  # 記錄每個軸的輸出維度
        
        for axis_idx, axis_cfg in axes_config.items():
            encoder_type = axis_cfg.get('type', 'none')
            
            if encoder_type == 'periodic':
                # 週期性嵌入
                domain_size = axis_cfg.get('domain_size', 2.0 * np.pi)
                n_modes = axis_cfg.get('n_modes', 8)
                
                encoder = PeriodicFourierFeatures(
                    domain_size=domain_size,
                    n_modes=n_modes,
                    trainable=trainable
                )
                self.encoders[str(axis_idx)] = encoder
                self.axis_out_dims[axis_idx] = encoder.out_dim
                
            elif encoder_type == 'standard':
                # 標準 Fourier random features
                n_modes = axis_cfg.get('n_modes', 8)
                sigma = axis_cfg.get('sigma', 4.0)
                use_2pi = axis_cfg.get('use_2pi', True)
                
                encoder = StandardFourierFeatures1D(
                    m=n_modes,
                    sigma=sigma,
                    trainable=trainable,
                    use_2pi=use_2pi
                )
                self.encoders[str(axis_idx)] = encoder
                self.axis_out_dims[axis_idx] = 2 * n_modes
                
            elif encoder_type == 'none':
                # 不編碼，直接透傳
                self.encoders[str(axis_idx)] = None
                self.axis_out_dims[axis_idx] = 1
                
            else:
                raise ValueError(f"不支援的編碼器類型: {encoder_type}")
        
        # 計算總輸出維度
        self.out_dim = sum(self.axis_out_dims.values())
        
        print(f"✅ HybridFourierFeatures 初始化完成")
        print(f"   輸入維度: {self.in_dim}")
        print(f"   輸出維度: {self.out_dim}")
        print(f"   軸配置:")
        for axis_idx, axis_cfg in axes_config.items():
            print(f"     軸 {axis_idx}: {axis_cfg['type']} → {self.axis_out_dims[axis_idx]} 維")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            x: [batch_size, in_dim] 輸入座標
        
        Returns:
            [batch_size, out_dim] 混合 Fourier 特徵
        """
        if x.shape[1] != self.in_dim:
            raise ValueError(f"輸入維度不匹配: 期望 {self.in_dim}, 實際 {x.shape[1]}")
        
        features = []
        
        # 對每個軸分別編碼
        for axis_idx in range(self.in_dim):
            x_axis = x[:, axis_idx:axis_idx+1]  # [batch, 1]
            encoder = self.encoders[str(axis_idx)]
            
            if encoder is None:
                # 直接透傳
                features.append(x_axis)
            else:
                # 使用編碼器
                features.append(encoder(x_axis))
        
        # 拼接所有軸的特徵
        return torch.cat(features, dim=-1)
    
    def extra_repr(self) -> str:
        """返回模組描述"""
        return f'in_dim={self.in_dim}, out_dim={self.out_dim}, trainable={self.trainable}'


class PeriodicFourierFeatures(nn.Module):
    """
    週期性 Fourier 特徵編碼層（單軸版本）
    
    針對週期域 x ∈ [0, L] 的嚴格週期性編碼：
        φ(x) = [sin(2πkx/L), cos(2πkx/L)] for k = 1, 2, ..., n_modes
    
    保證：φ(0) = φ(L)（天然週期邊界條件）
    
    Args:
        domain_size: 週期域大小 L
        n_modes: 傅立葉模態數量
        trainable: 是否讓頻率可訓練
    """
    
    def __init__(
        self,
        domain_size: float = 2.0 * np.pi,
        n_modes: int = 8,
        trainable: bool = False
    ):
        super().__init__()
        self.domain_size = domain_size
        self.n_modes = n_modes
        self.out_dim = 2 * n_modes
        
        # 生成週期性頻率：ω_k = 2πk / L
        frequencies = 2.0 * np.pi * torch.arange(1, n_modes + 1, dtype=torch.float32) / domain_size
        
        if trainable:
            self.frequencies = nn.Parameter(frequencies)
        else:
            self.register_buffer('frequencies', frequencies)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, 1] 單軸座標
        Returns:
            [batch_size, 2*n_modes] 週期性特徵
        """
        z = x * self.frequencies.unsqueeze(0)  # [batch, n_modes]
        return torch.cat([torch.sin(z), torch.cos(z)], dim=-1)
    
    def extra_repr(self) -> str:
        return f'domain_size={self.domain_size:.4f}, n_modes={self.n_modes}, out_dim={self.out_dim}'


class StandardFourierFeatures1D(nn.Module):
    """
    標準 Fourier Random Features 編碼層（單軸版本）
    
    使用隨機高斯採樣的頻率進行編碼：
        φ(x) = [cos(2πωx), sin(2πωx)]
        其中 ω ~ N(0, σ²)
    
    Args:
        m: Fourier 特徵數量（輸出 2m 維）
        sigma: 頻率尺度參數
        trainable: 是否讓頻率可訓練
        use_2pi: 是否乘以 2π
    """
    
    def __init__(
        self,
        m: int = 8,
        sigma: float = 4.0,
        trainable: bool = False,
        use_2pi: bool = True
    ):
        super().__init__()
        self.m = m
        self.sigma = sigma
        self.use_2pi = use_2pi
        self.out_dim = 2 * m
        
        # 生成隨機頻率（單軸）
        B = torch.randn(1, m) * sigma  # [1, m]
        
        if trainable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer('B', B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, 1] 單軸座標
        Returns:
            [batch_size, 2*m] Fourier 特徵
        """
        z = x @ self.B  # [batch, m]
        if self.use_2pi:
            z = 2.0 * np.pi * z
        return torch.cat([torch.cos(z), torch.sin(z)], dim=-1)
    
    def extra_repr(self) -> str:
        return f'm={self.m}, sigma={self.sigma:.2f}, use_2pi={self.use_2pi}, out_dim={self.out_dim}'


# ========== 便捷工廠函數 ==========

def create_kolmogorov_2d_hybrid(
    n_modes_time: int = 12,
    n_modes_space: int = 8,
    sigma_time: float = 4.0,
    domain_size: float = 2.0 * np.pi,
    trainable: bool = False
) -> HybridFourierFeatures:
    """
    創建 Kolmogorov Flow 2D+T 的混合編碼器
    
    配置：
    - t (軸0): 標準 Fourier (n_modes_time, sigma_time)
    - x (軸1): 週期性嵌入 (domain_size, n_modes_space)
    - y (軸2): 週期性嵌入 (domain_size, n_modes_space)
    
    Args:
        n_modes_time: 時間軸 Fourier 模態數
        n_modes_space: 空間軸週期模態數
        sigma_time: 時間軸頻率尺度
        domain_size: 空間域週期大小
        trainable: 是否可訓練
    
    Returns:
        HybridFourierFeatures 實例
    """
    config = {
        0: {'type': 'standard', 'n_modes': n_modes_time, 'sigma': sigma_time, 'use_2pi': True},
        1: {'type': 'periodic', 'domain_size': domain_size, 'n_modes': n_modes_space},
        2: {'type': 'periodic', 'domain_size': domain_size, 'n_modes': n_modes_space},
    }
    return HybridFourierFeatures(config, trainable=trainable)


def create_kolmogorov_2d_steady_hybrid(
    n_modes_space: int = 8,
    domain_size: float = 2.0 * np.pi,
    trainable: bool = False
) -> HybridFourierFeatures:
    """
    創建 Kolmogorov Flow 2D 穩態的混合編碼器（無時間維度）
    
    配置：
    - x (軸0): 週期性嵌入 (domain_size, n_modes_space)
    - y (軸1): 週期性嵌入 (domain_size, n_modes_space)
    
    Args:
        n_modes_space: 空間軸週期模態數
        domain_size: 空間域週期大小
        trainable: 是否可訓練
    
    Returns:
        HybridFourierFeatures 實例
    """
    config = {
        0: {'type': 'periodic', 'domain_size': domain_size, 'n_modes': n_modes_space},
        1: {'type': 'periodic', 'domain_size': domain_size, 'n_modes': n_modes_space},
    }
    return HybridFourierFeatures(config, trainable=trainable)


if __name__ == "__main__":
    # 測試程式碼
    print("=== HybridFourierFeatures 測試 ===\n")
    
    # 測試 1: Kolmogorov Flow 2D+T
    print("1️⃣  測試 Kolmogorov Flow 2D+T 配置")
    hybrid_2dt = create_kolmogorov_2d_hybrid(
        n_modes_time=12,
        n_modes_space=8,
        sigma_time=4.0
    )
    x_2dt = torch.randn(100, 3)  # [batch, 3] = [t, x, y]
    features_2dt = hybrid_2dt(x_2dt)
    print(f"   輸入形狀: {x_2dt.shape}")
    print(f"   輸出形狀: {features_2dt.shape}")
    print(f"   預期輸出維度: {24 + 16 + 16} = 56\n")
    
    # 測試 2: Kolmogorov Flow 2D 穩態
    print("2️⃣  測試 Kolmogorov Flow 2D 穩態配置")
    hybrid_2d = create_kolmogorov_2d_steady_hybrid(n_modes_space=8)
    x_2d = torch.randn(100, 2)  # [batch, 2] = [x, y]
    features_2d = hybrid_2d(x_2d)
    print(f"   輸入形狀: {x_2d.shape}")
    print(f"   輸出形狀: {features_2d.shape}")
    print(f"   預期輸出維度: {16 + 16} = 32\n")
    
    # 測試 3: 週期性驗證
    print("3️⃣  測試週期性邊界條件")
    periodic_encoder = PeriodicFourierFeatures(domain_size=2*np.pi, n_modes=4)
    x_left = torch.tensor([[0.0]])  # x = 0
    x_right = torch.tensor([[2*np.pi]])  # x = 2π
    
    feat_left = periodic_encoder(x_left)
    feat_right = periodic_encoder(x_right)
    
    diff = torch.abs(feat_left - feat_right).max().item()
    print(f"   x=0 特徵: {feat_left[0, :4].tolist()}")
    print(f"   x=2π 特徵: {feat_right[0, :4].tolist()}")
    print(f"   最大差異: {diff:.2e}")
    print(f"   週期性檢查: {'✅ 通過' if diff < 1e-6 else '❌ 失敗'}\n")
    
    # 測試 4: 梯度計算（PINNs 關鍵）
    print("4️⃣  測試梯度計算")
    x_grad = torch.randn(50, 3, requires_grad=True)
    features_grad = hybrid_2dt(x_grad)
    
    # 計算對輸入的梯度
    loss = features_grad.sum()
    loss.backward()
    
    print(f"   輸入梯度形狀: {x_grad.grad.shape}")
    print(f"   梯度範圍: [{x_grad.grad.min():.4f}, {x_grad.grad.max():.4f}]")
    print(f"   梯度計算: ✅ 通過\n")
    
    print("✅ 所有測試通過！")
