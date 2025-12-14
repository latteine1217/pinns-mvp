"""
模擬湍流場生成器 (Mock Turbulence Field Generators)

提供物理合理的湍流場生成，用於測試和離線開發：
- Channel Flow (通道流 Re_tau=1000)
- Isotropic Turbulence (各向同性湍流)

設計原則：
1. 物理合理性：滿足基本物理特徵（邊界條件、速度分布、湍流強度）
2. 可重現性：固定隨機種子確保結果一致
3. 靈活性：支援 cutout 和 points 兩種查詢模式
4. 獨立性：不依賴 JHTDB API，完全本地生成

參考文獻：
- Pope, S. B. (2000). Turbulent Flows. Cambridge University Press.
- Channel Flow DNS: Del Álamo & Jiménez (2003)
"""

import numpy as np
from typing import Dict, List
from abc import ABC, abstractmethod


class TurbulenceGenerator(ABC):
    """湍流場生成器基類"""
    
    def __init__(self, seed: int = 42):
        """
        Args:
            seed: 隨機種子，確保可重現性
        """
        self.seed = seed
        np.random.seed(seed)
    
    @abstractmethod
    def generate_velocity_field(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                variables: List[str]) -> Dict[str, np.ndarray]:
        """
        生成空間網格上的速度場
        
        Args:
            X, Y, Z: 空間座標網格 (meshgrid 格式)
            variables: 需要生成的變數列表 ['u', 'v', 'w', 'p']
            
        Returns:
            變數名 -> 場資料的字典
        """
        pass
    
    @abstractmethod
    def generate_points(self, points: np.ndarray, 
                       variables: List[str]) -> Dict[str, np.ndarray]:
        """
        生成散點上的速度值
        
        Args:
            points: 座標點陣列 [N, 3]
            variables: 需要生成的變數列表
            
        Returns:
            變數名 -> 散點值的字典
        """
        pass


class ChannelFlowGenerator(TurbulenceGenerator):
    """
    通道流湍流生成器 (Re_tau = 1000)
    
    物理特徵：
    - 流向速度：拋物型平均分布 + 湍流擾動
    - 壁面法向速度：小擾動，壁面處為零
    - 展向速度：中等強度湍流擾動
    - 壓力場：線性梯度 + 湍流壓力波動
    
    座標系統：
    - x: 流向 [0, 8π]
    - y: 壁面法向 [-1, 1] (y=±1 為壁面)
    - z: 展向 [0, 3π]
    """
    
    def __init__(self, seed: int = 42, 
                 u_bulk: float = 15.0,
                 turbulent_intensity: float = 0.5):
        """
        Args:
            seed: 隨機種子
            u_bulk: 主流速度 (m/s)
            turbulent_intensity: 湍流強度
        """
        super().__init__(seed)
        self.u_bulk = u_bulk
        self.turbulent_intensity = turbulent_intensity
    
    def generate_velocity_field(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                variables: List[str]) -> Dict[str, np.ndarray]:
        """生成 Channel Flow 速度場"""
        
        # 標準化座標
        domain = {'x': [0, 8*np.pi], 'y': [-1, 1], 'z': [0, 3*np.pi]}
        x_norm = 2 * np.pi * (X - domain['x'][0]) / (domain['x'][1] - domain['x'][0])
        y_norm = Y  # 已經是 [-1, 1]
        z_norm = 2 * np.pi * (Z - domain['z'][0]) / (domain['z'][1] - domain['z'][0])
        
        data = {}
        
        for var in variables:
            if var == 'u':
                # 流向速度：拋物型平均分布 + 湍流擾動
                u_mean = self.u_bulk * (1 - y_norm**2)  # 拋物型分布
                
                # 湍流擾動：近壁面小，中心大
                turbulent_intensity = self.turbulent_intensity * (1 - np.abs(y_norm)**3)
                u_fluctuation = (
                    turbulent_intensity * 3.0 * np.sin(2*x_norm) * np.cos(3*z_norm) +
                    turbulent_intensity * 1.5 * np.sin(4*x_norm) * np.cos(6*z_norm) +
                    turbulent_intensity * 0.3 * np.random.randn(*X.shape)
                )
                
                data[var] = u_mean + u_fluctuation
                
            elif var == 'v':
                # 壁面法向速度：很小，主要是湍流擾動
                # 邊界條件：壁面處 v=0
                wall_factor = 1 - y_norm**2  # 壁面處為 0
                
                data[var] = wall_factor * (
                    0.8 * np.cos(x_norm) * np.sin(2*z_norm) +
                    0.4 * np.cos(3*x_norm) * np.sin(4*z_norm) +
                    0.1 * np.random.randn(*X.shape)
                )
                
            elif var == 'w':
                # 展向速度：中等強度湍流擾動
                wall_factor = 1 - y_norm**2  # 壁面處為 0
                turbulent_intensity = 0.8 * (1 - np.abs(y_norm)**2)
                
                data[var] = wall_factor * (
                    8.0 * np.sin(x_norm) * np.cos(z_norm) +
                    4.0 * np.sin(3*x_norm) * np.cos(2*z_norm) +
                    turbulent_intensity * 4.0 * np.random.randn(*X.shape)
                )
                
            elif var == 'p':
                # 壓力場：受平均速度梯度和湍流影響
                u_val = data.get('u', np.zeros_like(X))
                v_val = data.get('v', np.zeros_like(X))
                w_val = data.get('w', np.zeros_like(X))
                
                # 基於連續性方程和動量方程的壓力估計
                p_mean = -2.0 * y_norm  # 線性壓力梯度（驅動流動）
                p_fluctuation = (
                    -0.3 * (u_val**2 + v_val**2 + w_val**2) +  # 動壓項
                    5.0 * np.cos(x_norm + z_norm) * (1 - y_norm**2) +  # 湍流壓力
                    0.1 * np.random.randn(*X.shape)
                )
                
                data[var] = p_mean + p_fluctuation
        
        return data
    
    def generate_points(self, points: np.ndarray, 
                       variables: List[str]) -> Dict[str, np.ndarray]:
        """生成 Channel Flow 散點資料"""
        
        # 標準化座標
        domain = {'x': [0, 8*np.pi], 'y': [-1, 1], 'z': [0, 3*np.pi]}
        x_norm = 2 * np.pi * (points[:, 0] - domain['x'][0]) / (domain['x'][1] - domain['x'][0])
        y_norm = points[:, 1]  # 已經是 [-1, 1]
        z_norm = 2 * np.pi * (points[:, 2] - domain['z'][0]) / (domain['z'][1] - domain['z'][0])
        
        n_points = len(points)
        data = {}
        
        for var in variables:
            if var == 'u':
                # 流向速度：拋物型平均分布 + 湍流擾動
                u_mean = self.u_bulk * (1 - y_norm**2)
                
                turbulent_intensity = self.turbulent_intensity * (1 - np.abs(y_norm)**3)
                u_fluctuation = (
                    turbulent_intensity * 3.0 * np.sin(2*x_norm) * np.cos(3*z_norm) +
                    turbulent_intensity * 1.5 * np.sin(4*x_norm) * np.cos(6*z_norm) +
                    turbulent_intensity * 0.3 * np.random.randn(n_points)
                )
                
                data[var] = u_mean + u_fluctuation
                
            elif var == 'v':
                # 壁面法向速度
                wall_factor = 1 - y_norm**2
                
                data[var] = wall_factor * (
                    0.8 * np.cos(x_norm) * np.sin(2*z_norm) +
                    0.4 * np.cos(3*x_norm) * np.sin(4*z_norm) +
                    0.1 * np.random.randn(n_points)
                )
                
            elif var == 'w':
                # 展向速度
                wall_factor = 1 - y_norm**2
                
                data[var] = wall_factor * (
                    2.0 * np.sin(x_norm) * np.cos(z_norm) +
                    1.0 * np.sin(3*x_norm) * np.cos(2*z_norm) +
                    0.2 * np.random.randn(n_points)
                )
                
            elif var == 'p':
                # 壓力場
                u_val = data.get('u', np.zeros(n_points))
                v_val = data.get('v', np.zeros(n_points))
                w_val = data.get('w', np.zeros(n_points))
                
                p_mean = -2.0 * y_norm
                p_fluctuation = (
                    -0.3 * (u_val**2 + v_val**2 + w_val**2) +
                    5.0 * np.cos(x_norm + z_norm) * (1 - y_norm**2) +
                    0.1 * np.random.randn(n_points)
                )
                
                data[var] = p_mean + p_fluctuation
        
        return data


class IsotropicTurbulenceGenerator(TurbulenceGenerator):
    """
    各向同性湍流生成器 (HIT)
    
    物理特徵：
    - 各向統計同性：無優先方向
    - 多尺度渦結構：大尺度 + 小尺度疊加
    - 無邊界效應：週期性邊界條件
    
    座標系統：
    - x, y, z: [0, 2π] (週期性)
    """
    
    def __init__(self, seed: int = 42,
                 velocity_scale: float = 5.0):
        """
        Args:
            seed: 隨機種子
            velocity_scale: 速度尺度 (m/s)
        """
        super().__init__(seed)
        self.velocity_scale = velocity_scale
    
    def generate_velocity_field(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                variables: List[str]) -> Dict[str, np.ndarray]:
        """生成各向同性湍流速度場"""
        
        # 標準化座標到 [0, 2π]
        domain = {'x': [0, 2*np.pi], 'y': [0, 2*np.pi], 'z': [0, 2*np.pi]}
        x_norm = 2 * np.pi * (X - domain['x'][0]) / (domain['x'][1] - domain['x'][0])
        y_norm = 2 * np.pi * (Y - domain['y'][0]) / (domain['y'][1] - domain['y'][0])
        z_norm = 2 * np.pi * (Z - domain['z'][0]) / (domain['z'][1] - domain['z'][0])
        
        data = {}
        
        for var in variables:
            if var == 'u':
                # 主流方向速度：包含多尺度渦結構
                data[var] = self.velocity_scale * (
                    np.sin(x_norm) * np.cos(y_norm) * np.sin(z_norm) +
                    0.4 * np.sin(2*x_norm) * np.cos(2*y_norm) * np.sin(2*z_norm) +
                    0.1 * np.sin(4*x_norm) * np.cos(4*y_norm) * np.sin(4*z_norm) +
                    0.02 * np.random.randn(*X.shape)
                )
                
            elif var == 'v':
                # 橫向速度
                data[var] = self.velocity_scale * 0.6 * (
                    np.cos(x_norm) * np.sin(y_norm) * np.cos(z_norm) +
                    0.5 * np.cos(2*x_norm) * np.sin(2*y_norm) * np.cos(2*z_norm) +
                    0.02 * np.random.randn(*X.shape)
                )
                
            elif var == 'w':
                # 展向速度
                data[var] = self.velocity_scale * 0.4 * (
                    np.sin(x_norm) * np.sin(y_norm) * np.cos(z_norm) +
                    0.5 * np.sin(2*x_norm) * np.sin(2*y_norm) * np.cos(2*z_norm) +
                    0.02 * np.random.randn(*X.shape)
                )
                
            elif var == 'p':
                # 壓力場：需要滿足連續性方程的約束
                u_val = data.get('u', np.zeros_like(X))
                v_val = data.get('v', np.zeros_like(X))
                w_val = data.get('w', np.zeros_like(X))
                
                data[var] = (
                    -0.5 * (u_val**2 + v_val**2 + w_val**2) +  # 動壓項
                    10.0 * np.cos(x_norm + y_norm + z_norm) +   # 波動壓力
                    0.01 * np.random.randn(*X.shape)
                )
        
        return data
    
    def generate_points(self, points: np.ndarray, 
                       variables: List[str]) -> Dict[str, np.ndarray]:
        """生成各向同性湍流散點資料"""
        
        # 標準化座標
        domain = {'x': [0, 2*np.pi], 'y': [0, 2*np.pi], 'z': [0, 2*np.pi]}
        x_norm = 2 * np.pi * (points[:, 0] - domain['x'][0]) / (domain['x'][1] - domain['x'][0])
        y_norm = 2 * np.pi * (points[:, 1] - domain['y'][0]) / (domain['y'][1] - domain['y'][0])
        z_norm = 2 * np.pi * (points[:, 2] - domain['z'][0]) / (domain['z'][1] - domain['z'][0])
        
        n_points = len(points)
        data = {}
        
        for var in variables:
            if var == 'u':
                data[var] = self.velocity_scale * (
                    np.sin(x_norm) * np.cos(y_norm) * np.sin(z_norm) +
                    0.4 * np.sin(2*x_norm) * np.cos(2*y_norm) * np.sin(2*z_norm) +
                    0.02 * np.random.randn(n_points)
                )
            elif var == 'v':
                data[var] = self.velocity_scale * 0.6 * (
                    np.cos(x_norm) * np.sin(y_norm) * np.cos(z_norm) +
                    0.5 * np.cos(2*x_norm) * np.sin(2*y_norm) * np.cos(2*z_norm) +
                    0.02 * np.random.randn(n_points)
                )
            elif var == 'w':
                data[var] = self.velocity_scale * 0.4 * (
                    np.sin(x_norm) * np.sin(y_norm) * np.cos(z_norm) +
                    0.5 * np.sin(2*x_norm) * np.sin(2*y_norm) * np.cos(2*z_norm) +
                    0.02 * np.random.randn(n_points)
                )
            elif var == 'p':
                u_val = data.get('u', np.zeros(n_points))
                v_val = data.get('v', np.zeros(n_points))
                w_val = data.get('w', np.zeros(n_points))
                data[var] = (
                    -0.5 * (u_val**2 + v_val**2 + w_val**2) +
                    10.0 * np.cos(x_norm + y_norm + z_norm) +
                    0.01 * np.random.randn(n_points)
                )
        
        return data


# Factory function
def create_generator(dataset: str, seed: int = 42, **kwargs) -> TurbulenceGenerator:
    """
    創建湍流生成器的工廠函數
    
    Args:
        dataset: 資料集名稱 ('channel', 'isotropic1024coarse', etc.)
        seed: 隨機種子
        **kwargs: 傳遞給生成器的額外參數
        
    Returns:
        對應的湍流生成器實例
        
    Example:
        >>> generator = create_generator('channel', seed=42)
        >>> X, Y, Z = np.meshgrid(...)
        >>> data = generator.generate_velocity_field(X, Y, Z, ['u', 'v', 'w', 'p'])
    """
    if dataset == 'channel':
        return ChannelFlowGenerator(seed=seed, **kwargs)
    else:
        # 對於其他資料集，默認使用各向同性湍流
        return IsotropicTurbulenceGenerator(seed=seed, **kwargs)


# 測試代碼
if __name__ == "__main__":
    print("🌊 測試湍流生成器...")
    
    # 測試 Channel Flow 生成器
    print("\n=== 測試 Channel Flow 生成器 ===")
    channel_gen = create_generator('channel', seed=42)
    
    # 生成 cutout 資料
    x = np.linspace(0, 8*np.pi, 32)
    y = np.linspace(-1, 1, 32)
    z = np.linspace(0, 3*np.pi, 32)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    cutout_data = channel_gen.generate_velocity_field(X, Y, Z, ['u', 'v', 'w', 'p'])
    
    print("Cutout 資料:")
    for var, arr in cutout_data.items():
        print(f"  {var}: shape={arr.shape}, range=[{arr.min():.3f}, {arr.max():.3f}], mean={arr.mean():.3f}")
    
    # 生成散點資料
    points = np.array([[np.pi, 0.0, np.pi], [2*np.pi, -0.5, 2*np.pi]])
    points_data = channel_gen.generate_points(points, ['u', 'v', 'w', 'p'])
    
    print("\n散點資料:")
    for var, arr in points_data.items():
        print(f"  {var}: {arr}")
    
    # 測試各向同性湍流生成器
    print("\n=== 測試各向同性湍流生成器 ===")
    iso_gen = create_generator('isotropic1024coarse', seed=42)
    
    x = np.linspace(0, 2*np.pi, 32)
    y = np.linspace(0, 2*np.pi, 32)
    z = np.linspace(0, 2*np.pi, 32)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    iso_cutout = iso_gen.generate_velocity_field(X, Y, Z, ['u', 'v', 'w', 'p'])
    
    print("Cutout 資料:")
    for var, arr in iso_cutout.items():
        print(f"  {var}: shape={arr.shape}, range=[{arr.min():.3f}, {arr.max():.3f}], mean={arr.mean():.3f}")
    
    print("\n✅ 湍流生成器測試完成！")
