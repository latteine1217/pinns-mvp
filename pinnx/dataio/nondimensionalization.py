"""
JHTDB數據無因次化處理模組
========================

提供Johns Hopkins Turbulence Database數據的無因次化處理功能：
1. 不同湍流資料集的特徵參數定義
2. 座標和物理量的無因次化/有因次化轉換
3. 雷諾數和無因次時間的處理
4. 多種無因次化方案支援

支援的資料集：
- Channel Flow (通道流)
- Homogeneous Isotropic Turbulence (各向同性湍流)
- Transitional Boundary Layer (過渡邊界層)

無因次化基準：
- 特徵長度：通道半高度h、積分尺度、邊界層厚度等
- 特徵速度：摩擦速度、特徵渦旋速度、自由流速度等
- 特徵時間：對流時間、渦旋翻轉時間、黏性時間等
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, Union
import warnings


class JHTDBNondimensionalizer:
    """
    JHTDB數據無因次化處理器
    
    提供統一的無因次化/有因次化接口
    """
    
    def __init__(self, 
                 dataset_type: str,
                 reynolds_number: Optional[float] = None,
                 custom_scales: Optional[Dict] = None):
        """
        初始化無因次化處理器
        
        Args:
            dataset_type: 資料集類型 ("channel", "hit", "tbl", "mhd")
            reynolds_number: 雷諾數
            custom_scales: 自定義特徵尺度
        """
        self.dataset_type = dataset_type.lower()
        self.reynolds_number = reynolds_number
        
        # 設定預設特徵尺度
        self._setup_default_scales()
        
        # 覆蓋自定義尺度
        if custom_scales:
            self.scales.update(custom_scales)
            
        # 計算無因次數
        self._compute_dimensionless_numbers()
    
    def _setup_default_scales(self):
        """設定各資料集的預設特徵尺度"""
        
        if self.dataset_type == "channel":
            # 通道流：基於摩擦速度的尺度
            self.scales = {
                # 幾何參數
                'L_ref': 1.0,           # 通道半高度 h (m)
                'h': 1.0,               # 通道半高度 (m)
                'delta': 2.0,           # 通道全高度 (m)
                
                # 流動參數
                'u_tau': 1.0,           # 摩擦速度 (m/s)  
                'u_bulk': 20.0,         # 體流速度 (m/s)
                'u_centerline': 30.0,   # 中心線速度 (m/s)
                
                # 流體性質  
                'nu': 1e-5,             # 運動黏滯係數 (m²/s)
                'rho': 1.0,             # 密度 (kg/m³)
                
                # 時間尺度
                't_viscous': 1.0,       # 黏性時間 h²/ν
                't_eddy': 1.0,          # 渦旋時間 h/u_τ
                
                # 壓力尺度
                'p_ref': 1.0,           # 參考壓力 ρu_τ²
            }
            
        elif self.dataset_type == "hit":
            # 各向同性湍流：基於積分尺度的尺度
            self.scales = {
                # 積分尺度
                'L_ref': 1.0,           # 積分長度尺度 (m)
                'eta': 0.01,            # Kolmogorov尺度 (m)
                'L_int': 1.0,           # 積分尺度 (m)
                
                # 速度尺度
                'u_rms': 1.0,           # RMS速度 (m/s)
                'u_eta': 0.1,           # Kolmogorov速度 (m/s)
                
                # 流體性質
                'nu': 1e-5,             # 運動黏滯係數 (m²/s)
                'rho': 1.0,             # 密度 (kg/m³)
                
                # 時間尺度
                't_eddy': 1.0,          # 大渦旋翻轉時間 L_int/u_rms
                't_eta': 0.01,          # Kolmogorov時間 (ν/ε)^0.5
                
                # 能量耗散率
                'epsilon': 1.0,         # 能量耗散率 (m²/s³)
            }
            
        elif self.dataset_type == "tbl":
            # 過渡邊界層：基於動量厚度的尺度
            self.scales = {
                # 邊界層參數
                'L_ref': 1.0,           # 動量厚度 θ (m)
                'delta_99': 10.0,       # 邊界層厚度 (m)
                'theta': 1.0,           # 動量厚度 (m)
                'delta_star': 1.3,      # 位移厚度 (m)
                
                # 速度尺度
                'U_inf': 10.0,          # 自由流速度 (m/s)
                'u_tau': 0.5,           # 摩擦速度 (m/s)
                
                # 流體性質
                'nu': 1e-5,             # 運動黏滯係數 (m²/s)
                'rho': 1.0,             # 密度 (kg/m³)
                
                # 時間尺度
                't_convective': 0.1,    # 對流時間 θ/U_∞
                't_viscous': 100.0,     # 黏性時間 θ²/ν
            }
            
        else:
            # 預設尺度
            self.scales = {
                'L_ref': 1.0,
                'u_ref': 1.0, 
                'nu': 1e-5,
                'rho': 1.0,
                't_ref': 1.0,
                'p_ref': 1.0
            }
            warnings.warn(f"未知資料集類型 '{self.dataset_type}'，使用預設尺度")
    
    def _compute_dimensionless_numbers(self):
        """計算相關的無因次數"""
        
        # 雷諾數
        if self.reynolds_number is None:
            if self.dataset_type == "channel":
                # Re_τ = u_τ * h / ν
                self.reynolds_number = self.scales['u_tau'] * self.scales['h'] / self.scales['nu']
            elif self.dataset_type == "hit":
                # Re_λ = u_rms * λ / ν (Taylor微尺度雷諾數)
                lambda_g = np.sqrt(15 * self.scales['nu'] / self.scales['epsilon']) * self.scales['u_rms']
                self.reynolds_number = self.scales['u_rms'] * lambda_g / self.scales['nu']
            elif self.dataset_type == "tbl":
                # Re_θ = U_∞ * θ / ν
                self.reynolds_number = self.scales['U_inf'] * self.scales['theta'] / self.scales['nu']
            else:
                self.reynolds_number = 1000.0  # 預設值
        
        # 其他無因次參數
        if self.dataset_type == "channel":
            self.Re_tau = self.reynolds_number
            self.Re_bulk = self.scales['u_bulk'] * self.scales['h'] / self.scales['nu']
        elif self.dataset_type == "hit":
            self.Re_lambda = self.reynolds_number  
            # 大尺度雷諾數
            self.Re_L = self.scales['u_rms'] * self.scales['L_int'] / self.scales['nu']
        elif self.dataset_type == "tbl":
            self.Re_theta = self.reynolds_number
            self.Re_delta = self.scales['U_inf'] * self.scales['delta_99'] / self.scales['nu']
    
    def nondimensionalize_coords(self, 
                               coords: Union[torch.Tensor, np.ndarray],
                               coord_type: str = "spatial") -> Union[torch.Tensor, np.ndarray]:
        """
        座標無因次化
        
        Args:
            coords: 座標陣列 [batch_size, dim]
            coord_type: 座標類型 ("spatial", "temporal", "spatiotemporal")
            
        Returns:
            無因次化座標
        """
        is_torch = isinstance(coords, torch.Tensor)
        if is_torch:
            coords_np = coords.detach().cpu().numpy()
        else:
            coords_np = coords.copy()
        
        if coord_type == "spatial":
            # 空間座標：除以特徵長度
            coords_nd = coords_np / self.scales['L_ref']
            
        elif coord_type == "temporal":
            # 時間座標：除以特徵時間
            if self.dataset_type == "channel":
                t_ref = self.scales['h'] / self.scales['u_tau']  # h/u_τ
            elif self.dataset_type == "hit":
                t_ref = self.scales['L_int'] / self.scales['u_rms']  # L/u_rms
            elif self.dataset_type == "tbl":
                t_ref = self.scales['theta'] / self.scales['U_inf']  # θ/U_∞
            else:
                t_ref = self.scales.get('t_ref', 1.0)
                
            coords_nd = coords_np / t_ref
            
        elif coord_type == "spatiotemporal":
            # 時空座標：[t, x, y, z] 或 [t, x, y]
            coords_nd = coords_np.copy()
            
            # 時間維度
            if self.dataset_type == "channel":
                t_ref = self.scales['h'] / self.scales['u_tau']
            elif self.dataset_type == "hit":
                t_ref = self.scales['L_int'] / self.scales['u_rms']
            elif self.dataset_type == "tbl":
                t_ref = self.scales['theta'] / self.scales['U_inf']
            else:
                t_ref = self.scales.get('t_ref', 1.0)
                
            coords_nd[:, 0] = coords_np[:, 0] / t_ref
            
            # 空間維度
            coords_nd[:, 1:] = coords_np[:, 1:] / self.scales['L_ref']
            
        else:
            raise ValueError(f"未知座標類型: {coord_type}")
        
        if is_torch:
            return torch.from_numpy(coords_nd).float()
        else:
            return coords_nd
    
    def nondimensionalize_velocity(self, 
                                 velocity: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        速度無因次化
        
        Args:
            velocity: 速度場 [batch_size, dim] 
            
        Returns:
            無因次化速度
        """
        is_torch = isinstance(velocity, torch.Tensor)
        
        if self.dataset_type == "channel":
            u_ref = self.scales['u_tau']
        elif self.dataset_type == "hit":
            u_ref = self.scales['u_rms']
        elif self.dataset_type == "tbl":
            u_ref = self.scales['U_inf']
        else:
            u_ref = self.scales.get('u_ref', 1.0)
        
        if is_torch:
            return velocity / u_ref
        else:
            return velocity / u_ref
    
    def nondimensionalize_pressure(self, 
                                 pressure: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        壓力無因次化
        
        Args:
            pressure: 壓力場 [batch_size, 1]
            
        Returns:
            無因次化壓力
        """
        is_torch = isinstance(pressure, torch.Tensor)
        
        if self.dataset_type == "channel":
            p_ref = self.scales['rho'] * self.scales['u_tau']**2
        elif self.dataset_type == "hit":
            p_ref = self.scales['rho'] * self.scales['u_rms']**2
        elif self.dataset_type == "tbl":
            p_ref = self.scales['rho'] * self.scales['U_inf']**2
        else:
            p_ref = self.scales.get('p_ref', 1.0)
        
        if is_torch:
            return pressure / p_ref
        else:
            return pressure / p_ref
    
    def dimensionalize_coords(self, 
                            coords_nd: Union[torch.Tensor, np.ndarray],
                            coord_type: str = "spatial") -> Union[torch.Tensor, np.ndarray]:
        """座標有因次化（無因次化的逆操作）"""
        is_torch = isinstance(coords_nd, torch.Tensor)
        if is_torch:
            coords_np = coords_nd.detach().cpu().numpy()
        else:
            coords_np = coords_nd.copy()
        
        if coord_type == "spatial":
            coords_d = coords_np * self.scales['L_ref']
            
        elif coord_type == "temporal":
            if self.dataset_type == "channel":
                t_ref = self.scales['h'] / self.scales['u_tau']
            elif self.dataset_type == "hit":
                t_ref = self.scales['L_int'] / self.scales['u_rms']
            elif self.dataset_type == "tbl":
                t_ref = self.scales['theta'] / self.scales['U_inf']
            else:
                t_ref = self.scales.get('t_ref', 1.0)
                
            coords_d = coords_np * t_ref
            
        elif coord_type == "spatiotemporal":
            coords_d = coords_np.copy()
            
            # 時間維度
            if self.dataset_type == "channel":
                t_ref = self.scales['h'] / self.scales['u_tau']
            elif self.dataset_type == "hit":
                t_ref = self.scales['L_int'] / self.scales['u_rms']
            elif self.dataset_type == "tbl":
                t_ref = self.scales['theta'] / self.scales['U_inf']
            else:
                t_ref = self.scales.get('t_ref', 1.0)
                
            coords_d[:, 0] = coords_np[:, 0] * t_ref
            coords_d[:, 1:] = coords_np[:, 1:] * self.scales['L_ref']
            
        else:
            raise ValueError(f"未知座標類型: {coord_type}")
        
        if is_torch:
            return torch.from_numpy(coords_d).float()
        else:
            return coords_d
    
    def dimensionalize_velocity(self, 
                              velocity_nd: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """速度有因次化"""
        is_torch = isinstance(velocity_nd, torch.Tensor)
        
        if self.dataset_type == "channel":
            u_ref = self.scales['u_tau']
        elif self.dataset_type == "hit":
            u_ref = self.scales['u_rms']
        elif self.dataset_type == "tbl":
            u_ref = self.scales['U_inf']
        else:
            u_ref = self.scales.get('u_ref', 1.0)
        
        if is_torch:
            return velocity_nd * u_ref
        else:
            return velocity_nd * u_ref
    
    def dimensionalize_pressure(self, 
                              pressure_nd: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """壓力有因次化"""
        is_torch = isinstance(pressure_nd, torch.Tensor)
        
        if self.dataset_type == "channel":
            p_ref = self.scales['rho'] * self.scales['u_tau']**2
        elif self.dataset_type == "hit":
            p_ref = self.scales['rho'] * self.scales['u_rms']**2
        elif self.dataset_type == "tbl":
            p_ref = self.scales['rho'] * self.scales['U_inf']**2
        else:
            p_ref = self.scales.get('p_ref', 1.0)
        
        if is_torch:
            return pressure_nd * p_ref
        else:
            return pressure_nd * p_ref
    
    def get_scale_info(self) -> Dict:
        """獲取尺度資訊"""
        info = {
            'dataset_type': self.dataset_type,
            'reynolds_number': self.reynolds_number,
            'scales': self.scales.copy()
        }
        
        # 添加計算出的無因次數
        if hasattr(self, 'Re_tau'):
            info['Re_tau'] = self.Re_tau
        if hasattr(self, 'Re_lambda'):
            info['Re_lambda'] = self.Re_lambda
        if hasattr(self, 'Re_theta'):
            info['Re_theta'] = self.Re_theta
            
        return info
    
    def update_scales(self, **kwargs):
        """更新尺度參數"""
        for key, value in kwargs.items():
            if key in self.scales:
                self.scales[key] = value
            else:
                warnings.warn(f"未知尺度參數: {key}")
        
        # 重新計算無因次數
        self._compute_dimensionless_numbers()


def create_channel_flow_nondimensionalizer(re_tau: float = 180) -> JHTDBNondimensionalizer:
    """
    創建通道流專用無因次化器
    
    Args:
        re_tau: 摩擦雷諾數 Re_τ = u_τ*h/ν
    """
    # 根據Re_τ設定特徵尺度
    h = 1.0          # 通道半高度
    nu = 1e-5        # 運動黏滯係數
    u_tau = re_tau * nu / h  # 摩擦速度
    
    scales = {
        'L_ref': h,
        'h': h,
        'u_tau': u_tau,
        'nu': nu,
        'rho': 1.0
    }
    
    return JHTDBNondimensionalizer(
        dataset_type="channel",
        reynolds_number=re_tau,
        custom_scales=scales
    )


def create_hit_nondimensionalizer(re_lambda: float = 100) -> JHTDBNondimensionalizer:
    """
    創建各向同性湍流專用無因次化器
    
    Args:
        re_lambda: Taylor微尺度雷諾數
    """
    # 典型HIT參數
    L_int = 1.0      # 積分尺度
    u_rms = 1.0      # RMS速度
    nu = 1e-5        # 運動黏滯係數
    
    scales = {
        'L_ref': L_int,
        'L_int': L_int,
        'u_rms': u_rms,
        'nu': nu,
        'rho': 1.0
    }
    
    return JHTDBNondimensionalizer(
        dataset_type="hit",
        reynolds_number=re_lambda,
        custom_scales=scales
    )


if __name__ == "__main__":
    # 測試無因次化系統
    print("🧪 測試JHTDB無因次化系統")
    
    # 1. 創建通道流無因次化器
    nondim = create_channel_flow_nondimensionalizer(re_tau=180)
    print(f"通道流尺度信息: {nondim.get_scale_info()}")
    
    # 2. 測試座標無因次化
    coords_dimensional = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],    # [t, x, y, z]
        [1.0, np.pi, 0.5, np.pi],
        [2.0, 2*np.pi, -0.8, 2*np.pi]
    ], dtype=torch.float32)
    
    coords_nd = nondim.nondimensionalize_coords(coords_dimensional, "spatiotemporal")
    coords_back = nondim.dimensionalize_coords(coords_nd, "spatiotemporal")
    
    print("\n座標無因次化測試:")
    print(f"原始座標: {coords_dimensional}")
    print(f"無因次座標: {coords_nd}")
    print(f"恢復座標: {coords_back}")
    print(f"誤差: {torch.max(torch.abs(coords_dimensional - coords_back)).item():.2e}")
    
    # 3. 測試速度無因次化
    velocity = torch.tensor([
        [1.0, 0.1, 0.05],  # [u, v, w]
        [0.8, -0.2, 0.03],
        [1.2, 0.15, -0.08]
    ], dtype=torch.float32)
    
    velocity_nd = nondim.nondimensionalize_velocity(velocity)
    velocity_back = nondim.dimensionalize_velocity(velocity_nd)
    
    print("\n速度無因次化測試:")
    print(f"原始速度: {velocity}")
    print(f"無因次速度: {velocity_nd}")
    print(f"恢復速度: {velocity_back}")
    print(f"誤差: {torch.max(torch.abs(velocity - velocity_back)).item():.2e}")
    
    print("✅ JHTDB無因次化系統測試完成")