"""
符號距離函數(SDF)權重系統
=========================

提供基於幾何約束的自適應權重計算功能：
1. 基本幾何SDF計算 (矩形、圓形、通道等)
2. 距離基權重函數
3. 邊界附近權重增強
4. 多重邊界條件處理
5. 與現有動態權重系統整合

在PINNs中，SDF權重主要用於：
- 邊界條件強化：接近邊界的點獲得更高權重
- 物理域分區：不同區域應用不同的物理約束
- 漸進式訓練：從簡單幾何逐步過渡到複雜幾何
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
import warnings


class SDFWeightCalculator:
    """
    符號距離函數權重計算器
    
    支援多種幾何體的SDF計算和權重分配策略
    """
    
    def __init__(self, 
                 domain_type: str = "channel",
                 domain_bounds: Optional[Dict] = None,
                 weight_strategy: str = "exponential"):
        """
        初始化SDF權重計算器
        
        Args:
            domain_type: 域類型 ("channel", "rectangle", "circle", "custom")
            domain_bounds: 域邊界定義，格式依domain_type而定
            weight_strategy: 權重策略 ("exponential", "linear", "gaussian", "inverse")
        """
        self.domain_type = domain_type
        self.weight_strategy = weight_strategy
        
        # 預設domain_bounds
        if domain_bounds is None:
            if domain_type == "channel":
                # 通道流：[0,4π] x [-1,1] x [0,2π]
                self.domain_bounds = {
                    'x_min': 0.0, 'x_max': 4*np.pi,
                    'y_min': -1.0, 'y_max': 1.0,
                    'z_min': 0.0, 'z_max': 2*np.pi
                }
            elif domain_type == "rectangle":
                # 預設矩形域：[-1,1] x [-1,1]
                self.domain_bounds = {
                    'x_min': -1.0, 'x_max': 1.0,
                    'y_min': -1.0, 'y_max': 1.0
                }
            else:
                raise ValueError(f"請為domain_type='{domain_type}'提供domain_bounds")
        else:
            self.domain_bounds = domain_bounds
            
        # 權重參數 - 優化邊界處理
        self.weight_params = {
            'boundary_width': 0.15,      # 邊界層寬度 (增加以改善邊界處理)
            'interior_weight': 1.0,      # 內部權重
            'boundary_weight': 3.0,      # 邊界權重 (降低以減少過度強化)
            'decay_rate': 1.5            # 衰減率 (降低以更平滑過渡)
        }
    
    def compute_sdf_2d(self, 
                       coords: torch.Tensor) -> torch.Tensor:
        """
        計算2D符號距離函數
        
        Args:
            coords: 座標點 [batch_size, 2] = [x, y]
            
        Returns:
            SDF值 [batch_size, 1]，負值表示內部，正值表示外部
        """
        if self.domain_type == "rectangle":
            return self._sdf_rectangle_2d(coords)
        elif self.domain_type == "circle":
            return self._sdf_circle_2d(coords)
        else:
            raise NotImplementedError(f"2D SDF for {self.domain_type} not implemented")
    
    def compute_sdf_3d(self, 
                       coords: torch.Tensor) -> torch.Tensor:
        """
        計算3D符號距離函數
        
        Args:
            coords: 座標點 [batch_size, 3] = [x, y, z]
            
        Returns:
            SDF值 [batch_size, 1]
        """
        if self.domain_type == "channel":
            return self._sdf_channel_3d(coords)
        elif self.domain_type == "box":
            return self._sdf_box_3d(coords)
        else:
            raise NotImplementedError(f"3D SDF for {self.domain_type} not implemented")
    
    def _sdf_rectangle_2d(self, coords: torch.Tensor) -> torch.Tensor:
        """2D矩形域SDF"""
        x, y = coords[:, 0:1], coords[:, 1:2]
        
        # 到邊界的距離
        dx = torch.minimum(x - self.domain_bounds['x_min'], 
                          self.domain_bounds['x_max'] - x)
        dy = torch.minimum(y - self.domain_bounds['y_min'], 
                          self.domain_bounds['y_max'] - y)
        
        # 內部：負值，外部：正值
        inside = torch.minimum(dx, dy)
        outside_x = torch.maximum(torch.abs(x) - (self.domain_bounds['x_max'] - self.domain_bounds['x_min'])/2, 
                                 torch.zeros_like(x))
        outside_y = torch.maximum(torch.abs(y) - (self.domain_bounds['y_max'] - self.domain_bounds['y_min'])/2, 
                                 torch.zeros_like(y))
        outside = torch.sqrt(outside_x**2 + outside_y**2)
        
        return torch.where(torch.logical_and(dx > 0, dy > 0), -inside, outside)
    
    def _sdf_channel_3d(self, coords: torch.Tensor) -> torch.Tensor:
        """3D通道流域SDF - 主要關注y方向的壁面"""
        x, y, z = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]
        
        # 通道流最重要的是y方向的壁面約束
        # y = ±1 是主要邊界
        wall_distance = 1.0 - torch.abs(y)  # 到壁面的距離
        
        # x, z方向週期性，不需要強邊界約束
        # 但可以添加軟約束避免過度外推
        x_dist = torch.minimum(x - self.domain_bounds['x_min'], 
                              self.domain_bounds['x_max'] - x)
        z_dist = torch.minimum(z - self.domain_bounds['z_min'], 
                              self.domain_bounds['z_max'] - z)
        
        # 主要使用壁面距離，輔以週期邊界軟約束
        sdf = wall_distance - 0.1 * torch.minimum(torch.minimum(x_dist, z_dist), torch.zeros_like(x_dist))
        
        return sdf
    
    def compute_boundary_weights(self, 
                                coords: torch.Tensor,
                                sdf_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        基於SDF計算邊界權重
        
        Args:
            coords: 座標點
            sdf_values: 預計算的SDF值（可選）
            
        Returns:
            權重值 [batch_size, 1]
        """
        if sdf_values is None:
            if coords.shape[1] == 2:
                sdf_values = self.compute_sdf_2d(coords)
            elif coords.shape[1] == 3:
                sdf_values = self.compute_sdf_3d(coords)
            else:
                raise ValueError(f"不支援的座標維度: {coords.shape[1]}")
        
        # 計算權重
        distance_to_boundary = torch.abs(sdf_values)
        
        if self.weight_strategy == "exponential":
            weights = self._exponential_weights(distance_to_boundary)
        elif self.weight_strategy == "linear":
            weights = self._linear_weights(distance_to_boundary)
        elif self.weight_strategy == "gaussian":
            weights = self._gaussian_weights(distance_to_boundary)
        elif self.weight_strategy == "inverse":
            weights = self._inverse_weights(distance_to_boundary)
        else:
            raise ValueError(f"未知權重策略: {self.weight_strategy}")
        
        return weights
    
    def _exponential_weights(self, distance: torch.Tensor) -> torch.Tensor:
        """指數衰減權重"""
        boundary_width = self.weight_params['boundary_width']
        boundary_weight = self.weight_params['boundary_weight']
        interior_weight = self.weight_params['interior_weight']
        decay_rate = self.weight_params['decay_rate']
        
        # 在邊界層內使用指數增長，外部使用基準權重
        mask = distance <= boundary_width
        
        weights = torch.ones_like(distance) * interior_weight
        weights[mask] = interior_weight + (boundary_weight - interior_weight) * \
                       torch.exp(-decay_rate * distance[mask] / boundary_width)
        
        return weights
    
    def _linear_weights(self, distance: torch.Tensor) -> torch.Tensor:
        """線性權重"""
        boundary_width = self.weight_params['boundary_width']
        boundary_weight = self.weight_params['boundary_weight']
        interior_weight = self.weight_params['interior_weight']
        
        # 線性插值
        ratio = torch.clamp(distance / boundary_width, 0.0, 1.0)
        weights = boundary_weight * (1 - ratio) + interior_weight * ratio
        
        return weights
    
    def _gaussian_weights(self, distance: torch.Tensor) -> torch.Tensor:
        """高斯權重"""
        boundary_width = self.weight_params['boundary_width']
        boundary_weight = self.weight_params['boundary_weight']
        interior_weight = self.weight_params['interior_weight']
        
        # 高斯分佈
        sigma = boundary_width / 3.0  # 3σ規則
        gaussian = torch.exp(-0.5 * (distance / sigma)**2)
        weights = interior_weight + (boundary_weight - interior_weight) * gaussian
        
        return weights
    
    def _inverse_weights(self, distance: torch.Tensor) -> torch.Tensor:
        """反比權重"""
        boundary_weight = self.weight_params['boundary_weight']
        interior_weight = self.weight_params['interior_weight']
        epsilon = 1e-6  # 避免除零
        
        weights = interior_weight + (boundary_weight - interior_weight) / (distance + epsilon)
        
        return weights
    
    def update_weight_params(self, **kwargs):
        """更新權重參數"""
        for key, value in kwargs.items():
            if key in self.weight_params:
                self.weight_params[key] = value
            else:
                warnings.warn(f"未知權重參數: {key}")


class SDFEnhancedWeightScheduler:
    """
    SDF增強的權重調度器
    
    整合原有的動態權重系統與SDF幾何權重
    """
    
    def __init__(self,
                 base_scheduler,  # DynamicWeightScheduler實例
                 sdf_calculator: SDFWeightCalculator,
                 sdf_weight: float = 0.1):
        """
        Args:
            base_scheduler: 基礎動態權重調度器
            sdf_calculator: SDF權重計算器
            sdf_weight: SDF權重的影響係數
        """
        self.base_scheduler = base_scheduler
        self.sdf_calculator = sdf_calculator
        self.sdf_weight = sdf_weight
    
    def compute_enhanced_weights(self,
                               coords: torch.Tensor,
                               data_loss: float,
                               physics_loss: float,
                               epoch: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        計算SDF增強的權重
        
        Returns:
            data_weights: 數據項權重 [batch_size, 1]
            physics_weights: 物理項權重 [batch_size, 1]
        """
        # 1. 獲取基礎動態權重（標量）
        base_data_weight, base_physics_weight = self.base_scheduler.update(
            data_loss, physics_loss, epoch
        )
        
        # 2. 計算SDF幾何權重（張量）
        sdf_weights = self.sdf_calculator.compute_boundary_weights(coords)
        
        # 3. 組合權重
        # 數據項：遠離邊界時保持基礎權重，接近邊界時適當降低
        data_weights = base_data_weight * torch.ones_like(sdf_weights) * \
                      (1.0 - self.sdf_weight * (sdf_weights - 1.0))
        
        # 物理項：接近邊界時增強權重
        physics_weights = base_physics_weight * torch.ones_like(sdf_weights) * \
                         (1.0 + self.sdf_weight * (sdf_weights - 1.0))
        
        return data_weights, physics_weights
    
    def get_point_wise_weights(self, coords: torch.Tensor) -> torch.Tensor:
        """獲取點的幾何權重（用於loss計算）"""
        return self.sdf_calculator.compute_boundary_weights(coords)


def create_channel_flow_sdf_weights() -> SDFWeightCalculator:
    """
    創建通道流專用的SDF權重計算器
    """
    return SDFWeightCalculator(
        domain_type="channel",
        domain_bounds={
            'x_min': 0.0, 'x_max': 4*np.pi,      # 流向：週期
            'y_min': -1.0, 'y_max': 1.0,          # 法向：壁面
            'z_min': 0.0, 'z_max': 2*np.pi       # 展向：週期
        },
        weight_strategy="exponential"
    )


def create_rectangle_sdf_weights(bounds: Dict) -> SDFWeightCalculator:
    """
    創建矩形域專用的SDF權重計算器
    """
    return SDFWeightCalculator(
        domain_type="rectangle",
        domain_bounds=bounds,
        weight_strategy="gaussian"
    )


if __name__ == "__main__":
    # 測試SDF權重系統
    print("🧪 測試SDF權重系統")
    
    # 1. 創建通道流SDF計算器
    sdf_calc = create_channel_flow_sdf_weights()
    
    # 2. 測試3D點
    coords_3d = torch.tensor([
        [np.pi, 0.0, np.pi],     # 中心點
        [np.pi, 0.9, np.pi],     # 接近上壁面
        [np.pi, -0.95, np.pi],   # 接近下壁面
        [np.pi, 0.5, np.pi],     # 中間位置
    ], dtype=torch.float32)
    
    # 3. 計算SDF和權重
    sdf_values = sdf_calc.compute_sdf_3d(coords_3d)
    weights = sdf_calc.compute_boundary_weights(coords_3d, sdf_values)
    
    print("測試點SDF值和權重:")
    for i, (coord, sdf, weight) in enumerate(zip(coords_3d, sdf_values, weights)):
        print(f"  點{i+1}: coord={coord.numpy()}, SDF={sdf.item():.4f}, weight={weight.item():.4f}")
    
    print("✅ SDF權重系統測試完成")