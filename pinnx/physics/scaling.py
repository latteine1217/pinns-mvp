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
                          scaler_type: str = "vs",
                          learnable: bool = False,
                          **kwargs) -> BaseScaler:
    """
    根據資料自動創建並擬合尺度器
    
    Args:
        input_data: 輸入資料
        output_data: 輸出資料 (可選)
        scaler_type: 尺度器類型 (僅支援 "vs")
        learnable: 是否可學習
        **kwargs: 額外參數
        
    Returns:
        已擬合的尺度器
        
    Note:
        本專案僅支援 VSScaler。資料前處理請使用 pinnx.utils.normalization.UnifiedNormalizer。
    """
    if scaler_type == "vs":
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
        
    raise ValueError(f"不支援的尺度器類型: {scaler_type}。僅支援 'vs'。")
    
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
        scaler: 使用的尺度器（僅支援 VSScaler）
        input_coords: 輸入座標
        output_vars: 輸出變數
        
    Returns:
        物理空間的梯度
        
    Note:
        僅支援 VSScaler 的梯度反標準化。
    """
    if isinstance(scaler, VSScaler):
        # VS-PINN 梯度變換
        # ∂f_phys/∂x_phys = (∂f_norm/∂x_norm) * (σ_f/σ_x)
        input_scale = scaler.input_std
        output_scale = scaler.output_std
        
        scale_factor = output_scale / input_scale
        physical_gradients = gradients * scale_factor
        
    else:
        # 如果不是 VSScaler，返回原始梯度（或拋出警告）
        import warnings
        warnings.warn(
            f"denormalize_gradients() 僅支援 VSScaler，收到 {type(scaler).__name__}。"
            "返回未修改的梯度。建議使用 VSScaler 或在資料預處理階段處理標準化。"
        )
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


# ==============================================================================
# JHTDB Channel Flow 專用無量綱化系統
# ==============================================================================

class NonDimensionalizer(nn.Module):
    """
    JHTDB Channel Flow Re=1000 專用無量綱化器
    
    基於 Buckingham π 定理和 arXiv 2308.08468 最佳實踐設計
    確保物理一致性和數值穩定性
    
    特徵量:
    - L_char = 1.0 (半通道高度)  
    - U_char = 1.0 (摩擦速度 u_τ)
    - t_char = 1.0 (h/u_τ)
    - P_char = 1.0 (ρu_τ²)
    - Re_τ = 1000 (摩擦雷諾數)
    """
    
    def __init__(self, config: Optional[Dict[str, float]] = None):
        super().__init__()
        
        # JHTDB Channel Flow Re=1000 物理參數
        default_config = {
            'L_char': 1.0,         # 半通道高度
            'U_char': 1.0,         # 摩擦速度 u_τ
            't_char': 1.0,         # L_char/U_char
            'P_char': 1.0,         # ρU_char²
            'nu': 1e-3,            # 動力黏度 ν = U_char*L_char/Re_τ
            'Re_tau': 1000.0,      # 摩擦雷諾數
            'rho': 1.0,            # 密度 (標準化為1)
        }
        
        if config is not None:
            default_config.update(config)
        
        self.config = default_config
        
        # 註冊特徵量為緩衝區 (不參與梯度計算)
        for key, value in default_config.items():
            self.register_buffer(key, torch.tensor(float(value)))
        
        # 驗證物理一致性
        self._verify_physical_consistency()
        
        # 統計量緩衝區 (將在 fit 方法中填充)
        self.register_buffer('coord_stats_fitted', torch.tensor(False))
        self.register_buffer('field_stats_fitted', torch.tensor(False))
        
    def _verify_physical_consistency(self):
        """驗證物理參數的一致性"""
        # 雷諾數一致性檢查: Re_τ = U_char * L_char / ν
        # 使用 float64 精度進行關鍵計算以避免數值誤差
        Re_computed = (self.U_char.double() * self.L_char.double() / self.nu.double()).float()
        
        # 調整容差以適應 float32 精度限制 (根據風險分析 #007)
        tolerance = 1e-4  # 從理論目標 1e-12 調整為實際可行的 1e-4
        
        if abs(Re_computed - self.Re_tau) > tolerance:
            raise ValueError(
                f"雷諾數不一致: 配置 Re_τ={self.Re_tau}, "
                f"計算值={Re_computed:.12f}, 差異={abs(Re_computed - self.Re_tau):.2e}, "
                f"容差={tolerance:.2e}"
            )
        
        # 特徵時間一致性檢查
        t_computed = self.L_char / self.U_char
        if abs(t_computed - self.t_char) > 1e-10:
            raise ValueError(f"特徵時間不一致: t_char={self.t_char}, 計算值={t_computed}")
        
        # 特徵壓力一致性檢查  
        P_computed = self.rho * self.U_char**2
        if abs(P_computed - self.P_char) > 1e-10:
            raise ValueError(f"特徵壓力不一致: P_char={self.P_char}, 計算值={P_computed}")
        
        print(f"✅ 物理一致性驗證通過: Re_τ={self.Re_tau}, ν={self.nu}")
    
    def fit_statistics(self, coords: torch.Tensor, fields: torch.Tensor, 
                      robust: bool = True) -> 'NonDimensionalizer':
        """
        根據 JHTDB 數據擬合統計量
        
        Args:
            coords: 座標 [N, 2] (x, y)
            fields: 物理場 [N, 3] (u, v, p) 
            robust: 是否使用穩健統計估計
        """
        with torch.no_grad():
            if robust:
                coord_stats = self._robust_statistics(coords)
                field_stats = self._robust_statistics(fields)
            else:
                coord_stats = self._standard_statistics(coords) 
                field_stats = self._standard_statistics(fields)
            
            # 座標統計量
            self.register_buffer('x_mean', coord_stats['mean'][:1])
            self.register_buffer('x_std', coord_stats['std'][:1])
            self.register_buffer('y_mean', coord_stats['mean'][1:2])
            self.register_buffer('y_std', coord_stats['std'][1:2])
            
            # 場變數統計量 (u, v, p)
            self.register_buffer('u_mean', field_stats['mean'][:1])
            self.register_buffer('u_std', field_stats['std'][:1])
            self.register_buffer('v_mean', field_stats['mean'][1:2])
            self.register_buffer('v_std', field_stats['std'][1:2])
            self.register_buffer('p_mean', field_stats['mean'][2:3])
            self.register_buffer('p_std', field_stats['std'][2:3])
            
            # 更新狀態
            self.coord_stats_fitted = torch.tensor(True)
            self.field_stats_fitted = torch.tensor(True)
            
            # 記錄統計資訊
            print(f"📊 統計量擬合完成:")
            print(f"  座標 - x: μ={self.x_mean.item():.3f}, σ={self.x_std.item():.3f}")
            print(f"  座標 - y: μ={self.y_mean.item():.3f}, σ={self.y_std.item():.3f}")
            print(f"  速度 - u: μ={self.u_mean.item():.3f}, σ={self.u_std.item():.3f}")
            print(f"  速度 - v: μ={self.v_mean.item():.3f}, σ={self.v_std.item():.3f}")
            print(f"  壓力 - p: μ={self.p_mean.item():.3f}, σ={self.p_std.item():.3f}")
        
        return self
    
    def _robust_statistics(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """穩健統計量估計 (基於分位數)"""
        q25, q50, q75 = torch.quantile(data, torch.tensor([0.25, 0.5, 0.75]), dim=0)
        iqr = q75 - q25
        
        # 異常值檢測和過濾
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        filtered_data = []
        for i in range(data.shape[1]):
            col_data = data[:, i]
            mask = (col_data >= lower_bound[i]) & (col_data <= upper_bound[i])
            filtered_col = col_data[mask]
            if len(filtered_col) < 0.5 * len(col_data):  # 如果過濾太多，使用原始數據
                filtered_col = col_data
            filtered_data.append(filtered_col)
        
        # 計算穩健統計量
        robust_mean = torch.stack([torch.median(col) for col in filtered_data])
        robust_std = iqr / 1.349  # 高斯分佈下 IQR→σ 轉換
        
        # 確保 std > 0
        robust_std = torch.where(robust_std < 1e-8, torch.ones_like(robust_std), robust_std)
        
        return {'mean': robust_mean, 'std': robust_std}
    
    def _standard_statistics(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """標準統計量估計"""
        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)
        std = torch.where(std < 1e-8, torch.ones_like(std), std)
        
        return {'mean': mean, 'std': std}
    
    def scale_coordinates(self, coords: torch.Tensor) -> torch.Tensor:
        """
        座標無量綱化: 保持物理意義的同時標準化數值範圍
        
        Args:
            coords: [N, 2] (x, y) 物理座標
            
        Returns:
            scaled_coords: [N, 2] 無量綱化座標
        """
        if not self.coord_stats_fitted:
            raise RuntimeError("座標統計量尚未擬合，請先調用 fit_statistics()")
        
        x, y = coords[:, 0:1], coords[:, 1:2]
        
        # 流向(x): 基於週期長度標準化到 [-1, 1] 
        # JHTDB Channel Flow: x ∈ [0, 8π] → [-1, 1]
        x_scaled = 2 * (x / (8 * np.pi)) - 1
        
        # 壁法向(y): 基於統計量標準化 (已在 [-1, 1])
        y_scaled = (y - self.y_mean) / self.y_std
        
        return torch.cat([x_scaled, y_scaled], dim=1)
    
    def scale_velocity(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        速度場無量綱化: 基於摩擦速度和統計標準化
        
        Args:
            velocity: [N, 2] (u, v) 物理速度
            
        Returns:
            scaled_velocity: [N, 2] 無量綱化速度
        """
        if not self.field_stats_fitted:
            raise RuntimeError("場統計量尚未擬合，請先調用 fit_statistics()")
        
        u, v = velocity[:, 0:1], velocity[:, 1:2]
        
        # 統計標準化 + 限制範圍避免極值
        u_scaled = (u - self.u_mean) / self.u_std
        v_scaled = (v - self.v_mean) / self.v_std
        
        # 限制到合理範圍 (3σ 原則)
        u_scaled = torch.clamp(u_scaled, -3.0, 3.0)
        v_scaled = torch.clamp(v_scaled, -3.0, 3.0)
        
        return torch.cat([u_scaled, v_scaled], dim=1)
    
    def scale_pressure(self, pressure: torch.Tensor) -> torch.Tensor:
        """
        壓力場無量綱化: 基於摩擦壓力標準化
        
        Args:
            pressure: [N, 1] 物理壓力
            
        Returns:
            scaled_pressure: [N, 1] 無量綱化壓力
        """
        if not self.field_stats_fitted:
            raise RuntimeError("場統計量尚未擬合，請先調用 fit_statistics()")
        
        # 統計標準化
        p_scaled = (pressure - self.p_mean) / self.p_std
        
        # 壓力允許更大波動範圍
        p_scaled = torch.clamp(p_scaled, -4.0, 4.0)
        
        return p_scaled
    
    def inverse_scale_coordinates(self, coords_scaled: torch.Tensor) -> torch.Tensor:
        """座標反向縮放"""
        x_scaled, y_scaled = coords_scaled[:, 0:1], coords_scaled[:, 1:2]
        
        # 反向流向縮放: [-1, 1] → [0, 8π]
        x = (x_scaled + 1) * (8 * np.pi) / 2
        
        # 反向壁法向縮放
        y = y_scaled * self.y_std + self.y_mean
        
        return torch.cat([x, y], dim=1)
    
    def inverse_scale_velocity(self, velocity_scaled: torch.Tensor) -> torch.Tensor:
        """速度場反向縮放"""
        u_scaled, v_scaled = velocity_scaled[:, 0:1], velocity_scaled[:, 1:2]
        
        u = u_scaled * self.u_std + self.u_mean
        v = v_scaled * self.v_std + self.v_mean
        
        return torch.cat([u, v], dim=1)
    
    def inverse_scale_pressure(self, pressure_scaled: torch.Tensor) -> torch.Tensor:
        """壓力場反向縮放"""
        return pressure_scaled * self.p_std + self.p_mean
    
    def transform_gradients(self, gradients_scaled: torch.Tensor, 
                          variable_type: str, coord_type: str) -> torch.Tensor:
        """
        梯度縮放變換: 鏈式法則應用
        
        ∂f_phys/∂x_phys = (∂f_scaled/∂x_scaled) × (scale_f/scale_x)
        
        Args:
            gradients_scaled: 縮放空間梯度
            variable_type: 變數類型 ('velocity', 'pressure')
            coord_type: 座標類型 ('spatial_x', 'spatial_y', 'temporal')
            
        Returns:
            physical_gradients: 物理空間梯度
        """
        if variable_type == 'velocity' and coord_type == 'spatial_x':
            # ∂u/∂x: 速度對流向的梯度
            scale_factor = self.u_std / (8 * np.pi / 2)  # u_scale / x_scale
            
        elif variable_type == 'velocity' and coord_type == 'spatial_y':
            # ∂u/∂y: 速度對壁法向的梯度  
            scale_factor = self.u_std / self.y_std
            
        elif variable_type == 'pressure' and coord_type == 'spatial_x':
            # ∂p/∂x: 壓力對流向的梯度
            scale_factor = self.p_std / (8 * np.pi / 2)
            
        elif variable_type == 'pressure' and coord_type == 'spatial_y':
            # ∂p/∂y: 壓力對壁法向的梯度
            scale_factor = self.p_std / self.y_std
            
        elif variable_type == 'velocity' and coord_type == 'temporal':
            # ∂u/∂t: 速度對時間的梯度 (如需時間相關性)
            scale_factor = self.u_std / self.t_char
            
        else:
            raise ValueError(f"不支援的梯度變換: {variable_type} vs {coord_type}")
        
        return gradients_scaled * scale_factor
    
    def compute_ns_scales(self) -> Dict[str, torch.Tensor]:
        """
        計算無量綱化 NS 方程各項的目標量級
        
        確保量級平衡:
        - 對流項: u∂u/∂x ~ O(1)
        - 壓力項: ∂p/∂x ~ O(1)  
        - 黏性項: ν∇²u ~ O(1/Re_τ) ~ 1e-3
        - 時間項: ∂u/∂t ~ O(1)
        """
        return {
            'convection_scale': torch.tensor(1.0),           # 對流項目標量級
            'pressure_scale': torch.tensor(1.0),             # 壓力項目標量級
            'viscous_scale': torch.tensor(1.0 / self.Re_tau), # 黏性項 ~ 1e-3
            'temporal_scale': torch.tensor(1.0),             # 時間項目標量級
            'continuity_scale': torch.tensor(1.0),           # 連續方程目標量級
        }
    
    def validate_scaling(self, coords: torch.Tensor, fields: torch.Tensor) -> Dict[str, bool]:
        """
        驗證縮放的物理一致性
        
        Returns:
            驗證結果字典
        """
        results = {}
        
        # 1. 雷諾數不變性檢查
        Re_original = self.U_char * self.L_char / self.nu
        Re_scaled = 1.0 * 1.0 / (self.nu / (self.U_char * self.L_char))
        reynolds_error = abs(Re_original - Re_scaled)
        results['reynolds_invariant'] = reynolds_error < 1e-12
        
        # 2. 座標變換可逆性
        coords_scaled = self.scale_coordinates(coords)
        coords_recovered = self.inverse_scale_coordinates(coords_scaled)
        coord_error = torch.max(torch.abs(coords - coords_recovered))
        results['coordinate_invertible'] = coord_error < 1e-6
        
        # 3. 場變數可逆性
        velocity = fields[:, :2]
        pressure = fields[:, 2:3]
        
        velocity_scaled = self.scale_velocity(velocity)
        velocity_recovered = self.inverse_scale_velocity(velocity_scaled)
        velocity_error = torch.max(torch.abs(velocity - velocity_recovered))
        results['velocity_invertible'] = velocity_error < 1e-6
        
        pressure_scaled = self.scale_pressure(pressure)
        pressure_recovered = self.inverse_scale_pressure(pressure_scaled)
        pressure_error = torch.max(torch.abs(pressure - pressure_recovered))
        results['pressure_invertible'] = pressure_error < 1e-6
        
        # 4. 邊界條件保持性
        zero_velocity = torch.zeros_like(velocity[:5])  # 測試前5個點
        zero_scaled = self.scale_velocity(zero_velocity)
        results['boundary_preserved'] = torch.allclose(zero_scaled, torch.zeros_like(zero_scaled), atol=1e-8)
        
        return results
    
    def get_scaling_info(self) -> Dict[str, Any]:
        """獲取完整的縮放資訊摘要"""
        return {
            'physical_parameters': {
                'L_char': self.L_char.item(),
                'U_char': self.U_char.item(), 
                't_char': self.t_char.item(),
                'P_char': self.P_char.item(),
                'nu': self.nu.item(),
                'Re_tau': self.Re_tau.item(),
            },
            'coordinate_statistics': {
                'x_mean': self.x_mean.item() if self.coord_stats_fitted else None,
                'x_std': self.x_std.item() if self.coord_stats_fitted else None,
                'y_mean': self.y_mean.item() if self.coord_stats_fitted else None,
                'y_std': self.y_std.item() if self.coord_stats_fitted else None,
            },
            'field_statistics': {
                'u_mean': self.u_mean.item() if self.field_stats_fitted else None,
                'u_std': self.u_std.item() if self.field_stats_fitted else None,
                'v_mean': self.v_mean.item() if self.field_stats_fitted else None,
                'v_std': self.v_std.item() if self.field_stats_fitted else None,
                'p_mean': self.p_mean.item() if self.field_stats_fitted else None,
                'p_std': self.p_std.item() if self.field_stats_fitted else None,
            },
            'fitted_status': {
                'coordinates': self.coord_stats_fitted.item(),
                'fields': self.field_stats_fitted.item(),
            }
        }
