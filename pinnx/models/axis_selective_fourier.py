"""
軸向選擇性 Fourier Features 模組

針對非均勻幾何（如通道流）的專門優化：
- 週期軸（x, z）使用固定循環數頻率
- 非週期軸（y-壁法向）避免或極少使用 Fourier
- 支持頻率退火：分階段逐步解鎖高頻

設計原則：
1. 週期軸保守頻率：K_x={1,2,4,8}, K_z={1,2,4}
2. 非週期軸避免高頻：y 軸空列表或僅1個最低尺度
3. 物理域歸一化：將各軸映射到 [0, 2π] 以匹配標準 Fourier 假設
"""

import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional, cast
import numpy as np


class AxisSelectiveFourierFeatures(nn.Module):
    """
    軸向選擇性 Fourier Features 編碼層
    
    與標準 FourierFeatures 的主要差異：
    1. 每個軸使用不同的頻率組（非統一隨機高斯）
    2. 頻率基於物理循環數（wavenumber k = 2πn/L）
    3. 支持部分軸不使用 Fourier（空列表）
    
    Args:
        axes_config: 字典指定各軸**當前啟用**的頻率
            - 鍵：軸名稱（'x', 'y', 'z', 't' 等）
            - 值：循環數列表（如 [1, 2, 4, 8]）或空列表（不用 Fourier）
            - 順序必須與輸入張量維度對應
            
        domain_lengths: 各軸物理域長度（用於歸一化）
            - 鍵：軸名稱
            - 值：物理長度（如 x: 8π, y: 2, z: 3π）
            - 若未提供則假設已歸一化（長度=2π）
            
        trainable: 是否讓 Fourier 係數可訓練（預設 False）
        
        full_axes_config: 完整頻率配置（用於 Fourier 退火驗證）
            - 必須包含所有可能解鎖的頻率
            - 範例：axes_config=[1,2], full_axes_config=[1,2,4,8]
        
    輸出維度：
        - 對於有 n 個頻率的軸：貢獻 2n 維（cos + sin）
        - 對於空列表軸：貢獻 0 維（該軸直接略過）
        - 總輸出維度 = Σ(2 * len(freqs_i))
    
    Examples:
        >>> # 範例 1：基本使用（通道流配置）
        >>> config = {
        ...     'x': [1, 2, 4, 8],  # 4 個頻率 → 8 維
        ...     'y': [],            # 不用 Fourier → 0 維
        ...     'z': [1, 2, 4]      # 3 個頻率 → 6 維
        ... }
        >>> domain_lengths = {'x': 25.13, 'y': 2.0, 'z': 9.42}  # 8π, 2h, 3π
        >>> fourier = AxisSelectiveFourierFeatures(
        ...     axes_config=config,
        ...     full_axes_config=config,
        ...     domain_lengths=domain_lengths
        ... )
        >>> x = torch.randn(100, 3)  # [batch, 3] 對應 (x, y, z)
        >>> features = fourier(x)    # [batch, 14] = 8 + 0 + 6
        
        >>> # 範例 2：頻率退火（雙配置）
        >>> # 初始只啟用低頻，預留高頻空間
        >>> initial_config = {'x': [1, 2], 'y': [], 'z': [1, 2]}
        >>> full_config = {'x': [1, 2, 4, 8], 'y': [], 'z': [1, 2, 4, 8]}
        >>> fourier = AxisSelectiveFourierFeatures(
        ...     axes_config=initial_config,
        ...     full_axes_config=full_config,
        ...     domain_lengths=domain_lengths
        ... )
        >>> features = fourier(x)  # [batch, 16] = 固定輸出維度（基於 full_config）
        >>> # 在訓練過程中動態解鎖高頻
        >>> fourier.set_active_frequencies({'x': [1, 2, 4], 'y': [], 'z': [1, 2, 4]})
        >>> features = fourier(x)  # [batch, 16] = 維度不變，未啟用頻率被掩碼置零
    """
    
    def __init__(
        self,
        axes_config: Dict[str, List[int]],
        domain_lengths: Optional[Dict[str, float]] = None,
        trainable: bool = False,
        full_axes_config: Optional[Dict[str, List[int]]] = None,
    ):
        super().__init__()
        
        # 驗證配置
        self.axes_names = list(axes_config.keys())
        self.in_dim = len(self.axes_names)
        
        if self.in_dim == 0:
            raise ValueError("axes_config 不能為空字典")
        
        # 驗證頻率值（必須是非負整數）
        for axis, freqs in axes_config.items():
            for k in freqs:
                if not isinstance(k, (int, float)) or k < 0:
                    raise ValueError(
                        f"軸 '{axis}' 的頻率 {k} 無效（必須是非負數）"
                    )
        
        if full_axes_config is None:
            raise ValueError("AxisSelectiveFourierFeatures 需要提供 full_axes_config")

        # 儲存配置
        # - full_axes_config：完整頻率配置（用於構建 Fourier 矩陣 → 固定維度）
        # - axes_config：當前啟用頻率（用於退火控制 → 動態掩碼）
        self._full_axes_config = {k: list(v) for k, v in full_axes_config.items()}
        self.axes_config = {k: list(v) for k, v in axes_config.items()}
        self.trainable = trainable
        
        # 處理域長度（用於歸一化到 [0, 2π]）
        if domain_lengths is None:
            # 預設：假設已歸一化到 2π
            domain_lengths = {axis: 2.0 * math.pi for axis in self.axes_names}
        
        self.domain_lengths = domain_lengths
        
        # 計算每個軸的歸一化因子（L → 2π）
        self.normalization_factors = torch.tensor(
            [2.0 * math.pi / domain_lengths.get(axis, 2.0 * math.pi) 
             for axis in self.axes_names],
            dtype=torch.float32
        )
        
        # 生成 Fourier 係數矩陣 B
        # 🔧 TASK-007: 矩陣基於 _full_axes_config 構建（固定維度）
        # 維度：[in_dim, total_frequencies_in_full_config]
        B_matrix, has_features = self._build_fourier_matrix(self._full_axes_config)
        self._has_features = has_features
        
        # 計算輸出維度
        self.out_dim = 2 * B_matrix.shape[1]  # cos + sin
        
        # 🔧 TASK-007: 初始化頻率掩碼（退火支援）
        self._update_frequency_mask()
        
        # 註冊為 buffer 或 parameter
        if trainable:
            self.B = nn.Parameter(B_matrix)
            self.register_buffer('_normalization_factors', self.normalization_factors)
        else:
            self.register_buffer("B", B_matrix)
            self.register_buffer('_normalization_factors', self.normalization_factors)
    
    def _build_fourier_matrix(self, config: Optional[Dict[str, List[int]]] = None) -> tuple[torch.Tensor, bool]:
        """
        構建 Fourier 係數矩陣 B
        
        Args:
            config: 頻率配置（若未提供則使用 self._full_axes_config）
        
        策略：
        1. 對於每個軸 i 和每個循環數 k，設定 B[i, j] = k
        2. 其他軸設為 0（正交性）
        3. 跳過空列表軸（該維度不產生頻率列）
        
        例：config = {'x': [1, 2], 'y': [], 'z': [1]}
            B = [[1, 2, 0],     # x 軸貢獻 2 個頻率
                 [0, 0, 0],     # y 軸不貢獻（但此行會被移除）
                 [0, 0, 1]]     # z 軸貢獻 1 個頻率
            實際 B = [[1, 2, 0],
                      [0, 0, 0],
                      [0, 0, 1]]
        
        Returns:
            (B_matrix, has_features): 
                - B_matrix: [in_dim, total_frequencies]
                - has_features: 是否有任何非零頻率
        """
        if config is None:
            config = self._full_axes_config
        
        # 收集所有頻率
        frequency_columns = []
        
        for axis_idx, axis_name in enumerate(self.axes_names):
            freqs = config[axis_name]
            
            if len(freqs) == 0:
                # 該軸不使用 Fourier，跳過
                continue
            
            for k in freqs:
                # 創建一個頻率列：只有當前軸非零
                column = torch.zeros(self.in_dim, dtype=torch.float32)
                column[axis_idx] = float(k)
                frequency_columns.append(column)
        
        if len(frequency_columns) == 0:
            # 所有軸都是空列表，無任何 Fourier 特徵
            # 創建一個 dummy 矩陣（將在 forward 中返回零維輸出）
            B_matrix = torch.zeros(self.in_dim, 1, dtype=torch.float32)
            has_features = False
        else:
            # 拼接成矩陣 [in_dim, total_frequencies]
            B_matrix = torch.stack(frequency_columns, dim=1)
            has_features = True
        
        return B_matrix, has_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            x: [batch_size, in_dim] 輸入座標
               - 順序必須與 axes_config 的鍵順序一致
               - 例：axes_config.keys() = ['x', 'y', 'z']
                    則 x[:, 0] 為 x, x[:, 1] 為 y, x[:, 2] 為 z
        
        Returns:
            [batch_size, 2*m] Fourier 特徵 [cos(2π*k*x_normalized), sin(...)]
            若無任何頻率（所有軸空列表），返回 [batch_size, 0]
            
        Notes:
            🔧 TASK-007: 支援 Fourier 退火
            - 矩陣 B 基於 full_axes_config 構建（固定維度）
            - 根據當前 axes_config 應用掩碼（置零未啟用頻率）
        """
        if not self._has_features:
            # 無任何 Fourier 特徵，返回空張量
            return torch.empty(x.shape[0], 0, dtype=x.dtype, device=x.device)
        
        # Step 1: 歸一化輸入到 [0, 2π] 範圍
        # x_normalized = x * (2π / L)
        # 🚀 效能優化: _normalization_factors 是 buffer，自動在正確的 device 上
        norm_factors = cast(torch.Tensor, self._normalization_factors)
        x_norm = x * norm_factors

        # Step 2: 計算相位 z = x_normalized @ B
        # 維度：[batch, in_dim] @ [in_dim, total_freqs] = [batch, total_freqs]
        # 🚀 效能優化: self.B 是 buffer，自動在正確的 device 上
        z = x_norm @ self.B

        # Step 3: 🔧 TASK-007 頻率掩碼（退火支援）
        # 若當前 axes_config 是 full_axes_config 的子集，需要掩碼未啟用頻率
        if hasattr(self, '_frequency_mask'):
            # 🚀 效能優化: _frequency_mask 是 buffer，自動在正確的 device 上
            mask = cast(torch.Tensor, self._frequency_mask)
            z = z * mask
        
        # Step 4: 計算 cos/sin 特徵
        # 注意：已包含 2π 在歸一化中，故 z 已是弧度
        cos_features = torch.cos(z)
        sin_features = torch.sin(z)
        
        # Step 5: 拼接輸出
        return torch.cat([cos_features, sin_features], dim=-1)
    
    def get_active_frequencies(self) -> Dict[str, List[int]]:
        """
        返回當前啟用的頻率配置（用於頻率退火監控）
        
        Returns:
            字典 {軸名: [循環數列表]}（深拷貝，修改不影響原配置）
        """
        import copy
        return copy.deepcopy(self.axes_config)
    
    def _update_frequency_mask(self):
        """
        🔧 TASK-007: 根據當前 axes_config 更新頻率掩碼
        
        策略：
        - B 矩陣基於 full_axes_config 構建（固定維度）
        - 掩碼標記哪些頻率列當前啟用（1）或禁用（0）
        - forward() 時將未啟用頻率的相位置零
        
        範例：
        full_axes_config = {'x': [1,2,4], 'y': [], 'z': [1,2]}
        axes_config = {'x': [1,2], 'y': [], 'z': [1,2]}  # x 軸頻率 4 未啟用
        → B 矩陣列: [x:1, x:2, x:4, z:1, z:2] (5 列)
        → 掩碼: [1, 1, 0, 1, 1] (x:4 被禁用)
        """
        if not self._has_features:
            return
        
        # 構建掩碼：遍歷 B 矩陣的每一列，檢查對應頻率是否在 axes_config 中
        mask = []
        for axis_idx, axis_name in enumerate(self.axes_names):
            full_freqs = self._full_axes_config[axis_name]
            active_freqs = self.axes_config.get(axis_name, [])
            
            for k in full_freqs:
                # 若頻率 k 在當前啟用列表中，掩碼為 1，否則為 0
                mask.append(1.0 if k in active_freqs else 0.0)
        
        # 註冊為 buffer（不需要梯度）
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        self.register_buffer('_frequency_mask', mask_tensor, persistent=False)
    
    def set_active_frequencies(self, new_config: Dict[str, List[int]]):
        """
        動態更新頻率配置（用於頻率退火）
        
        Args:
            new_config: 新的頻率配置（必須是 full_axes_config 的子集）
        
        注意：
            🔧 TASK-007: 不再重建 B 矩陣，僅更新掩碼（輸出維度保持不變）
        """
        # 驗證：新配置的頻率必須是 **full_axes_config** 的子集
        for axis, new_freqs in new_config.items():
            if axis not in self._full_axes_config:
                raise ValueError(f"軸 '{axis}' 不在完整配置中")
            
            full_freqs = self._full_axes_config[axis]
            for k in new_freqs:
                if k not in full_freqs:
                    raise ValueError(
                        f"頻率 {k} 不在軸 '{axis}' 的完整配置 {full_freqs} 中"
                    )
        
        # 🔧 TASK-007 Phase 2: 僅更新配置與掩碼，不重建矩陣
        self.axes_config = {k: list(v) for k, v in new_config.items()}
        self._update_frequency_mask()
        
        # 輸出維度保持不變（基於 full_axes_config）
        # 註：out_dim 在初始化時已基於完整配置設定
    
    def extra_repr(self) -> str:
        """模組資訊字串"""
        config_str = ", ".join(
            f"{axis}:{freqs}" for axis, freqs in self.axes_config.items()
        )
        return (
            f"axes_config={{{config_str}}}, "
            f"out_dim={self.out_dim}, "
            f"trainable={self.trainable}"
        )


class FourierFeatureFactory:
    """
    Fourier Features 工廠類
    
    提供統一接口創建不同類型的 Fourier 特徵層。
    """
    
    @staticmethod
    def create(
        config: Optional[Dict] = None,
        in_dim: Optional[int] = None,
        **kwargs
    ) -> nn.Module:
        """
        創建 Fourier Features 層
        
        Args:
            config: 配置（支持兩種格式）
                - Dict with 'type': 'axis_selective' → AxisSelectiveFourierFeatures
                - Dict with 'type': 'standard' → FourierFeatures（標準版）
                - None → 返回零維輸出模組
            in_dim: 輸入維度（當 config 為 None 時必須提供）
            **kwargs: 傳遞給具體類的額外參數
        
        Returns:
            FourierFeatures 或 AxisSelectiveFourierFeatures 實例
        """
        # 處理 None 配置（返回零維輸出）
        if config is None:
            if in_dim is None:
                raise ValueError("當 config=None 時必須提供 in_dim 參數")
            # 創建所有軸為空列表的配置
            axes_names = kwargs.get('axes_names', [f'dim{i}' for i in range(in_dim)])
            empty_config = {name: [] for name in axes_names[:in_dim]}
            return AxisSelectiveFourierFeatures(
                axes_config=empty_config,
                full_axes_config=empty_config,
                domain_lengths=kwargs.get('domain_lengths'),
                trainable=kwargs.get('trainable', False)
            )
        
        # 字典配置
        if not isinstance(config, dict):
            raise TypeError(f"config 必須是 dict, int 或 None，收到 {type(config)}")
        
        feature_type = config.get('type', 'standard')
        
        if feature_type == 'axis_selective':
            # 軸向選擇性版本
            axes_config = config.get('axes_config')
            domain_lengths = config.get('domain_lengths', kwargs.get('domain_lengths'))
            trainable = config.get('trainable', kwargs.get('trainable', False))
            full_axes_config = config.get('full_axes_config')
            
            if axes_config is None:
                raise ValueError("axis_selective 類型需要 'axes_config' 參數")
            if full_axes_config is None:
                raise ValueError("axis_selective 類型需要 'full_axes_config' 參數")
            
            return AxisSelectiveFourierFeatures(
                axes_config=axes_config,
                full_axes_config=full_axes_config,
                domain_lengths=domain_lengths,
                trainable=trainable
            )
        
        elif feature_type == 'standard':
            # 標準版本
            if in_dim is None:
                raise ValueError("standard 類型需要 in_dim 參數")
            from .fourier_mlp import FourierFeatures
            return FourierFeatures(
                in_dim=in_dim,
                m=config.get('m', 32),
                sigma=config.get('sigma', 5.0),
                multiscale=config.get('multiscale', False),
                trainable=config.get('trainable', False)
            )
        
        else:
            raise ValueError(f"未知的 Fourier 特徵類型: {feature_type}")


if __name__ == "__main__":
    # 快速測試
    print("=== 測試 AxisSelectiveFourierFeatures ===\n")
    
    # 測試 1: 通道流配置
    print("【測試 1】通道流配置（x/z 週期，y 不用 Fourier）")
    config = {
        'x': [1, 2, 4, 8],
        'y': [],
        'z': [1, 2, 4]
    }
    domain_lengths = {'x': 25.13274, 'y': 2.0, 'z': 9.42478}
    
    fourier = AxisSelectiveFourierFeatures(
        axes_config=config,
        domain_lengths=domain_lengths,
        full_axes_config=config
    )
    print(f"  輸入維度: {fourier.in_dim}")
    print(f"  輸出維度: {fourier.out_dim}")
    print(f"  Fourier 矩陣形狀: {fourier.B.shape}")
    print(f"  {fourier}\n")
    
    x = torch.randn(10, 3)
    features = fourier(x)
    print(f"  輸入形狀: {x.shape}")
    print(f"  輸出形狀: {features.shape}")
    print(f"  ✅ 預期輸出維度: {2*(4+0+3)} = 14\n")
    
    # 測試 2: 所有軸使用 Fourier
    print("【測試 2】所有軸使用 Fourier")
    config2 = {
        'x': [1, 2],
        'y': [1],
        'z': [1, 2]
    }
    fourier2 = AxisSelectiveFourierFeatures(
        axes_config=config2,
        full_axes_config=config2
    )
    print(f"  輸出維度: {fourier2.out_dim}")
    features2 = fourier2(x)
    print(f"  輸出形狀: {features2.shape}")
    print(f"  ✅ 預期輸出維度: {2*(2+1+2)} = 10\n")
    
    # 測試 3: 所有軸空列表（無 Fourier）
    print("【測試 3】所有軸空列表（無 Fourier 特徵）")
    config3 = {
        'x': [],
        'y': [],
        'z': []
    }
    fourier3 = AxisSelectiveFourierFeatures(
        axes_config=config3,
        full_axes_config=config3
    )
    print(f"  輸出維度: {fourier3.out_dim}")
    features3 = fourier3(x)
    print(f"  輸出形狀: {features3.shape}")
    print(f"  ✅ 預期輸出維度: 0\n")
    
    # 測試 4: 工廠模式
    print("【測試 4】使用工廠創建")
    factory_config = {
        'type': 'axis_selective',
        'axes_config': {'x': [1, 2], 'y': [1]},
        'full_axes_config': {'x': [1, 2], 'y': [1]},
        'domain_lengths': {'x': 6.28, 'y': 2.0}
    }
    fourier4 = FourierFeatureFactory.create(in_dim=2, config=factory_config)
    print(f"  {fourier4}")
    features4 = fourier4(torch.randn(5, 2))
    print(f"  輸出形狀: {features4.shape}")
    print(f"  ✅ 預期輸出維度: {2*(2+1)} = 6\n")
    
    print("=== 所有測試完成 ===")
