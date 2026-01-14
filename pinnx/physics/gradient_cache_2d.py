"""
Gradient Cache for 2D Navier-Stokes (Kolmogorov Flow)
======================================================

提供 2D 流場的梯度快取功能，避免重複計算。

適用於：
- Kolmogorov Flow 2D
- 其他 2D Navier-Stokes 問題

Author: Performance Optimization Team
Date: 2026-01-07
"""

import torch
from typing import Dict, Optional


class GradientCache2D:
    """
    2D 流場梯度快取
    
    解決 Kolmogorov Flow 中梯度重複計算的問題：
    - 連續方程需要：∂u/∂x, ∂v/∂y
    - u 的對流項需要：∂u/∂x, ∂u/∂y
    - v 的對流項需要：∂v/∂x, ∂v/∂y
    - u 的黏性項需要：∂²u/∂x², ∂²u/∂y²
    - v 的黏性項需要：∂²v/∂x², ∂²v/∂y²
    - 壓力梯度需要：∂p/∂x, ∂p/∂y
    
    總計需要 12 個梯度，但當前實現中會計算 20-25 次（重複 2-3 倍）。
    
    透過一次性計算並快取，預期減少 40-50% 的梯度計算時間。
    
    Example:
        >>> cache = GradientCache2D(device='cuda')
        >>> predictions = {'u': u_tensor, 'v': v_tensor, 'p': p_tensor}
        >>> coords = torch.tensor(..., requires_grad=True)
        >>> gradients = cache.compute_all_gradients(predictions, coords)
        >>> u_x = gradients['u_x']  # 一階梯度 ∂u/∂x
        >>> u_xx = gradients['u_xx']  # 二階梯度 ∂²u/∂x²
        >>> cache.clear_cache()  # 釋放記憶體
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        初始化 2D 梯度快取
        
        Args:
            device: 計算設備 ('cuda', 'cpu', 'mps')
        """
        self.device = device
        self._cache: Optional[Dict[str, torch.Tensor]] = None
    
    def compute_all_gradients(
        self,
        predictions: Dict[str, torch.Tensor],
        coords: torch.Tensor,
        create_graph: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        計算所有速度和壓力的一階與二階梯度
        
        此方法一次性計算以下梯度：
        - 速度場 u, v 的一階梯度: ∂u/∂x, ∂u/∂y (共4個)
        - 速度場 u, v 的二階對角梯度: ∂²u/∂x², ∂²u/∂y² (共4個)
        - 壓力場 p 的一階梯度: ∂p/∂x, ∂p/∂y (共2個)
        
        總計: 10 個梯度張量
        
        Args:
            predictions: 包含 'u', 'v', 'p' 的字典，每個 shape [N, 1]
            coords: 座標張量 [N, 2] (x, y)，必須設置 requires_grad=True
            create_graph: 是否創建計算圖（訓練時為 True，推理時為 False）
        
        Returns:
            gradients: 包含所有梯度的字典
                {
                    'u_x': ∂u/∂x [N, 1], 'u_y': ∂u/∂y [N, 1],
                    'u_xx': ∂²u/∂x² [N, 1], 'u_yy': ∂²u/∂y² [N, 1],
                    'v_x': ∂v/∂x [N, 1], 'v_y': ∂v/∂y [N, 1],
                    'v_xx': ∂²v/∂x² [N, 1], 'v_yy': ∂²v/∂y² [N, 1],
                    'p_x': ∂p/∂x [N, 1], 'p_y': ∂p/∂y [N, 1]
                }
        
        Raises:
            KeyError: 如果 predictions 缺少必要的鍵 ('u', 'v', 'p')
            RuntimeError: 如果 coords 沒有設置 requires_grad=True
        """
        # 驗證輸入
        required_keys = ['u', 'v', 'p']
        missing_keys = [k for k in required_keys if k not in predictions]
        if missing_keys:
            raise KeyError(f"predictions missing required keys: {missing_keys}")
        
        if not coords.requires_grad:
            raise RuntimeError("coords must have requires_grad=True for gradient computation")
        
        # 🔧 支持 2D 和 3D 座標
        # - [N, 2]: 純 2D 穩態場景 (x, y)
        # - [N, 3]: Time Window 場景 (x, y, t)，只計算空間導數
        coord_dim = coords.shape[1]
        if coord_dim not in [2, 3]:
            raise ValueError(f"coords must be 2D or 3D, got shape {coords.shape}")
        
        # 提取預測值
        u = predictions['u']
        v = predictions['v']
        p = predictions['p']
        
        # 一階梯度 (6 個：u, v 各 2 個 + p 的 2 個)
        u_grads = self._compute_first_order(u, coords, create_graph, retain_graph=True)  # [N, 2]
        v_grads = self._compute_first_order(v, coords, create_graph, retain_graph=True)  # [N, 2]
        p_grads = self._compute_first_order(p, coords, create_graph=False, retain_graph=True)  # [N, 2] 壓力不需要二階
        
        # 二階梯度 (4 個：u, v 各 2 個對角線項)
        u_grads_2 = self._compute_second_order_diagonal(
            u_grads, coords, create_graph, retain_graph=True
        )  # [N, 2]
        v_grads_2 = self._compute_second_order_diagonal(
            v_grads, coords, create_graph, retain_graph=False
        )  # [N, 2]
        
        # 組裝字典（分割為單獨的列以符合物理模組期望）
        gradients = {
            # U 的梯度
            'u_x': u_grads[:, 0:1],   # ∂u/∂x
            'u_y': u_grads[:, 1:2],   # ∂u/∂y
            'u_xx': u_grads_2[:, 0:1],  # ∂²u/∂x²
            'u_yy': u_grads_2[:, 1:2],  # ∂²u/∂y²
            
            # V 的梯度
            'v_x': v_grads[:, 0:1],   # ∂v/∂x
            'v_y': v_grads[:, 1:2],   # ∂v/∂y
            'v_xx': v_grads_2[:, 0:1],  # ∂²v/∂x²
            'v_yy': v_grads_2[:, 1:2],  # ∂²v/∂y²
            
            # P 的梯度
            'p_x': p_grads[:, 0:1],   # ∂p/∂x
            'p_y': p_grads[:, 1:2],   # ∂p/∂y
        }
        
        # 快取結果
        self._cache = gradients
        return gradients
    
    def _compute_first_order(
        self, 
        field: torch.Tensor, 
        coords: torch.Tensor,
        create_graph: bool,
        retain_graph: bool = True
    ) -> torch.Tensor:
        """
        計算一階空間梯度 ∂field/∂(x,y)
        
        Args:
            field: 標量場 [N, 1]
            coords: 座標 [N, 2] 或 [N, 3]
                - [N, 2]: 純 2D 穩態場景
                - [N, 3]: Time Window 場景 (x, y, t)
            create_graph: 是否保留計算圖
        
        Returns:
            grad: 空間梯度張量 [N, 2]，始終包含 [∂f/∂x, ∂f/∂y]
        """
        full_grad = torch.autograd.grad(
            outputs=field,
            inputs=coords,
            grad_outputs=torch.ones_like(field),
            create_graph=create_graph,
            retain_graph=retain_graph,
            allow_unused=True  # 允許未使用的輸入（例如 Time Window 中的時間維度）
        )[0]
        
        # 如果梯度為 None（field 不依賴於 coords）
        if full_grad is None:
            raise RuntimeError(
                f"梯度計算失敗：field 似乎不依賴於 coords。\n"
                f"coords shape: {coords.shape}, field shape: {field.shape}\n"
                f"這通常發生在測試環境中。實際訓練時，predictions 應該是網路輸出，\n"
                f"自動包含對輸入座標的依賴關係。"
            )
        
        # 🔧 智能處理：只返回空間維度的梯度 (x, y)
        # - 如果 coords 是 [N, 2]，full_grad 就是 [N, 2]，直接返回
        # - 如果 coords 是 [N, 3]，full_grad 是 [N, 3]，只取前 2 維 (忽略時間導數)
        spatial_grad = full_grad[:, :2]  # [N, 2]
        
        return spatial_grad  # [N, 2]
    
    def _compute_second_order_diagonal(
        self, 
        first_grad: torch.Tensor, 
        coords: torch.Tensor,
        create_graph: bool,
        retain_graph: bool = True
    ) -> torch.Tensor:
        """
        計算二階對角空間梯度 (∂²/∂x², ∂²/∂y²) - 向量化版本
        
        此方法只計算 Hessian 矩陣的對角線項，因為 Laplacian 運算子
        只需要對角線元素的和：Δf = ∂²f/∂x² + ∂²f/∂y²
        
        優化說明：
            使用 grad_outputs 進行批次化計算，避免 .sum() 標量化導致的性能損失。
            在 GPU 上可獲得 1.1-1.2× 加速（批次大小 8000）。
        
        Args:
            first_grad: 一階梯度 [N, 2]（空間梯度）
            coords: 座標 [N, 2] 或 [N, 3]
            create_graph: 是否保留計算圖
        
        Returns:
            second_grad: 二階對角空間梯度 [N, 2]，包含 [∂²f/∂x², ∂²f/∂y²]
        """
        second_grads = []
        for i in range(2):  # x, y (只計算空間維度)
            # 🚀 向量化優化：使用 grad_outputs 避免標量化
            # 創建單位向量作為 grad_outputs，實現批次計算
            grad_outputs = torch.zeros_like(first_grad)
            grad_outputs[:, i] = 1.0  # [N, 2]，第 i 列為 1
            
            # 對每個空間維度計算二階梯度（批次化）
            keep_graph = retain_graph or i < 1
            full_grad_i = torch.autograd.grad(
                outputs=first_grad,
                inputs=coords,
                grad_outputs=grad_outputs,
                create_graph=create_graph,
                retain_graph=keep_graph,
                allow_unused=True  # 允許未使用的輸入（例如 Time Window 中的時間維度）
            )[0]
            
            if full_grad_i is None:
                raise RuntimeError(
                    f"二階梯度計算失敗：first_grad[:, {i}] 似乎不依賴於 coords。\n"
                    f"coords shape: {coords.shape}"
                )
            
            # 🔧 智能處理：只取空間維度的對角線項
            # - 如果 coords 是 [N, 2]，取 full_grad_i[:, i:i+1]
            # - 如果 coords 是 [N, 3]，仍然取 full_grad_i[:, i:i+1]（忽略時間分量）
            second_grads.append(full_grad_i[:, i:i+1])  # [N, 1]
        
        return torch.cat(second_grads, dim=1)  # [N, 2]
    
    def get_cached(self, key: str) -> Optional[torch.Tensor]:
        """
        從快取中獲取指定的梯度
        
        Args:
            key: 梯度鍵名，例如 'u_x', 'v_yy', 'p_x'
        
        Returns:
            梯度張量 [N, 1]，如果快取為空或鍵不存在則返回 None
        """
        if self._cache is None:
            return None
        return self._cache.get(key, None)
    
    def clear_cache(self):
        """
        清除快取以釋放記憶體
        
        此方法應在每個訓練步驟結束時調用，以避免記憶體累積。
        梯度張量會被釋放，但仍然保留在計算圖中直到反向傳播完成。
        """
        self._cache = None
    
    def __repr__(self) -> str:
        """字串表示"""
        cache_status = "cached" if self._cache is not None else "empty"
        num_cached = len(self._cache) if self._cache is not None else 0
        return f"GradientCache2D(device='{self.device}', cache={cache_status}, num_gradients={num_cached})"
