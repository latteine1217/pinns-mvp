"""
Gradient computation with caching for VS-PINN physics.

This module provides efficient gradient computation by calculating all required
gradients once and caching them for reuse across multiple physics residual calculations.

Author: Performance Optimization Team
Date: 2025-12-15
"""

import torch
from typing import Dict, Optional


class GradientCache:
    """
    一次性計算並快取所有物理方程需要的梯度
    
    此類解決 VS-PINN 中梯度重複計算的問題：
    - Momentum 方程需要速度的一階和二階梯度
    - Continuity 方程需要速度的一階梯度（與 Momentum 重複）
    - 壓力梯度在三個 Momentum 方程中都需要（重複計算）
    
    透過一次性計算所有梯度並快取，避免重複的 autograd 調用，
    預期可減少 15-25% 的梯度計算時間。
    
    Example:
        >>> cache = GradientCache(device='cuda')
        >>> predictions = {'u': u_tensor, 'v': v_tensor, 'w': w_tensor, 'p': p_tensor}
        >>> coords = torch.tensor(..., requires_grad=True)
        >>> gradients = cache.compute_all_gradients(predictions, coords)
        >>> u_x = gradients['u_x']  # 一階梯度 ∂u/∂x
        >>> u_xx = gradients['u_xx']  # 二階梯度 ∂²u/∂x²
        >>> cache.clear_cache()  # 釋放記憶體
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        初始化梯度快取
        
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
        - 速度場 u, v, w 的一階梯度: ∂u/∂x, ∂u/∂y, ∂u/∂z (共9個)
        - 速度場 u, v, w 的二階對角梯度: ∂²u/∂x², ∂²u/∂y², ∂²u/∂z² (共9個)
        - 壓力場 p 的一階梯度: ∂p/∂x, ∂p/∂y, ∂p/∂z (共3個)
        
        總計: 21 個梯度張量
        
        Args:
            predictions: 包含 'u', 'v', 'w', 'p' 的字典，每個 shape [N, 1]
            coords: 座標張量 [N, 3] (x, y, z)，必須設置 requires_grad=True
            create_graph: 是否創建計算圖（訓練時為 True，推理時為 False）
        
        Returns:
            gradients: 包含所有梯度的字典
                {
                    'u_x': ∂u/∂x [N, 1], 'u_y': ∂u/∂y [N, 1], 'u_z': ∂u/∂z [N, 1],
                    'u_xx': ∂²u/∂x² [N, 1], 'u_yy': ∂²u/∂y² [N, 1], 'u_zz': ∂²u/∂z² [N, 1],
                    'v_x': ∂v/∂x [N, 1], 'v_y': ∂v/∂y [N, 1], 'v_z': ∂v/∂z [N, 1],
                    'v_xx': ∂²v/∂x² [N, 1], 'v_yy': ∂²v/∂y² [N, 1], 'v_zz': ∂²v/∂z² [N, 1],
                    'w_x': ∂w/∂x [N, 1], 'w_y': ∂w/∂y [N, 1], 'w_z': ∂w/∂z [N, 1],
                    'w_xx': ∂²w/∂x² [N, 1], 'w_yy': ∂²w/∂y² [N, 1], 'w_zz': ∂²w/∂z² [N, 1],
                    'p_x': ∂p/∂x [N, 1], 'p_y': ∂p/∂y [N, 1], 'p_z': ∂p/∂z [N, 1]
                }
        
        Raises:
            KeyError: 如果 predictions 缺少必要的鍵 ('u', 'v', 'w', 'p')
            RuntimeError: 如果 coords 沒有設置 requires_grad=True
        """
        # 驗證輸入
        required_keys = ['u', 'v', 'w', 'p']
        missing_keys = [k for k in required_keys if k not in predictions]
        if missing_keys:
            raise KeyError(f"predictions missing required keys: {missing_keys}")
        
        if not coords.requires_grad:
            raise RuntimeError("coords must have requires_grad=True for gradient computation")
        
        # 提取預測值
        u = predictions['u']
        v = predictions['v']
        w = predictions['w']
        p = predictions['p']
        
        # 一階梯度 (9 個)
        u_grads = self._compute_first_order(u, coords, create_graph, retain_graph=True)  # [N, 3]
        v_grads = self._compute_first_order(v, coords, create_graph, retain_graph=True)  # [N, 3]
        w_grads = self._compute_first_order(w, coords, create_graph, retain_graph=True)  # [N, 3]
        p_grads = self._compute_first_order(p, coords, create_graph=False, retain_graph=False)  # [N, 3] 壓力不需要二階
        
        # 二階梯度 (9 個 - 只需對角線項)
        u_grads_2 = self._compute_second_order_diagonal(u_grads, coords, create_graph)  # [N, 3]
        v_grads_2 = self._compute_second_order_diagonal(v_grads, coords, create_graph)  # [N, 3]
        w_grads_2 = self._compute_second_order_diagonal(w_grads, coords, create_graph)  # [N, 3]
        
        # 組裝字典（分割為單獨的列以符合物理模組期望）
        gradients = {
            # U 的梯度
            'u_x': u_grads[:, 0:1],   # ∂u/∂x
            'u_y': u_grads[:, 1:2],   # ∂u/∂y
            'u_z': u_grads[:, 2:3],   # ∂u/∂z
            'u_xx': u_grads_2[:, 0:1],  # ∂²u/∂x²
            'u_yy': u_grads_2[:, 1:2],  # ∂²u/∂y²
            'u_zz': u_grads_2[:, 2:3],  # ∂²u/∂z²
            
            # V 的梯度
            'v_x': v_grads[:, 0:1],   # ∂v/∂x
            'v_y': v_grads[:, 1:2],   # ∂v/∂y
            'v_z': v_grads[:, 2:3],   # ∂v/∂z
            'v_xx': v_grads_2[:, 0:1],  # ∂²v/∂x²
            'v_yy': v_grads_2[:, 1:2],  # ∂²v/∂y²
            'v_zz': v_grads_2[:, 2:3],  # ∂²v/∂z²
            
            # W 的梯度
            'w_x': w_grads[:, 0:1],   # ∂w/∂x
            'w_y': w_grads[:, 1:2],   # ∂w/∂y
            'w_z': w_grads[:, 2:3],   # ∂w/∂z
            'w_xx': w_grads_2[:, 0:1],  # ∂²w/∂x²
            'w_yy': w_grads_2[:, 1:2],  # ∂²w/∂y²
            'w_zz': w_grads_2[:, 2:3],  # ∂²w/∂z²
            
            # P 的梯度
            'p_x': p_grads[:, 0:1],   # ∂p/∂x
            'p_y': p_grads[:, 1:2],   # ∂p/∂y
            'p_z': p_grads[:, 2:3],   # ∂p/∂z
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
        計算一階梯度 ∂field/∂coords
        
        Args:
            field: 標量場 [N, 1]
            coords: 座標 [N, 3]
            create_graph: 是否保留計算圖
        
        Returns:
            grad: 梯度張量 [N, 3]，包含 [∂f/∂x, ∂f/∂y, ∂f/∂z]
        """
        grad = torch.autograd.grad(
            outputs=field,
            inputs=coords,
            grad_outputs=torch.ones_like(field),
            create_graph=create_graph,
            retain_graph=retain_graph,
            allow_unused=False
        )[0]
        return grad  # [N, 3]
    
    def _compute_second_order_diagonal(
        self,
        first_grad: torch.Tensor,
        coords: torch.Tensor,
        create_graph: bool
    ) -> torch.Tensor:
        """
        計算二階對角梯度 (∂²/∂x², ∂²/∂y², ∂²/∂z²)
        
        此方法只計算 Hessian 矩陣的對角線項，因為 Laplacian 運算子
        只需要對角線元素的和：Δf = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²
        
        Args:
            first_grad: 一階梯度 [N, 3]
            coords: 座標 [N, 3]
            create_graph: 是否保留計算圖
        
        Returns:
            second_grad: 二階對角梯度 [N, 3]，包含 [∂²f/∂x², ∂²f/∂y², ∂²f/∂z²]
        """
        second_grads = []
        for i in range(3):  # x, y, z
            # 對每個維度計算二階梯度
            retain_graph = i < 2
            grad_i = torch.autograd.grad(
                outputs=first_grad[:, i].sum(),  # 標量化以計算梯度
                inputs=coords,
                create_graph=create_graph,
                retain_graph=retain_graph,
                allow_unused=False
            )[0][:, i:i+1]  # 只取對角線項 (i, i)
            second_grads.append(grad_i)
        
        return torch.cat(second_grads, dim=1)  # [N, 3]
    
    def get_cached(self, key: str) -> Optional[torch.Tensor]:
        """
        從快取中獲取指定的梯度
        
        Args:
            key: 梯度鍵名，例如 'u_x', 'v_yy', 'p_z'
        
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
        return f"GradientCache(device='{self.device}', cache={cache_status}, num_gradients={num_cached})"
