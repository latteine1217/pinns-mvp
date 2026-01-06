"""
課程訓練調度器

逐步提升物理參數難度（如雷諾數），從簡單到複雜進行訓練
支援動態調整 Re、nu、壓力梯度、損失權重和採樣策略
"""

import logging
from typing import Dict, Any, Optional
import torch

from .base import WeightScheduler


class CurriculumScheduler(WeightScheduler):
    """
    課程訓練調度器 - 逐步提升雷諾數，從層流到湍流
    
    特性：
    - 動態調整 Re_tau, nu, pressure_gradient
    - 階段式權重切換
    - 階段式學習率調整
    - 階段式採樣策略調整
    """
    
    def __init__(self, stages: list, physics_module):
        """
        Args:
            stages: 課程階段列表，每個元素包含:
                - name: 階段名稱
                - epoch_range: [start, end]
                - Re_tau: 雷諾數
                - nu: 黏度
                - pressure_gradient: 壓力梯度
                - weights: 損失權重字典
                - sampling: 採樣配置
                - lr: 學習率
            physics_module: 物理方程模組（用於更新參數）
        """
        self.stages = stages
        self.physics = physics_module
        self.current_stage_idx = 0
        self.current_stage = stages[0] if stages else None
        
        # 按 epoch_range 排序
        self.stages.sort(key=lambda s: s['epoch_range'][0])
        
        logging.info("="*80)
        logging.info("🚀 CurriculumScheduler initialized - Progressive Weight & Sampling Training")
        logging.info("="*80)
        for i, s in enumerate(self.stages, 1):
            logging.info(f"Stage {i}: {s['name']}")
            logging.info(f"  Epochs: {s['epoch_range'][0]}-{s['epoch_range'][1]}")

            # 根據場景類型顯示不同的物理參數（如果存在）
            if 'Re_tau' in s and s['Re_tau'] is not None:  # Channel Flow
                if s.get('pressure_gradient') is not None:
                    logging.info(f"  Re_tau: {s['Re_tau']:.1f}, nu: {s['nu']:.6f}, dP/dx: {s['pressure_gradient']:.3f}")
                else:
                    logging.info(f"  Re_tau: {s['Re_tau']:.1f}, nu: {s['nu']:.6f}")
            elif 'Re' in s and s['Re'] is not None:  # Kolmogorov Flow
                logging.info(f"  Re: {s['Re']:.1f}, nu: {s['nu']:.6f}")
            elif 'nu' in s and s['nu'] is not None:
                logging.info(f"  nu: {s['nu']:.6f}")
            else:
                # 沒有物理參數 - curriculum 只控制權重和採樣
                logging.info(f"  Physics params: inherited from main config")

            # 顯示採樣配置
            if 'sampling' in s:
                pde_pts = s['sampling'].get('N_pde', 'N/A')
                bc_pts = s['sampling'].get('boundary_points', 0)
                logging.info(f"  PDE points: {pde_pts}, BC points: {bc_pts}")

            # 處理 lr（可選參數）
            if 'lr' in s:
                lr_value = float(s['lr']) if isinstance(s['lr'], (str, int)) else s['lr']
                logging.info(f"  Learning rate: {lr_value:.6f} (explicit)")
            else:
                logging.info(f"  Learning rate: controlled by global scheduler")

            # 顯示主要權重變化（如果存在）
            if 'weights' in s:
                key_weights = {k: v for k, v in s['weights'].items() if k in ['data', 'continuity', 'prior']}
                if key_weights:
                    weights_str = ', '.join([f"{k}={v}" for k, v in key_weights.items()])
                    logging.info(f"  Key weights: {weights_str}")
        logging.info("="*80)
    
    def _find_stage_and_check_transition(self, epoch: int) -> tuple[Dict[str, Any], bool]:
        """
        內部方法：找到當前階段並檢測是否為切換點
        
        Returns:
            (stage_config, is_transition)
        """
        # 找到當前階段
        for idx, stage in enumerate(self.stages):
            start, end = stage['epoch_range']
            if start <= epoch < end:
                # 檢測階段切換
                is_transition = (idx != self.current_stage_idx)
                
                if is_transition:
                    self.current_stage_idx = idx
                    self.current_stage = stage

                    # 更新物理參數（如果有）
                    if any(key in stage for key in ['Re_tau', 'Re', 'nu', 'pressure_gradient']):
                        self._update_physics_parameters(stage)

                    logging.info("="*80)
                    logging.info(f"🎯 CURRICULUM STAGE TRANSITION at Epoch {epoch}")
                    logging.info(f"📚 New Stage: {stage['name']}")

                    # 根據場景類型顯示不同的物理參數（如果存在）
                    if 'Re_tau' in stage and stage['Re_tau'] is not None:  # Channel Flow
                        logging.info(f"🔬 Re_tau: {stage['Re_tau']:.1f}, nu: {stage['nu']:.6f}")
                    elif 'Re' in stage and stage['Re'] is not None:  # Kolmogorov Flow
                        logging.info(f"🔬 Re: {stage['Re']:.1f}, nu: {stage['nu']:.6f}")
                    elif 'nu' in stage and stage['nu'] is not None:
                        logging.info(f"🔬 nu: {stage['nu']:.6f}")
                    else:
                        logging.info(f"🔬 Physics params: inherited from main config")

                    # 顯示採樣配置
                    if 'sampling' in stage:
                        pde_pts = stage['sampling'].get('N_pde', 'N/A')
                        bc_pts = stage['sampling'].get('boundary_points', 0)
                        logging.info(f"⚙️  PDE/BC points: {pde_pts}/{bc_pts}")

                    # 顯示權重
                    if 'weights' in stage:
                        logging.info(f"📊 Weights: {stage['weights']}")

                    # 顯示學習率（如果指定）
                    if 'lr' in stage:
                        lr_value = float(stage['lr']) if isinstance(stage['lr'], (str, int)) else stage['lr']
                        logging.info(f"📉 Learning rate: {lr_value:.6f} (explicit reset)")
                    else:
                        logging.info(f"📉 Learning rate: controlled by global scheduler")
                    logging.info("="*80)
                
                return stage, is_transition
        
        # 超出範圍，返回最後階段
        return self.stages[-1], False
    
    def _update_physics_parameters(self, stage: Dict[str, Any]):
        """更新物理方程模組的參數（避免梯度計算問題）"""
        updated_params = []

        # nu 是 torch.nn.Buffer，使用 .data 來避免梯度追蹤
        if 'nu' in stage and hasattr(self.physics, 'nu'):
            if isinstance(self.physics.nu, torch.Tensor):
                # 使用 .data.fill_() 來避免影響計算圖
                self.physics.nu.data = torch.tensor(stage['nu'], dtype=self.physics.nu.dtype, device=self.physics.nu.device)
            else:
                self.physics.nu = stage['nu']
            updated_params.append(f"nu={stage['nu']:.6f}")

        # Channel Flow 參數
        if hasattr(self.physics, 'Re_tau') and 'Re_tau' in stage:
            if isinstance(self.physics.Re_tau, torch.Tensor):
                self.physics.Re_tau.data = torch.tensor(stage['Re_tau'], dtype=self.physics.Re_tau.dtype, device=self.physics.Re_tau.device)
            else:
                self.physics.Re_tau = stage['Re_tau']
            updated_params.append(f"Re_tau={stage['Re_tau']:.1f}")

        if hasattr(self.physics, 'pressure_gradient') and 'pressure_gradient' in stage:
            if isinstance(self.physics.pressure_gradient, torch.Tensor):
                self.physics.pressure_gradient.data = torch.tensor(stage['pressure_gradient'], dtype=self.physics.pressure_gradient.dtype, device=self.physics.pressure_gradient.device)
            else:
                self.physics.pressure_gradient = stage['pressure_gradient']
            updated_params.append(f"dP/dx={stage['pressure_gradient']:.3f}")

        # Kolmogorov Flow 參數
        if hasattr(self.physics, 'Re') and 'Re' in stage:
            if isinstance(self.physics.Re, torch.Tensor):
                self.physics.Re.data = torch.tensor(stage['Re'], dtype=self.physics.Re.dtype, device=self.physics.Re.device)
            else:
                self.physics.Re = stage['Re']
            updated_params.append(f"Re={stage['Re']:.1f}")

        # 日誌輸出
        if updated_params:
            logging.debug(f"✅ Physics parameters updated: {', '.join(updated_params)}")
        else:
            logging.debug("ℹ️  No physics parameters to update (controlled by main config)")
    
    # ========================================================================
    # 統一接口實作（WeightScheduler ABC）
    # ========================================================================
    
    def get_weights(self, epoch: int) -> Dict[str, float]:
        """
        獲取當前 epoch 的損失權重（統一接口）
        
        Args:
            epoch: 當前訓練輪次
            
        Returns:
            權重字典，例如：
            {
                'data': 100.0,
                'momentum_x': 1.0,
                'continuity': 1.0,
                'boundary': 10.0,
                'prior': 1.0
            }
        """
        stage, _ = self._find_stage_and_check_transition(epoch)
        return stage.get('weights', {})
    
    def get_metadata(self, epoch: int) -> Optional[Dict[str, Any]]:
        """
        獲取當前 epoch 的課程元數據（統一接口）
        
        Args:
            epoch: 當前訓練輪次
            
        Returns:
            元數據字典，包含：
            - stage_name: 階段名稱
            - is_transition: 是否為階段切換點
            - lr: 學習率（如果課程控制 LR）
            - sampling: 採樣配置（如果課程控制採樣）
            - Re_tau, nu, pressure_gradient: 物理參數（如果課程控制）
        """
        stage, is_transition = self._find_stage_and_check_transition(epoch)
        
        # 構建元數據（排除 weights，因為已在 get_weights() 中返回）
        metadata = {
            'stage_name': stage['name'],
            'is_transition': is_transition,
            'sampling': stage.get('sampling', {})
        }
        
        # 可選物理參數
        optional_keys = ['lr', 'Re_tau', 'Re', 'nu', 'pressure_gradient']
        for key in optional_keys:
            if key in stage:
                metadata[key] = stage[key]
        
        return metadata
