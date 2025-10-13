"""
自適應殘差點採樣可視化工具

功能：
1. 繪製重採樣點的空間分佈（新點 vs 舊點）
2. 疊加殘差熱圖（顯示高殘差區域）
3. 對比新舊點的殘差統計
4. 時序追蹤：重採樣歷史與損失曲線
5. QR-pivot 選點質量評估

用法：
    python scripts/visualize_adaptive_sampling.py \
        --checkpoint checkpoints/model_epoch_2000.pth \
        --config configs/inverse_reconstruction_main.yml \
        --output results/adaptive_sampling_vis/
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure
from pathlib import Path
import yaml
import logging
from typing import Dict, List, Tuple, Optional, Any
import seaborn as sns

# 設置繪圖樣式
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

logger = logging.getLogger(__name__)


class AdaptiveSamplingVisualizer:
    """自適應採樣可視化器"""
    
    def __init__(self, 
                 model_path: str,
                 config_path: str,
                 output_dir: str,
                 device: str = 'cpu'):
        """
        Args:
            model_path: 模型檢查點路徑
            config_path: 配置文件路徑
            output_dir: 輸出目錄
            device: 計算設備
        """
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.device = device
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 載入配置
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 載入檢查點
        self.checkpoint = torch.load(self.model_path, map_location=device)
        
        # 提取自適應採樣統計（如果存在）
        self.sampling_stats = self.checkpoint.get('sampling_stats', {})
        self.training_history = self.checkpoint.get('training_history', {})
        
        logger.info(f"✅ 載入檢查點: {self.model_path}")
        logger.info(f"   訓練 Epoch: {self.checkpoint.get('epoch', 'N/A')}")
        logger.info(f"   重採樣次數: {self.sampling_stats.get('resample_count', 0)}")
    
    def plot_spatial_distribution(self,
                                 current_points: torch.Tensor,
                                 new_point_mask: torch.Tensor,
                                 residuals: Optional[torch.Tensor] = None,
                                 domain_bounds: Optional[Dict] = None,
                                 title: str = "Adaptive Collocation Point Distribution") -> Figure:
        """
        繪製殘差點空間分佈
        
        Args:
            current_points: 當前殘差點 [N, dim]
            new_point_mask: 新點掩碼 [N,] (bool)
            residuals: 殘差值 [N, n_eqs]（可選，用於顏色映射）
            domain_bounds: 域邊界（可選）
            title: 圖表標題
            
        Returns:
            figure: matplotlib 圖表
        """
        dim = current_points.shape[1]
        
        # 分離舊點和新點
        old_points = current_points[~new_point_mask].numpy()
        new_points = current_points[new_point_mask].numpy()
        
        # 計算殘差範數（如果提供）
        if residuals is not None:
            residual_norms = torch.norm(residuals, dim=-1).numpy()
            old_residuals = residual_norms[~new_point_mask]
            new_residuals = residual_norms[new_point_mask]
        else:
            old_residuals = None
            new_residuals = None
        
        # 創建圖表
        if dim == 2:
            fig = self._plot_2d_distribution(
                old_points, new_points, old_residuals, new_residuals,
                domain_bounds, title
            )
        elif dim == 3:
            fig = self._plot_3d_distribution(
                old_points, new_points, old_residuals, new_residuals,
                domain_bounds, title
            )
        else:
            raise ValueError(f"不支援的維度: {dim}")
        
        return fig
    
    def _plot_2d_distribution(self,
                             old_points: np.ndarray,
                             new_points: np.ndarray,
                             old_residuals: Optional[np.ndarray],
                             new_residuals: Optional[np.ndarray],
                             domain_bounds: Optional[Dict],
                             title: str) -> Figure:
        """繪製 2D 分佈"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左圖：點分佈（區分新舊）
        ax1 = axes[0]
        ax1.scatter(old_points[:, 0], old_points[:, 1], 
                   c='blue', alpha=0.3, s=20, label=f'Kept Points ({len(old_points)})')
        ax1.scatter(new_points[:, 0], new_points[:, 1],
                   c='red', alpha=0.8, s=50, marker='^', 
                   label=f'New Points ({len(new_points)})')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Point Distribution (Old vs New)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        if domain_bounds:
            ax1.set_xlim(domain_bounds.get('x', (None, None)))
            ax1.set_ylim(domain_bounds.get('y', (None, None)))
        
        # 右圖：殘差熱圖（如果提供）
        ax2 = axes[1]
        if old_residuals is not None and new_residuals is not None:
            # 合併所有點和殘差
            all_points = np.vstack([old_points, new_points])
            all_residuals = np.concatenate([old_residuals, new_residuals])
            
            # 散點圖 + 顏色映射
            scatter = ax2.scatter(all_points[:, 0], all_points[:, 1],
                                 c=all_residuals, cmap='YlOrRd', 
                                 s=30, alpha=0.7)
            
            # 標記新點邊界
            ax2.scatter(new_points[:, 0], new_points[:, 1],
                       facecolors='none', edgecolors='black', 
                       s=80, linewidths=1.5, marker='o',
                       label='New Points')
            
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Residual Norm', rotation=270, labelpad=20)
            
            ax2.set_title('Residual Heat Map')
        else:
            ax2.text(0.5, 0.5, 'No Residual Data', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Residual Heat Map (N/A)')
        
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        if domain_bounds:
            ax2.set_xlim(domain_bounds.get('x', (None, None)))
            ax2.set_ylim(domain_bounds.get('y', (None, None)))
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def _plot_3d_distribution(self,
                             old_points: np.ndarray,
                             new_points: np.ndarray,
                             old_residuals: Optional[np.ndarray],
                             new_residuals: Optional[np.ndarray],
                             domain_bounds: Optional[Dict],
                             title: str) -> Figure:
        """繪製 3D 分佈"""
        from mpl_toolkits.mplot3d import Axes3D  # type: ignore
        
        fig = plt.figure(figsize=(14, 6))
        
        # 左圖：點分佈（區分新舊）
        ax1 = fig.add_subplot(121, projection='3d')  # type: ignore
        ax1.scatter(old_points[:, 0], old_points[:, 1], old_points[:, 2],  # type: ignore[call-overload]
                   c='blue', alpha=0.3, s=10, label=f'Kept Points ({len(old_points)})')
        ax1.scatter(new_points[:, 0], new_points[:, 1], new_points[:, 2],  # type: ignore[call-overload]
                   c='red', alpha=0.8, s=30, marker='^',
                   label=f'New Points ({len(new_points)})')
        
        ax1.set_xlabel('X')  # type: ignore
        ax1.set_ylabel('Y')  # type: ignore
        ax1.set_zlabel('Z')  # type: ignore
        ax1.set_title('3D Point Distribution')  # type: ignore
        ax1.legend()  # type: ignore
        
        # 右圖：殘差熱圖（如果提供）
        ax2 = fig.add_subplot(122, projection='3d')  # type: ignore
        if old_residuals is not None and new_residuals is not None:
            all_points = np.vstack([old_points, new_points])
            all_residuals = np.concatenate([old_residuals, new_residuals])
            
            scatter = ax2.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2],  # type: ignore[call-overload]
                                 c=all_residuals, cmap='YlOrRd', s=20, alpha=0.6)
            
            cbar = plt.colorbar(scatter, ax=ax2, shrink=0.6)
            cbar.set_label('Residual Norm', rotation=270, labelpad=15)
            
            ax2.set_title('3D Residual Heat Map')  # type: ignore
        else:
            ax2.text2D(0.5, 0.5, 'No Residual Data', ha='center', va='center', transform=ax2.transAxes)  # type: ignore
            ax2.set_title('Residual Heat Map (N/A)')  # type: ignore
        
        ax2.set_xlabel('X')  # type: ignore
        ax2.set_ylabel('Y')  # type: ignore
        ax2.set_zlabel('Z')  # type: ignore
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_residual_comparison(self,
                                old_points: torch.Tensor,
                                new_points: torch.Tensor,
                                old_residuals: torch.Tensor,
                                new_residuals: torch.Tensor) -> Figure:
        """
        對比新舊點的殘差統計
        
        Args:
            old_points: 保留點 [N_old, dim]
            new_points: 新點 [N_new, dim]
            old_residuals: 保留點殘差 [N_old, n_eqs]
            new_residuals: 新點殘差 [N_new, n_eqs]
            
        Returns:
            figure: matplotlib 圖表
        """
        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 計算殘差範數
        old_norms = torch.norm(old_residuals, dim=-1).numpy()
        new_norms = torch.norm(new_residuals, dim=-1).numpy()
        
        # 1. 殘差分佈直方圖
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(old_norms, bins=50, alpha=0.6, label='Kept Points', color='blue')
        ax1.hist(new_norms, bins=50, alpha=0.6, label='New Points', color='red')
        ax1.set_xlabel('Residual Norm')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Residual Distribution')
        ax1.legend()
        ax1.set_yscale('log')
        
        # 2. 箱型圖比較
        ax2 = fig.add_subplot(gs[0, 1])
        bp = ax2.boxplot([old_norms, new_norms])
        ax2.set_xticklabels(['Kept', 'New'])
        ax2.set_ylabel('Residual Norm')
        ax2.set_title('Residual Statistics')
        ax2.grid(True, alpha=0.3)
        
        # 3. CDF 比較
        ax3 = fig.add_subplot(gs[0, 2])
        old_sorted = np.sort(old_norms)
        new_sorted = np.sort(new_norms)
        ax3.plot(old_sorted, np.linspace(0, 1, len(old_sorted)), 
                label='Kept Points', color='blue', linewidth=2)
        ax3.plot(new_sorted, np.linspace(0, 1, len(new_sorted)),
                label='New Points', color='red', linewidth=2)
        ax3.set_xlabel('Residual Norm')
        ax3.set_ylabel('Cumulative Probability')
        ax3.set_title('Cumulative Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 分方程殘差比較
        n_eqs = old_residuals.shape[1]
        eq_names = [f'Eq {i+1}' for i in range(n_eqs)]
        
        ax4 = fig.add_subplot(gs[1, :])
        
        old_means = old_residuals.abs().mean(dim=0).numpy()
        new_means = new_residuals.abs().mean(dim=0).numpy()
        
        x = np.arange(n_eqs)
        width = 0.35
        
        ax4.bar(x - width/2, old_means, width, label='Kept Points', color='blue', alpha=0.7)
        ax4.bar(x + width/2, new_means, width, label='New Points', color='red', alpha=0.7)
        
        ax4.set_xlabel('Equation Index')
        ax4.set_ylabel('Mean Absolute Residual')
        ax4.set_title('Per-Equation Residual Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(eq_names)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 統計摘要
        stats_text = (
            f"Kept Points: μ={old_norms.mean():.2e}, σ={old_norms.std():.2e}, "
            f"max={old_norms.max():.2e}\n"
            f"New Points:  μ={new_norms.mean():.2e}, σ={new_norms.std():.2e}, "
            f"max={new_norms.max():.2e}"
        )
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle('Residual Comparison: Kept vs New Points', 
                    fontsize=14, fontweight='bold')
        
        return fig
    
    def plot_resampling_timeline(self,
                                training_history: Dict[str, List],
                                resample_epochs: List[int]) -> Figure:
        """
        繪製重採樣時序圖（損失曲線 + 重採樣標記）
        
        Args:
            training_history: 訓練歷史 {'epoch': [...], 'loss': [...], ...}
            resample_epochs: 重採樣觸發的 epoch 列表
            
        Returns:
            figure: matplotlib 圖表
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        epochs = training_history.get('epoch', [])
        total_loss = training_history.get('total_loss', [])
        pde_loss = training_history.get('pde_loss', [])
        data_loss = training_history.get('data_loss', [])
        
        # 上圖：總損失 + 重採樣標記
        ax1 = axes[0]
        ax1.semilogy(epochs, total_loss, label='Total Loss', color='black', linewidth=1.5)
        
        # 標記重採樣點
        for re_epoch in resample_epochs:
            if re_epoch in epochs:
                idx = epochs.index(re_epoch)
                ax1.axvline(re_epoch, color='red', linestyle='--', alpha=0.6)
                ax1.scatter(re_epoch, total_loss[idx], color='red', s=100, 
                           marker='v', zorder=5)
        
        ax1.set_ylabel('Total Loss (log scale)')
        ax1.set_title('Training Loss with Resampling Events')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 下圖：分項損失
        ax2 = axes[1]
        if pde_loss:
            ax2.semilogy(epochs, pde_loss, label='PDE Loss', color='blue', linewidth=1.2)
        if data_loss:
            ax2.semilogy(epochs, data_loss, label='Data Loss', color='green', linewidth=1.2)
        
        # 標記重採樣點
        for re_epoch in resample_epochs:
            ax2.axvline(re_epoch, color='red', linestyle='--', alpha=0.6)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Component Loss (log scale)')
        ax2.set_title('Loss Components')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 添加重採樣次數註解
        fig.text(0.99, 0.99, f'Total Resamples: {len(resample_epochs)}',
                ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        
        return fig
    
    def plot_qr_quality_metrics(self,
                               svd_ranks: List[int],
                               svd_energies: List[float],
                               n_new_points: List[int],
                               resample_epochs: List[int]) -> Figure:
        """
        繪製 QR-pivot 選點質量指標
        
        Args:
            svd_ranks: SVD 秩歷史
            svd_energies: SVD 能量保留比例歷史
            n_new_points: 新點數量歷史
            resample_epochs: 重採樣 epoch 歷史
            
        Returns:
            figure: matplotlib 圖表
        """
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        
        # 上圖：SVD 秩
        ax1 = axes[0]
        ax1.plot(resample_epochs, svd_ranks, marker='o', color='blue', linewidth=2)
        ax1.set_ylabel('SVD Rank')
        ax1.set_title('SVD Rank Evolution')
        ax1.grid(True, alpha=0.3)
        
        # 中圖：SVD 能量保留
        ax2 = axes[1]
        ax2.plot(resample_epochs, svd_energies, marker='s', color='green', linewidth=2)
        ax2.axhline(y=0.99, color='red', linestyle='--', label='Target (0.99)')
        ax2.set_ylabel('Energy Ratio')
        ax2.set_ylim([0.95, 1.0])
        ax2.set_title('SVD Energy Retention')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 下圖：新點數量
        ax3 = axes[2]
        ax3.plot(resample_epochs, n_new_points, marker='^', color='red', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Number of New Points')
        ax3.set_title('New Points Added per Resample')
        ax3.grid(True, alpha=0.3)
        
        fig.suptitle('QR-Pivot Selection Quality Metrics', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def generate_full_report(self,
                            save_plots: bool = True) -> Dict[str, Figure]:
        """
        生成完整可視化報告
        
        Args:
            save_plots: 是否保存圖表到文件
            
        Returns:
            figures: 圖表字典 {name: figure}
        """
        figures = {}
        
        logger.info("=" * 80)
        logger.info("生成自適應採樣可視化報告...")
        logger.info("=" * 80)
        
        # 1. 空間分佈圖（如果有點數據）
        if 'current_pde_points' in self.sampling_stats:
            current_points = self.sampling_stats['current_pde_points']
            new_point_mask = self.sampling_stats.get('new_point_mask', 
                                                     torch.zeros(len(current_points), dtype=torch.bool))
            residuals = self.sampling_stats.get('current_residuals', None)
            
            domain_bounds = self._extract_domain_bounds()
            
            fig = self.plot_spatial_distribution(
                current_points, new_point_mask, residuals, domain_bounds
            )
            figures['spatial_distribution'] = fig
            
            if save_plots:
                fig.savefig(self.output_dir / 'spatial_distribution.png', dpi=200, bbox_inches='tight')
                logger.info(f"✅ 保存: spatial_distribution.png")
        
        # 2. 殘差比較圖（如果有分離的舊點/新點數據）
        if all(k in self.sampling_stats for k in ['old_residuals', 'new_residuals']):
            fig = self.plot_residual_comparison(
                self.sampling_stats['old_points'],
                self.sampling_stats['new_points'],
                self.sampling_stats['old_residuals'],
                self.sampling_stats['new_residuals']
            )
            figures['residual_comparison'] = fig
            
            if save_plots:
                fig.savefig(self.output_dir / 'residual_comparison.png', dpi=200, bbox_inches='tight')
                logger.info(f"✅ 保存: residual_comparison.png")
        
        # 3. 重採樣時序圖
        if self.training_history and 'resample_epochs' in self.sampling_stats:
            fig = self.plot_resampling_timeline(
                self.training_history,
                self.sampling_stats['resample_epochs']
            )
            figures['resampling_timeline'] = fig
            
            if save_plots:
                fig.savefig(self.output_dir / 'resampling_timeline.png', dpi=200, bbox_inches='tight')
                logger.info(f"✅ 保存: resampling_timeline.png")
        
        # 4. QR 質量指標
        if all(k in self.sampling_stats for k in ['svd_ranks', 'svd_energies', 'n_new_points']):
            fig = self.plot_qr_quality_metrics(
                self.sampling_stats['svd_ranks'],
                self.sampling_stats['svd_energies'],
                self.sampling_stats['n_new_points'],
                self.sampling_stats['resample_epochs']
            )
            figures['qr_quality_metrics'] = fig
            
            if save_plots:
                fig.savefig(self.output_dir / 'qr_quality_metrics.png', dpi=200, bbox_inches='tight')
                logger.info(f"✅ 保存: qr_quality_metrics.png")
        
        logger.info("=" * 80)
        logger.info(f"✅ 報告生成完成！共 {len(figures)} 張圖表")
        logger.info(f"   輸出目錄: {self.output_dir}")
        logger.info("=" * 80)
        
        return figures
    
    def _extract_domain_bounds(self) -> Dict[str, Tuple[float, float]]:
        """從配置中提取域邊界"""
        domain_cfg = self.config.get('domain', {})
        
        bounds = {}
        for coord in ['x', 'y', 'z']:
            coord_cfg = domain_cfg.get(coord, {})
            if coord_cfg:
                bounds[coord] = (coord_cfg.get('min', 0.0), coord_cfg.get('max', 1.0))
        
        return bounds


def main():
    parser = argparse.ArgumentParser(description='自適應殘差點採樣可視化工具')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型檢查點路徑')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路徑')
    parser.add_argument('--output', type=str, default='results/adaptive_sampling_vis/',
                       help='輸出目錄')
    parser.add_argument('--device', type=str, default='cpu',
                       help='計算設備')
    parser.add_argument('--no-save', action='store_true',
                       help='不保存圖表（僅顯示）')
    
    args = parser.parse_args()
    
    # 設置日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 創建可視化器
    visualizer = AdaptiveSamplingVisualizer(
        model_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output,
        device=args.device
    )
    
    # 生成報告
    figures = visualizer.generate_full_report(save_plots=not args.no_save)
    
    # 顯示圖表（可選）
    if args.no_save:
        plt.show()
    
    logger.info("✅ 可視化完成！")


if __name__ == "__main__":
    main()
