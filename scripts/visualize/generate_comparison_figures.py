#!/usr/bin/env python3
"""
對比實驗圖表生成工具
根據 docs/EXPERIMENT_COMPARISON_PLAN.md 自動生成論文所需的對比圖表

功能：
  - F-S1: Random vs QR 感測器佈點對比圖
  - F-K1: K-scan error 曲線圖（Random vs QR）
  - F-P1: Prior weight sweep 曲線圖
  - F-A1: Ablation bar chart（消融實驗條形圖）
  - F-R1: 全場重建三聯圖（DNS / PINN / Error）
  - F-R2: Channel flow 統計圖（U⁺(y⁺), τ_w）

使用方式：
    # 生成感測器佈點對比圖
    python scripts/visualize/generate_comparison_figures.py \
        --mode sensor_comparison \
        --random-sensors data/sensors/random_K100.npz \
        --qr-sensors data/sensors/qr_K100.npz \
        --background-field data/jhtdb/slice_z0.npz \
        --output results/figures/F-S1_random_vs_qr.png
    
    # 生成 K-scan error 曲線
    python scripts/visualize/generate_comparison_figures.py \
        --mode k_scan \
        --results-dir results/experiments/S2_k_scan \
        --output results/figures/F-K1_k_scan.png
    
    # 生成 Prior weight sweep 曲線
    python scripts/visualize/generate_comparison_figures.py \
        --mode prior_sweep \
        --results-dir results/experiments/C2_prior_sweep \
        --output results/figures/F-P1_prior_sweep.png
    
    # 生成 Ablation bar chart
    python scripts/visualize/generate_comparison_figures.py \
        --mode ablation \
        --results-dir results/experiments/A1_ablation_fourier \
        --output results/figures/F-A1_ablation.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging

# 添加專案路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===========================
# 全域繪圖規範（遵循 9.1 節規則）
# ===========================
PLOT_CONFIG = {
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2.0,
    'lines.markersize': 8,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
}

# 感測點標記規格（固定）
SENSOR_MARKER_CONFIG = {
    'size': 40,               # marker 大小
    'alpha': 0.8,             # 透明度
    'edgecolors': 'black',    # 邊框顏色
    'linewidths': 1.0,        # 邊框寬度
}

# 顏色方案
COLORS = {
    'random': '#E74C3C',      # 紅色
    'qr': '#3498DB',          # 藍色
    'prior': '#2ECC71',       # 綠色
    'baseline': '#95A5A6',    # 灰色
    'target': '#F39C12',      # 橙色（目標門檻線）
}


class ComparisonFigureGenerator:
    """對比實驗圖表生成器"""
    
    def __init__(self, output_dir: str = "results/figures"):
        """
        初始化圖表生成器
        
        Args:
            output_dir: 輸出目錄
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 應用全域設定
        plt.rcParams.update(PLOT_CONFIG)
        
        logger.info(f"圖表生成器初始化完成，輸出目錄: {self.output_dir}")
    
    def _resolve_output_path(self, output_name: str) -> Path:
        """
        解析輸出路徑（處理絕對路徑與相對路徑）
        
        Args:
            output_name: 輸出檔名或完整路徑
            
        Returns:
            解析後的輸出路徑
        """
        output_path = Path(output_name)
        if output_path.is_absolute():
            # 絕對路徑：確保父目錄存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return output_path
        elif '/' in output_name or '\\' in output_name:
            # 包含路徑分隔符的相對路徑：從當前工作目錄解析
            output_path = Path(output_name).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return output_path
        else:
            # 單純檔名：使用 output_dir
            return self.output_dir / output_name
    
    # =============================
    # F-S1: Random vs QR 感測器對比圖
    # =============================
    def generate_sensor_comparison(
        self,
        random_sensor_path: str,
        qr_sensor_path: str,
        background_field_path: str,
        output_name: str = "F-S1_random_vs_qr.png",
        field_name: str = "vorticity",
        view: str = "xy"
    ) -> str:
        """
        生成 F-S1: Random vs QR 感測器佈點對比圖
        
        Args:
            random_sensor_path: Random 感測器資料路徑
            qr_sensor_path: QR 感測器資料路徑
            background_field_path: 背景場資料路徑（用於顏色編碼）
            output_name: 輸出檔名
            field_name: 背景場變數名稱（'vorticity', '|u|', '|∇u|', 'Q'）
            view: 視角（'xy', 'xz', 'yz'）
            
        Returns:
            輸出圖片路徑
        """
        logger.info("生成 F-S1: Random vs QR 感測器佈點對比圖...")
        
        # 載入資料
        random_data = np.load(random_sensor_path)
        qr_data = np.load(qr_sensor_path)
        background_data = np.load(background_field_path)
        
        # 提取座標
        random_coords = self._extract_coordinates(random_data)
        qr_coords = self._extract_coordinates(qr_data)
        
        # 提取背景場
        background_field, bg_coords = self._extract_background_field(
            background_data, field_name
        )
        
        # 確定視角軸
        x_idx, y_idx, x_label, y_label = self._get_view_axes(view, random_coords.shape[1])
        
        # ===== 統一色階（關鍵規則） =====
        vmin, vmax = np.percentile(background_field, [1, 99])
        
        # 創建圖表
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # ===== 左圖：Random sensors =====
        ax = axes[0]
        
        # 背景場（使用 pcolormesh 或 contourf）
        if len(bg_coords[0].shape) == 2:  # 已經是網格
            im = ax.pcolormesh(
                bg_coords[0], bg_coords[1], background_field,
                vmin=vmin, vmax=vmax, cmap='jet', shading='auto', alpha=0.7
            )
        else:
            # 散點圖背景
            im = ax.scatter(
                bg_coords[0], bg_coords[1], c=background_field,
                vmin=vmin, vmax=vmax, cmap='jet', s=5, alpha=0.5
            )
        
        # 感測點（統一規格）
        ax.scatter(
            random_coords[:, x_idx], random_coords[:, y_idx],
            c=COLORS['random'], label='Random',
            s=SENSOR_MARKER_CONFIG['size'],
            alpha=SENSOR_MARKER_CONFIG['alpha'],
            edgecolors=SENSOR_MARKER_CONFIG['edgecolors'],
            linewidths=SENSOR_MARKER_CONFIG['linewidths'],
            marker='o', zorder=10
        )
        
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f'Random Sensors (K={len(random_coords)})', fontsize=14, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # ===== 右圖：QR sensors =====
        ax = axes[1]
        
        # 背景場（與左圖相同色階）
        if len(bg_coords[0].shape) == 2:
            im = ax.pcolormesh(
                bg_coords[0], bg_coords[1], background_field,
                vmin=vmin, vmax=vmax, cmap='jet', shading='auto', alpha=0.7
            )
        else:
            im = ax.scatter(
                bg_coords[0], bg_coords[1], c=background_field,
                vmin=vmin, vmax=vmax, cmap='jet', s=5, alpha=0.5
            )
        
        # 感測點（統一規格）
        ax.scatter(
            qr_coords[:, x_idx], qr_coords[:, y_idx],
            c=COLORS['qr'], label='QR-Pivot',
            s=SENSOR_MARKER_CONFIG['size'],
            alpha=SENSOR_MARKER_CONFIG['alpha'],
            edgecolors=SENSOR_MARKER_CONFIG['edgecolors'],
            linewidths=SENSOR_MARKER_CONFIG['linewidths'],
            marker='o', zorder=10
        )
        
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f'QR-Pivot Sensors (K={len(qr_coords)})', fontsize=14, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 共用色條
        cbar = plt.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
        cbar.set_label(f'{field_name}', fontsize=11)
        
        # 標註資訊
        fig.suptitle(
            f'Sensor Placement Comparison: Random vs QR-Pivot\n'
            f'Background: {field_name} | View: {view.upper()} plane',
            fontsize=15, fontweight='bold', y=0.98
        )
        
        plt.tight_layout()
        
        # 儲存
        output_path = self._resolve_output_path(output_name)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ F-S1 已儲存: {output_path}")
        return str(output_path)
    
    # =============================
    # F-K1: K-scan Error 曲線圖
    # =============================
    def generate_k_scan_curve(
        self,
        results_dir: str,
        output_name: str = "F-K1_k_scan.png",
        target_threshold: float = 0.15,
        k_values: Optional[List[int]] = None
    ) -> str:
        """
        生成 F-K1: K-scan error 曲線圖
        
        Args:
            results_dir: 實驗結果目錄（包含不同 K 的子目錄）
            output_name: 輸出檔名
            target_threshold: 目標誤差門檻（10-15%）
            k_values: K 值列表（若為 None 則自動偵測）
            
        Returns:
            輸出圖片路徑
        """
        logger.info("生成 F-K1: K-scan error 曲線圖...")
        
        results_path = Path(results_dir)
        
        # 自動偵測 K 值
        if k_values is None:
            k_values = self._detect_k_values(results_path)
        
        # 收集數據
        random_errors = {}
        qr_errors = {}
        
        for k in k_values:
            # Random 策略
            random_result = results_path / f"random_K{k}" / "metrics.json"
            if random_result.exists():
                random_errors[k] = self._extract_error_metrics(random_result)
            
            # QR 策略
            qr_result = results_path / f"qr_K{k}" / "metrics.json"
            if qr_result.exists():
                qr_errors[k] = self._extract_error_metrics(qr_result)
        
        # 繪圖
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Random 曲線
        if random_errors:
            k_list = sorted(random_errors.keys())
            mean_errors = [random_errors[k]['mean'] for k in k_list]
            std_errors = [random_errors[k]['std'] for k in k_list]
            
            ax.plot(k_list, mean_errors, 'o-', color=COLORS['random'],
                   linewidth=2.5, markersize=10, label='Random')
            ax.fill_between(k_list,
                           np.array(mean_errors) - np.array(std_errors),
                           np.array(mean_errors) + np.array(std_errors),
                           color=COLORS['random'], alpha=0.2)
        
        # QR 曲線
        if qr_errors:
            k_list = sorted(qr_errors.keys())
            mean_errors = [qr_errors[k]['mean'] for k in k_list]
            std_errors = [qr_errors[k]['std'] for k in k_list]
            
            ax.plot(k_list, mean_errors, 's-', color=COLORS['qr'],
                   linewidth=2.5, markersize=10, label='QR-Pivot')
            ax.fill_between(k_list,
                           np.array(mean_errors) - np.array(std_errors),
                           np.array(mean_errors) + np.array(std_errors),
                           color=COLORS['qr'], alpha=0.2)
        
        # 目標門檻線
        ax.axhline(y=target_threshold, color=COLORS['target'], linestyle='--',
                  linewidth=2, label=f'Target ({target_threshold*100:.0f}%)')
        
        ax.set_xlabel('Number of Sensors (K)', fontsize=13)
        ax.set_ylabel('Relative L2 Error', fontsize=13)
        ax.set_title('K-scan: Error vs Number of Sensors\n(mean ± std, 3 seeds)',
                    fontsize=15, fontweight='bold')
        ax.legend(loc='best', fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 設定 x 軸刻度
        if k_values:
            ax.set_xticks(k_values)
        
        plt.tight_layout()
        
        # 儲存
        output_path = self._resolve_output_path(output_name)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ F-K1 已儲存: {output_path}")
        return str(output_path)
    
    # =============================
    # F-P1: Prior Weight Sweep 曲線圖
    # =============================
    def generate_prior_sweep_curve(
        self,
        results_dir: str,
        output_name: str = "F-P1_prior_sweep.png",
        prior_weights: Optional[List[float]] = None
    ) -> str:
        """
        生成 F-P1: Prior weight sweep 曲線圖
        
        Args:
            results_dir: 實驗結果目錄
            output_name: 輸出檔名
            prior_weights: prior_weight 列表（若為 None 則自動偵測）
            
        Returns:
            輸出圖片路徑
        """
        logger.info("生成 F-P1: Prior weight sweep 曲線圖...")
        
        results_path = Path(results_dir)
        
        # 自動偵測 prior_weight 值
        if prior_weights is None:
            prior_weights = self._detect_prior_weights(results_path)
        
        # 收集數據
        errors = {}
        div_errors = {}
        tau_w_errors = {}
        
        for pw in prior_weights:
            result_file = results_path / f"prior_weight_{pw:.1f}" / "metrics.json"
            if result_file.exists():
                metrics = self._load_json(result_file)
                errors[pw] = metrics.get('relative_l2_overall', np.nan)
                div_errors[pw] = metrics.get('divergence_error_mean', np.nan)
                tau_w_errors[pw] = metrics.get('tau_w_relative_error', np.nan)
        
        # 繪圖
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        pw_list = sorted(errors.keys())
        
        # 子圖1: Overall Error
        ax = axes[0]
        error_list = [errors[pw] for pw in pw_list]
        ax.plot(pw_list, error_list, 'o-', color=COLORS['prior'],
               linewidth=2.5, markersize=10)
        ax.set_xlabel('Prior Weight', fontsize=12)
        ax.set_ylabel('Relative L2 Error', fontsize=12)
        ax.set_title('Overall Error vs Prior Weight', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 子圖2: Divergence Error
        ax = axes[1]
        div_list = [div_errors[pw] for pw in pw_list if not np.isnan(div_errors.get(pw, np.nan))]
        if div_list:
            ax.semilogy(pw_list[:len(div_list)], div_list, 's-', color='red',
                       linewidth=2.5, markersize=10)
            ax.set_xlabel('Prior Weight', fontsize=12)
            ax.set_ylabel('‖∇·u‖ (log scale)', fontsize=12)
            ax.set_title('Continuity Error vs Prior Weight', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
        
        # 子圖3: τ_w Error
        ax = axes[2]
        tau_list = [tau_w_errors[pw] for pw in pw_list if not np.isnan(tau_w_errors.get(pw, np.nan))]
        if tau_list:
            ax.plot(pw_list[:len(tau_list)], tau_list, '^-', color='blue',
                   linewidth=2.5, markersize=10)
            ax.set_xlabel('Prior Weight', fontsize=12)
            ax.set_ylabel('τ_w Relative Error', fontsize=12)
            ax.set_title('Wall Shear Stress Error vs Prior Weight', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
        
        fig.suptitle('Prior Weight Sweep Analysis: Finding the Sweet Spot',
                    fontsize=15, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        # 儲存
        output_path = self._resolve_output_path(output_name)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ F-P1 已儲存: {output_path}")
        return str(output_path)
    
    # =============================
    # F-A1: Ablation Bar Chart
    # =============================
    def generate_ablation_chart(
        self,
        results_dir: str,
        output_name: str = "F-A1_ablation.png",
        baseline_name: str = "full"
    ) -> str:
        """
        生成 F-A1: Ablation bar chart（消融實驗條形圖）
        
        Args:
            results_dir: 實驗結果目錄
            output_name: 輸出檔名
            baseline_name: 基線實驗名稱（用於計算 Δerror）
            
        Returns:
            輸出圖片路徑
        """
        logger.info("生成 F-A1: Ablation bar chart...")
        
        results_path = Path(results_dir)
        
        # 定義實驗配置
        experiments = {
            'Full': baseline_name,
            '- Fourier': 'without_fourier',
            '- GradNorm': 'without_gradnorm',
            '- RWF': 'without_rwf',
            '- ResNet': 'without_resnet',
            '- VS-PINN': 'without_vspinn'
        }
        
        # 收集數據
        errors = {}
        epochs_to_converge = {}
        
        for label, exp_name in experiments.items():
            result_file = results_path / exp_name / "metrics.json"
            if result_file.exists():
                metrics = self._load_json(result_file)
                errors[label] = metrics.get('relative_l2_overall', np.nan)
                epochs_to_converge[label] = metrics.get('epochs_to_threshold', np.nan)
        
        # 計算 Δerror（相對於 Full）
        baseline_error = errors.get('Full', np.nan)
        delta_errors = {k: (v - baseline_error) * 100 for k, v in errors.items() if k != 'Full'}
        
        # 繪圖
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 子圖1: Δerror Bar Chart
        ax = axes[0]
        labels = list(delta_errors.keys())
        values = list(delta_errors.values())
        colors_list = [COLORS['baseline'] if v > 0 else COLORS['prior'] for v in values]
        
        bars = ax.barh(labels, values, color=colors_list, edgecolor='black', linewidth=1.5)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Δerror (% relative to Full)', fontsize=12)
        ax.set_title('Ablation Study: Impact on Error\n(negative = improvement)',
                    fontsize=13, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        
        # 標註數值
        for bar, value in zip(bars, values):
            width = bar.get_width()
            ax.text(width + (1 if width > 0 else -1), bar.get_y() + bar.get_height()/2,
                   f'{value:+.1f}%', ha='left' if width > 0 else 'right',
                   va='center', fontsize=10, fontweight='bold')
        
        # 子圖2: 收斂速度
        ax = axes[1]
        epochs_labels = [k for k in epochs_to_converge.keys() if k in labels or k == 'Full']
        epochs_values = [epochs_to_converge.get(k, 0) for k in epochs_labels]
        
        if any(v > 0 for v in epochs_values):
            bars = ax.barh(epochs_labels, epochs_values, color=COLORS['qr'],
                          edgecolor='black', linewidth=1.5)
            ax.set_xlabel('Epochs to Converge (threshold 15%)', fontsize=12)
            ax.set_title('Training Efficiency', fontsize=13, fontweight='bold')
            ax.grid(True, axis='x', alpha=0.3, linestyle='--')
            
            # 標註數值
            for bar, value in zip(bars, epochs_values):
                if value > 0:
                    ax.text(value + 50, bar.get_y() + bar.get_height()/2,
                           f'{int(value)}', ha='left', va='center',
                           fontsize=10, fontweight='bold')
        
        fig.suptitle('Ablation Study: Component Contributions',
                    fontsize=15, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # 儲存
        output_path = self._resolve_output_path(output_name)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ F-A1 已儲存: {output_path}")
        return str(output_path)
    
    # =============================
    # 輔助函數
    # =============================
    def _extract_coordinates(self, data: dict) -> np.ndarray:
        """從感測器資料中提取座標"""
        # 優先使用 'coordinates' 或 'coords'
        if 'coordinates' in data:
            return np.array(data['coordinates'])
        elif 'coords' in data:
            return np.array(data['coords'])
        elif 'sensor_coords' in data:
            return np.array(data['sensor_coords'])
        
        # 嘗試從 x, y, z 重建
        if 'sensor_x' in data and 'sensor_y' in data:
            if 'sensor_z' in data:
                return np.stack([data['sensor_x'], data['sensor_y'], data['sensor_z']], axis=1)
            else:
                return np.stack([data['sensor_x'], data['sensor_y']], axis=1)
        
        raise KeyError("無法找到座標資料：'coordinates', 'coords', 'sensor_coords', 或 'sensor_x/y/z'")
    
    def _extract_background_field(
        self, data: dict, field_name: str
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """提取背景場數據"""
        # 提取場數據
        if field_name == 'vorticity':
            # 計算渦度 ω_z = ∂v/∂x - ∂u/∂y
            u = data.get('u', np.zeros((128, 128)))
            v = data.get('v', np.zeros((128, 128)))
            # 簡化：使用中心差分
            vorticity = np.gradient(v, axis=1) - np.gradient(u, axis=0)
            field = vorticity
        elif field_name == '|u|':
            u = data.get('u', np.zeros((128, 128)))
            v = data.get('v', np.zeros((128, 128)))
            field = np.sqrt(u**2 + v**2)
        elif field_name in data:
            field = data[field_name]
        else:
            # 預設使用 u 分量
            field = data.get('u', np.zeros((128, 128)))
        
        # 提取座標
        if 'x' in data and 'y' in data:
            x = data['x']
            y = data['y']
            if x.ndim == 1:
                X, Y = np.meshgrid(x, y, indexing='ij')
            else:
                X, Y = x, y
        else:
            # 生成預設網格
            nx, ny = field.shape[:2]
            X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
        
        return field, (X, Y)
    
    def _get_view_axes(self, view: str, ndim: int) -> Tuple[int, int, str, str]:
        """根據視角返回軸索引和標籤"""
        if view == 'xy':
            return 0, 1, 'x', 'y'
        elif view == 'xz' and ndim >= 3:
            return 0, 2, 'x', 'z'
        elif view == 'yz' and ndim >= 3:
            return 1, 2, 'y', 'z'
        else:
            # 預設 xy
            return 0, 1, 'x', 'y'
    
    def _detect_k_values(self, results_dir: Path) -> List[int]:
        """自動偵測 K 值"""
        k_values = set()
        for subdir in results_dir.iterdir():
            if subdir.is_dir():
                # 匹配格式：random_K30, qr_K50 等
                match = subdir.name.split('_K')
                if len(match) == 2 and match[1].isdigit():
                    k_values.add(int(match[1]))
        return sorted(k_values)
    
    def _detect_prior_weights(self, results_dir: Path) -> List[float]:
        """自動偵測 prior_weight 值"""
        weights = set()
        for subdir in results_dir.iterdir():
            if subdir.is_dir():
                # 匹配格式：prior_weight_0.1, prior_weight_0.5 等
                match = subdir.name.split('prior_weight_')
                if len(match) == 2:
                    try:
                        weights.add(float(match[1]))
                    except ValueError:
                        pass
        return sorted(weights)
    
    def _extract_error_metrics(self, metrics_file: Path) -> Dict[str, float]:
        """提取誤差指標（含統計）"""
        metrics = self._load_json(metrics_file)
        
        # 假設有多次運行的結果
        if 'seeds' in metrics:
            errors = [seed['relative_l2_overall'] for seed in metrics['seeds']]
            return {
                'mean': float(np.mean(errors)),
                'std': float(np.std(errors)),
                'min': float(np.min(errors)),
                'max': float(np.max(errors))
            }
        else:
            # 單次運行
            error = metrics.get('relative_l2_overall', np.nan)
            error_float = float(error) if not np.isnan(error) else 0.0
            return {
                'mean': error_float,
                'std': 0.0,
                'min': error_float,
                'max': error_float
            }
    
    def _load_json(self, file_path: Path) -> dict:
        """載入 JSON 文件"""
        with open(file_path, 'r') as f:
            return json.load(f)


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='對比實驗圖表生成工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 通用參數
    parser.add_argument('--mode', type=str, required=True,
                       choices=['sensor_comparison', 'k_scan', 'prior_sweep', 'ablation'],
                       help='生成模式')
    parser.add_argument('--output', type=str, default=None,
                       help='輸出檔案路徑（若為 None 則使用預設名稱）')
    parser.add_argument('--output-dir', type=str, default='results/figures',
                       help='輸出目錄')
    
    # sensor_comparison 模式參數
    parser.add_argument('--random-sensors', type=str, default=None,
                       help='Random 感測器資料路徑')
    parser.add_argument('--qr-sensors', type=str, default=None,
                       help='QR 感測器資料路徑')
    parser.add_argument('--background-field', type=str, default=None,
                       help='背景場資料路徑')
    parser.add_argument('--field-name', type=str, default='vorticity',
                       choices=['vorticity', '|u|', '|∇u|', 'Q', 'u', 'v', 'w', 'p'],
                       help='背景場變數名稱')
    parser.add_argument('--view', type=str, default='xy',
                       choices=['xy', 'xz', 'yz'],
                       help='視角')
    
    # k_scan / prior_sweep / ablation 模式參數
    parser.add_argument('--results-dir', type=str, default=None,
                       help='實驗結果目錄')
    parser.add_argument('--k-values', type=int, nargs='+', default=None,
                       help='K 值列表（k_scan 模式）')
    parser.add_argument('--prior-weights', type=float, nargs='+', default=None,
                       help='Prior weight 列表（prior_sweep 模式）')
    parser.add_argument('--baseline-name', type=str, default='full',
                       help='基線實驗名稱（ablation 模式）')
    
    args = parser.parse_args()
    
    # 初始化生成器
    generator = ComparisonFigureGenerator(args.output_dir)
    
    # 根據模式生成圖表
    output_path: str = ""
    
    if args.mode == 'sensor_comparison':
        if not all([args.random_sensors, args.qr_sensors, args.background_field]):
            parser.error("sensor_comparison 模式需要 --random-sensors, --qr-sensors, --background-field")
        
        output_name = args.output or "F-S1_random_vs_qr.png"
        output_path = generator.generate_sensor_comparison(
            args.random_sensors,
            args.qr_sensors,
            args.background_field,
            output_name,
            args.field_name,
            args.view
        )
    
    elif args.mode == 'k_scan':
        if not args.results_dir:
            parser.error("k_scan 模式需要 --results-dir")
        
        output_name = args.output or "F-K1_k_scan.png"
        output_path = generator.generate_k_scan_curve(
            args.results_dir,
            output_name,
            k_values=args.k_values
        )
    
    elif args.mode == 'prior_sweep':
        if not args.results_dir:
            parser.error("prior_sweep 模式需要 --results-dir")
        
        output_name = args.output or "F-P1_prior_sweep.png"
        output_path = generator.generate_prior_sweep_curve(
            args.results_dir,
            output_name,
            prior_weights=args.prior_weights
        )
    
    elif args.mode == 'ablation':
        if not args.results_dir:
            parser.error("ablation 模式需要 --results-dir")
        
        output_name = args.output or "F-A1_ablation.png"
        output_path = generator.generate_ablation_chart(
            args.results_dir,
            output_name,
            baseline_name=args.baseline_name
        )
    
    print(f"\n{'='*60}")
    print("🎉 圖表生成完成!")
    print(f"📊 輸出檔案: {output_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
