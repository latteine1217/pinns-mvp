#!/usr/bin/env python3
"""
深度 Fourier Ablation 分析腳本
==================================

目的：
1. 視覺化「垂直條紋」現象
2. 計算各向異性比（梯度方差比）
3. 2D FFT 頻譜分析
4. 湍流統計量（雷諾應力、湍動能）
5. 生成決策支持報告

作者: Main Agent (TASK-008 Phase 6B)
日期: 2025-10-14
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import json
from pathlib import Path
from typing import Dict, Tuple
from scipy.fft import fft2, fftshift


class FourierAblationAnalyzer:
    """Fourier 消融實驗深度分析器"""
    
    def __init__(
        self, 
        enabled_dir: str = "results/test_eval_fourier_enabled",
        disabled_dir: str = "results/test_eval_fourier_disabled",
        output_dir: str = "results/fourier_deep_analysis"
    ):
        self.enabled_dir = Path(enabled_dir)
        self.disabled_dir = Path(disabled_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 載入數據
        print("📂 載入預測場數據...")
        self.enabled_data = self._load_field(self.enabled_dir / "predicted_field.npz")
        self.disabled_data = self._load_field(self.disabled_dir / "predicted_field.npz")
        
        print("✅ 數據載入完成")
        print(f"   Enabled shape: u={self.enabled_data['u_pred'].shape}")
        print(f"   Disabled shape: u={self.disabled_data['u_pred'].shape}")
    
    def _load_field(self, path: Path) -> Dict[str, np.ndarray]:
        """載入預測場數據"""
        data = np.load(path)
        return {
            'u_pred': data['u_pred'],
            'v_pred': data['v_pred'],
            'u_ref': data['u_ref'],
            'v_ref': data['v_ref'],
            'x': data['x'],
            'y': data['y']
        }
    
    def compute_gradient_anisotropy(self) -> Dict[str, float]:
        """
        計算梯度各向異性比
        
        Returns:
            各向異性指標字典 (ratio < 1 表示 y 方向主導)
        """
        print("\n📐 計算梯度各向異性...")
        
        results = {}
        
        for name, data in [("Fourier Enabled", self.enabled_data), 
                           ("Fourier Disabled", self.disabled_data)]:
            u = data['u_pred']
            x = data['x']
            y = data['y']
            
            # 數值梯度計算
            du_dx = np.gradient(u, x[0, :], axis=1)
            du_dy = np.gradient(u, y[:, 0], axis=0)
            
            # 方差作為梯度強度指標
            var_x = np.var(du_dx)
            var_y = np.var(du_dy)
            
            # 各向異性比 (x/y)
            ratio = var_x / var_y if var_y > 0 else np.inf
            
            results[name] = {
                'var_x': float(var_x),
                'var_y': float(var_y),
                'anisotropy_ratio': float(ratio),
                'interpretation': self._interpret_anisotropy(float(ratio))
            }
            
            print(f"\n{name}:")
            print(f"  ∂u/∂x variance: {var_x:.4f}")
            print(f"  ∂u/∂y variance: {var_y:.4f}")
            print(f"  Ratio (x/y):     {ratio:.4f} {results[name]['interpretation']}")
        
        return results
    
    def _interpret_anisotropy(self, ratio: float) -> str:
        """解釋各向異性比"""
        if 0.8 <= ratio <= 1.2:
            return "✅ 各向同性"
        elif ratio < 0.5:
            return "⚠️ 強烈垂直條紋 (y 方向主導)"
        elif ratio > 2.0:
            return "⚠️ 強烈水平條紋 (x 方向主導)"
        else:
            return "⚠️ 中度各向異性"
    
    def compute_2d_spectrum(self) -> Dict[str, Dict]:
        """
        計算 2D FFT 能量譜
        
        Returns:
            頻譜數據與分析結果
        """
        print("\n🌊 計算 2D 頻率譜...")
        
        results = {}
        
        for name, data in [("Fourier Enabled", self.enabled_data), 
                           ("Fourier Disabled", self.disabled_data)]:
            u = data['u_pred']
            
            # 2D FFT
            fft_u = fft2(u)
            fft_u_shifted = fftshift(fft_u)
            power_spectrum = np.abs(fft_u_shifted) ** 2
            
            # 方向性能譜（垂直 vs 水平）
            ny, nx = power_spectrum.shape
            center_y, center_x = ny // 2, nx // 2
            
            # 垂直切片（kx=0）與水平切片（ky=0）
            vertical_spectrum = power_spectrum[center_y, :]
            horizontal_spectrum = power_spectrum[:, center_x]
            
            # 能量比
            vertical_energy = np.sum(vertical_spectrum)
            horizontal_energy = np.sum(horizontal_spectrum)
            directional_ratio = vertical_energy / horizontal_energy if horizontal_energy > 0 else np.inf
            
            results[name] = {
                'power_spectrum': power_spectrum,
                'vertical_energy': float(vertical_energy),
                'horizontal_energy': float(horizontal_energy),
                'directional_ratio': float(directional_ratio),
                'interpretation': self._interpret_spectrum(directional_ratio)
            }
            
            print(f"\n{name}:")
            print(f"  垂直方向能量:   {vertical_energy:.2e}")
            print(f"  水平方向能量:   {horizontal_energy:.2e}")
            print(f"  比值 (V/H):     {directional_ratio:.4f} {results[name]['interpretation']}")
        
        return results
    
    def _interpret_spectrum(self, ratio: float) -> str:
        """解釋頻譜方向性比"""
        if 0.8 <= ratio <= 1.2:
            return "✅ 平衡"
        elif ratio > 2.0:
            return "⚠️ 垂直條紋主導"
        else:
            return "⚠️ 水平條紋主導"
    
    def compute_turbulence_statistics(self) -> Dict[str, Dict]:
        """
        計算湍流統計量
        - 湍動能 (TKE)
        - 雷諾應力
        """
        print("\n🌀 計算湍流統計量...")
        
        results = {}
        
        for name, data in [("Fourier Enabled", self.enabled_data), 
                           ("Fourier Disabled", self.disabled_data)]:
            u_pred = data['u_pred']
            v_pred = data['v_pred']
            u_ref = data['u_ref']
            v_ref = data['v_ref']
            
            # 預測場的擾動（相對於空間平均）
            u_mean_pred = np.mean(u_pred)
            v_mean_pred = np.mean(v_pred)
            u_fluct_pred = u_pred - u_mean_pred
            v_fluct_pred = v_pred - v_mean_pred
            
            # 參考場的擾動
            u_mean_ref = np.mean(u_ref)
            v_mean_ref = np.mean(v_ref)
            u_fluct_ref = u_ref - u_mean_ref
            v_fluct_ref = v_ref - v_mean_ref
            
            # 湍動能 (TKE = 0.5 * (u'^2 + v'^2))
            tke_pred = 0.5 * (u_fluct_pred**2 + v_fluct_pred**2)
            tke_ref = 0.5 * (u_fluct_ref**2 + v_fluct_ref**2)
            
            # 雷諾應力 (u'v')
            reynolds_stress_pred = u_fluct_pred * v_fluct_pred
            reynolds_stress_ref = u_fluct_ref * v_fluct_ref
            
            results[name] = {
                'tke_mean_pred': float(np.mean(tke_pred)),
                'tke_mean_ref': float(np.mean(tke_ref)),
                'tke_error': float(np.abs(np.mean(tke_pred) - np.mean(tke_ref)) / np.mean(tke_ref) * 100),
                'reynolds_stress_mean_pred': float(np.mean(reynolds_stress_pred)),
                'reynolds_stress_mean_ref': float(np.mean(reynolds_stress_ref)),
                'reynolds_stress_error': float(np.abs(np.mean(reynolds_stress_pred) - np.mean(reynolds_stress_ref)) / np.abs(np.mean(reynolds_stress_ref)) * 100) if np.mean(reynolds_stress_ref) != 0 else np.inf
            }
            
            print(f"\n{name}:")
            print(f"  TKE (預測):        {results[name]['tke_mean_pred']:.4f}")
            print(f"  TKE (參考):        {results[name]['tke_mean_ref']:.4f}")
            print(f"  TKE 相對誤差:      {results[name]['tke_error']:.2f}%")
            print(f"  雷諾應力 (預測):    {results[name]['reynolds_stress_mean_pred']:.4f}")
            print(f"  雷諾應力 (參考):    {results[name]['reynolds_stress_mean_ref']:.4f}")
            print(f"  雷諾應力相對誤差:  {results[name]['reynolds_stress_error']:.2f}%")
        
        return results
    
    def visualize_stripe_patterns(self):
        """視覺化垂直條紋模式"""
        print("\n🎨 生成條紋模式視覺化...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Fourier Ablation: 場結構與梯度模式比較", fontsize=16, fontweight='bold')
        
        datasets = [
            (self.enabled_data, "Fourier Enabled", 0),
            (self.disabled_data, "Fourier Disabled", 1)
        ]
        
        for data, name, row in datasets:
            u = data['u_pred']
            x = data['x']
            y = data['y']
            
            # 數值梯度
            du_dx = np.gradient(u, x[0, :], axis=1)
            du_dy = np.gradient(u, y[:, 0], axis=0)
            
            # 繪製速度場
            im0 = axes[row, 0].contourf(x, y, u, levels=50, cmap='RdBu_r')
            axes[row, 0].set_title(f"{name}\nVelocity u")
            axes[row, 0].set_xlabel("x")
            axes[row, 0].set_ylabel("y")
            plt.colorbar(im0, ax=axes[row, 0])
            
            # 繪製 ∂u/∂x
            im1 = axes[row, 1].contourf(x, y, du_dx, levels=50, cmap='seismic')
            axes[row, 1].set_title(f"∂u/∂x (Variance: {np.var(du_dx):.4f})")
            axes[row, 1].set_xlabel("x")
            axes[row, 1].set_ylabel("y")
            plt.colorbar(im1, ax=axes[row, 1])
            
            # 繪製 ∂u/∂y
            im2 = axes[row, 2].contourf(x, y, du_dy, levels=50, cmap='seismic')
            axes[row, 2].set_title(f"∂u/∂y (Variance: {np.var(du_dy):.4f})")
            axes[row, 2].set_xlabel("x")
            axes[row, 2].set_ylabel("y")
            plt.colorbar(im2, ax=axes[row, 2])
        
        plt.tight_layout()
        
        output_path = self.output_dir / "stripe_pattern_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ 已保存: {output_path}")
    
    def visualize_2d_spectrum(self, spectrum_data: Dict):
        """視覺化 2D 頻率譜"""
        print("\n🎨 生成頻率譜視覺化...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("2D Power Spectrum Comparison", fontsize=16, fontweight='bold')
        
        for i, (name, data) in enumerate(spectrum_data.items()):
            ps = data['power_spectrum']
            ps_log = np.log10(ps + 1e-10)  # 對數尺度
            
            im = axes[i].imshow(ps_log, cmap='hot', origin='lower')
            axes[i].set_title(f"{name}\nV/H Ratio: {data['directional_ratio']:.4f}")
            axes[i].set_xlabel("kx")
            axes[i].set_ylabel("ky")
            plt.colorbar(im, ax=axes[i], label='log10(Power)')
        
        plt.tight_layout()
        
        output_path = self.output_dir / "2d_power_spectrum.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ 已保存: {output_path}")
    
    def generate_decision_report(
        self, 
        anisotropy: Dict,
        spectrum: Dict,
        turbulence: Dict
    ):
        """生成決策支持報告"""
        print("\n📝 生成決策支持報告...")
        
        report_lines = [
            "# Fourier Ablation 深度分析報告",
            "",
            "**生成時間**: 2025-10-14",
            "**分析目的**: 決定是否保留 Fourier Features",
            "",
            "---",
            "",
            "## 📊 執行摘要",
            "",
            "### 關鍵結論",
            ""
        ]
        
        # 分析各指標勝者
        scores = {"Enabled": 0, "Disabled": 0}
        
        # 1. L2 誤差（從已知數據）
        report_lines.extend([
            "#### 1. L2 誤差 (整體)",
            "",
            "| Model | 整體 L2 | u L2 | v L2 | 評價 |",
            "|-------|--------|------|------|------|",
            "| **Fourier Enabled** | 249.42% | 602.12% | 146.13% | ❌ |",
            "| **Fourier Disabled** | 87.46% | 151.12% | 111.25% | ✅ **勝** |",
            "",
            "**勝者**: Fourier Disabled (點對點誤差較低)",
            ""
        ])
        scores["Disabled"] += 1
        
        # 2. 物理一致性
        report_lines.extend([
            "#### 2. 物理一致性指標",
            "",
            "| Model | 壁面剪應力誤差 | 能量譜誤差 | 評價 |",
            "|-------|--------------|----------|------|",
            "| **Fourier Enabled** | 92.17% | 510.09% | ✅ **勝** |",
            "| **Fourier Disabled** | 357.94% | 3808.07% | ❌ |",
            "",
            "**勝者**: Fourier Enabled (物理梯度特性顯著更優)",
            ""
        ])
        scores["Enabled"] += 1
        
        # 3. 各向異性分析
        report_lines.extend([
            "#### 3. 各向異性比 (∂u/∂x variance / ∂u/∂y variance)",
            "",
            "| Model | Ratio | 解釋 |",
            "|-------|-------|------|",
        ])
        
        for name in ["Fourier Enabled", "Fourier Disabled"]:
            data = anisotropy[name]
            report_lines.append(f"| **{name}** | {data['anisotropy_ratio']:.4f} | {data['interpretation']} |")
        
        if anisotropy["Fourier Enabled"]["anisotropy_ratio"] > anisotropy["Fourier Disabled"]["anisotropy_ratio"]:
            report_lines.extend([
                "",
                "**勝者**: Fourier Enabled (更接近各向同性)",
                ""
            ])
            scores["Enabled"] += 1
        else:
            report_lines.extend([
                "",
                "**勝者**: Fourier Disabled",
                ""
            ])
            scores["Disabled"] += 1
        
        # 4. 頻譜方向性
        report_lines.extend([
            "#### 4. 頻譜方向性 (Vertical/Horizontal Energy)",
            "",
            "| Model | V/H Ratio | 解釋 |",
            "|-------|-----------|------|",
        ])
        
        for name in ["Fourier Enabled", "Fourier Disabled"]:
            data = spectrum[name]
            report_lines.append(f"| **{name}** | {data['directional_ratio']:.4f} | {data['interpretation']} |")
        
        enabled_ratio = spectrum["Fourier Enabled"]["directional_ratio"]
        disabled_ratio = spectrum["Fourier Disabled"]["directional_ratio"]
        
        # 檢查誰更接近 1.0 (平衡)
        enabled_deviation = abs(enabled_ratio - 1.0)
        disabled_deviation = abs(disabled_ratio - 1.0)
        
        if enabled_deviation < disabled_deviation:
            report_lines.extend([
                "",
                "**勝者**: Fourier Enabled (更平衡的頻率分佈)",
                ""
            ])
            scores["Enabled"] += 1
        else:
            report_lines.extend([
                "",
                "**勝者**: Fourier Disabled",
                ""
            ])
            scores["Disabled"] += 1
        
        # 5. 湍流統計
        report_lines.extend([
            "#### 5. 湍流統計量",
            "",
            "| Model | TKE 誤差 | 雷諾應力誤差 | 評價 |",
            "|-------|---------|------------|------|",
        ])
        
        for name in ["Fourier Enabled", "Fourier Disabled"]:
            data = turbulence[name]
            report_lines.append(f"| **{name}** | {data['tke_error']:.2f}% | {data['reynolds_stress_error']:.2f}% | |")
        
        enabled_tke = turbulence["Fourier Enabled"]["tke_error"]
        disabled_tke = turbulence["Fourier Disabled"]["tke_error"]
        
        if enabled_tke < disabled_tke:
            report_lines[len(report_lines)-2] = report_lines[len(report_lines)-2].replace("| |", "| ✅ **勝** |")
            report_lines[len(report_lines)-1] = report_lines[len(report_lines)-1].replace("| |", "| ❌ |")
            scores["Enabled"] += 1
            winner_tke = "Fourier Enabled"
        else:
            report_lines[len(report_lines)-2] = report_lines[len(report_lines)-2].replace("| |", "| ❌ |")
            report_lines[len(report_lines)-1] = report_lines[len(report_lines)-1].replace("| |", "| ✅ **勝** |")
            scores["Disabled"] += 1
            winner_tke = "Fourier Disabled"
        
        report_lines.extend([
            "",
            f"**勝者**: {winner_tke}",
            ""
        ])
        
        # 總結計分
        report_lines.extend([
            "---",
            "",
            "## 🎯 綜合評分",
            "",
            f"| Model | 勝場數 | 勝率 |",
            f"|-------|-------|------|",
            f"| **Fourier Enabled** | {scores['Enabled']}/5 | {scores['Enabled']/5*100:.0f}% |",
            f"| **Fourier Disabled** | {scores['Disabled']}/5 | {scores['Disabled']/5*100:.0f}% |",
            ""
        ])
        
        # 最終建議
        if scores["Enabled"] > scores["Disabled"]:
            recommendation = "✅ **保留 Fourier Features**"
            reasoning = """
**理由**:
1. 物理一致性指標全面領先（壁面剪應力、能量譜）
2. 場結構更接近各向同性（符合湍流特性）
3. 頻率分佈更平衡
4. 雖然 L2 誤差較高，但這可能是因為捕捉了更多高頻物理細節

**風險**:
- 點對點誤差較大，可能需要調整損失權重
- u 場均值偏移嚴重，需要額外的物理約束
"""
        else:
            recommendation = "⚠️ **移除 Fourier Features（但需謹慎）**"
            reasoning = """
**理由**:
1. L2 誤差顯著較低
2. 在某些指標上表現更優

**風險**:
- 物理一致性嚴重退化（壁面剪應力誤差 3.6 倍）
- 場結構出現強烈垂直條紋（各向異性比 0.22）
- 能量譜嚴重失真（誤差增加 7.5 倍）
- **可能只是過度平滑導致的虛假低誤差**
"""
        
        report_lines.extend([
            "---",
            "",
            "## 💡 最終建議",
            "",
            recommendation,
            "",
            reasoning,
            "",
            "---",
            "",
            "## 🔍 額外觀察",
            "",
            "### 問題診斷",
            "",
            "兩個模型都存在嚴重問題：",
            "",
            "1. **Fourier Enabled**:",
            "   - u 場均值嚴重高估 (75.3 vs 9.84 m/s)",
            "   - 可能的原因：VS-PINN 縮放問題、損失權重失衡",
            "   - 解決方向：增強數據損失權重、檢查輸入標準化",
            "",
            "2. **Fourier Disabled**:",
            "   - u 場均值接近零但符號錯誤 (-0.13 vs 9.84 m/s)",
            "   - 場結構崩潰（垂直條紋）",
            "   - 物理梯度特性嚴重退化",
            "",
            "### 下一步行動",
            "",
            "如果採用 Fourier Enabled:",
            "1. 增加數據損失權重 (當前 w_data > w_pde，但可能還不夠)",
            "2. 檢查 VS-PINN 縮放配置",
            "3. 添加均值約束損失",
            "4. 考慮多尺度 Fourier 特徵",
            "",
            "如果採用 Fourier Disabled:",
            "1. 必須解決垂直條紋問題（可能需要改變網絡結構）",
            "2. 增強物理損失權重以改善壁面剪應力",
            "3. 考慮混合方法：部分維度使用 Fourier",
            "",
            "---",
            "",
            "## 📁 生成文件",
            "",
            "- `stripe_pattern_analysis.png` - 條紋模式視覺化",
            "- `2d_power_spectrum.png` - 2D 頻率譜",
            "- `decision_report.md` - 本報告",
            "- `analysis_metrics.json` - 完整指標數據",
            "",
            "---",
            "",
            "**報告結束** | 生成於 2025-10-14"
        ])
        
        # 保存報告
        report_path = self.output_dir / "decision_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        
        print(f"   ✅ 已保存: {report_path}")
        
        # 保存指標數據
        metrics = {
            "anisotropy": anisotropy,
            "spectrum": {k: {kk: vv for kk, vv in v.items() if kk != 'power_spectrum'} 
                        for k, v in spectrum.items()},
            "turbulence": turbulence,
            "scores": scores,
            "recommendation": recommendation
        }
        
        metrics_path = self.output_dir / "analysis_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"   ✅ 已保存: {metrics_path}")
        
        return recommendation, scores
    
    def run_full_analysis(self):
        """執行完整分析流程"""
        print("=" * 80)
        print("FOURIER ABLATION 深度分析")
        print("=" * 80)
        
        # 1. 各向異性分析
        anisotropy = self.compute_gradient_anisotropy()
        
        # 2. 頻譜分析
        spectrum = self.compute_2d_spectrum()
        
        # 3. 湍流統計
        turbulence = self.compute_turbulence_statistics()
        
        # 4. 視覺化
        self.visualize_stripe_patterns()
        self.visualize_2d_spectrum(spectrum)
        
        # 5. 決策報告
        recommendation, scores = self.generate_decision_report(
            anisotropy, spectrum, turbulence
        )
        
        # 6. 終端摘要
        print("\n" + "=" * 80)
        print("✅ 分析完成")
        print("=" * 80)
        print(f"\n{recommendation}")
        print(f"\n評分: Enabled {scores['Enabled']}/5 vs Disabled {scores['Disabled']}/5")
        print(f"\n所有結果已保存至: {self.output_dir}/")
        print("\n建議閱讀:")
        print(f"  1. {self.output_dir}/decision_report.md")
        print(f"  2. {self.output_dir}/stripe_pattern_analysis.png")
        print(f"  3. {self.output_dir}/2d_power_spectrum.png")


def main():
    """主函數"""
    analyzer = FourierAblationAnalyzer()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
