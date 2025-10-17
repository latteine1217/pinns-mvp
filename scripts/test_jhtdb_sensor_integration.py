#!/usr/bin/env python3
"""
完整整合測試：JHTDB 數據載入 + 分層 QR-Pivot 感測點提取

驗證項目：
1. JHTDB HDF5 數據載入（速度場、壓力場、座標）
2. 分層 QR-Pivot 感測點座標驗證
3. 感測點數據提取與插值
4. 物理合理性檢查
5. 數據完整性檢查（無 NaN、無 Inf）

使用方式：
    python scripts/test_jhtdb_sensor_integration.py
"""

import numpy as np
import logging
from pathlib import Path
import json
from typing import Dict, Tuple
import sys

# 添加專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinnx.dataio.jhtdb_cutout_loader import JHTDBCutoutLoader

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegrationTestRunner:
    """整合測試執行器"""
    
    def __init__(self,
                 jhtdb_data_dir: str = "data/jhtdb/channel_flow_re1000/raw",
                 sensor_file: str = "results/qr_pivot_stratified/stratified_qr_K50.npz",
                 output_dir: str = "results/validation"):
        """
        Args:
            jhtdb_data_dir: JHTDB 數據目錄
            sensor_file: 感測點檔案路徑
            output_dir: 輸出目錄
        """
        self.jhtdb_data_dir = Path(jhtdb_data_dir)
        self.sensor_file = Path(sensor_file)
        self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_results = {
            'passed': [],
            'failed': [],
            'warnings': []
        }
    
    def print_section(self, title: str, char: str = '='):
        """打印分隔線"""
        width = 80
        print(f"\n{char * width}")
        print(f"{title:^{width}}")
        print(f"{char * width}\n")
    
    def test_jhtdb_loader(self) -> bool:
        """測試 1: JHTDB 數據載入"""
        self.print_section("測試 1: JHTDB 數據載入", char='=')
        
        try:
            # 初始化載入器
            logger.info(f"初始化 JHTDB 載入器: {self.jhtdb_data_dir}")
            self.loader = JHTDBCutoutLoader(str(self.jhtdb_data_dir))
            
            # 檢查文件
            if self.loader.pressure_file is None:
                raise FileNotFoundError("壓力場文件未找到")
            if self.loader.velocity_file is None:
                raise FileNotFoundError("速度場文件未找到")
            
            logger.info("✅ 文件檢查通過")
            
            # 載入座標
            coords = self.loader.load_coordinates()
            self.grid_shape = self.loader.get_grid_shape()
            self.physical_domain = self.loader.get_physical_domain()
            
            logger.info(f"網格形狀: {self.grid_shape}")
            logger.info(f"物理域: {self.physical_domain}")
            
            # 載入場數據
            state = self.loader.load_full_state()
            
            # 驗證數據完整性
            for var in ['u', 'v', 'w', 'p']:
                if state[var] is None:
                    raise ValueError(f"變數 {var} 載入失敗")
                
                # 檢查 NaN 和 Inf
                n_nan = np.isnan(state[var]).sum()
                n_inf = np.isinf(state[var]).sum()
                
                if n_nan > 0:
                    raise ValueError(f"{var} 包含 {n_nan} 個 NaN 值")
                if n_inf > 0:
                    raise ValueError(f"{var} 包含 {n_inf} 個 Inf 值")
                
                logger.info(f"✅ {var}: 形狀 {state[var].shape}, "
                           f"範圍 [{state[var].min():.4f}, {state[var].max():.4f}]")
            
            self.test_results['passed'].append('JHTDB 數據載入')
            return True
            
        except Exception as e:
            logger.error(f"❌ 測試失敗: {e}")
            self.test_results['failed'].append(f'JHTDB 數據載入: {e}')
            return False
    
    def test_sensor_loading(self) -> bool:
        """測試 2: 感測點載入與驗證"""
        self.print_section("測試 2: 感測點載入與驗證", char='=')
        
        try:
            # 載入感測點
            logger.info(f"載入感測點: {self.sensor_file}")
            sensor_data = np.load(self.sensor_file, allow_pickle=True)
            
            self.sensor_coords = sensor_data['coords']  # [K, 3]
            self.sensor_metadata = sensor_data['metadata'].item()
            
            K = len(self.sensor_coords)
            logger.info(f"感測點數量: {K}")
            logger.info(f"方法: {self.sensor_metadata['method']}")
            logger.info(f"分層策略: {self.sensor_metadata['stratification']}")
            
            # 驗證座標範圍
            x_coords = self.sensor_coords[:, 0]
            y_coords = self.sensor_coords[:, 1]
            z_coords = self.sensor_coords[:, 2]
            
            logger.info(f"座標範圍:")
            logger.info(f"  X: [{x_coords.min():.4f}, {x_coords.max():.4f}]")
            logger.info(f"  Y: [{y_coords.min():.4f}, {y_coords.max():.4f}]")
            logger.info(f"  Z: [{z_coords.min():.4f}, {z_coords.max():.4f}]")
            
            # 檢查是否在 JHTDB 物理域內
            x_in_range = (x_coords >= self.physical_domain['x'][0]) & \
                        (x_coords <= self.physical_domain['x'][1])
            y_in_range = (y_coords >= self.physical_domain['y'][0]) & \
                        (y_coords <= self.physical_domain['y'][1])
            z_in_range = (z_coords >= self.physical_domain['z'][0]) & \
                        (z_coords <= self.physical_domain['z'][1])
            
            all_in_range = x_in_range & y_in_range & z_in_range
            
            if not all_in_range.all():
                n_out = (~all_in_range).sum()
                raise ValueError(f"{n_out} 個感測點超出 JHTDB 物理域範圍")
            
            logger.info(f"✅ 所有 {K} 個感測點座標都在物理域內")
            
            self.test_results['passed'].append('感測點載入與驗證')
            return True
            
        except Exception as e:
            logger.error(f"❌ 測試失敗: {e}")
            self.test_results['failed'].append(f'感測點載入與驗證: {e}')
            return False
    
    def test_sensor_extraction(self) -> bool:
        """測試 3: 感測點數據提取"""
        self.print_section("測試 3: 感測點數據提取", char='=')
        
        try:
            # 提取感測點數據
            logger.info(f"從 JHTDB 場數據提取 {len(self.sensor_coords)} 個感測點...")
            
            extracted = self.loader.extract_points(
                self.sensor_coords,
                variables=['u', 'v', 'w', 'p']
            )
            
            self.extracted_data = extracted
            
            # 驗證提取結果
            for var in ['u', 'v', 'w', 'p']:
                if var not in extracted:
                    raise ValueError(f"變數 {var} 未提取")
                
                values = extracted[var]
                
                # 檢查形狀
                if values.shape[0] != len(self.sensor_coords):
                    raise ValueError(f"{var} 形狀錯誤: {values.shape}")
                
                # 檢查 NaN 和 Inf
                n_nan = np.isnan(values).sum()
                n_inf = np.isinf(values).sum()
                
                if n_nan > 0:
                    raise ValueError(f"{var} 包含 {n_nan} 個 NaN 值")
                if n_inf > 0:
                    raise ValueError(f"{var} 包含 {n_inf} 個 Inf 值")
                
                logger.info(f"✅ {var}: 形狀 {values.shape}, "
                           f"範圍 [{np.nanmin(values):.6f}, {np.nanmax(values):.6f}], "
                           f"均值 {np.nanmean(values):.6f}")
            
            self.test_results['passed'].append('感測點數據提取')
            return True
            
        except Exception as e:
            logger.error(f"❌ 測試失敗: {e}")
            self.test_results['failed'].append(f'感測點數據提取: {e}')
            return False
    
    def test_physical_validity(self) -> bool:
        """測試 4: 物理合理性檢查"""
        self.print_section("測試 4: 物理合理性檢查", char='=')
        
        try:
            u = self.extracted_data['u']
            v = self.extracted_data['v']
            w = self.extracted_data['w']
            p = self.extracted_data['p']
            
            # 通道流物理特徵檢查
            checks_passed = []
            warnings = []
            
            # 1. U（流向）應主導且 > 0
            if u.mean() > 0:
                checks_passed.append("U 均值為正（順流方向）")
            else:
                warnings.append(f"警告: U 均值為 {u.mean():.6f}（應 > 0）")
            
            # 2. V 和 W 應近乎對稱（均值接近 0）
            v_mean = abs(v.mean())
            w_mean = abs(w.mean())
            
            if v_mean < 0.05:
                checks_passed.append(f"V 均值接近 0 ({v.mean():.6f})")
            else:
                warnings.append(f"警告: V 均值偏離 0 ({v.mean():.6f})")
            
            if w_mean < 0.05:
                checks_passed.append(f"W 均值接近 0 ({w.mean():.6f})")
            else:
                warnings.append(f"警告: W 均值偏離 0 ({w.mean():.6f})")
            
            # 3. 速度量級檢查（Re_tau=1000 通道流）
            if u.max() > 0 and u.max() < 2.0:
                checks_passed.append(f"U 最大值合理 ({u.max():.4f})")
            else:
                warnings.append(f"警告: U 最大值異常 ({u.max():.4f})")
            
            # 4. 壓力波動檢查
            p_std = p.std()
            if p_std > 0:
                checks_passed.append(f"壓力場有波動 (std={p_std:.6f})")
            else:
                warnings.append("警告: 壓力場無波動（可能異常）")
            
            # 輸出結果
            logger.info("物理合理性檢查結果:")
            for check in checks_passed:
                logger.info(f"  ✅ {check}")
            
            for warning in warnings:
                logger.warning(f"  ⚠️ {warning}")
                self.test_results['warnings'].append(warning)
            
            if warnings:
                self.test_results['passed'].append('物理合理性檢查（有警告）')
            else:
                self.test_results['passed'].append('物理合理性檢查')
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 測試失敗: {e}")
            self.test_results['failed'].append(f'物理合理性檢查: {e}')
            return False
    
    def save_results(self):
        """保存測試結果"""
        self.print_section("保存測試結果", char='=')
        
        # 保存提取的感測點數據
        output_npz = self.output_dir / "sensor_data_integrated.npz"
        np.savez(
            output_npz,
            sensor_coords=self.sensor_coords,
            u=self.extracted_data['u'],
            v=self.extracted_data['v'],
            w=self.extracted_data['w'],
            p=self.extracted_data['p'],
            metadata=self.sensor_metadata,
            grid_shape=np.array(self.grid_shape),
            physical_domain=self.physical_domain
        )
        logger.info(f"✅ 感測點數據已保存: {output_npz}")
        
        # 保存測試報告
        report = {
            'timestamp': str(np.datetime64('now')),
            'jhtdb_data_dir': str(self.jhtdb_data_dir),
            'sensor_file': str(self.sensor_file),
            'grid_shape': [int(x) for x in self.grid_shape],
            'physical_domain': {
                k: (float(v[0]), float(v[1]))
                for k, v in self.physical_domain.items()
            },
            'n_sensors': len(self.sensor_coords),
            'test_results': self.test_results,
            'extracted_data_stats': {
                var: {
                    'min': float(np.nanmin(self.extracted_data[var])),
                    'max': float(np.nanmax(self.extracted_data[var])),
                    'mean': float(np.nanmean(self.extracted_data[var])),
                    'std': float(np.nanstd(self.extracted_data[var]))
                }
                for var in ['u', 'v', 'w', 'p']
            }
        }
        
        report_file = self.output_dir / "integration_test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"✅ 測試報告已保存: {report_file}")
    
    def print_summary(self):
        """打印測試摘要"""
        self.print_section("測試摘要", char='=')
        
        n_passed = len(self.test_results['passed'])
        n_failed = len(self.test_results['failed'])
        n_warnings = len(self.test_results['warnings'])
        
        print(f"通過: {n_passed}")
        for test in self.test_results['passed']:
            print(f"  ✅ {test}")
        
        if n_failed > 0:
            print(f"\n失敗: {n_failed}")
            for test in self.test_results['failed']:
                print(f"  ❌ {test}")
        
        if n_warnings > 0:
            print(f"\n警告: {n_warnings}")
            for warning in self.test_results['warnings']:
                print(f"  ⚠️ {warning}")
        
        print(f"\n{'=' * 80}")
        if n_failed == 0:
            print("✅ 所有整合測試通過！")
        else:
            print(f"❌ {n_failed} 個測試失敗")
        print(f"{'=' * 80}\n")
    
    def run_all_tests(self):
        """執行所有測試"""
        self.print_section("JHTDB + 分層 QR-Pivot 整合測試", char='=')
        
        # 測試 1: JHTDB 載入
        if not self.test_jhtdb_loader():
            logger.error("JHTDB 載入失敗，後續測試中止")
            self.print_summary()
            return False
        
        # 測試 2: 感測點載入
        if not self.test_sensor_loading():
            logger.error("感測點載入失敗，後續測試中止")
            self.print_summary()
            return False
        
        # 測試 3: 數據提取
        if not self.test_sensor_extraction():
            logger.error("數據提取失敗，後續測試中止")
            self.print_summary()
            return False
        
        # 測試 4: 物理驗證
        self.test_physical_validity()
        
        # 保存結果
        self.save_results()
        
        # 打印摘要
        self.print_summary()
        
        return len(self.test_results['failed']) == 0


def main():
    """主函數"""
    runner = IntegrationTestRunner(
        jhtdb_data_dir="data/jhtdb/channel_flow_re1000/raw",
        sensor_file="results/qr_pivot_stratified/stratified_qr_K50.npz",
        output_dir="results/validation"
    )
    
    success = runner.run_all_tests()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
