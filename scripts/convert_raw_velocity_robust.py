"""
JHTDB Raw Binary 速度數據轉換工具 (Robust 版本)

特點:
- 檢測並過濾異常值/NaN
- 支援損壞或不完整的數據檔案
- 輸出詳細診斷報告
- 生成品質報告與視覺化

使用方式:
    python scripts/convert_raw_velocity_robust.py
"""

import numpy as np
import h5py
from pathlib import Path
import logging
import json
from typing import Tuple, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_raw_velocity_file(
    raw_file: str,
    max_samples: int = 1000000
) -> Dict[str, Any]:
    """
    分析 raw binary 速度檔案的數據品質
    
    Args:
        raw_file: Raw binary 檔案路徑
        max_samples: 最大取樣點數（避免記憶體溢出）
        
    Returns:
        分析結果字典
    """
    logger.info(f"=== 分析檔案: {raw_file} ===")
    
    # 讀取檔案
    file_size = Path(raw_file).stat().st_size
    total_floats = file_size // 4
    usable_floats = (total_floats // 3) * 3
    
    logger.info(f"檔案大小: {file_size:,} bytes")
    logger.info(f"總 float32 數量: {total_floats:,}")
    logger.info(f"可用數據 (3的倍數): {usable_floats:,} = {usable_floats//3:,} 點 × 3 分量")
    
    # 讀取數據（可能很大，分塊處理）
    data = np.fromfile(raw_file, dtype=np.float32, count=usable_floats)
    velocity = data.reshape(-1, 3)
    
    n_points = len(velocity)
    logger.info(f"重塑為: {velocity.shape}")
    
    # 分析每個分量
    analysis = {
        'file_size': file_size,
        'total_floats': total_floats,
        'n_spatial_points': n_points,
        'components': {}
    }
    
    for comp_idx, comp_name in enumerate(['U', 'V', 'W']):
        comp_data = velocity[:, comp_idx]
        
        # 基本統計
        n_nan = int(np.sum(np.isnan(comp_data)))
        n_inf = int(np.sum(np.isinf(comp_data)))
        
        # 正常值定義：有限且 |val| < 2.0 (通道流速度不應超過此值)
        normal_mask = np.isfinite(comp_data) & (np.abs(comp_data) < 2.0)
        normal_vals = comp_data[normal_mask]
        n_normal = len(normal_vals)
        
        comp_analysis = {
            'n_nan': n_nan,
            'n_inf': n_inf,
            'n_normal': n_normal,
            'normal_ratio': n_normal / n_points,
            'min': float(normal_vals.min()) if len(normal_vals) > 0 else None,
            'max': float(normal_vals.max()) if len(normal_vals) > 0 else None,
            'mean': float(normal_vals.mean()) if len(normal_vals) > 0 else None,
            'std': float(normal_vals.std()) if len(normal_vals) > 0 else None,
        }
        
        analysis['components'][comp_name] = comp_analysis
        
        logger.info(f"\n{comp_name} 分量:")
        logger.info(f"  NaN: {n_nan} ({n_nan/n_points*100:.4f}%)")
        logger.info(f"  Inf: {n_inf} ({n_inf/n_points*100:.4f}%)")
        logger.info(f"  正常值: {n_normal:,} ({n_normal/n_points*100:.2f}%)")
        if n_normal > 0:
            logger.info(f"  範圍: [{comp_analysis['min']:.4f}, {comp_analysis['max']:.4f}]")
            logger.info(f"  均值: {comp_analysis['mean']:.4f} ± {comp_analysis['std']:.4f}")
    
    return analysis


def convert_raw_velocity_robust(
    raw_file: str,
    pressure_file: str,
    output_file: str,
    output_report: str | None = None,
    filter_outliers: bool = True,
    outlier_threshold: float = 2.0,
) -> bool:
    """
    穩健地轉換 raw binary 速度數據為 HDF5 格式
    
    Args:
        raw_file: Raw binary 速度檔案路徑
        pressure_file: 壓力場 HDF5 檔案（用於讀取座標）
        output_file: 輸出 HDF5 檔案路徑
        output_report: 輸出品質報告檔案 (JSON)
        filter_outliers: 是否過濾異常值
        outlier_threshold: 異常值閾值（|val| > threshold 視為異常）
        
    Returns:
        轉換是否成功
    """
    logger.info("=== 開始穩健轉換 ===")
    
    # 1. 分析原始檔案
    analysis = analyze_raw_velocity_file(raw_file)
    
    # 2. 從壓力場讀取座標
    logger.info(f"\n從壓力場讀取座標: {pressure_file}")
    with h5py.File(pressure_file, 'r') as f:
        xcoor = f['xcoor'][:]
        ycoor = f['ycoor'][:]
        zcoor = f['zcoor'][:]
        
        # 獲取網格形狀
        pressure_key = [k for k in f.keys() if 'pressure' in k.lower()][0]
        p_shape = f[pressure_key].shape
        expected_shape = (p_shape[0], p_shape[1], p_shape[2], 3)
        logger.info(f"期望速度場形狀: {expected_shape}")
    
    logger.info(f"網格資訊:")
    logger.info(f"  X: {len(xcoor)} 點, [{xcoor.min():.4f}, {xcoor.max():.4f}]")
    logger.info(f"  Y: {len(ycoor)} 點, [{ycoor.min():.4f}, {ycoor.max():.4f}]")
    logger.info(f"  Z: {len(zcoor)} 點, [{zcoor.min():.4f}, {zcoor.max():.4f}]")
    
    # 3. 讀取 raw binary 數據
    logger.info(f"\n讀取 raw binary 檔案: {raw_file}")
    file_size = Path(raw_file).stat().st_size
    total_floats = file_size // 4
    usable_floats = (total_floats // 3) * 3
    
    data = np.fromfile(raw_file, dtype=np.float32, count=usable_floats)
    velocity_flat = data.reshape(-1, 3)
    logger.info(f"✅ 讀取 {len(velocity_flat):,} 個空間點")
    
    # 4. 檢查數據大小是否匹配
    expected_size = np.prod(expected_shape)
    actual_size = len(velocity_flat)
    
    if actual_size != expected_size // 3:  # expected_shape 包含 3 分量
        logger.warning(f"⚠️ 數據大小不匹配:")
        logger.warning(f"  期望: {expected_size // 3:,} 點")
        logger.warning(f"  實際: {actual_size:,} 點")
        logger.warning(f"  差距: {expected_size // 3 - actual_size:,} 點")
        
        # 檢查資料完整性
        completeness = actual_size / (expected_size // 3)
        logger.warning(f"  完整度: {completeness*100:.2f}%")
        
        if completeness < 0.85:
            logger.error("❌ 數據完整度過低 (<85%)，無法轉換")
            return False
    
    # 5. 過濾異常值（如果啟用）
    if filter_outliers:
        logger.info(f"\n過濾異常值 (閾值: |val| < {outlier_threshold})...")
        
        # 創建遮罩：所有分量都必須正常
        normal_mask = np.all(np.isfinite(velocity_flat) & (np.abs(velocity_flat) < outlier_threshold), axis=1)
        
        n_outliers = np.sum(~normal_mask)
        logger.info(f"  異常點數: {n_outliers:,} ({n_outliers/len(velocity_flat)*100:.2f}%)")
        logger.info(f"  正常點數: {np.sum(normal_mask):,} ({np.sum(normal_mask)/len(velocity_flat)*100:.2f}%)")
        
        if n_outliers > len(velocity_flat) * 0.20:  # 超過20%異常
            logger.error(f"❌ 異常值比例過高 (>{20}%)，數據品質不佳")
            logger.error(f"建議檢查原始數據來源或重新下載")
            return False
        
        # 對於無法 reshape 的情況，填充 NaN
        velocity_clean = velocity_flat.copy()
        velocity_clean[~normal_mask] = np.nan
    else:
        velocity_clean = velocity_flat
    
    # 6. 嘗試 reshape 到目標形狀
    target_shape = (p_shape[0], p_shape[1], p_shape[2], 3)
    
    if len(velocity_clean) == np.prod(target_shape) // 3:
        # 可以完美 reshape
        velocity_reshaped = velocity_clean.reshape(target_shape)
        logger.info(f"✅ 重塑為形狀: {velocity_reshaped.shape}")
    else:
        # 無法完美 reshape，創建 NaN 填充的數組
        logger.warning(f"⚠️ 無法完美 reshape，使用 NaN 填充")
        velocity_reshaped = np.full(target_shape, np.nan, dtype=np.float32)
        
        # 填充可用的數據
        available_points = min(len(velocity_clean), np.prod(target_shape) // 3)
        velocity_reshaped.reshape(-1, 3)[:available_points] = velocity_clean[:available_points]
        
        logger.info(f"  填充: {available_points:,} / {np.prod(target_shape)//3:,} 點")
    
    # 7. 統計檢查
    logger.info(f"\n最終速度場統計:")
    for i, comp in enumerate(['U', 'V', 'W']):
        comp_data = velocity_reshaped[..., i]
        finite_mask = np.isfinite(comp_data)
        finite_vals = comp_data[finite_mask]
        
        logger.info(f"  {comp}: [{finite_vals.min():8.4f}, {finite_vals.max():8.4f}], "
                   f"mean={finite_vals.mean():8.4f}, "
                   f"有效率={np.sum(finite_mask)/comp_data.size*100:.2f}%")
    
    # 8. 寫入 HDF5 檔案
    logger.info(f"\n寫入 HDF5 檔案: {output_file}")
    try:
        with h5py.File(output_file, 'w') as f:
            # 座標
            f.create_dataset('xcoor', data=xcoor, dtype=np.float32)
            f.create_dataset('ycoor', data=ycoor, dtype=np.float32)
            f.create_dataset('zcoor', data=zcoor, dtype=np.float32)
            
            # 速度場
            f.create_dataset(
                'Velocity_0001',
                data=velocity_reshaped,
                dtype=np.float32,
                compression='gzip',
                compression_opts=4
            )
            
            # 元數據
            f.attrs['description'] = 'JHTDB Channel Flow Re_tau=1000 Velocity Field (Robust Conversion)'
            f.attrs['source'] = 'Converted from raw binary format with outlier filtering'
            f.attrs['grid_shape'] = f"{target_shape[2]}x{target_shape[1]}x{target_shape[0]}"
            f.attrs['components'] = 'u, v, w'
            f.attrs['outlier_threshold'] = outlier_threshold
            f.attrs['filter_outliers'] = filter_outliers
            
        logger.info(f"✅ 成功寫入 HDF5 檔案")
        
    except Exception as e:
        logger.error(f"❌ 寫入失敗: {e}")
        return False
    
    # 9. 輸出品質報告
    if output_report:
        report = {
            'input_file': str(raw_file),
            'output_file': str(output_file),
            'analysis': analysis,
            'conversion': {
                'filter_outliers': filter_outliers,
                'outlier_threshold': outlier_threshold,
                'target_shape': list(target_shape),
                'success': True,
            }
        }
        
        with open(output_report, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"✅ 品質報告已保存: {output_report}")
    
    logger.info("=== 轉換完成 ===")
    return True


if __name__ == "__main__":
    # 配置檔案路徑
    data_dir = Path("data/jhtdb/channel_flow_re1000/raw")
    
    # 輸入檔案
    raw_velocity_file = data_dir / "SciServer Channel.h5"
    pressure_file = data_dir / "JHU Turbulence Channel_t1_pressure.h5"
    
    # 輸出檔案
    output_velocity_file = data_dir / "channel_t1_velocity_converted_robust.h5"
    output_report_file = data_dir / "velocity_conversion_report.json"
    
    # 檢查檔案是否存在
    if not raw_velocity_file.exists():
        logger.error(f"❌ Raw velocity 檔案不存在: {raw_velocity_file}")
        exit(1)
    
    if not pressure_file.exists():
        logger.error(f"❌ 壓力場檔案不存在: {pressure_file}")
        exit(1)
    
    # 執行轉換
    logger.info(f"\n{'='*70}")
    logger.info(f"JHTDB Raw Binary 速度數據穩健轉換工具")
    logger.info(f"{'='*70}\n")
    
    success = convert_raw_velocity_robust(
        raw_file=str(raw_velocity_file),
        pressure_file=str(pressure_file),
        output_file=str(output_velocity_file),
        output_report=str(output_report_file),
        filter_outliers=True,
        outlier_threshold=2.0,
    )
    
    if success:
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ 轉換成功！")
        logger.info(f"{'='*70}")
        logger.info(f"\n輸出檔案:")
        logger.info(f"  HDF5: {output_velocity_file}")
        logger.info(f"  報告: {output_report_file}")
        logger.info(f"\n⚠️  注意事項:")
        logger.info(f"  - 原始數據中約有 14% 的異常值已被過濾")
        logger.info(f"  - 異常點已被設為 NaN")
        logger.info(f"  - 建議在使用前檢查品質報告")
        logger.info(f"\n下一步:")
        logger.info(f"  1. 查看品質報告: cat {output_report_file}")
        logger.info(f"  2. 測試載入: python pinnx/dataio/jhtdb_cutout_loader.py")
        logger.info(f"  3. 或考慮從 JHTDB 重新下載完整數據")
    else:
        logger.error("\n❌ 轉換失敗，請檢查上述錯誤訊息")
        logger.error("建議:")
        logger.error("  1. 檢查原始檔案是否損壞")
        logger.error("  2. 嘗試從 JHTDB 重新下載")
        logger.error("  3. 聯繫數據提供者確認檔案格式")
        exit(1)
