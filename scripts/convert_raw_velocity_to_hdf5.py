"""
將 JHTDB Raw Binary 速度數據轉換為標準 HDF5 格式

用途：
- 處理從 JHTDB 下載的 raw binary 速度檔案（沒有 HDF5 封裝）
- 生成與壓力場格式一致的 HDF5 檔案

使用方式：
    python scripts/convert_raw_velocity_to_hdf5.py
"""

import numpy as np
import h5py
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_raw_velocity_to_hdf5(
    raw_file: str,
    pressure_file: str,  # 用於獲取座標和網格形狀
    output_file: str,
    shape: tuple = None,  # 如果已知形狀可直接指定 (nz, ny, nx, 3)
):
    """
    轉換 raw binary 速度數據為 HDF5 格式
    
    Args:
        raw_file: Raw binary 速度檔案路徑
        pressure_file: 壓力場 HDF5 檔案（用於讀取座標）
        output_file: 輸出 HDF5 檔案路徑
        shape: 數據形狀 (nz, ny, nx, 3)，如果為 None 則從壓力場推斷
    """
    logger.info("=== 開始轉換 Raw Binary 速度數據 ===")
    
    # 1. 從壓力場讀取座標和網格資訊
    logger.info(f"從壓力場讀取座標: {pressure_file}")
    with h5py.File(pressure_file, 'r') as f:
        xcoor = f['xcoor'][:]
        ycoor = f['ycoor'][:]
        zcoor = f['zcoor'][:]
        
        # 獲取網格形狀
        if shape is None:
            # 從壓力場推斷（假設壓力場為 [nz, ny, nx, 1]）
            pressure_key = [k for k in f.keys() if 'pressure' in k.lower()][0]
            p_shape = f[pressure_key].shape
            shape = (p_shape[0], p_shape[1], p_shape[2], 3)  # 速度有 3 個分量
            logger.info(f"從壓力場推斷速度場形狀: {shape}")
    
    logger.info(f"網格資訊:")
    logger.info(f"  X: {len(xcoor)} 點, [{xcoor.min():.4f}, {xcoor.max():.4f}]")
    logger.info(f"  Y: {len(ycoor)} 點, [{ycoor.min():.4f}, {ycoor.max():.4f}]")
    logger.info(f"  Z: {len(zcoor)} 點, [{zcoor.min():.4f}, {zcoor.max():.4f}]")
    
    # 2. 讀取 raw binary 數據（跳過前 4 bytes header）
    logger.info(f"讀取 raw binary 檔案: {raw_file}")
    expected_size = np.prod(shape)
    
    try:
        with open(raw_file, 'rb') as f:
            # 跳過前 4 bytes（可能是檔案標識或 padding）
            f.seek(4)
            velocity_data = np.fromfile(f, dtype=np.float32, count=expected_size)
        logger.info(f"✅ 成功讀取 {len(velocity_data)} 個數值（從 offset=4 開始）")
    except Exception as e:
        logger.error(f"❌ 讀取失敗: {e}")
        return False
    
    # 檢查數據大小是否匹配
    if len(velocity_data) != expected_size:
        logger.warning(f"數據大小不匹配:")
        logger.warning(f"  期望: {expected_size}")
        logger.warning(f"  實際: {len(velocity_data)}")
        
        # 嘗試重塑到最接近的形狀
        if len(velocity_data) % 3 == 0:
            total_points = len(velocity_data) // 3
            logger.info(f"嘗試重塑為 ({total_points}, 3) 然後 reshape...")
            velocity_data = velocity_data[:expected_size]  # 截斷到期望大小
        else:
            logger.error("無法重塑數據，大小不符合預期")
            return False
    
    # 3. 重塑為 [nz, ny, nx, 3]
    try:
        velocity_data = velocity_data.reshape(shape)
        logger.info(f"✅ 重塑為形狀: {velocity_data.shape}")
    except Exception as e:
        logger.error(f"❌ 重塑失敗: {e}")
        return False
    
    # 4. 統計檢查
    logger.info(f"速度場統計:")
    logger.info(f"  U (分量 0): [{velocity_data[..., 0].min():.4f}, {velocity_data[..., 0].max():.4f}], "
               f"mean={velocity_data[..., 0].mean():.4f}")
    logger.info(f"  V (分量 1): [{velocity_data[..., 1].min():.4f}, {velocity_data[..., 1].max():.4f}], "
               f"mean={velocity_data[..., 1].mean():.4f}")
    logger.info(f"  W (分量 2): [{velocity_data[..., 2].min():.4f}, {velocity_data[..., 2].max():.4f}], "
               f"mean={velocity_data[..., 2].mean():.4f}")
    
    # 檢查是否有異常值
    if np.any(np.isnan(velocity_data)):
        logger.warning(f"⚠️ 發現 NaN 值: {np.sum(np.isnan(velocity_data))} 個")
    if np.any(np.isinf(velocity_data)):
        logger.warning(f"⚠️ 發現 Inf 值: {np.sum(np.isinf(velocity_data))} 個")
    if np.abs(velocity_data).max() > 10:
        logger.warning(f"⚠️ 速度值可能異常大: max={np.abs(velocity_data).max():.4f}")
    
    # 5. 寫入 HDF5 檔案
    logger.info(f"寫入 HDF5 檔案: {output_file}")
    try:
        with h5py.File(output_file, 'w') as f:
            # 創建座標數據集（與壓力場一致）
            f.create_dataset('xcoor', data=xcoor, dtype=np.float32)
            f.create_dataset('ycoor', data=ycoor, dtype=np.float32)
            f.create_dataset('zcoor', data=zcoor, dtype=np.float32)
            
            # 創建速度場數據集（命名格式與壓力場一致）
            f.create_dataset(
                'Velocity_0001',
                data=velocity_data,
                dtype=np.float32,
                compression='gzip',
                compression_opts=4  # 適度壓縮
            )
            
            # 添加元數據
            f.attrs['description'] = 'JHTDB Channel Flow Re_tau=1000 Velocity Field'
            f.attrs['source'] = 'Converted from raw binary format'
            f.attrs['grid_shape'] = f"{shape[2]}x{shape[1]}x{shape[0]}"  # nx x ny x nz
            f.attrs['components'] = 'u, v, w'
        
        logger.info(f"✅ 成功寫入 HDF5 檔案")
        
    except Exception as e:
        logger.error(f"❌ 寫入失敗: {e}")
        return False
    
    # 6. 驗證生成的檔案
    logger.info("驗證生成的 HDF5 檔案...")
    try:
        with h5py.File(output_file, 'r') as f:
            logger.info(f"  Keys: {list(f.keys())}")
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    logger.info(f"  {key}: shape={f[key].shape}, dtype={f[key].dtype}")
        logger.info("✅ 檔案驗證通過")
        
    except Exception as e:
        logger.error(f"❌ 驗證失敗: {e}")
        return False
    
    logger.info("=== 轉換完成 ===")
    return True


if __name__ == "__main__":
    # 配置檔案路徑
    data_dir = Path("data/jhtdb/channel_flow_re1000/raw")
    
    # 輸入檔案
    raw_velocity_file = data_dir / "SciServer Channel.h5"  # Raw binary 速度檔案
    pressure_file = data_dir / "JHU Turbulence Channel_t1_pressure.h5"  # 壓力場（用於座標）
    
    # 輸出檔案
    output_velocity_file = data_dir / "channel_t1_velocity_converted.h5"
    
    # 檢查檔案是否存在
    if not raw_velocity_file.exists():
        logger.error(f"Raw velocity 檔案不存在: {raw_velocity_file}")
        exit(1)
    
    if not pressure_file.exists():
        logger.error(f"壓力場檔案不存在: {pressure_file}")
        exit(1)
    
    # 執行轉換
    success = convert_raw_velocity_to_hdf5(
        raw_file=str(raw_velocity_file),
        pressure_file=str(pressure_file),
        output_file=str(output_velocity_file)
    )
    
    if success:
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ 轉換成功！")
        logger.info(f"輸出檔案: {output_velocity_file}")
        logger.info(f"{'='*60}")
        logger.info(f"\n下一步：")
        logger.info(f"1. 使用 JHTDBCutoutLoader 測試載入:")
        logger.info(f"   python pinnx/dataio/jhtdb_cutout_loader.py")
        logger.info(f"\n2. 或手動重命名為標準名稱:")
        logger.info(f"   mv '{output_velocity_file}' '{data_dir / 'channel_t1_velocity.h5'}'")
    else:
        logger.error("❌ 轉換失敗，請檢查上述錯誤訊息")
        exit(1)
