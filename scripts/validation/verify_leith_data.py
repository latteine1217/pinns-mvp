#!/usr/bin/env python3
"""
LES 數據驗證腳本
檢查 LES 湍流模型數據文件是否存在且格式正確
"""

import os
import sys
import h5py
import numpy as np

def check_les_field():
    """檢查 LES 場數據文件"""
    les_file = 'data/lowfi/kolmogorov_rans/rans_re50_kf4_les.h5'
    
    print("=" * 70)
    print("📊 檢查 LES 場數據")
    print("=" * 70)
    
    if not os.path.exists(les_file):
        print(f"❌ 文件不存在: {les_file}")
        print("\n💡 建議：")
        print("   運行 LES 場生成腳本創建數據文件")
        return False
    
    print(f"✅ 文件存在: {les_file}")
    print(f"   大小: {os.path.getsize(les_file) / 1e6:.2f} MB\n")
    
    # 檢查 HDF5 結構
    try:
        with h5py.File(les_file, 'r') as f:
            # 檢查群組
            if 'mean_field' not in f:
                print("❌ 缺少必要群組: /mean_field")
                return False
            print("✅ 群組結構正確: /mean_field")
            
            # 檢查必要變量
            required_vars = ['u', 'v', 'nu_t', 'x', 'y']
            mean_field = f['mean_field']
            
            missing = [var for var in required_vars if var not in mean_field]
            if missing:
                print(f"❌ 缺少變量: {', '.join(missing)}")
                return False
            
            print(f"✅ 變量完整: {', '.join(required_vars)}\n")
            
            # 檢查變量形狀與數值
            u = mean_field['u'][:]
            v = mean_field['v'][:]
            nu_t = mean_field['nu_t'][:]
            x = mean_field['x'][:]
            y = mean_field['y'][:]
            
            print("📐 數據形狀:")
            print(f"   u:    {u.shape}")
            print(f"   v:    {v.shape}")
            print(f"   nu_t: {nu_t.shape}")
            print(f"   x:    {x.shape} (應為 1D)")
            print(f"   y:    {y.shape} (應為 1D)\n")
            
            # 驗證 1D 座標
            if x.ndim != 1 or y.ndim != 1:
                print(f"❌ 座標格式錯誤：x 為 {x.ndim}D, y 為 {y.ndim}D（應為 1D）")
                return False
            print("✅ 座標格式正確: 1D arrays\n")
            
            # 驗證場變量形狀一致
            if not (u.shape == v.shape == nu_t.shape):
                print("❌ 場變量形狀不一致")
                return False
            print("✅ 場變量形狀一致\n")
            
            # 檢查數值範圍
            print("📊 數值範圍:")
            print(f"   u:    [{u.min():.4f}, {u.max():.4f}]")
            print(f"   v:    [{v.min():.4f}, {v.max():.4f}]")
            print(f"   nu_t: [{nu_t.min():.4e}, {nu_t.max():.4e}]")
            print(f"   x:    [{x.min():.4f}, {x.max():.4f}]")
            print(f"   y:    [{y.min():.4f}, {y.max():.4f}]\n")
            
            # 檢查 NaN/Inf
            if np.any(np.isnan(u)) or np.any(np.isnan(v)) or np.any(np.isnan(nu_t)):
                print("❌ 數據包含 NaN")
                return False
            if np.any(np.isinf(u)) or np.any(np.isinf(v)) or np.any(np.isinf(nu_t)):
                print("❌ 數據包含 Inf")
                return False
            print("✅ 數據無 NaN/Inf\n")
            
            # 檢查 nu_t 非負性
            if np.any(nu_t < 0):
                print("⚠️  渦黏度包含負值（物理上應為正）")
            else:
                print("✅ 渦黏度非負性正確\n")
            
            # 檢查不應存在的 k-ε 變量
            forbidden_vars = ['k', 'epsilon']
            found_forbidden = [var for var in forbidden_vars if var in mean_field]
            if found_forbidden:
                print(f"⚠️  發現 k-ε 變量: {', '.join(found_forbidden)}")
                print("   LES 模型不應包含這些變量\n")
            else:
                print("✅ 無 k-ε 殘留變量\n")
            
    except Exception as e:
        print(f"❌ 讀取文件時出錯: {e}")
        return False
    
    return True


def check_les_sensors():
    """檢查 LES Sensor 文件"""
    sensor_file = 'data/lowfi/kolmogorov_rans/sensors_K100_les.npz'
    
    print("=" * 70)
    print("📍 檢查 LES Sensor 數據")
    print("=" * 70)
    
    if not os.path.exists(sensor_file):
        print(f"❌ 文件不存在: {sensor_file}")
        print("\n💡 建議：")
        print("   運行 QR-Pivot sensor 生成腳本創建感測點")
        return False
    
    print(f"✅ 文件存在: {sensor_file}")
    print(f"   大小: {os.path.getsize(sensor_file) / 1e3:.2f} KB\n")
    
    # 檢查 NPZ 結構
    try:
        sensors = np.load(sensor_file, allow_pickle=True)
        
        # 檢查必要鍵
        required_keys = ['K', 'method', 'source', 'sensor_x', 'sensor_y', 'metrics']
        missing = [key for key in required_keys if key not in sensors]
        if missing:
            print(f"❌ 缺少鍵: {', '.join(missing)}")
            return False
        
        print(f"✅ 鍵完整: {', '.join(required_keys)}\n")
        
        # 檢查數值
        K = int(sensors['K'])
        method = str(sensors['method'])
        source = str(sensors['source'])
        sensor_x = sensors['sensor_x']
        sensor_y = sensors['sensor_y']
        metrics = sensors['metrics'].item()
        
        print("📊 Sensor 配置:")
        print(f"   K:      {K}")
        print(f"   Method: {method}")
        print(f"   Source: {source}")
        print(f"   X 範圍: [{sensor_x.min():.4f}, {sensor_x.max():.4f}]")
        print(f"   Y 範圍: [{sensor_y.min():.4f}, {sensor_y.max():.4f}]\n")
        
        # 檢查感測點數量
        if len(sensor_x) != K or len(sensor_y) != K:
            print(f"❌ 感測點數量不匹配: K={K}, len(x)={len(sensor_x)}, len(y)={len(sensor_y)}")
            return False
        print(f"✅ 感測點數量正確: {K}\n")
        
        # 檢查條件數
        cond = metrics.get('condition_number', None)
        if cond is None:
            print("⚠️  未找到條件數")
        else:
            print(f"📐 條件數: {cond:.2e}")
            if cond < 100:
                print("   ✅ 優秀 (< 100)")
            elif cond < 500:
                print("   ✅ 良好 (100-500)")
            else:
                print("   ⚠️  可接受 (> 500，建議重新生成)")
        print()
        
        # 檢查 source 標識
        if 'les' not in source.lower():
            print(f"⚠️  Source 不包含 'les': {source}")
            print("   建議確認是否為 LES 模型生成\n")
        else:
            print("✅ Source 標識正確\n")
            
    except Exception as e:
        print(f"❌ 讀取文件時出錯: {e}")
        return False
    
    return True


def main():
    """主檢查流程"""
    print("\n" + "=" * 70)
    print("🔍 LES 數據驗證工具")
    print("=" * 70)
    print("檢查 Kolmogorov Flow LES 湍流模型數據的完整性與格式\n")
    
    field_ok = check_les_field()
    sensor_ok = check_les_sensors()
    
    print("=" * 70)
    print("📋 驗證總結")
    print("=" * 70)
    print(f"LES 場數據:   {'✅ 通過' if field_ok else '❌ 失敗'}")
    print(f"LES Sensor:   {'✅ 通過' if sensor_ok else '❌ 失敗'}")
    print("=" * 70)
    
    if field_ok and sensor_ok:
        print("\n🎉 所有驗證通過！可以開始訓練。\n")
        print("📝 訓練命令：")
        print("   python scripts/train/train.py \\")
        print("     --cfg configs/kolmogorov_re50_kf4_K100.yml \\")
        print("     --device cuda")
        print()
        return 0
    else:
        print("\n⚠️  驗證失敗，請先生成必要的數據文件。")
        print("\n📝 數據生成步驟（示例）：")
        print("   1. 生成 LES 場：")
        print("      python scripts/generate/generate_les_field.py \\")
        print("        --Re 50 --kf 4 \\")
        print("        --output data/lowfi/kolmogorov_rans/rans_re50_kf4_les.h5")
        print()
        print("   2. 生成 QR Sensor：")
        print("      python scripts/generate/sensors/generate_sensors_qr_les.py \\")
        print("        --les-file data/lowfi/kolmogorov_rans/rans_re50_kf4_les.h5 \\")
        print("        --K 100 \\")
        print("        --output data/lowfi/kolmogorov_rans/sensors_K100_les.npz")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())
