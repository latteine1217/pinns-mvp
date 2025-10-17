#!/usr/bin/env python3
"""
測試 pyJHTDB 官方庫連接 JHTDB
"""

import sys
import numpy as np
from pathlib import Path

# 添加專案路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 載入環境變數
from dotenv import load_dotenv
env_file = project_root / '.env'
if env_file.exists():
    load_dotenv(env_file)
    print(f"✅ 已載入環境變數: {env_file}")
else:
    print(f"⚠️  未找到 .env 文件: {env_file}")

import os
import pyJHTDB

def test_pyjhtdb_connection():
    """測試 pyJHTDB 連接"""
    
    # 獲取 token
    token = os.getenv('JHTDB_AUTH_TOKEN')
    if not token:
        print("❌ 未找到 JHTDB_AUTH_TOKEN 環境變數")
        return False
    
    print(f"🔑 Token: {token[:20]}...")
    
    try:
        # 初始化 libJHTDB
        print("\n📡 初始化 pyJHTDB.libJHTDB()...")
        lJHTDB = pyJHTDB.libJHTDB()
        lJHTDB.initialize()
        
        # 設置 token
        lJHTDB.add_token(token)
        print("✅ Token 設置成功")
        
        # Channel Flow 數據集（Re_τ=1000）
        dataset = 'channel'
        
        # 獲取有效時間範圍
        print("\n📊 獲取數據集資訊...")
        time_range = pyJHTDB.dbinfo.channel['time']
        print(f"   Channel 時間範圍: [{time_range[0]:.4f}, {time_range[-1]:.4f}]")
        print(f"   總時間步數: {len(time_range)}")
        
        # 使用中間時間點
        time = float(time_range[len(time_range)//2])
        print(f"   使用時間: {time:.4f}")
        
        # 空間和時間插值設定（參考官方範例）
        spatialInterp = 6   # 6 點 Lagrange 插值
        temporalInterp = 0  # 無時間插值
        
        # 測試 1: 獲取單個時間步的速度數據（使用簡單座標）
        print("\n🧪 測試 1: 獲取單點速度數據")
        
        # 測試點（物理座標）
        # Channel Flow: x ∈ [0, 8π], y ∈ [-1, 1], z ∈ [0, 3π]
        x = 4.0  # 中心位置
        y = 0.0  # 中心高度
        z = 4.0  # 中心位置
        
        print(f"   位置: ({x}, {y}, {z}), 時間: {time:.4f}")
        
        # 使用 getData 方法（單點查詢）
        result = lJHTDB.getData(
            time, 
            np.array([[x, y, z]], dtype=np.float32),
            sinterp=spatialInterp,  # 6 點 Lagrange 插值
            tinterp=temporalInterp,  # 無時間插值
            data_set=dataset,
            getFunction='getVelocity'
        )
        
        print(f"✅ 速度數據: u={result[0,0]:.6f}, v={result[0,1]:.6f}, w={result[0,2]:.6f}")
        
        # 測試 2: 獲取多點數據
        print("\n🧪 測試 2: 獲取多點速度數據（5 個點）")
        
        # 生成隨機點（在合理範圍內）
        np.random.seed(42)
        n_points = 5
        points = np.zeros((n_points, 3), dtype=np.float32)
        points[:, 0] = np.random.uniform(0.0, 8*np.pi, n_points)  # x
        points[:, 1] = np.random.uniform(-0.9, 0.9, n_points)     # y
        points[:, 2] = np.random.uniform(0.0, 3*np.pi, n_points)  # z
        
        print(f"   點 1: ({points[0,0]:.3f}, {points[0,1]:.3f}, {points[0,2]:.3f})")
        print(f"   點 2: ({points[1,0]:.3f}, {points[1,1]:.3f}, {points[1,2]:.3f})")
        print(f"   ...")
        
        result = lJHTDB.getData(
            time,
            points,
            sinterp=spatialInterp,
            tinterp=temporalInterp,
            data_set=dataset,
            getFunction='getVelocity'
        )
        
        print(f"✅ 獲取 {result.shape[0]} 個點的速度數據")
        print(f"   點 1 速度: ({result[0,0]:.6f}, {result[0,1]:.6f}, {result[0,2]:.6f})")
        print(f"   點 2 速度: ({result[1,0]:.6f}, {result[1,1]:.6f}, {result[1,2]:.6f})")
        
        # 測試 3: 獲取壓力數據
        print("\n🧪 測試 3: 獲取壓力數據")
        
        result_p = lJHTDB.getData(
            time,
            np.array([[x, y, z]], dtype=np.float32),
            sinterp=spatialInterp,
            tinterp=temporalInterp,
            data_set=dataset,
            getFunction='getPressure'
        )
        
        print(f"✅ 壓力數據: p={result_p[0,0]:.6f}")
        
        # 測試 4: 檢查數據合理性
        print("\n🧪 測試 4: 數據合理性檢查")
        
        # Channel Flow Re_tau=1000 的典型速度範圍
        # U_b ~ 1.0, U_c ~ 1.13, u_tau ~ 0.05
        # 速度應該在 [-0.5, 1.5] 範圍內
        
        u_mag = np.linalg.norm(result, axis=1)
        print(f"   速度大小範圍: [{u_mag.min():.6f}, {u_mag.max():.6f}]")
        
        if u_mag.max() > 5.0:
            print("⚠️  警告: 速度值異常高（可能是量綱問題）")
        elif u_mag.max() < 0.01:
            print("⚠️  警告: 速度值異常低")
        else:
            print("✅ 速度值在合理範圍內")
        
        print("\n" + "="*60)
        print("🎉 所有測試通過！pyJHTDB 連接成功！")
        print("="*60)
        
        # 清理
        lJHTDB.finalize()
        
        return True
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pyjhtdb_connection()
    sys.exit(0 if success else 1)
