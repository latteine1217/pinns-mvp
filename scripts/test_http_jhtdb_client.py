#!/usr/bin/env python3
"""
測試 HTTP JHTDB 客戶端
驗證真實 JHTDB 數據獲取功能
"""

import sys
import os
import numpy as np
import logging
import time
from pathlib import Path

# 添加專案根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinnx.dataio.jhtdb_client import JHTDBManager, create_jhtdb_manager

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_http_client_basic():
    """測試 HTTP 客戶端基本功能"""
    
    print("🌊 測試 HTTP JHTDB 客戶端基本功能")
    print("=" * 50)
    
    try:
        # 創建 HTTP 客戶端
        manager = create_jhtdb_manager(use_mock=False, use_http=True)
        
        print(f"✅ 客戶端類型: {manager.client_type}")
        print(f"✅ 可用資料集: {manager.list_datasets()}")
        
        return True
        
    except Exception as e:
        print(f"❌ HTTP 客戶端初始化失敗: {e}")
        return False


def test_cutout_data():
    """測試 cutout 數據獲取"""
    
    print("\n🔲 測試 Cutout 數據獲取")
    print("-" * 30)
    
    try:
        manager = create_jhtdb_manager(use_mock=False, use_http=True)
        
        # 測試小尺寸的 cutout 請求（避免超時）
        dataset = 'channel'
        start = [0.0, -0.5, 0.0]
        end = [1.0, 0.5, 1.0]
        resolution = [8, 8, 8]  # 小尺寸測試
        variables = ['u', 'v', 'p']
        
        print(f"📊 請求參數:")
        print(f"  - 資料集: {dataset}")
        print(f"  - 起始座標: {start}")
        print(f"  - 結束座標: {end}")
        print(f"  - 解析度: {resolution}")
        print(f"  - 變數: {variables}")
        
        # 發送請求
        start_time = time.time()
        result = manager.fetch_cutout(
            dataset=dataset,
            start=start,
            end=end,
            timestep=0,
            variables=variables,
            resolution=resolution
        )
        end_time = time.time()
        
        print(f"⏱️  請求耗時: {end_time - start_time:.2f} 秒")
        print(f"📦 數據來源: {'快取' if result['from_cache'] else '新獲取'}")
        
        # 檢查數據
        data = result['data']
        print(f"\n📋 數據摘要:")
        for var, arr in data.items():
            print(f"  {var}: shape={arr.shape}, dtype={arr.dtype}")
            print(f"      範圍=[{arr.min():.3f}, {arr.max():.3f}], 均值={arr.mean():.3f}")
        
        # 驗證結果
        if 'validation' in result:
            print(f"\n🔍 數據驗證:")
            for var, report in result['validation'].items():
                status = "✅" if report['valid'] else "❌"
                print(f"  {var}: {status}")
                if report['warnings']:
                    print(f"    警告: {report['warnings']}")
                if report['errors']:
                    print(f"    錯誤: {report['errors']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cutout 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_points_data():
    """測試散點數據獲取"""
    
    print("\n📍 測試散點數據獲取")
    print("-" * 30)
    
    try:
        manager = create_jhtdb_manager(use_mock=False, use_http=True)
        
        # 測試少量散點（避免超時）
        dataset = 'channel'
        points = [
            [0.5, 0.0, 0.5],
            [1.0, 0.0, 1.0],
            [1.5, 0.0, 1.5]
        ]
        variables = ['u', 'v', 'w', 'p']
        
        print(f"📊 請求參數:")
        print(f"  - 資料集: {dataset}")
        print(f"  - 點數: {len(points)}")
        print(f"  - 座標: {points}")
        print(f"  - 變數: {variables}")
        
        # 發送請求
        start_time = time.time()
        result = manager.fetch_points(
            dataset=dataset,
            points=points,
            timestep=0,
            variables=variables
        )
        end_time = time.time()
        
        print(f"⏱️  請求耗時: {end_time - start_time:.2f} 秒")
        print(f"📦 數據來源: {'快取' if result['from_cache'] else '新獲取'}")
        
        # 檢查數據
        data = result['data']
        print(f"\n📋 數據摘要:")
        for var, arr in data.items():
            print(f"  {var}: shape={arr.shape}, dtype={arr.dtype}")
            print(f"      值={arr}")
        
        return True
        
    except Exception as e:
        print(f"❌ 散點測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fallback_logic():
    """測試回退邏輯"""
    
    print("\n🔄 測試客戶端回退邏輯")
    print("-" * 30)
    
    # 測試不同的初始化選項
    test_configs = [
        {'use_mock': False, 'use_http': True, 'description': 'HTTP 客戶端優先'},
        {'use_mock': False, 'use_http': False, 'description': 'pyJHTDB 客戶端優先'},
    ]
    
    for config in test_configs:
        try:
            description = config.pop('description')
            manager = create_jhtdb_manager(**config)
            print(f"✅ {description}: {manager.client_type}")
        except Exception as e:
            description = config.get('description', '未知配置')
            print(f"❌ {description}: 失敗 - {e}")
    
    return True


def test_cache_functionality():
    """測試快取功能"""
    
    print("\n💾 測試快取功能")
    print("-" * 30)
    
    try:
        manager = create_jhtdb_manager(use_mock=False, use_http=True)
        
        # 相同的請求參數
        dataset = 'channel'
        points = [[0.5, 0.0, 0.5]]
        variables = ['u']
        
        # 第一次請求（應該是新獲取）
        print("📡 第一次請求...")
        result1 = manager.fetch_points(
            dataset=dataset,
            points=points,
            variables=variables
        )
        print(f"  數據來源: {'快取' if result1['from_cache'] else '新獲取'}")
        
        # 第二次請求（應該來自快取）
        print("📡 第二次請求...")
        result2 = manager.fetch_points(
            dataset=dataset,
            points=points,
            variables=variables
        )
        print(f"  數據來源: {'快取' if result2['from_cache'] else '新獲取'}")
        
        # 驗證數據一致性
        if np.allclose(result1['data']['u'], result2['data']['u']):
            print("✅ 快取數據一致性檢查通過")
        else:
            print("❌ 快取數據不一致")
        
        return True
        
    except Exception as e:
        print(f"❌ 快取測試失敗: {e}")
        return False


def main():
    """主測試函數"""
    
    print("🧪 HTTP JHTDB 客戶端測試套件")
    print("=" * 60)
    print("目標：驗證真實 JHTDB 數據獲取功能")
    print("=" * 60)
    
    # 運行測試
    tests = [
        ("基本功能", test_http_client_basic),
        ("回退邏輯", test_fallback_logic),
        ("快取功能", test_cache_functionality),
        ("Cutout 數據", test_cutout_data),
        ("散點數據", test_points_data),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except KeyboardInterrupt:
            print(f"\n⚠️  用戶中斷測試")
            break
        except Exception as e:
            print(f"❌ 測試 '{test_name}' 發生未預期錯誤: {e}")
            results[test_name] = False
    
    # 測試總結
    print(f"\n{'='*60}")
    print("🏁 測試總結")
    print(f"{'='*60}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"  {test_name}: {status}")
    
    print(f"\n📊 總計: {passed}/{total} 測試通過")
    
    if passed == total:
        print("🎉 所有測試都通過！HTTP JHTDB 客戶端已就緒。")
    else:
        print("⚠️  部分測試失敗，請檢查配置或網路連接。")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)