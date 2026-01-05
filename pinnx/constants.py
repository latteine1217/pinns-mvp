"""
物理常數與參考統計量

此模組定義專案中使用的物理常數、參考統計量，避免硬編碼 magic numbers。

References:
    - Johns Hopkins Turbulence Database: https://turbulence.pha.jhu.edu/
"""

from typing import Dict, Tuple

# ===================================================================
# JHTDB Channel Flow Re_tau=1000 參考統計量
# ===================================================================

JHTDB_RETAU1000_STATS: Dict[str, Dict[str, float]] = {
    'means': {
        'u': 9.921185,
        'v': -0.000085,
        'w': -0.002202,
        'p': -40.374241
    },
    'stds': {
        'u': 4.593879,
        'v': 0.329614,
        'w': 3.865396,
        'p': 28.619722
    }
}
"""
JHTDB Channel Flow Re_tau=1000 統計量（Z-score normalization）

來源: Johns Hopkins Turbulence Database
場景: Channel Flow at Re_tau=1000
用途: 訓練時 Z-score 標準化的預設值

注意: 
    - 這些值應優先從實際訓練數據統計中獲得
    - 僅在 checkpoint 和 config 均缺失統計資訊時使用
"""

JHTDB_RETAU1000_RANGES: Dict[str, Tuple[float, float]] = {
    'u': (0.0, 20.0),
    'v': (-1.0, 1.0),
    'w': (-5.0, 5.0),
    'p': (-100.0, 10.0)
}
"""
JHTDB Channel Flow Re_tau=1000 典型範圍（ManualScalingWrapper）

用途: 當配置缺失輸出範圍時的預設值
警告: 這些範圍僅為粗略估計，實際應從數據統計中獲得
"""

# ===================================================================
# Kolmogorov Flow 2D 參考統計量
# ===================================================================

KOLMOGOROV_RE50_STATS: Dict[str, Dict[str, float]] = {
    'means': {
        'u': 0.0,
        'v': 0.0,
        'p': 0.0
    },
    'stds': {
        'u': 1.0,
        'v': 1.0,
        'p': 1.0
    }
}
"""
Kolmogorov Flow Re=50, kf=4 統計量（Z-score normalization）

來源: 歸一化後的訓練數據（理論上 mean=0, std=1）
場景: 2D Kolmogorov Flow (periodic box)
用途: 訓練時 Z-score 標準化的預設值

注意:
    - Kolmogorov Flow 通常在訓練前已進行 Z-score 標準化
    - 若訓練數據已歸一化，這些值應為 mean=0, std=1
    - 實際使用時應從 checkpoint 或 config 中獲取真實統計量
"""

KOLMOGOROV_RE50_RANGES: Dict[str, Tuple[float, float]] = {
    'u': (-2.0, 2.0),
    'v': (-2.0, 2.0),
    'p': (-1.0, 1.0)
}
"""
Kolmogorov Flow Re=50, kf=4 典型範圍（歸一化後）

用途: 當配置缺失輸出範圍時的預設值
警告: 這些範圍為歸一化後的估計值，實際應從數據統計中獲得
"""

# ===================================================================
# 物理常數
# ===================================================================

DEFAULT_RHO: float = 1.0
"""流體密度預設值（無因次化情況）"""

DEFAULT_U_TAU: float = 1.0
"""摩擦速度預設值（無因次化情況）"""

# ===================================================================
# 驗證函數
# ===================================================================

def validate_jhtdb_stats() -> bool:
    """
    驗證 JHTDB 統計量的完整性
    
    Returns:
        bool: 所有必要統計量存在且有效
    """
    required_vars = ['u', 'v', 'w', 'p']
    
    # 檢查 means
    if not all(var in JHTDB_RETAU1000_STATS['means'] for var in required_vars):
        return False
    
    # 檢查 stds
    if not all(var in JHTDB_RETAU1000_STATS['stds'] for var in required_vars):
        return False
    
    # 檢查數值有效性（stds > 0）
    if not all(JHTDB_RETAU1000_STATS['stds'][var] > 0 for var in required_vars):
        return False
    
    return True


def validate_kolmogorov_stats() -> bool:
    """
    驗證 Kolmogorov Flow 統計量的完整性
    
    Returns:
        bool: 所有必要統計量存在且有效
    """
    required_vars = ['u', 'v', 'p']
    
    # 檢查 means
    if not all(var in KOLMOGOROV_RE50_STATS['means'] for var in required_vars):
        return False
    
    # 檢查 stds
    if not all(var in KOLMOGOROV_RE50_STATS['stds'] for var in required_vars):
        return False
    
    # 檢查數值有效性（stds > 0）
    if not all(KOLMOGOROV_RE50_STATS['stds'][var] > 0 for var in required_vars):
        return False
    
    return True


if __name__ == "__main__":
    # 測試常數定義
    print("=" * 60)
    print("📋 物理場景統計量")
    print("=" * 60)
    
    # JHTDB Channel Flow
    print("\n【JHTDB Channel Flow Re_tau=1000】")
    print("\nMeans:")
    for var, val in JHTDB_RETAU1000_STATS['means'].items():
        print(f"  {var}: {val:12.6f}")
    
    print("\nStandard Deviations:")
    for var, val in JHTDB_RETAU1000_STATS['stds'].items():
        print(f"  {var}: {val:12.6f}")
    
    print("\nTypical Ranges:")
    for var, (min_val, max_val) in JHTDB_RETAU1000_RANGES.items():
        print(f"  {var}: [{min_val:8.2f}, {max_val:8.2f}]")
    
    is_valid = validate_jhtdb_stats()
    status = "✅ 通過" if is_valid else "❌ 失敗"
    print(f"\n驗證: {status}")
    
    # Kolmogorov Flow
    print("\n" + "-" * 60)
    print("\n【Kolmogorov Flow Re=50, kf=4】")
    print("\nMeans:")
    for var, val in KOLMOGOROV_RE50_STATS['means'].items():
        print(f"  {var}: {val:12.6f}")
    
    print("\nStandard Deviations:")
    for var, val in KOLMOGOROV_RE50_STATS['stds'].items():
        print(f"  {var}: {val:12.6f}")
    
    print("\nTypical Ranges:")
    for var, (min_val, max_val) in KOLMOGOROV_RE50_RANGES.items():
        print(f"  {var}: [{min_val:8.2f}, {max_val:8.2f}]")
    
    is_valid = validate_kolmogorov_stats()
    status = "✅ 通過" if is_valid else "❌ 失敗"
    print(f"\n驗證: {status}")
    
    print("\n" + "=" * 60)
