#!/usr/bin/env python3
"""
JHTDB Data Import Verification Script
Step-by-step verification of Channel Flow data loaded from JHTDB

Goals:
1. Load JHTDB Channel Flow data 
2. Display basic flow field statistics
3. Check physical reasonableness of data
4. Generate visualization plots to verify turbulence characteristics
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns  # Temporarily commented to avoid dependency issues

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pinnx.dataio.channel_flow_loader import ChannelFlowLoader
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_jhtdb_data_step_by_step(load_mode='sensor'):
    """
    Step-by-step JHTDB data verification
    
    Args:
        load_mode: Loading mode ('sensor' for sensor points only, 'full_field' for complete field)
    """
    
    print("=" * 80)
    print("🔍 JHTDB Channel Flow Data Verification")
    print("=" * 80)
    
    # Step 1: Check configuration and data files
    print("\n📁 Step 1: Check Configuration and Data Files")
    print("-" * 50)
    
    config_path = project_root / "configs" / "channel_flow_re1000.yml"
    cache_dir = project_root / "data" / "jhtdb" / "channel_flow_re1000"
    
    print(f"Config file path: {config_path}")
    print(f"Config file exists: {config_path.exists()}")
    print(f"Data directory path: {cache_dir}")
    print(f"Data directory exists: {cache_dir.exists()}")
    
    if cache_dir.exists():
        data_files = list(cache_dir.glob("*.npz"))
        print(f"Found {len(data_files)} data files:")
        for file in data_files:
            print(f"  - {file.name}")
    
    # Step 2: Initialize loader
    print("\n🔧 Step 2: Initialize Channel Flow Loader")
    print("-" * 50)
    
    try:
        loader = ChannelFlowLoader(config_path=config_path, cache_dir=cache_dir)
        print("✅ Loader initialization successful")
        
        # Check available datasets
        available_datasets = loader.get_available_datasets()
        print(f"Available datasets: {available_datasets}")
        
    except Exception as e:
        print(f"❌ Loader initialization failed: {e}")
        return False
    
    # Step 3: Load data
    if load_mode == 'full_field':
        print("\n📊 Step 3: Load Complete Flow Field Data (8×8×8=512 points)")
        print("-" * 50)
        
        try:
            # Load complete flow field data
            channel_data = loader.load_full_field_data()
            print("✅ Complete flow field data loaded successfully")
            
            # Display basic information
            print(f"Number of data points: {len(channel_data.sensor_points)}")
            print(f"Coordinate shape: {channel_data.sensor_points.shape}")
            print(f"Available field data: {list(channel_data.sensor_data.keys())}")
            print(f"Grid shape: {channel_data.selection_info.get('grid_shape', 'Unknown')}")
            
            # Display coordinate ranges
            coords = channel_data.sensor_points
            print(f"x coordinate range: [{coords[:, 0].min():.3f}, {coords[:, 0].max():.3f}]")
            print(f"y coordinate range: [{coords[:, 1].min():.3f}, {coords[:, 1].max():.3f}]")
            print(f"z coordinate range: [{coords[:, 2].min():.3f}, {coords[:, 2].max():.3f}]")
            
        except Exception as e:
            print(f"❌ Complete flow field data loading failed: {e}")
            return False
    else:
        print("\n📊 Step 3: Load QR-Pivot Sensor Point Data")
        print("-" * 50)
        
        try:
            # Try to load K=8 QR-pivot data
            channel_data = loader.load_sensor_data(strategy='qr_pivot', K=8)
            print("✅ Sensor point data loaded successfully")
            
            # Display basic information
            print(f"Number of sensor points: {len(channel_data.sensor_points)}")
            print(f"Sensor point coordinate shape: {channel_data.sensor_points.shape}")
            print(f"Available field data: {list(channel_data.sensor_data.keys())}")
            
            # Display sensor point coordinate ranges
            x_coords = channel_data.sensor_points[:, 0] 
            y_coords = channel_data.sensor_points[:, 1]
            print(f"x coordinate range: [{x_coords.min():.3f}, {x_coords.max():.3f}]")
            print(f"y coordinate range: [{y_coords.min():.3f}, {y_coords.max():.3f}]")
            
        except Exception as e:
            print(f"❌ Sensor point data loading failed: {e}")
            return False
    
    # 步驟4：檢查流場數據
    print("\n🌊 步驟4：檢查流場數據的物理特性")
    print("-" * 50)
    
    # 如果是完整流場，進行無散條件檢查
    if load_mode == 'full_field' and 'w' in channel_data.sensor_data:
        print("\n🔍 3D無散條件檢查 (∇·u = 0):")
        
        # 重構3D場
        grid_shape = channel_data.selection_info.get('grid_shape', (8, 8, 8))
        u_3d = channel_data.sensor_data['u'].reshape(grid_shape)
        v_3d = channel_data.sensor_data['v'].reshape(grid_shape)
        w_3d = channel_data.sensor_data['w'].reshape(grid_shape)
        
        # 計算散度 (中心差分) - 修復軸序
        # meshgrid(indexing='ij'): axis=0->x, axis=1->y, axis=2->z
        du_dx = np.gradient(u_3d, axis=0)  # x方向 (修復: axis=2 -> axis=0)
        dv_dy = np.gradient(v_3d, axis=1)  # y方向 (正確)
        dw_dz = np.gradient(w_3d, axis=2)  # z方向 (修復: axis=0 -> axis=2)
        
        divergence = du_dx + dv_dy + dw_dz
        
        # 計算散度統計
        div_mean = np.mean(divergence)
        div_std = np.std(divergence)
        div_rms = np.sqrt(np.mean(divergence**2))
        div_max = np.max(np.abs(divergence))
        
        # 計算相對散度
        velocity_rms = np.sqrt(np.mean(u_3d**2 + v_3d**2 + w_3d**2))
        relative_divergence = div_rms / velocity_rms if velocity_rms > 0 else float('inf')
        
        print(f"  散度統計:")
        print(f"    均值: {div_mean:.8f}")
        print(f"    標準差: {div_std:.8f}")
        print(f"    RMS: {div_rms:.8f}")
        print(f"    最大絕對值: {div_max:.8f}")
        print(f"    相對散度: {relative_divergence:.8f}")
        
        # 評估無散程度
        if relative_divergence < 1e-6:
            print("  ✅ 極佳的無散條件滿足")
        elif relative_divergence < 1e-4:
            print("  ✅ 良好的無散條件滿足")
        elif relative_divergence < 1e-2:
            print("  ⚠️  可接受的無散條件滿足")
        else:
            print("  ❌ 無散條件不滿足，可能不是真實流場")
    
    # 檢查每個場的統計特性
    for field_name, field_data in channel_data.sensor_data.items():
        print(f"\n{field_name.upper()} 場統計:")
        print(f"  形狀: {field_data.shape}")
        print(f"  平均值: {np.mean(field_data):.6f}")
        print(f"  標準差: {np.std(field_data):.6f}")
        print(f"  最小值: {np.min(field_data):.6f}")
        print(f"  最大值: {np.max(field_data):.6f}")
        print(f"  是否有NaN: {np.isnan(field_data).any()}")
        print(f"  是否有Inf: {np.isinf(field_data).any()}")
        
        # 物理合理性檢查
        if field_name == 'u':
            # u場在Channel Flow中應該為正值且有合理範圍
            print(f"  u場合理性檢查:")
            print(f"    - 所有值為正: {np.all(field_data >= 0)}")
            print(f"    - 最大值合理 (<30): {np.max(field_data) < 30}")
        elif field_name == 'v':
            # v場在2D channel flow中應該接近零
            print(f"  v場合理性檢查:")
            print(f"    - 接近零 (|v| < 1): {np.all(np.abs(field_data) < 1)}")
        elif field_name == 'p':
            # 壓力場應該有梯度但不能有極值
            print(f"  p場合理性檢查:")
            print(f"    - 有變化 (std > 0): {np.std(field_data) > 0}")
    
    # 步驟5：載入Enhanced Mock數據進行對比
    print("\n🎭 步驟5：Mock功能已移除")
    print("-" * 50)
    print("❌ Enhanced Mock數據生成功能已移除，專案現在僅使用真實JHTDB數據")
    mock_data = None
    
    # 步驟6：數據驗證
    print("\n✅ 步驟6：執行完整數據驗證")
    print("-" * 50)
    
    validation_results = loader.validate_data(channel_data)
    
    print("驗證結果:")
    passed_checks = 0
    total_checks = len(validation_results)
    
    for check_name, result in validation_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {check_name}: {status}")
        if result:
            passed_checks += 1
    
    print(f"\n驗證總結: {passed_checks}/{total_checks} 檢查通過")
    
    # 步驟7：創建可視化圖表
    print("\n📈 步驟7：創建數據可視化")
    print("-" * 50)
    
    try:
        create_verification_plots(channel_data, mock_data)
        print("✅ 可視化圖表創建成功")
    except Exception as e:
        print(f"❌ 可視化創建失敗: {e}")
    
    # 步驟8：檢查域配置
    print("\n🏗️ 步驟8：檢查域配置參數")
    print("-" * 50)
    
    domain_config = channel_data.domain_config
    print("域配置參數:")
    for key, value in domain_config.items():
        print(f"  {key}: {value}")
    
    # 檢查物理參數
    phys_params = channel_data.get_physical_parameters()
    print("\n物理參數:")
    for key, value in phys_params.items():
        print(f"  {key}: {value}")
    
    # 總結
    print("\n" + "=" * 80)
    print("🎯 數據驗證總結")
    print("=" * 80)
    
    verification_score = passed_checks / total_checks * 100
    
    if verification_score >= 90:
        print(f"✅ 數據質量優秀 ({verification_score:.1f}%)")
        print("✅ JHTDB數據載入驗證成功")
        return True
    elif verification_score >= 70:
        print(f"⚠️ 數據質量良好 ({verification_score:.1f}%)，但有輕微問題")
        return True
    else:
        print(f"❌ 數據質量不佳 ({verification_score:.1f}%)，需要檢查")
        return False

def create_verification_plots(channel_data, mock_data=None):
    """創建數據驗證可視化圖表"""
    
    plt.style.use('default')  # 使用default樣式避免seaborn依賴
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('JHTDB Channel Flow Data Verification', fontsize=16, fontweight='bold')
    
    # 子圖1：感測點分佈
    ax = axes[0, 0]
    sensor_points = channel_data.sensor_points
    scatter = ax.scatter(sensor_points[:, 0], sensor_points[:, 1], 
                        c=range(len(sensor_points)), cmap='viridis', s=100, alpha=0.8)
    ax.set_xlabel('x [Streamwise]')
    ax.set_ylabel('y [Wall-normal]')
    ax.set_title('QR-Pivot Sensor Distribution')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Sensor Index')
    
    # 子圖2：u場分佈
    ax = axes[0, 1]
    if 'u' in channel_data.sensor_data:
        u_data = channel_data.sensor_data['u']
        scatter = ax.scatter(sensor_points[:, 0], sensor_points[:, 1], 
                           c=u_data, cmap='coolwarm', s=100, alpha=0.8)
        ax.set_xlabel('x [Streamwise]')
        ax.set_ylabel('y [Wall-normal]')
        ax.set_title(f'u Field Distribution (Mean={np.mean(u_data):.3f})')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='u [m/s]')
    
    # 子圖3：v場分佈
    ax = axes[0, 2]
    if 'v' in channel_data.sensor_data:
        v_data = channel_data.sensor_data['v']
        scatter = ax.scatter(sensor_points[:, 0], sensor_points[:, 1], 
                           c=v_data, cmap='coolwarm', s=100, alpha=0.8)
        ax.set_xlabel('x [Streamwise]')
        ax.set_ylabel('y [Wall-normal]')
        ax.set_title(f'v Field Distribution (Mean={np.mean(v_data):.3f})')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='v [m/s]')
    
    # 子圖4：壓力場分佈
    ax = axes[1, 0]
    if 'p' in channel_data.sensor_data:
        p_data = channel_data.sensor_data['p']
        scatter = ax.scatter(sensor_points[:, 0], sensor_points[:, 1], 
                           c=p_data, cmap='coolwarm', s=100, alpha=0.8)
        ax.set_xlabel('x [Streamwise]')
        ax.set_ylabel('y [Wall-normal]')
        ax.set_title(f'p Field Distribution (Mean={np.mean(p_data):.3f})')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='p [Pa]')
    
    # 子圖5：場統計對比
    ax = axes[1, 1]
    fields = ['u', 'v', 'p']
    means = []
    stds = []
    
    for field in fields:
        if field in channel_data.sensor_data:
            data = channel_data.sensor_data[field]
            means.append(np.mean(data))
            stds.append(np.std(data))
        else:
            means.append(0)
            stds.append(0)
    
    x_pos = np.arange(len(fields))
    bars1 = ax.bar(x_pos - 0.2, means, 0.4, label='Mean', alpha=0.7)
    bars2 = ax.bar(x_pos + 0.2, stds, 0.4, label='Std Dev', alpha=0.7)
    
    ax.set_xlabel('Field Variables')
    ax.set_ylabel('Values')
    ax.set_title('Field Statistics')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(fields)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加數值標籤
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax.text(bar1.get_x() + bar1.get_width()/2., height1,
                f'{height1:.3f}', ha='center', va='bottom', fontsize=8)
        ax.text(bar2.get_x() + bar2.get_width()/2., height2,
                f'{height2:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 子圖6：Mock數據對比 (如果有)
    ax = axes[1, 2]
    if mock_data is not None:
        # 從Mock數據隨機取樣8個點進行對比
        n_total = len(mock_data['u'])
        sample_indices = np.random.choice(n_total, size=8, replace=False)
        
        mock_u_sample = mock_data['u'][sample_indices]
        mock_v_sample = mock_data['v'][sample_indices]
        
        sensor_u = channel_data.sensor_data.get('u', np.zeros(8))
        sensor_v = channel_data.sensor_data.get('v', np.zeros(8))
        
        ax.scatter(sensor_u, sensor_v, c='red', s=100, alpha=0.7, label='JHTDB Sensors')
        ax.scatter(mock_u_sample, mock_v_sample, c='blue', s=100, alpha=0.7, label='Mock Samples')
        ax.set_xlabel('u [m/s]')
        ax.set_ylabel('v [m/s]')
        ax.set_title('JHTDB vs Mock Velocity Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No Mock Data\nAvailable', ha='center', va='center', 
               transform=ax.transAxes, fontsize=12)
        ax.set_title('Mock Data Comparison')
    
    plt.tight_layout()
    
    # 保存圖表
    output_path = project_root / 'jhtdb_verification_plots.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='驗證JHTDB數據')
    parser.add_argument('--mode', choices=['sensor', 'full_field'], default='full_field',
                        help='載入模式：sensor (只載入感測點) 或 full_field (載入完整流場)')
    args = parser.parse_args()
    
    print(f"開始JHTDB數據驗證... (模式: {args.mode})")
    success = verify_jhtdb_data_step_by_step(load_mode=args.mode)
    
    if success:
        print(f"\n🎉 驗證完成！JHTDB數據載入正常 (模式: {args.mode})")
    else:
        print(f"\n⚠️ 驗證發現問題，請檢查數據和配置 (模式: {args.mode})")
        sys.exit(1)