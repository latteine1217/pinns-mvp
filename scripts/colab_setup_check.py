#!/usr/bin/env python3
"""
Colab A100 環境配置檢查
用於驗證 Colab 環境是否滿足訓練需求
"""

import sys
import torch
import numpy as np
from pathlib import Path

def check_cuda():
    """檢查 CUDA 可用性"""
    print("=" * 60)
    print("1. CUDA 檢查")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用！")
        return False
    
    print(f"✅ CUDA 可用")
    # 類型忽略：torch.version.cuda 在運行時存在但靜態檢查無法識別
    cuda_version = getattr(torch.version, 'cuda', 'N/A')  # type: ignore
    print(f"   CUDA 版本: {cuda_version}")
    print(f"   GPU 數量: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {props.name}")
        print(f"      記憶體: {props.total_memory / 1024**3:.2f} GB")
        print(f"      計算能力: {props.major}.{props.minor}")
    
    return True

def check_memory():
    """檢查記憶體"""
    print("\n" + "=" * 60)
    print("2. 記憶體檢查")
    print("=" * 60)
    
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        reserved = torch.cuda.memory_reserved(0)
        allocated = torch.cuda.memory_allocated(0)
        
        print(f"✅ GPU 記憶體:")
        print(f"   總量: {total / 1024**3:.2f} GB")
        print(f"   已保留: {reserved / 1024**3:.2f} GB")
        print(f"   已分配: {allocated / 1024**3:.2f} GB")
        print(f"   可用: {(total - reserved) / 1024**3:.2f} GB")

def check_project_structure():
    """檢查專案結構"""
    print("\n" + "=" * 60)
    print("3. 專案結構檢查")
    print("=" * 60)
    
    required_dirs = [
        'configs',
        'scripts',
        'pinnx',
        'data/jhtdb/channel_flow_re1000',
        'checkpoints',
        'log',
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} (缺失)")
            all_ok = False
    
    return all_ok

def check_config_file():
    """檢查配置文件"""
    print("\n" + "=" * 60)
    print("4. 配置文件檢查")
    print("=" * 60)
    
    config_path = Path('configs/test_normalization_main_2000epochs.yml')
    if config_path.exists():
        print(f"✅ {config_path}")
        
        # 檢查感測點文件
        sensor_path = Path('data/jhtdb/channel_flow_re1000/sensors_K50_qr_pivot.npz')
        if sensor_path.exists():
            data = np.load(sensor_path)
            print(f"✅ 感測點文件: {sensor_path}")
            print(f"   包含鍵: {list(data.keys())}")
            if 'coords' in data:
                print(f"   感測點數量: {data['coords'].shape[0]}")
        else:
            print(f"❌ 感測點文件缺失: {sensor_path}")
            return False
    else:
        print(f"❌ 配置文件缺失: {config_path}")
        return False
    
    return True

def check_dependencies():
    """檢查依賴套件"""
    print("\n" + "=" * 60)
    print("5. 依賴套件檢查")
    print("=" * 60)
    
    required_packages = {
        'torch': torch.__version__,
        'numpy': np.__version__,
    }
    
    try:
        import yaml
        required_packages['yaml'] = yaml.__version__
    except ImportError:
        print("❌ PyYAML 未安裝")
        return False
    
    try:
        import h5py
        required_packages['h5py'] = h5py.__version__
    except ImportError:
        print("⚠️  h5py 未安裝（若使用 HDF5 資料需安裝）")
    
    for pkg, ver in required_packages.items():
        print(f"✅ {pkg}: {ver}")
    
    return True

def estimate_training_resources():
    """估算訓練資源需求"""
    print("\n" + "=" * 60)
    print("6. 訓練資源估算")
    print("=" * 60)
    
    # 模型參數量
    model_params = 296_804
    param_size_mb = model_params * 4 / 1024**2  # float32
    
    # 批次資料
    batch_size = 10_000
    pde_points = 20_000
    boundary_points = 5_000
    
    # 估算記憶體需求（粗略）
    data_size_mb = (batch_size + pde_points + boundary_points) * 3 * 4 / 1024**2
    gradient_size_mb = param_size_mb * 2  # 梯度 + 優化器狀態
    
    total_estimate_mb = param_size_mb + data_size_mb + gradient_size_mb
    
    print(f"📊 估算記憶體需求:")
    print(f"   模型參數: {param_size_mb:.2f} MB ({model_params:,} params)")
    print(f"   批次資料: {data_size_mb:.2f} MB")
    print(f"   梯度/優化器: {gradient_size_mb:.2f} MB")
    print(f"   總計（粗估）: {total_estimate_mb:.2f} MB")
    print(f"   建議 GPU 記憶體: ≥ {total_estimate_mb * 4:.0f} MB (含安全餘量)")
    
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_mem_gb * 1024 >= total_estimate_mb * 4:
            print(f"✅ GPU 記憶體充足 ({gpu_mem_gb:.1f} GB)")
        else:
            print(f"⚠️  GPU 記憶體可能不足 ({gpu_mem_gb:.1f} GB)")

def main():
    """主檢查流程"""
    print("🔍 Colab A100 環境配置檢查")
    print()
    
    checks = [
        ("CUDA", check_cuda),
        ("記憶體", check_memory),
        ("專案結構", check_project_structure),
        ("配置文件", check_config_file),
        ("依賴套件", check_dependencies),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            result = check_func()
            results[name] = result if result is not None else True
        except Exception as e:
            print(f"❌ {name} 檢查失敗: {e}")
            results[name] = False
    
    # 資源估算
    estimate_training_resources()
    
    # 總結
    print("\n" + "=" * 60)
    print("檢查總結")
    print("=" * 60)
    
    all_passed = all(results.values())
    for name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {name}")
    
    if all_passed:
        print("\n✅ 所有檢查通過！可以開始訓練。")
        print("\n📝 啟動訓練指令:")
        print("   python scripts/train.py --cfg configs/test_normalization_main_2000epochs.yml")
        return 0
    else:
        print("\n❌ 部分檢查未通過，請先修正問題。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
