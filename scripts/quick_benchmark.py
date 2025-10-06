#!/usr/bin/env python3
"""
快速性能基準測試腳本
用於驗證基本功能和收集初步性能數據
"""

import os
import sys
import time
import json
import yaml
import platform
from datetime import datetime
from pathlib import Path

import torch
import psutil
import numpy as np

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def quick_hardware_info():
    """快速收集硬體資訊"""
    try:
        cpu_freq = psutil.cpu_freq()
        cpu_freq_max = f"{cpu_freq.max:.1f}" if cpu_freq and cpu_freq.max else "Unknown"
        
        return {
            'platform': platform.platform(),
            'cpu_count': psutil.cpu_count(logical=True),
            'cpu_freq_max': cpu_freq_max,
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
        }
    except Exception as e:
        return {'error': str(e)}

def quick_training_test():
    """快速訓練測試"""
    from pinnx.models.fourier_mlp import PINNNet
    from pinnx.models.wrappers import ScaledPINNWrapper
    
    # 載入配置
    with open('configs/defaults.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 設定設備
    device = torch.device('cpu')  # 強制使用 CPU 來確保一致性
    
    # 創建小型模型進行快速測試
    model = PINNNet(
        in_dim=3,  # t, x, y
        out_dim=3,  # u, v, p
        width=32,  # 小寬度
        depth=3,   # 淺深度
        activation='tanh',
        fourier_m=16,  # 少量 Fourier features
        fourier_sigma=1.0
    ).to(device)
    
    # 包裝為 ScaledPINNWrapper
    model = ScaledPINNWrapper(model).to(device)
    
    # 計算參數數量
    total_params = sum(p.numel() for p in model.parameters())
    
    # 創建優化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 生成小量測試資料
    n_points = 100
    x = torch.rand(n_points, 3).to(device)  # [t, x, y]
    
    # 快速訓練循環
    times = []
    losses = []
    
    for epoch in range(10):  # 只訓練 10 步
        start_time = time.time()
        
        optimizer.zero_grad()
        
        # 前向傳播
        x.requires_grad_(True)
        pred = model(x)
        
        # 簡單損失（預測值的平方和）
        loss = torch.mean(pred**2)
        
        # 反向傳播
        loss.backward()
        optimizer.step()
        
        epoch_time = time.time() - start_time
        times.append(epoch_time)
        losses.append(loss.item())
    
    return {
        'total_params': total_params,
        'avg_epoch_time': np.mean(times),
        'final_loss': losses[-1],
        'all_losses': losses,
        'epoch_times': times
    }

def main():
    """主函數"""
    print("🚀 執行快速性能基準測試...")
    
    # 收集硬體資訊
    hw_info = quick_hardware_info()
    print(f"硬體平台: {hw_info.get('platform', 'Unknown')}")
    print(f"CPU 核心數: {hw_info.get('cpu_count', 'Unknown')}")
    print(f"記憶體: {hw_info.get('memory_total_gb', 'Unknown'):.1f} GB")
    print(f"PyTorch 版本: {hw_info.get('torch_version', 'Unknown')}")
    print(f"CUDA 可用: {hw_info.get('cuda_available', False)}")
    
    # 執行快速訓練測試
    print("\n⚡ 執行快速訓練測試...")
    start_time = time.time()
    try:
        training_result = quick_training_test()
        total_time = time.time() - start_time
        
        print(f"✅ 訓練測試完成！")
        print(f"總測試時間: {total_time:.2f} 秒")
        print(f"模型參數數量: {training_result['total_params']:,}")
        print(f"平均 epoch 時間: {training_result['avg_epoch_time']:.4f} 秒")
        print(f"最終損失: {training_result['final_loss']:.6f}")
        
        # 儲存結果
        results = {
            'timestamp': datetime.now().isoformat(),
            'hardware_info': hw_info,
            'training_test': training_result,
            'total_test_time': total_time
        }
        
        # 確保輸出目錄存在
        output_dir = Path('tasks/task-002')
        output_dir.mkdir(exist_ok=True)
        
        # 儲存結果
        with open(output_dir / 'quick_baseline_data.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 結果已儲存到: {output_dir / 'quick_baseline_data.json'}")
        
        # 評估是否符合基準
        meets_standards = True
        if training_result['avg_epoch_time'] > 0.1:
            print(f"⚠️  單 epoch 時間 {training_result['avg_epoch_time']:.4f}s 超過目標 0.1s")
            meets_standards = False
        
        if training_result['final_loss'] > 1e-3:
            print(f"⚠️  最終損失 {training_result['final_loss']:.6f} 超過目標 1e-3")
            meets_standards = False
        
        if meets_standards:
            print("✅ 所有性能指標符合基準要求！")
        else:
            print("❌ 部分性能指標需要優化")
            
        return results
        
    except Exception as e:
        print(f"❌ 訓練測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()