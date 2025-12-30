#!/usr/bin/env python3
"""
WandB 集成測試腳本

驗證 WandB 配置是否正確，以及基本日誌記錄功能是否正常。
"""

import os
import sys
from pathlib import Path

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import wandb
import torch


def test_wandb_config():
    """測試 WandB 配置文件"""
    print("=" * 60)
    print("測試 1: 檢查 WandB 配置文件")
    print("=" * 60)
    
    config_path = project_root / '.wandb_config'
    if not config_path.exists():
        print("❌ 錯誤：.wandb_config 文件不存在")
        print(f"   請創建文件： {config_path}")
        return False
    
    # 讀取配置
    api_key = None
    project = None
    with open(config_path, 'r') as f:
        for line in f:
            if line.startswith('WANDB_API_KEY='):
                api_key = line.split('=')[1].strip()
            elif line.startswith('WANDB_PROJECT='):
                project = line.split('=')[1].strip()
    
    if not api_key:
        print("❌ 錯誤：WANDB_API_KEY 未設定")
        return False
    
    print(f"✅ API Key: {'*' * 10}{api_key[-4:]} (已隱藏)")
    print(f"✅ Project: {project or 'pinns-turbulence-reconstruction (預設)'}")
    
    # 設定環境變數
    os.environ['WANDB_API_KEY'] = api_key
    return True


def test_wandb_init():
    """測試 WandB 初始化"""
    print("\n" + "=" * 60)
    print("測試 2: WandB 初始化")
    print("=" * 60)
    
    try:
        run = wandb.init(
            project='pinns-test',
            name='wandb-integration-test',
            config={'test': True, 'purpose': 'validation'},
            mode='online'  # 使用 online 模式測試連接
        )
        print("✅ WandB 初始化成功")
        print(f"   Run ID: {run.id}")
        print(f"   Run URL: {run.url}")
        return run
    except Exception as e:
        print(f"❌ WandB 初始化失敗：{e}")
        return None


def test_wandb_logging(run):
    """測試 WandB 日誌記錄"""
    print("\n" + "=" * 60)
    print("測試 3: WandB 日誌記錄")
    print("=" * 60)
    
    if run is None:
        print("⏭️  跳過（初始化失敗）")
        return False
    
    try:
        # 測試 scalar 記錄
        for i in range(5):
            wandb.log({
                'Loss/total': 1.0 / (i + 1),
                'Loss/pde': 0.5 / (i + 1),
                'Loss/data': 0.3 / (i + 1),
                'Training/learning_rate': 0.001,
                'epoch': i
            })
        print("✅ Scalar 日誌記錄成功")
        
        # 測試 histogram 記錄
        dummy_tensor = torch.randn(100)
        wandb.log({
            'Weights/test': wandb.Histogram(dummy_tensor.numpy()),
            'epoch': 5
        })
        print("✅ Histogram 日誌記錄成功")
        
        # 測試 summary
        wandb.run.summary['final_loss'] = 0.123
        wandb.run.summary['best_epoch'] = 3
        print("✅ Summary 記錄成功")
        
        return True
    except Exception as e:
        print(f"❌ 日誌記錄失敗：{e}")
        return False


def test_wandb_finalize(run):
    """測試 WandB 完成"""
    print("\n" + "=" * 60)
    print("測試 4: WandB 完成")
    print("=" * 60)
    
    if run is None:
        print("⏭️  跳過（初始化失敗）")
        return False
    
    try:
        wandb.finish()
        print("✅ WandB 運行已結束")
        print(f"   請訪問 {run.url} 查看結果")
        return True
    except Exception as e:
        print(f"❌ WandB 完成失敗：{e}")
        return False


def main():
    """主測試流程"""
    print("\n" + "=" * 60)
    print("WandB 集成測試")
    print("=" * 60)
    
    results = {}
    
    # 測試 1: 配置文件
    results['config'] = test_wandb_config()
    if not results['config']:
        print("\n❌ 測試終止：配置文件檢查失敗")
        return False
    
    # 測試 2: 初始化
    run = test_wandb_init()
    results['init'] = run is not None
    
    # 測試 3: 日誌記錄
    results['logging'] = test_wandb_logging(run)
    
    # 測試 4: 完成
    results['finalize'] = test_wandb_finalize(run)
    
    # 彙總結果
    print("\n" + "=" * 60)
    print("測試結果彙總")
    print("=" * 60)
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ 通過" if passed_test else "❌ 失敗"
        print(f"  {test_name:15s}: {status}")
    
    print("\n" + "=" * 60)
    if passed == total:
        print(f"🎉 所有測試通過 ({passed}/{total})")
        print("=" * 60)
        return True
    else:
        print(f"⚠️  部分測試失敗 ({passed}/{total})")
        print("=" * 60)
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
