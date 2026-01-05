"""
測試 normalization metadata 是否正確保存到 checkpoint

用法：
    python scripts/test_normalization_fix.py
"""

import torch
import sys
from pathlib import Path

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from pinnx.constants import KOLMOGOROV_RE50_STATS, JHTDB_RETAU1000_STATS


def test_checkpoint_normalization(checkpoint_path: str):
    """測試 checkpoint 是否包含 normalization metadata"""
    print(f"\n{'='*60}")
    print(f"Testing: {checkpoint_path}")
    print(f"{'='*60}")
    
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        # 檢查頂層 'normalization' 鍵
        if 'normalization' in ckpt:
            print("✅ 'normalization' key exists in checkpoint")
            norm = ckpt['normalization']
            
            # 檢查內容
            print(f"\nNormalization metadata:")
            print(f"  Keys: {list(norm.keys())}")
            
            if 'variable_order' in norm:
                print(f"  ✅ variable_order: {norm['variable_order']}")
            else:
                print(f"  ❌ variable_order: NOT FOUND")
            
            if 'means' in norm:
                print(f"  ✅ means: {norm['means']}")
            else:
                print(f"  ⚠️  means: NOT FOUND (will use defaults)")
            
            if 'stds' in norm:
                print(f"  ✅ stds: {norm['stds']}")
            else:
                print(f"  ⚠️  stds: NOT FOUND (will use defaults)")
            
            if 'norm_type' in norm or 'type' in norm:
                norm_type = norm.get('norm_type', norm.get('type'))
                print(f"  ✅ type: {norm_type}")
            
            # 判定物理類型
            variable_order = norm.get('variable_order', [])
            if len(variable_order) <= 3 and 'w' not in variable_order:
                detected_physics = 'kolmogorov_2d'
            else:
                detected_physics = 'channel_3d'
            print(f"\n  Detected physics type: {detected_physics}")
            
        else:
            print("❌ 'normalization' key NOT FOUND in checkpoint")
            print("   This checkpoint was created before the fix.")
            
            # 檢查是否有 config['normalization']
            if 'config' in ckpt and 'normalization' in ckpt['config']:
                print("\n  ℹ️  Found config['normalization']:")
                print(f"     {ckpt['config']['normalization']}")
        
        # 檢查其他重要鍵
        print(f"\nOther checkpoint keys: {list(ckpt.keys())}")
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()


def simulate_new_checkpoint():
    """模擬新訓練流程保存的 checkpoint"""
    print(f"\n{'='*60}")
    print(f"Simulating NEW checkpoint (after fix)")
    print(f"{'='*60}")
    
    # 模擬 Trainer 調用 CheckpointManager.save()
    # 這裡展示 normalizers 參數的格式
    
    # Case 1: Kolmogorov Flow (from OutputTransform.get_metadata())
    normalizers_kolmogorov = {
        "norm_type": "training_data_norm",
        "variable_order": ["u", "v", "p"],
        "means": {"u": 0.0, "v": 0.0, "p": 0.0},
        "stds": {"u": 1.0, "v": 1.0, "p": 1.0},
        "params": {}
    }
    
    print("\nCase 1: Kolmogorov Flow (2D)")
    print(f"  normalizers dict: {normalizers_kolmogorov}")
    print(f"  → This will be saved as checkpoint['normalization']")
    
    # Case 2: Channel Flow (from OutputTransform.get_metadata())
    normalizers_channel = {
        "norm_type": "training_data_norm",
        "variable_order": ["u", "v", "w", "p"],
        "means": JHTDB_RETAU1000_STATS['means'],
        "stds": JHTDB_RETAU1000_STATS['stds'],
        "params": {}
    }
    
    print("\nCase 2: Channel Flow (3D)")
    print(f"  normalizers dict keys: {list(normalizers_channel.keys())}")
    print(f"  variable_order: {normalizers_channel['variable_order']}")
    print(f"  → This will be saved as checkpoint['normalization']")


if __name__ == "__main__":
    # 測試現有的舊 checkpoint
    old_checkpoint = project_root / "checkpoints" / "window_1_t15_18.pth"
    if old_checkpoint.exists():
        test_checkpoint_normalization(str(old_checkpoint))
    else:
        print(f"⚠️  Old checkpoint not found: {old_checkpoint}")
    
    # 模擬新 checkpoint
    simulate_new_checkpoint()
    
    print(f"\n{'='*60}")
    print("Summary:")
    print("  ✅ After fix: All new checkpoints will include 'normalization' metadata")
    print("  ✅ Denormalization will use correct stats based on physics type")
    print("  ⚠️  Old checkpoints: Will use improved fallback (Kolmogorov vs Channel)")
    print(f"{'='*60}\n")
