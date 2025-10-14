"""簡單檢查點測試 - 驗證模型載入與基本預測"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    ckpt_path = "checkpoints/test_physics_fix_1k_v2/best_model.pth"
    
    print("="*70)
    print("📦 檢查點內容檢查")
    print("="*70)
    
    # 載入檢查點
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    print(f"\n✅ 檢查點載入成功")
    print(f"   Epoch: {ckpt['epoch']}")
    print(f"   Keys: {list(ckpt.keys())}")
    
    # 檢查歷史
    if 'history' in ckpt:
        history = ckpt['history']
        print(f"\n📊 訓練歷史:")
        
        # 檢查歷史格式
        if isinstance(history, dict):
            print(f"   歷史鍵: {list(history.keys())}")
            
            if 'total_loss' in history:
                total_loss_hist = history['total_loss']
                if isinstance(total_loss_hist, list) and len(total_loss_hist) > 0:
                    print(f"   記錄筆數: {len(total_loss_hist)}")
                    print(f"   最終 Total Loss: {total_loss_hist[-1]:.6f}")
            
            if 'data_loss' in history:
                data_loss_hist = history['data_loss']
                if isinstance(data_loss_hist, list) and len(data_loss_hist) > 0:
                    print(f"   最終 Data Loss: {data_loss_hist[-1]:.6f}")
            
            if 'pde_loss' in history:
                pde_loss_hist = history['pde_loss']
                if isinstance(pde_loss_hist, list) and len(pde_loss_hist) > 0:
                    print(f"   最終 PDE Loss: {pde_loss_hist[-1]:.6f}")
            
            if 'continuity_loss' in history:
                cont_loss_hist = history['continuity_loss']
                if isinstance(cont_loss_hist, list) and len(cont_loss_hist) > 0:
                    print(f"   最終 Continuity Loss: {cont_loss_hist[-1]:.6f}")
            
            if 'wall_loss' in history:
                wall_loss_hist = history['wall_loss']
                if isinstance(wall_loss_hist, list) and len(wall_loss_hist) > 0:
                    print(f"   最終 Wall Loss: {wall_loss_hist[-1]:.6f}")
    
    # 檢查配置
    if 'config' in ckpt:
        cfg = ckpt['config']
        print(f"\n⚙️  嵌入配置:")
        print(f"   Model type: {cfg.get('model', {}).get('type', 'N/A')}")
        print(f"   Width: {cfg.get('model', {}).get('width', 'N/A')}")
        print(f"   Depth: {cfg.get('model', {}).get('depth', 'N/A')}")
        print(f"   Activation: {cfg.get('model', {}).get('activation', 'N/A')}")
        
        physics_cfg = cfg.get('physics', {})
        print(f"\n🔬 物理配置:")
        print(f"   Type: {physics_cfg.get('type', 'N/A')}")
        print(f"   nu: {physics_cfg.get('nu', 'N/A')}")
        print(f"   Re_tau: {physics_cfg.get('channel_flow', {}).get('Re_tau', 'N/A')}")
    
    # 檢查 metrics
    if 'metrics' in ckpt:
        metrics = ckpt['metrics']
        print(f"\n📈 評估指標 (epoch {ckpt['epoch']}):")
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                print(f"   {key}: {val:.6f}")
    
    # 檢查模型權重
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        print(f"\n🏗️  模型狀態:")
        print(f"   參數總數: {len(state_dict)}")
        
        # 統計參數量
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"   總參數量: {total_params:,}")
        
        # 顯示前幾層
        print(f"\n   前5個參數鍵:")
        for i, key in enumerate(list(state_dict.keys())[:5]):
            shape = state_dict[key].shape
            print(f"     {i+1}. {key}: {shape}")
    
    # 嘗試簡單前向傳播測試
    print(f"\n🧪 前向傳播測試:")
    
    if 'model_state_dict' not in ckpt:
        print(f"   ❌ 檢查點中無模型權重")
        return
    
    state_dict = ckpt['model_state_dict']
    
    try:
        from pinnx.models.fourier_mlp import create_enhanced_pinn
        
        cfg = ckpt['config']
        model_cfg = cfg['model']
        
        # 獲取 Fourier 配置
        fourier_cfg = model_cfg.get('fourier_features', {})
        fourier_m = fourier_cfg.get('fourier_m', model_cfg.get('fourier_m', 32))
        fourier_sigma = fourier_cfg.get('fourier_sigma', model_cfg.get('fourier_sigma', 1.0))
        
        # 創建模型
        model = create_enhanced_pinn(
            in_dim=3,
            out_dim=4,
            width=model_cfg['width'],
            depth=model_cfg['depth'],
            activation=model_cfg['activation'],
            use_fourier=True,
            fourier_m=fourier_m,
            fourier_sigma=fourier_sigma,
            use_rwf=model_cfg.get('use_rwf', False),
            rwf_scale_std=model_cfg.get('rwf_scale_std', 0.1)
        )
        
        # 載入權重
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        print(f"   ✅ 模型創建成功")
        
        # 測試前向傳播
        x_test = torch.rand(10, 3)  # 10 個測試點
        with torch.no_grad():
            y_pred = model(x_test)
        
        print(f"   ✅ 前向傳播成功")
        print(f"   輸入形狀: {x_test.shape}")
        print(f"   輸出形狀: {y_pred.shape}")
        print(f"\n   輸出統計 (10 隨機點):")
        print(f"     u: mean={y_pred[:, 0].mean():.4f}, std={y_pred[:, 0].std():.4f}")
        print(f"     v: mean={y_pred[:, 1].mean():.4f}, std={y_pred[:, 1].std():.4f}")
        print(f"     w: mean={y_pred[:, 2].mean():.4f}, std={y_pred[:, 2].std():.4f}")
        print(f"     p: mean={y_pred[:, 3].mean():.4f}, std={y_pred[:, 3].std():.4f}")
        
    except Exception as e:
        print(f"   ❌ 前向傳播失敗: {e}")
    
    print("\n" + "="*70)
    print("✅ 檢查點測試完成")
    print("="*70)

if __name__ == "__main__":
    main()
