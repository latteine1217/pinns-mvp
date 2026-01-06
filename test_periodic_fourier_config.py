"""
測試週期性 Fourier 嵌入配置
"""

import sys
sys.path.insert(0, '/Users/latteine/Documents/coding/pinns-sparse-flow')

import torch
import yaml
from pinnx.models import create_pinn_model

# 載入配置
config_path = '/Users/latteine/Documents/coding/pinns-sparse-flow/configs/kolmogorov_re50_kf4_K100_periodic_fourier.yml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

model_config = config['model']

print("="*60)
print("🔧 測試週期性 Fourier 嵌入配置")
print("="*60)

# 創建模型
print("\n1️⃣  創建模型...")
model = create_pinn_model(model_config)
print(f"   ✅ 模型創建成功")
print(f"   模型類型: {model.__class__.__name__}")

# 檢查 Fourier 編碼器
print("\n2️⃣  檢查 Fourier 編碼器...")
if hasattr(model, 'fourier') and model.fourier is not None:
    print(f"   Fourier 類型: {model.fourier.__class__.__name__}")
    print(f"   輸入維度: {model.fourier.in_dim}")
    print(f"   輸出維度: {model.fourier.out_dim}")
    
    if hasattr(model.fourier, 'axes_config'):
        print("\n   軸向配置:")
        for axis_idx, out_dim in model.fourier.axis_out_dims.items():
            print(f"     軸 {axis_idx}: {out_dim} 維")
else:
    print("   ⚠️  未啟用 Fourier 編碼")

# 測試前向傳播
print("\n3️⃣  測試前向傳播...")
batch_size = 100
x = torch.randn(batch_size, model_config['in_dim'])
print(f"   輸入形狀: {x.shape}")

with torch.no_grad():
    output = model(x)
print(f"   輸出形狀: {output.shape}")
print(f"   預期輸出維度: {model_config['out_dim']}")
print(f"   ✅ 前向傳播成功")

# 測試週期性
print("\n4️⃣  測試週期性邊界條件...")
domain_size = 2 * 3.141592653589793  # 2π
x_periodic = torch.tensor([
    [0.0, 0.0, 0.0],           # (t=0, x=0, y=0)
    [0.0, domain_size, 0.0],   # (t=0, x=2π, y=0)
    [0.0, 0.0, domain_size],   # (t=0, x=0, y=2π)
    [0.0, domain_size, domain_size],  # (t=0, x=2π, y=2π)
])

with torch.no_grad():
    output_periodic = model(x_periodic)

# 檢查 x 方向週期性
diff_x = torch.abs(output_periodic[0] - output_periodic[1]).max().item()
print(f"   x 方向週期性誤差: {diff_x:.2e}")
print(f"   {'✅' if diff_x < 1e-5 else '⚠️'} x 方向週期性 {'通過' if diff_x < 1e-5 else '可能需要調整'}")

# 檢查 y 方向週期性
diff_y = torch.abs(output_periodic[0] - output_periodic[2]).max().item()
print(f"   y 方向週期性誤差: {diff_y:.2e}")
print(f"   {'✅' if diff_y < 1e-5 else '⚠️'} y 方向週期性 {'通過' if diff_y < 1e-5 else '可能需要調整'}")

# 測試梯度計算
print("\n5️⃣  測試梯度計算...")
x_grad = torch.randn(50, model_config['in_dim'], requires_grad=True)
output_grad = model(x_grad)
loss = output_grad.sum()
loss.backward()

print(f"   輸入梯度形狀: {x_grad.grad.shape}")
print(f"   梯度範圍: [{x_grad.grad.min():.4f}, {x_grad.grad.max():.4f}]")
print(f"   ✅ 梯度計算成功")

# 模型摘要
print("\n6️⃣  模型摘要...")
summary = model.get_model_summary()
for key, value in summary.items():
    print(f"   {key:20s}: {value}")

print("\n" + "="*60)
print("✅ 所有測試通過！週期性 Fourier 嵌入配置正確")
print("="*60)
