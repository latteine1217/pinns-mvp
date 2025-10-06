#!/usr/bin/env python3
"""
Task-014: 3D PINNs重建精度優化 - QR Pivoting 50點 + 5000 Epochs 超長訓練版
===============================================================================

終極挑戰策略:
1. 🎯 使用QR Pivoting算法選擇50個最優感測點
2. 🚀 超長5000 epochs訓練以充分學習稀疏數據
3. 🎯 保持成功的V場權重平衡策略
4. 🎯 添加學習率衰減與早停機制
5. 🎯 增強的正則化策略防止過擬合

目標: 驗證極少點數(50)下的重建極限能力
"""

import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from scipy.linalg import qr
warnings.filterwarnings("ignore")

def load_and_process_jhtdb_3d():
    """載入並處理JHTDB 3D數據"""
    print("📁 載入JHTDB 3D數據...")
    
    cache_file = Path("data/jhtdb/channel_34e525c703a89036170603d28e552870.h5")
    if not cache_file.exists():
        raise FileNotFoundError(f"數據文件不存在: {cache_file}")
    
    with h5py.File(cache_file, 'r') as f:
        # 讀取3D網格數據 [64,32,32]
        u_grid = np.array(f['u'])
        v_grid = np.array(f['v'])
        w_grid = np.array(f['w'])  
        p_grid = np.array(f['p'])
        
        print(f"   原始網格: {u_grid.shape}")
        
        # 創建3D坐標網格
        nz, ny, nx = u_grid.shape
        
        z_coords = np.linspace(0, 2*np.pi, nz)
        y_coords = np.linspace(-1, 1, ny)
        x_coords = np.linspace(0, np.pi, nx)
        t_coord = 0.0
        
        T, Z, Y, X = np.meshgrid([t_coord], z_coords, y_coords, x_coords, indexing='ij')
        
        coords = np.column_stack([
            T.flatten(),  # t
            X.flatten(),  # x (展向)
            Y.flatten(),  # y (壁面法向)
            Z.flatten()   # z (流向)
        ])
        
        values = np.column_stack([
            u_grid.flatten(),
            v_grid.flatten(), 
            w_grid.flatten(),
            p_grid.flatten()
        ])
        
        print(f"   坐標維度: {coords.shape}")
        print(f"   狀態維度: {values.shape}")
        
        for i, name in enumerate(['U', 'V', 'W', 'P']):
            print(f"   {name}: mean={values[:,i].mean():.6f}, std={values[:,i].std():.6f}")
        
        return coords, values

def qr_pivot_sensor_selection(coords, values, n_sensors=50):
    """使用QR Pivoting選擇最優感測點"""
    print(f"\n🎯 使用QR Pivoting選擇{n_sensors}個最優感測點...")
    
    # 準備數據矩陣：使用速度場構建感測矩陣
    # 取前3個場 (u,v,w) 作為感測目標，壓力場作為重建目標
    data_matrix = values[:, :3]  # [n_points, 3] (u,v,w)
    
    print(f"   數據矩陣: {data_matrix.shape}")
    print(f"   選擇點數: {n_sensors}")
    
    # 對數據矩陣進行QR分解選主元
    try:
        # 轉置數據矩陣以便選擇空間點
        X = data_matrix.T  # [3, n_points]
        Q, R, piv = qr(X, mode='economic', pivoting=True)
        
        # 選擇前n_sensors個主元點
        selected_indices = piv[:n_sensors]
        
        print(f"   ✅ QR Pivoting成功選擇{len(selected_indices)}個點")
        
        # 計算選擇點的空間分佈
        selected_coords = coords[selected_indices]
        x_range = [selected_coords[:, 1].min(), selected_coords[:, 1].max()]
        y_range = [selected_coords[:, 2].min(), selected_coords[:, 2].max()]
        z_range = [selected_coords[:, 3].min(), selected_coords[:, 3].max()]
        
        print(f"   📍 選擇點空間分佈:")
        print(f"      X(展向): [{x_range[0]:.3f}, {x_range[1]:.3f}]")
        print(f"      Y(壁面): [{y_range[0]:.3f}, {y_range[1]:.3f}]")
        print(f"      Z(流向): [{z_range[0]:.3f}, {z_range[1]:.3f}]")
        
        # 計算條件數作為品質指標
        selected_data = data_matrix[selected_indices, :]
        cond_number = np.linalg.cond(selected_data)
        print(f"   📊 選擇數據條件數: {cond_number:.2e}")
        
        return selected_indices
        
    except Exception as e:
        print(f"   ❌ QR Pivoting失敗: {e}")
        print(f"   🔄 回退到隨機選擇...")
        return np.random.choice(len(coords), n_sensors, replace=False)

class UltraLongTrain3DNet(torch.nn.Module):
    """超長訓練專用3D PINNs網路"""
    
    def __init__(self, layers=[4, 128, 128, 128, 128, 4]):
        super().__init__()
        
        self.layers = torch.nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(torch.nn.Linear(layers[i], layers[i+1]))
            
        # Dropout防止過擬合（僅在極長訓練時啟用）
        self.dropout = torch.nn.Dropout(0.1)
        self.training_mode = True
        
        # 成功的初始化策略
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        
        print(f"   🎯 UltraLongTrain3D網路: {layers}")
        print(f"   📊 參數總數: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x):
        """前向傳播（支援dropout）"""
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.tanh(layer(x))
            # 在中間層添加輕微dropout
            if self.training_mode and i > 0:
                x = self.dropout(x)
        x = self.layers[-1](x)
        return x

def compute_physics_residual_ultra(model, coords_tensor):
    """超長訓練的物理約束計算"""
    coords_tensor.requires_grad_(True)
    pred = model(coords_tensor)
    
    u = pred[:, 0:1]
    v = pred[:, 1:2]
    w = pred[:, 2:3]
    p = pred[:, 3:4]
    
    # 梯度計算
    u_grads = torch.autograd.grad(u.sum(), coords_tensor, create_graph=True, retain_graph=True)[0]
    v_grads = torch.autograd.grad(v.sum(), coords_tensor, create_graph=True, retain_graph=True)[0]
    w_grads = torch.autograd.grad(w.sum(), coords_tensor, create_graph=True, retain_graph=True)[0]
    
    # 連續性方程
    u_x = u_grads[:, 1:2]
    v_y = v_grads[:, 2:3]
    w_z = w_grads[:, 3:4]
    
    continuity = u_x + v_y + w_z
    
    # V場壁面約束
    y_coords = coords_tensor[:, 2:3]
    wall_mask = (torch.abs(torch.abs(y_coords) - 1.0) < 0.1).squeeze()
    
    if wall_mask.sum() > 0:
        v_wall = v[wall_mask]
        wall_constraint = torch.mean(v_wall**2)
    else:
        wall_constraint = torch.tensor(0.0)
    
    return continuity, wall_constraint

def train_ultra_long_5000epochs():
    """QR Pivoting 50點 + 5000 epochs超長訓練"""
    print("🚀 QR Pivoting 50點 + 5000 Epochs 超長訓練...")
    
    # 載入數據
    coords, values = load_and_process_jhtdb_3d()
    
    # 🎯 使用QR Pivoting選擇50個最優點
    selected_indices = qr_pivot_sensor_selection(coords, values, n_sensors=50)
    train_coords = coords[selected_indices]
    train_values = values[selected_indices]
    
    print(f"\n📈 最終訓練配置:")
    print(f"   訓練點數: {len(train_coords)} (QR Pivoting選擇)")
    print(f"   訓練輪數: 5000 epochs")
    
    # 轉換為tensor
    coords_tensor = torch.FloatTensor(train_coords)
    values_tensor = torch.FloatTensor(train_values)
    
    # 物理約束點（使用更多點以保持物理一致性）
    n_physics = 200
    physics_indices = np.random.choice(len(coords), n_physics, replace=False)
    physics_tensor = torch.FloatTensor(coords[physics_indices])
    
    # === 尺度權重計算 ===
    field_names = ['U', 'V', 'W', 'P']
    field_stds = [values_tensor[:, i].std().item() for i in range(4)]
    base_std = field_stds[0]
    
    scale_weights = []
    for i, name in enumerate(field_names):
        if field_stds[i] > 1e-8:
            weight = base_std / field_stds[i]
            if name == 'V':
                weight *= 1.5  # V場增強
        else:
            weight = 1.0
        scale_weights.append(weight)
        print(f"   {name}場 std={field_stds[i]:.6f}, 權重={weight:.4f}")
    
    scale_weights = torch.FloatTensor(scale_weights)
    
    # 初始化網路
    model = UltraLongTrain3DNet()
    
    # 🎯 自適應學習率調度 (針對超長訓練)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 多階段學習率衰減
    def lr_scheduler(epoch):
        if epoch < 1000:
            return 1.0  # 初期保持原學習率
        elif epoch < 2000:
            return 0.5  # 中期減半
        elif epoch < 4000:
            return 0.2  # 後期大幅降低
        else:
            return 0.1  # 最終階段精細調整
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scheduler)
    
    # 早停機制
    best_loss = float('inf')
    patience = 500
    patience_counter = 0
    
    losses = []
    v_errors = []  # 專門追蹤V場誤差
    
    print("\n🔄 開始超長訓練 (5000 epochs)...")
    
    for epoch in range(5000):
        optimizer.zero_grad()
        
        # === 數據損失計算 ===
        pred_data = model(coords_tensor)
        
        field_losses = {}
        total_weighted_loss = 0
        
        for i, name in enumerate(field_names):
            field_pred = pred_data[:, i:i+1]
            field_true = values_tensor[:, i:i+1]
            field_mse = torch.nn.MSELoss()(field_pred, field_true)
            
            weighted_loss = scale_weights[i] * field_mse
            field_losses[name] = weighted_loss
            total_weighted_loss += weighted_loss
            
            # V場專項追蹤
            if name == 'V':
                v_rmse = torch.sqrt(field_mse)
                v_errors.append(v_rmse.item())
        
        data_loss = total_weighted_loss / len(field_names)
        
        # === 物理約束損失 ===
        try:
            continuity, wall_v = compute_physics_residual_ultra(model, physics_tensor)
            continuity_loss = torch.mean(continuity**2)
            wall_loss = wall_v
        except:
            continuity_loss = torch.tensor(0.0)
            wall_loss = torch.tensor(0.0)
        
        # === 正則化損失 (防止過擬合) ===
        l2_reg = 0
        for param in model.parameters():
            l2_reg += torch.norm(param)
        l2_lambda = 1e-6  # 輕微L2正則化
        
        # === 總損失 ===
        lambda_data = 10.0
        lambda_physics = 1.0
        lambda_wall = 2.0
        
        total_loss = (lambda_data * data_loss + 
                     lambda_physics * continuity_loss +
                     lambda_wall * wall_loss +
                     l2_lambda * l2_reg)
        
        total_loss.backward()
        
        # 梯度裁剪 (穩定超長訓練)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        losses.append(total_loss.item())
        
        # 早停檢查
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 詳細進度報告
        if epoch % 500 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            v_current = v_errors[-1] if v_errors else 0
            print(f"Epoch {epoch:4d}: 總損失={total_loss.item():.4f}, "
                  f"數據={data_loss.item():.4f}, 物理={continuity_loss.item():.6f}, "
                  f"V場RMSE={v_current:.6f}, LR={current_lr:.6f}")
        
        # 早停條件
        if patience_counter >= patience and epoch > 2000:
            print(f"   🛑 早停於 epoch {epoch} (patience={patience})")
            break
    
    print("✅ 超長訓練完成!")
    print(f"   🎯 最佳損失: {best_loss:.6f}")
    print(f"   🎯 實際訓練輪數: {epoch+1}")
    
    # 關閉dropout進行評估
    model.training_mode = False
    
    # 評估
    errors, msg = evaluate_ultra_performance(model, coords, values, train_coords, train_values)
    
    return model, losses, errors, v_errors

def evaluate_ultra_performance(model, coords, values, train_coords, train_values):
    """評估超長訓練版性能"""
    print("\n🔍 評估QR Pivoting + 5000 Epochs性能...")
    
    coords_tensor = torch.FloatTensor(coords)
    
    with torch.no_grad():
        pred_full = model(coords_tensor).numpy()
    
    field_names = ['U', 'V', 'W', 'P']
    print("=== QR Pivoting + 超長訓練重建誤差分析 ===")
    
    errors = []
    for i, name in enumerate(field_names):
        true_field = values[:, i]
        pred_field = pred_full[:, i]
        
        l2_error = np.sqrt(np.mean((pred_field - true_field)**2))
        rel_error = l2_error / (np.sqrt(np.mean(true_field**2)) + 1e-10) * 100
        
        errors.append(rel_error)
        print(f"{name}場: L2={l2_error:.6f}, 相對誤差={rel_error:.1f}%")
    
    avg_error = np.mean(errors)
    print(f"\n📊 QR Pivoting + 超長訓練平均相對誤差: {avg_error:.1f}%")
    
    # 與目標對比
    if avg_error < 30.0:
        print("🎉 成功達到<30%目標!")
        success_msg = "✅ QR Pivoting + 5000 epochs 成功達標"
    else:
        print(f"⚠️  向目標努力 (目標<30%)")
        success_msg = f"🔄 QR Pivoting + 5000 epochs ({avg_error:.1f}% vs 30%)"
    
    # 與之前版本對比
    print(f"\n📈 與數據增強版對比:")
    data_enhanced_avg = 33.0
    improvement = data_enhanced_avg - avg_error
    print(f"   數據增強版(2048點): {data_enhanced_avg:.1f}%")
    print(f"   QR Pivoting版(50點): {avg_error:.1f}%")
    print(f"   改善: {improvement:.1f}% (使用{2048/50:.1f}x更少數據)")
    
    # 訓練點性能分析
    print(f"\n🎯 訓練點重建性能:")
    train_tensor = torch.FloatTensor(train_coords)
    with torch.no_grad():
        pred_train = model(train_tensor).numpy()
    
    train_errors = []
    for i, name in enumerate(field_names):
        true_train = train_values[:, i]
        pred_train_field = pred_train[:, i]
        train_error = np.sqrt(np.mean((pred_train_field - true_train)**2)) / (np.sqrt(np.mean(true_train**2)) + 1e-10) * 100
        train_errors.append(train_error)
        print(f"   {name}場訓練誤差: {train_error:.1f}%")
    
    train_avg = np.mean(train_errors)
    generalization = avg_error - train_avg
    print(f"   訓練平均誤差: {train_avg:.1f}%")
    print(f"   泛化差距: {generalization:.1f}%")
    
    # V場專項分析
    v_error = errors[1]
    v_improvement = 214.6 - v_error
    print(f"\n🎯 V場專項成就:")
    print(f"   原始V場誤差: 214.6% → QR版: {v_error:.1f}% (改善 {v_improvement:.1f}%)")
    
    # QR Pivoting效果分析
    print(f"\n🧮 QR Pivoting選點效果:")
    print(f"   選擇點數: 50 (總數的{50/len(coords)*100:.3f}%)")
    print(f"   數據效率: {2048/50:.1f}x點數減少")
    print(f"   計算效率: 5000 vs 600 epochs (8.3x訓練時間)")
    
    return errors, success_msg

if __name__ == "__main__":
    print("=" * 80)
    print("🎯 Task-014: QR Pivoting 50點 + 5000 Epochs 超長訓練挑戰")
    print("=" * 80)
    
    try:
        model, losses, errors, v_errors = train_ultra_long_5000epochs()
        print("\n🎉 超長訓練挑戰完成!")
        
        # 可視化V場收斂過程
        if len(v_errors) > 100:
            plt.figure(figsize=(10, 6))
            plt.plot(v_errors[::10])  # 每10個epoch取一個點
            plt.title('V Field RMSE Convergence (QR Pivoting + 5000 Epochs)')
            plt.xlabel('Epoch (×10)')
            plt.ylabel('V Field RMSE')
            plt.yscale('log')
            plt.grid(True)
            plt.savefig('qr_pivot_v_convergence.png', dpi=150, bbox_inches='tight')
            print("   📊 V場收斂圖已保存: qr_pivot_v_convergence.png")
        
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()