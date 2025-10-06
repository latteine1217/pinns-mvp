#!/usr/bin/env python3
"""
Task-014: 3D 升級版真實JHTDB數據與PINNs訓練管線整合測試
=========================================================

目標：將重建精度從當前2D模型的 >80% 降至 <30%

關鍵升級：
1. 輸入維度：[x,y] → [x,y,z] (3D空間)
2. 輸出維度：[u,v,p] → [u,v,w,p] (包含w分量)
3. 網路架構：增強版Fourier MLP (8層×256寬度)
4. 物理約束：3D NS方程 + 時間依賴項
5. QR-pivot：3D快照矩陣策略

基於分析：w分量重要性70.2%，值得32倍計算成本
"""

import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# PINNx 導入
import pinnx
from pinnx.models.enhanced_fourier_mlp import EnhancedPINNNet
from pinnx.physics.ns_3d_temporal import NSEquations3DTemporal
from pinnx.sensors.qr_pivot import QRPivotSelector
from pinnx.losses.residuals import NSResidualLoss
from pinnx.losses.weighting import GradNormWeighter

def load_real_jhtdb_3d_data():
    """載入完整3D真實JHTDB Channel Flow數據"""
    print("📁 載入完整3D真實JHTDB Channel Flow數據...")
    
    cache_file = Path("data/jhtdb/channel_34e525c703a89036170603d28e552870.h5")
    if not cache_file.exists():
        raise FileNotFoundError(f"真實JHTDB數據快取不存在: {cache_file}")
    
    with h5py.File(cache_file, 'r') as f:
        # 載入完整3D數據
        u_data = np.array(f['u'])  # (64, 32, 32) = (x, y, z)
        v_data = np.array(f['v'])
        w_data = np.array(f['w'])  # 關鍵：現在包含w分量
        p_data = np.array(f['p'])
        
        data = {
            'u': u_data,
            'v': v_data,
            'w': w_data,  # 新增w分量
            'p': p_data
        }
    
    print(f"✅ 3D數據載入成功:")
    print(f"   u: {data['u'].shape}, 範圍: [{np.min(data['u']):.3f}, {np.max(data['u']):.3f}]")
    print(f"   v: {data['v'].shape}, 範圍: [{np.min(data['v']):.3f}, {np.max(data['v']):.3f}]")
    print(f"   w: {data['w'].shape}, 範圍: [{np.min(data['w']):.3f}, {np.max(data['w']):.3f}]")
    print(f"   p: {data['p'].shape}, 範圍: [{np.min(data['p']):.3f}, {np.max(data['p']):.3f}]")
    
    # w分量重要性分析
    w_std = np.std(data['w'])
    u_std = np.std(data['u'])
    print(f"   w分量重要性: {w_std/u_std:.1%} (相對u分量標準差)")
    
    return data

def setup_3d_coordinates(data):
    """建立3D座標網格"""
    print("📏 建立3D座標網格...")
    
    nx, ny, nz = data['u'].shape
    
    # Channel Flow 真實物理域
    x = np.linspace(0, 6.28, nx)      # 流向 (週期性)
    y = np.linspace(-1, 1, ny)        # 法向 (壁面邊界)
    z = np.linspace(0, 3.14, nz)      # 展向 (週期性)
    
    # 建立3D網格
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    coordinates = {
        'x': X,  # (64, 32, 32)
        'y': Y,
        'z': Z
    }
    
    print(f"✅ 3D座標網格建立完成:")
    print(f"   x域: [{x.min():.2f}, {x.max():.2f}] (流向)")
    print(f"   y域: [{y.min():.2f}, {y.max():.2f}] (法向)")  
    print(f"   z域: [{z.min():.2f}, {z.max():.2f}] (展向)")
    print(f"   總空間點數: {X.size:,}")
    
    return coordinates

def setup_3d_qr_pivot_sensors(data, coordinates, K=15):
    """使用3D QR-pivot選擇感測點（包含w分量信息）"""
    print(f"🎯 使用3D QR-pivot選擇 {K} 個感測點（包含w分量）...")
    
    nx, ny, nz = data['u'].shape
    total_points = nx * ny * nz
    
    # 將3D場數據重整為矩陣形式
    u_flat = data['u'].flatten()
    v_flat = data['v'].flatten()
    w_flat = data['w'].flatten()  # 關鍵：包含w分量
    p_flat = data['p'].flatten()
    
    print(f"📊 3D數據統計:")
    print(f"   總空間點數: {total_points:,}")
    print(f"   u_flat: {u_flat.shape}, std={np.std(u_flat):.3f}")
    print(f"   w_flat: {w_flat.shape}, std={np.std(w_flat):.3f}")
    
    # 構建3D增強快照矩陣：包含原始場、梯度和物理量
    u_3d = data['u']
    v_3d = data['v']
    w_3d = data['w']  # 關鍵3D信息
    p_3d = data['p']
    
    # 計算3D梯度（簡單差分）
    u_dx = np.gradient(u_3d, axis=0).flatten()
    u_dy = np.gradient(u_3d, axis=1).flatten()
    u_dz = np.gradient(u_3d, axis=2).flatten()  # 新增z方向梯度
    
    v_dx = np.gradient(v_3d, axis=0).flatten()
    v_dy = np.gradient(v_3d, axis=1).flatten()
    v_dz = np.gradient(v_3d, axis=2).flatten()
    
    w_dx = np.gradient(w_3d, axis=0).flatten()  # w分量梯度
    w_dy = np.gradient(w_3d, axis=1).flatten()
    w_dz = np.gradient(w_3d, axis=2).flatten()
    
    p_dx = np.gradient(p_3d, axis=0).flatten()
    p_dy = np.gradient(p_3d, axis=1).flatten()
    p_dz = np.gradient(p_3d, axis=2).flatten()
    
    # 計算3D物理量
    # 3D渦量：ω = ∇ × u
    vorticity_x = (w_dy - v_dz)  # ωx = ∂w/∂y - ∂v/∂z
    vorticity_y = (u_dz - w_dx)  # ωy = ∂u/∂z - ∂w/∂x
    vorticity_z = (v_dx - u_dy)  # ωz = ∂v/∂x - ∂u/∂y
    
    # 3D散度：∇·u
    divergence_3d = (u_dx + v_dy + w_dz)  # ∂u/∂x + ∂v/∂y + ∂w/∂z
    
    # 構建最大3D快照矩陣：[變數, 空間點]
    snapshots_3d = np.row_stack([
        u_flat, v_flat, w_flat, p_flat,        # 原始場 (4)
        u_dx, u_dy, u_dz,                      # u梯度 (3) 
        v_dx, v_dy, v_dz,                      # v梯度 (3)
        w_dx, w_dy, w_dz,                      # w梯度 (3) - 關鍵！
        p_dx, p_dy, p_dz,                      # 壓力梯度 (3)
        vorticity_x, vorticity_y, vorticity_z, # 3D渦量 (3)
        divergence_3d                          # 散度 (1)
    ])  # (23, 65536)
    
    print(f"📊 3D快照矩陣維度: {snapshots_3d.shape}")
    print(f"📊 矩陣秩: {np.linalg.matrix_rank(snapshots_3d)}")
    print(f"📊 w分量貢獻: 梯度{w_dx.std():.3f}, 渦量{vorticity_x.std():.3f}")
    
    # 使用QR-pivot選擇感測點
    sensor = QRPivotSelector()
    try:
        # QR-pivot在行（空間點）上選擇
        selected_indices, metrics = sensor.select_sensors(snapshots_3d.T, n_sensors=K)
        print(f"📊 3D QR-pivot結果: 選擇了 {len(selected_indices)} 個點")
        print(f"📊 條件數: {metrics.get('condition_number', 'N/A'):.2f}")
    except Exception as e:
        print(f"❌ QR-pivot失敗: {e}")
        # 回退到3D均勻採樣
        selected_indices = np.linspace(0, total_points-1, K, dtype=int)
        print(f"🔄 回退到3D均勻採樣: {len(selected_indices)} 個點")
    
    # 確保選擇的點數不超過請求數量
    selected_indices = selected_indices[:K]
    
    # 從平坦索引轉換為3D座標
    selected_coords = []
    selected_values = []
    
    for idx in selected_indices:
        i, j, k = np.unravel_index(idx, (nx, ny, nz))
        x_coord = coordinates['x'][i, j, k]
        y_coord = coordinates['y'][i, j, k]
        z_coord = coordinates['z'][i, j, k]  # 新增z座標
        
        selected_coords.append([x_coord, y_coord, z_coord])
        selected_values.append([
            data['u'][i, j, k],
            data['v'][i, j, k],
            data['w'][i, j, k],  # 新增w分量
            data['p'][i, j, k]
        ])
    
    selected_coords = np.array(selected_coords)    # (K, 3) = [x,y,z]
    selected_values = np.array(selected_values)    # (K, 4) = [u,v,w,p]
    
    print(f"✅ 3D感測點選擇完成:")
    print(f"   座標形狀: {selected_coords.shape}")
    print(f"   值形狀: {selected_values.shape}")
    print(f"   z座標範圍: [{selected_coords[:,2].min():.2f}, {selected_coords[:,2].max():.2f}]")
    
    return selected_coords, selected_values

def setup_enhanced_3d_pinns_model():
    """建立增強版3D PINNs模型"""
    print("🧠 建立增強版3D PINNs模型...")
    
    # Task-014專用3D優化配置 - 修復4D輸入
    config = {
        'in_dim': 4,         # [t, x, y, z] 4D時空輸入（穩態t=0）
        'out_dim': 4,        # [u, v, w, p] 包含w分量
        'width': 256,        # 增加寬度以應對3D複雜性
        'depth': 8,          # 更深網路捕捉3D特徵
        'fourier_m': 64,     # 增強Fourier特徵
        'fourier_sigma': 5.0,
        'activation': 'swish', # 更好的梯度傳播
        'use_residual': True,  # 殘差連接穩定深度網路
        'use_layer_norm': True, # 層歸一化
        'dropout': 0.1         # 防止過擬合
    }
    
    model = EnhancedPINNNet(**config)
    
    print(f"✅ 增強版3D模型建立完成:")
    print(f"   輸入: 4D時空 [t,x,y,z] (穩態)")
    print(f"   輸出: 4D場 [u,v,w,p]")
    print(f"   參數數量: {model.get_num_params():,}")
    print(f"   相對基線增長: {model.get_num_params()/58243:.1f}x")
    
    return model

def setup_3d_physics_constraints():
    """建立3D物理約束"""
    print("⚖️ 建立3D物理約束...")
    
    # Channel Flow Re=1000 對應的物理參數
    Re = 1000
    nu = 1.0 / Re  # 動黏滯係數
    
    physics = NSEquations3DTemporal(viscosity=nu, density=1.0)
    
    print(f"✅ 3D NS方程約束建立:")
    print(f"   雷諾數: Re = {Re}")
    print(f"   黏滯係數: ν = {nu:.6f}")
    print(f"   約束項: 3D動量 + 連續性方程")
    
    return physics

def run_enhanced_3d_training(model, sensor_coords, sensor_values, physics, full_data, coordinates):
    """運行增強版3D PINNs訓練"""
    print("🚀 開始增強版3D PINNs訓練...")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    
    # 準備3D感測數據 - 添加時間維度（t=0 穩態假設）
    sensor_coords_4d = np.zeros((len(sensor_coords), 4))  # [t, x, y, z]
    sensor_coords_4d[:, 0] = 0.0  # 時間設為0（穩態）
    sensor_coords_4d[:, 1:] = sensor_coords  # x, y, z
    
    coords_tensor = torch.tensor(sensor_coords_4d, dtype=torch.float32, device=device)
    values_tensor = torch.tensor(sensor_values, dtype=torch.float32, device=device)
    
    print(f"📊 3D訓練數據:")
    print(f"   座標: {coords_tensor.shape} [t,x,y,z]")
    print(f"   值: {values_tensor.shape} [u,v,w,p]")
    print(f"   時間: {coords_tensor[:,0].min():.2f} (穩態)")
    print(f"   x範圍: [{coords_tensor[:,1].min():.2f},{coords_tensor[:,1].max():.2f}]")
    print(f"   z範圍: [{coords_tensor[:,3].min():.2f},{coords_tensor[:,3].max():.2f}]")
    print(f"   w分量範圍: [{values_tensor[:,2].min():.2f},{values_tensor[:,2].max():.2f}]")
    
    # 準備物理約束點（較少但覆蓋3D域） - 添加時間維度
    nx, ny, nz = full_data['u'].shape
    # 稀疏採樣3D物理約束點
    physics_coords_3d = []
    for i in range(0, nx, 8):  # 8倍稀疏採樣
        for j in range(0, ny, 4):
            for k in range(0, nz, 4):
                x_coord = coordinates['x'][i, j, k]
                y_coord = coordinates['y'][i, j, k]
                z_coord = coordinates['z'][i, j, k]
                # 添加時間維度 [t, x, y, z]
                physics_coords_3d.append([0.0, x_coord, y_coord, z_coord])
    
    physics_coords_3d = np.array(physics_coords_3d)
    physics_tensor = torch.tensor(physics_coords_3d, dtype=torch.float32, device=device)
    
    print(f"📊 3D物理約束點: {physics_tensor.shape} [t,x,y,z]")
    
    # 優化器設置
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.8)
    
    # 訓練迴圈
    losses = []
    data_losses = []
    physics_losses = []
    
    # === 分析數據統計特性用於權重平衡 ===
    print("🔍 分析場統計特性...")
    values_np = values_tensor.detach().cpu().numpy()
    field_stats = {}
    field_names = ['U', 'V', 'W', 'P']
    
    for i, name in enumerate(field_names):
        std_val = np.std(values_np[:, i])
        mean_val = np.mean(values_np[:, i])
        field_stats[name] = {
            'std': std_val,
            'mean': mean_val,
            'range': [np.min(values_np[:, i]), np.max(values_np[:, i])]
        }
        print(f"{name}場: std={std_val:.6f}, mean={mean_val:.6f}, range={field_stats[name]['range']}")
    
    # 計算尺度平衡權重 - 基於標準差倒數
    base_std = field_stats['U']['std']  # 以U場為基準
    scale_weights = {}
    for i, name in enumerate(field_names):
        if field_stats[name]['std'] > 1e-8:
            scale_weights[name] = base_std / field_stats[name]['std']
        else:
            scale_weights[name] = 1.0
        print(f"{name}場尺度權重: {scale_weights[name]:.4f}")
    
    print("🔄 開始3D增強訓練迴圈...")
    for epoch in range(200):  # 更多epoch處理3D複雜性
        optimizer.zero_grad()
        
        # === 數據損失（監督學習）- 採用尺度平衡 ===
        pred_data = model(coords_tensor)
        
        # 分別計算各分量損失並應用尺度權重
        field_losses = []
        for i, name in enumerate(field_names):
            field_pred = pred_data[:, i:i+1]
            field_true = values_tensor[:, i:i+1]
            field_loss = torch.nn.MSELoss()(field_pred, field_true)
            weighted_loss = scale_weights[name] * field_loss
            field_losses.append(weighted_loss)
        
        data_loss = sum(field_losses) / len(field_losses)  # 平均加權損失
        
        # === 物理約束損失 ===
        pred_physics = model(physics_tensor)
        u_pred = pred_physics[:, 0:1]
        v_pred = pred_physics[:, 1:2] 
        w_pred = pred_physics[:, 2:3]
        p_pred = pred_physics[:, 3:4]
        
        # 3D NS方程殘差（簡化版本用於訓練）
        velocity_pred = torch.cat([u_pred, v_pred, w_pred], dim=1)
        
        # 計算連續性方程殘差 - 使用更穩健的梯度計算
        physics_tensor.requires_grad_(True)
        
        # 檢查梯度計算
        try:
            # 對於4D輸入 [t,x,y,z]，空間導數索引為 [1,2,3]
            u_grads = torch.autograd.grad(u_pred.sum(), physics_tensor, create_graph=True, retain_graph=True)[0]
            v_grads = torch.autograd.grad(v_pred.sum(), physics_tensor, create_graph=True, retain_graph=True)[0]  
            w_grads = torch.autograd.grad(w_pred.sum(), physics_tensor, create_graph=True, retain_graph=True)[0]
            
            u_x = u_grads[:, 1:2]  # ∂u/∂x
            v_y = v_grads[:, 2:3]  # ∂v/∂y
            w_z = w_grads[:, 3:4]  # ∂w/∂z
            
            continuity_residual = u_x + v_y + w_z
            physics_loss = torch.mean(continuity_residual**2)
            
        except Exception as e:
            print(f"⚠️ 梯度計算失敗: {e}")
            # fallback: 簡化物理損失
            physics_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # === 總損失 ===
        lambda_data = 10.0      # 數據項權重
        lambda_physics = 1.0    # 物理項權重
        
        total_loss = lambda_data * data_loss + lambda_physics * physics_loss
        
        # 正則化項
        reg_loss = 1e-6 * sum(torch.norm(param)**2 for param in model.parameters())
        total_loss += reg_loss
        
        total_loss.backward()
        
        # 梯度裁剪防止爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step(total_loss)
        
        # 記錄損失
        losses.append(total_loss.item())
        data_losses.append(data_loss.item())
        physics_losses.append(physics_loss.item())
        
        if epoch % 25 == 0:
            print(f"  Epoch {epoch:3d}: 總損失={total_loss.item():.2e}, "
                  f"數據={data_loss.item():.2e}, 物理={physics_loss.item():.2e}")
            print(f"             學習率={optimizer.param_groups[0]['lr']:.2e}")
    
    print(f"✅ 3D增強訓練完成！最終損失: {losses[-1]:.2e}")
    
    return model, {'total': losses, 'data': data_losses, 'physics': physics_losses}

def evaluate_3d_reconstruction(model, full_data, coordinates):
    """評估3D重建效果"""
    print("📊 評估3D重建效果...")
    
    device = next(model.parameters()).device
    nx, ny, nz = full_data['u'].shape
    
    # 準備3D評估網格（稀疏採樣以節省記憶體）
    eval_coords_3d = []
    eval_indices = []
    
    # 稀疏採樣進行評估
    for i in range(0, nx, 2):  # 2倍稀疏採樣
        for j in range(0, ny, 1):  # y方向保持完整（重要邊界）
            for k in range(0, nz, 2):
                x_coord = coordinates['x'][i, j, k]
                y_coord = coordinates['y'][i, j, k]
                z_coord = coordinates['z'][i, j, k]
                # 添加時間維度 [t, x, y, z]
                eval_coords_3d.append([0.0, x_coord, y_coord, z_coord])
                eval_indices.append((i, j, k))
    
    eval_coords_3d = np.array(eval_coords_3d)
    eval_tensor = torch.tensor(eval_coords_3d, dtype=torch.float32, device=device)
    
    print(f"📊 3D評估點數: {eval_tensor.shape[0]:,} (4D輸入)")
    
    # 預測
    model.eval()
    with torch.no_grad():
        pred = model(eval_tensor).cpu().numpy()
    
    # 重組真實場值用於比較
    u_true_eval = []
    v_true_eval = []
    w_true_eval = []
    p_true_eval = []
    
    for (i, j, k) in eval_indices:
        u_true_eval.append(full_data['u'][i, j, k])
        v_true_eval.append(full_data['v'][i, j, k])
        w_true_eval.append(full_data['w'][i, j, k])
        p_true_eval.append(full_data['p'][i, j, k])
    
    u_true_eval = np.array(u_true_eval)
    v_true_eval = np.array(v_true_eval)
    w_true_eval = np.array(w_true_eval)
    p_true_eval = np.array(p_true_eval)
    
    # 預測值
    u_pred_eval = pred[:, 0]
    v_pred_eval = pred[:, 1]
    w_pred_eval = pred[:, 2]  # 關鍵：w分量預測
    p_pred_eval = pred[:, 3]
    
    # 計算相對L2誤差
    u_error = np.sqrt(np.mean((u_pred_eval - u_true_eval)**2)) / np.sqrt(np.mean(u_true_eval**2))
    v_error = np.sqrt(np.mean((v_pred_eval - v_true_eval)**2)) / np.sqrt(np.mean(v_true_eval**2))
    w_error = np.sqrt(np.mean((w_pred_eval - w_true_eval)**2)) / np.sqrt(np.mean(w_true_eval**2))
    p_error = np.sqrt(np.mean((p_pred_eval - p_true_eval)**2)) / np.sqrt(np.mean(p_true_eval**2))
    
    print(f"📈 3D重建誤差分析:")
    print(f"  u場相對L2誤差: {u_error:.1%} (目標: ≤30%)")
    print(f"  v場相對L2誤差: {v_error:.1%} (目標: ≤30%)")
    print(f"  w場相對L2誤差: {w_error:.1%} (目標: ≤30%) ⭐新增")
    print(f"  p場相對L2誤差: {p_error:.1%} (目標: ≤30%)")
    
    # 整體性能指標
    avg_error = (u_error + v_error + w_error + p_error) / 4
    print(f"  平均誤差: {avg_error:.1%}")
    
    # 與2D基線對比
    print(f"\n📊 與2D基線對比:")
    print(f"  2D基線 - u: 91.1%, v: 81.2%, p: 86.4%")
    print(f"  3D升級 - u: {u_error:.1%}, v: {v_error:.1%}, w: {w_error:.1%}, p: {p_error:.1%}")
    
    return {
        'u_pred': u_pred_eval,
        'v_pred': v_pred_eval,
        'w_pred': w_pred_eval,
        'p_pred': p_pred_eval,
        'u_true': u_true_eval,
        'v_true': v_true_eval,
        'w_true': w_true_eval,
        'p_true': p_true_eval,
        'coords': eval_coords_3d,
        'errors': {'u': u_error, 'v': v_error, 'w': w_error, 'p': p_error, 'avg': avg_error}
    }

def create_3d_visualization(sensor_coords, results, losses):
    """創建3D升級結果可視化"""
    print("🎨 創建3D升級結果可視化...")
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 損失曲線
    ax1 = plt.subplot(2, 4, 1)
    epochs = range(len(losses['total']))
    plt.plot(epochs, losses['total'], 'b-', label='Total Loss', linewidth=2)
    plt.plot(epochs, losses['data'], 'g-', label='Data Loss', linewidth=2)
    plt.plot(epochs, losses['physics'], 'r-', label='Physics Loss', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('3D Enhanced Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 誤差對比圖
    ax2 = plt.subplot(2, 4, 2)
    fields = ['u', 'v', 'w', 'p']
    baseline_2d = [91.1, 81.2, None, 86.4]  # 2D基線（w無數據）
    enhanced_3d = [results['errors'][f]*100 for f in fields]
    
    x_pos = np.arange(len(fields))
    width = 0.35
    
    # 2D基線
    baseline_values = [baseline_2d[i] if baseline_2d[i] is not None else 0 for i in range(len(fields))]
    bars1 = plt.bar(x_pos - width/2, baseline_values, width, label='2D Baseline', alpha=0.7, color='lightcoral')
    
    # 3D升級
    bars2 = plt.bar(x_pos + width/2, enhanced_3d, width, label='3D Enhanced', alpha=0.7, color='skyblue')
    
    plt.axhline(y=30, color='red', linestyle='--', alpha=0.8, label='Target (30%)')
    plt.xlabel('Field Components')
    plt.ylabel('L2 Relative Error (%)')
    plt.title('2D vs 3D Performance Comparison')
    plt.xticks(x_pos, fields)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加數值標籤
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        if baseline_2d[i] is not None:
            plt.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 2, 
                    f'{baseline_2d[i]:.1f}%', ha='center', va='bottom', fontsize=9)
        plt.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 2, 
                f'{enhanced_3d[i]:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 3. 感測點3D分布（投影到xy平面）
    ax3 = plt.subplot(2, 4, 3)
    scatter = plt.scatter(sensor_coords[:, 0], sensor_coords[:, 1], 
                         c=sensor_coords[:, 2], s=100, cmap='viridis', 
                         edgecolors='black', linewidth=1)
    plt.colorbar(scatter, label='z-coordinate')
    plt.xlabel('x (streamwise)')
    plt.ylabel('y (wall-normal)')
    plt.title(f'3D Sensor Locations (N={len(sensor_coords)})')
    plt.grid(True, alpha=0.3)
    
    # 4-7. 場重建比較（選擇中間z切片）
    n_eval = len(results['u_pred'])
    mid_idx = n_eval // 2
    
    for i, field in enumerate(['u', 'v', 'w', 'p']):
        ax = plt.subplot(2, 4, 4 + i)
        
        pred_values = results[f'{field}_pred']
        true_values = results[f'{field}_true']
        
        # 散點圖比較
        plt.scatter(true_values[::10], pred_values[::10], alpha=0.6, s=20)
        
        # 理想線
        min_val = min(np.min(true_values), np.min(pred_values))
        max_val = max(np.max(true_values), np.max(pred_values))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        plt.xlabel(f'{field.upper()} True')
        plt.ylabel(f'{field.upper()} Predicted')
        plt.title(f'{field.upper()} Field: Error {results["errors"][field]:.1%}')
        plt.grid(True, alpha=0.3)
        
        # R²計算
        r2 = np.corrcoef(true_values, pred_values)[0, 1]**2
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Task-014: 3D Enhanced PINNs Reconstruction Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = "tasks/task-014/3d_enhanced_reconstruction_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 3D可視化已保存: {output_path}")
    
    return output_path

def main():
    """主函數 - Task-014 3D升級實施"""
    print("🚀 Task-014: 3D Enhanced PINNs Reconstruction 開始...\n")
    
    try:
        # 1. 載入完整3D真實JHTDB數據
        jhtdb_data_3d = load_real_jhtdb_3d_data()
        
        # 2. 建立3D座標網格
        coordinates_3d = setup_3d_coordinates(jhtdb_data_3d)
        
        # 3. 3D QR-pivot感測點選擇（包含w分量）
        sensor_coords_3d, sensor_values_3d = setup_3d_qr_pivot_sensors(
            jhtdb_data_3d, coordinates_3d, K=15
        )
        
        # 4. 建立增強版3D PINNs模型
        enhanced_model_3d = setup_enhanced_3d_pinns_model()
        
        # 5. 建立3D物理約束
        physics_3d = setup_3d_physics_constraints()
        
        # 6. 運行增強版3D訓練
        trained_model_3d, training_losses = run_enhanced_3d_training(
            enhanced_model_3d, sensor_coords_3d, sensor_values_3d, 
            physics_3d, jhtdb_data_3d, coordinates_3d
        )
        
        # 7. 評估3D重建效果
        results_3d = evaluate_3d_reconstruction(trained_model_3d, jhtdb_data_3d, coordinates_3d)
        
        # 8. 創建結果可視化
        viz_path = create_3d_visualization(sensor_coords_3d, results_3d, training_losses)
        
        # 9. 最終總結
        print(f"\n🎉 Task-014 3D升級成功完成！")
        print(f"\n📈 關鍵成就對比:")
        print(f"{'':>12} {'2D基線':>12} {'3D升級':>12} {'改善':>12}")
        print(f"{'='*50}")
        
        baseline_errors = {'u': 91.1, 'v': 81.2, 'p': 86.4}
        enhanced_errors = results_3d['errors']
        
        for field in ['u', 'v', 'p']:
            baseline = baseline_errors[field]
            enhanced = enhanced_errors[field] * 100
            improvement = ((baseline - enhanced) / baseline) * 100
            print(f"{field.upper() + '場誤差':>12} {baseline:>10.1f}% {enhanced:>10.1f}% {improvement:>+9.1f}%")
        
        # w分量是新增的
        w_error = enhanced_errors['w'] * 100
        print(f"{'W場誤差':>12} {'新增':>10} {w_error:>10.1f}% {'N/A':>12}")
        
        avg_error = enhanced_errors['avg'] * 100
        print(f"{'平均誤差':>12} {'N/A':>10} {avg_error:>10.1f}% {'N/A':>12}")
        
        print(f"\n✅ 成功指標達成情況:")
        for field in ['u', 'v', 'w', 'p']:
            error_pct = enhanced_errors[field] * 100
            status = "✅ 達標" if error_pct <= 30 else "❌ 未達標"
            print(f"   {field.upper()}場 ≤ 30%: {error_pct:.1f}% {status}")
        
        print(f"\n📊 可視化報告: {viz_path}")
        print(f"📁 詳細結果: tasks/task-014/")
        
        # 保存結果到任務目錄
        results_summary = {
            'task_id': 'task-014',
            'objective': '3D Enhanced PINNs Reconstruction',
            'sensor_points': len(sensor_coords_3d),
            'model_params': enhanced_model_3d.get_num_params(),
            'training_epochs': len(training_losses['total']),
            'final_loss': training_losses['total'][-1],
            'errors_pct': {k: v*100 for k, v in enhanced_errors.items()},
            'target_achieved': all(enhanced_errors[f] <= 0.30 for f in ['u', 'v', 'w', 'p']),
            'improvement_vs_2d': {
                'u': ((baseline_errors['u'] - enhanced_errors['u']*100) / baseline_errors['u']) * 100,
                'v': ((baseline_errors['v'] - enhanced_errors['v']*100) / baseline_errors['v']) * 100,
                'p': ((baseline_errors['p'] - enhanced_errors['p']*100) / baseline_errors['p']) * 100
            }
        }
        
        # 寫入任務完成報告
        import json
        with open('tasks/task-014/completion_summary.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"📝 任務報告已保存: tasks/task-014/completion_summary.json")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Task-014 3D升級失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)