#!/usr/bin/env python3
"""
分層採樣視覺化腳本
驗證邊界點和內部點的分佈是否符合預期
"""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 導入採樣函數
import yaml
from scripts.train import sample_boundary_points, sample_interior_points


def visualize_sampling():
    """視覺化分層採樣結果"""
    
    # 載入配置
    config_path = project_root / "configs" / "vs_pinn_stratified_sampling_test.yml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # 提取採樣參數
    sampling_cfg = cfg['training']['sampling']
    domain_cfg = cfg['physics']['domain']
    
    # 域範圍
    x_range = domain_cfg['x_range']
    y_range = domain_cfg['y_range']
    z_range = domain_cfg['z_range']
    
    # 邊界點配置
    boundary_dist = sampling_cfg['boundary_distribution']
    n_boundary = sum(boundary_dist.values())
    
    # 內部點配置
    n_interior = sampling_cfg['interior_points']
    use_sobol = sampling_cfg['use_sobol']
    
    print("=" * 60)
    print("分層採樣視覺化測試")
    print("=" * 60)
    print(f"域範圍: x={x_range}, y={y_range}, z={z_range}")
    print(f"邊界點: {n_boundary} 個")
    print(f"  - 壁面: {boundary_dist['wall']}")
    print(f"  - 週期性: {boundary_dist['periodic']}")
    print(f"  - Inlet: {boundary_dist['inlet']}")
    print(f"內部點: {n_interior} 個 (Sobol={use_sobol})")
    print()
    
    # 生成採樣點
    print("🔄 生成採樣點...")
    
    # 構建域邊界字典
    domain_bounds = {
        'x': tuple(x_range),
        'y': tuple(y_range),
        'z': tuple(z_range)
    }
    
    # 設定 device
    device = torch.device('cpu')
    
    boundary_points = sample_boundary_points(
        n_boundary,
        domain_bounds,
        device,
        distribution=boundary_dist
    )
    
    interior_points = sample_interior_points(
        n_interior,
        domain_bounds,
        device,
        exclude_boundary_tol=0.01,
        use_sobol=use_sobol
    )
    
    print(f"✅ 邊界點形狀: {boundary_points.shape}")
    print(f"✅ 內部點形狀: {interior_points.shape}")
    print()
    
    # 轉換為 numpy 以便繪圖
    boundary_np = boundary_points.cpu().numpy()
    interior_np = interior_points.cpu().numpy()
    
    # 統計資訊
    print("📊 採樣統計:")
    print(f"邊界點範圍:")
    print(f"  x: [{boundary_np[:, 0].min():.4f}, {boundary_np[:, 0].max():.4f}]")
    print(f"  y: [{boundary_np[:, 1].min():.4f}, {boundary_np[:, 1].max():.4f}]")
    print(f"  z: [{boundary_np[:, 2].min():.4f}, {boundary_np[:, 2].max():.4f}]")
    print(f"內部點範圍:")
    print(f"  x: [{interior_np[:, 0].min():.4f}, {interior_np[:, 0].max():.4f}]")
    print(f"  y: [{interior_np[:, 1].min():.4f}, {interior_np[:, 1].max():.4f}]")
    print(f"  z: [{interior_np[:, 2].min():.4f}, {interior_np[:, 2].max():.4f}]")
    print()
    
    # 驗證邊界點
    tol = 0.01
    n_wall_lower = np.sum(np.abs(boundary_np[:, 1] - y_range[0]) < tol)
    n_wall_upper = np.sum(np.abs(boundary_np[:, 1] - y_range[1]) < tol)
    n_periodic_x = np.sum((np.abs(boundary_np[:, 0] - x_range[0]) < tol) | 
                          (np.abs(boundary_np[:, 0] - x_range[1]) < tol))
    n_periodic_z = np.sum((np.abs(boundary_np[:, 2] - z_range[0]) < tol) | 
                          (np.abs(boundary_np[:, 2] - z_range[1]) < tol))
    
    print("✅ 邊界點驗證:")
    print(f"  下壁面 (y={y_range[0]}): {n_wall_lower} 點")
    print(f"  上壁面 (y={y_range[1]}): {n_wall_upper} 點")
    print(f"  x 週期性邊界: {n_periodic_x} 點")
    print(f"  z 週期性邊界: {n_periodic_z} 點")
    print()
    
    # 創建可視化
    fig = plt.figure(figsize=(16, 10))
    
    # 3D 散點圖
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(interior_np[:, 0], interior_np[:, 1], interior_np[:, 2],
                c='blue', marker='.', alpha=0.3, label='Interior')
    ax1.scatter(boundary_np[:, 0], boundary_np[:, 1], boundary_np[:, 2],
                c='red', marker='o', alpha=0.8, label='Boundary')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')  # type: ignore
    ax1.set_title('3D Stratified Sampling')
    ax1.legend()
    
    # xy 平面投影
    ax2 = fig.add_subplot(222)
    ax2.scatter(interior_np[:, 0], interior_np[:, 1], c='blue', s=1, alpha=0.3, label='Interior')
    ax2.scatter(boundary_np[:, 0], boundary_np[:, 1], c='red', s=5, alpha=0.8, label='Boundary')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('XY Projection')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # xz 平面投影
    ax3 = fig.add_subplot(223)
    ax3.scatter(interior_np[:, 0], interior_np[:, 2], c='blue', s=1, alpha=0.3, label='Interior')
    ax3.scatter(boundary_np[:, 0], boundary_np[:, 2], c='red', s=5, alpha=0.8, label='Boundary')
    ax3.set_xlabel('x')
    ax3.set_ylabel('z')
    ax3.set_title('XZ Projection')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # yz 平面投影
    ax4 = fig.add_subplot(224)
    ax4.scatter(interior_np[:, 1], interior_np[:, 2], c='blue', s=1, alpha=0.3, label='Interior')
    ax4.scatter(boundary_np[:, 1], boundary_np[:, 2], c='red', s=5, alpha=0.8, label='Boundary')
    ax4.set_xlabel('y')
    ax4.set_ylabel('z')
    ax4.set_title('YZ Projection')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存圖片
    output_path = project_root / "results" / "stratified_sampling_visualization.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 可視化圖片已保存: {output_path}")
    
    # 顯示圖片（如果在互動環境中）
    # plt.show()


if __name__ == "__main__":
    visualize_sampling()
