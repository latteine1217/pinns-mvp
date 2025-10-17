#!/usr/bin/env python3
"""
生成 Wall-Clustered 無中心層感測點
用於消融實驗 3：模擬 QR-Pivot 的分層分佈（僅壁面層 + 對數層）
"""

import numpy as np
import argparse
from pathlib import Path


def generate_wall_no_center_sensors(
    K: int = 50,
    domain_x: tuple = (0.0, 25.13),
    domain_y: tuple = (-1.0, 1.0),
    domain_z: float = 4.71,  # 2D 切片 z 位置
    wall_fraction: float = 0.52,  # 52% 壁面層（與 QR-Pivot 相同）
    log_fraction: float = 0.48,   # 48% 對數層（與 QR-Pivot 相同）
    seed: int = 42,
):
    """
    生成僅含壁面層與對數層的感測點（無中心層）
    
    分層定義：
    - 壁面層：y ∈ [0.8, 1.0] ∪ [-1.0, -0.8]
    - 對數層：y ∈ [0.2, 0.8] ∪ [-0.8, -0.2]
    - 中心層：無點
    """
    np.random.seed(seed)
    
    # 計算各層點數
    K_wall = int(K * wall_fraction)
    K_log = K - K_wall  # 剩餘點數全給對數層
    
    print(f"生成無中心層感測點配置：")
    print(f"  總點數：{K}")
    print(f"  壁面層：{K_wall} 點 ({wall_fraction*100:.1f}%)")
    print(f"  對數層：{K_log} 點 ({log_fraction*100:.1f}%)")
    print(f"  中心層：0 點 (0.0%)")
    
    # 壁面層：上下壁面各半
    K_wall_upper = K_wall // 2
    K_wall_lower = K_wall - K_wall_upper
    
    x_wall_upper = np.random.uniform(domain_x[0], domain_x[1], K_wall_upper)
    y_wall_upper = np.random.uniform(0.8, 1.0, K_wall_upper)
    
    x_wall_lower = np.random.uniform(domain_x[0], domain_x[1], K_wall_lower)
    y_wall_lower = np.random.uniform(-1.0, -0.8, K_wall_lower)
    
    # 對數層：上下對稱
    K_log_upper = K_log // 2
    K_log_lower = K_log - K_log_upper
    
    x_log_upper = np.random.uniform(domain_x[0], domain_x[1], K_log_upper)
    y_log_upper = np.random.uniform(0.2, 0.8, K_log_upper)
    
    x_log_lower = np.random.uniform(domain_x[0], domain_x[1], K_log_lower)
    y_log_lower = np.random.uniform(-0.8, -0.2, K_log_lower)
    
    # 合併所有點
    x = np.concatenate([x_wall_upper, x_wall_lower, x_log_upper, x_log_lower])
    y = np.concatenate([y_wall_upper, y_wall_lower, y_log_upper, y_log_lower])
    z = np.full(K, domain_z)  # 2D 切片，z 固定
    
    # 組裝感測點
    sensors = np.stack([x, y, z], axis=1)  # (K, 3)
    
    # 驗證點數
    assert sensors.shape[0] == K, f"感測點數量不匹配：{sensors.shape[0]} != {K}"
    
    # 統計分層分佈
    y_normalized = (y + 1.0) / 2.0  # 歸一化到 [0, 1]
    n_wall = np.sum((y_normalized >= 0.8) | (y_normalized <= 0.2))
    n_log = np.sum((y_normalized > 0.2) & (y_normalized < 0.8))
    
    print(f"\n驗證分層分佈：")
    print(f"  壁面層：{n_wall} 點")
    print(f"  對數層：{n_log} 點")
    print(f"  中心層：0 點")
    
    return sensors


def main():
    parser = argparse.ArgumentParser(description="生成無中心層的 Wall-Clustered 感測點")
    parser.add_argument("--K", type=int, default=50, help="感測點總數")
    parser.add_argument("--output", type=str, required=True, help="輸出 .npz 檔案路徑")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子")
    args = parser.parse_args()
    
    # 生成感測點
    sensors = generate_wall_no_center_sensors(K=args.K, seed=args.seed)
    
    # 儲存為 .npz
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        output_path,
        sensors=sensors,
        K=args.K,
        strategy="wall_no_center",
        description="Wall-Clustered without center layer (for ablation study)",
    )
    
    print(f"\n✅ 感測點已儲存：{output_path}")
    print(f"   形狀：{sensors.shape}")
    print(f"   範圍：x=[{sensors[:, 0].min():.2f}, {sensors[:, 0].max():.2f}], "
          f"y=[{sensors[:, 1].min():.2f}, {sensors[:, 1].max():.2f}], "
          f"z={sensors[:, 2].mean():.2f}")


if __name__ == "__main__":
    main()
