#!/usr/bin/env python3
"""
Re=100 DNS 自動化後處理腳本
用途：DNS 生成完成後，自動執行所有驗證、感測點生成和訓練準備
"""

import subprocess
import sys
from pathlib import Path
import argparse

def run_command(cmd, description):
    """執行命令並處理錯誤"""
    print("\n" + "=" * 60)
    print(f"🚀 {description}")
    print("=" * 60)
    print(f"執行: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"\n❌ 錯誤: {description} 失敗！")
        print(f"   返回碼: {result.returncode}")
        sys.exit(1)
    
    print(f"\n✅ {description} 完成！")
    return result

def main():
    parser = argparse.ArgumentParser(description="Re=100 DNS 自動化後處理")
    parser.add_argument("--dns-file", type=str, 
                       default="data/kolmogorov_dns_re100_512x512_v2.h5",
                       help="DNS 數據文件路徑")
    parser.add_argument("--K", type=int, default=100,
                       help="感測點數量")
    parser.add_argument("--snapshot", type=str, default="t_10.0",
                       help="用於生成感測點的快照")
    parser.add_argument("--skip-validation", action="store_true",
                       help="跳過 DNS 驗證步驟")
    parser.add_argument("--skip-training", action="store_true",
                       help="跳過啟動訓練步驟")
    
    args = parser.parse_args()
    
    # 檢查 DNS 文件是否存在
    if not Path(args.dns_file).exists():
        print(f"❌ 錯誤: DNS 文件不存在: {args.dns_file}")
        print("   請等待 DNS 生成完成後再執行此腳本。")
        print("\n💡 檢查進度: python scripts/check_dns_re100_v2.py")
        sys.exit(1)
    
    print("=" * 60)
    print("🎯 Re=100 DNS 自動化後處理")
    print("=" * 60)
    print(f"DNS 文件: {args.dns_file}")
    print(f"感測點數量: {args.K}")
    print(f"快照: {args.snapshot}")
    print("=" * 60)
    
    # 步驟 1: 驗證 DNS 數據
    if not args.skip_validation:
        run_command(
            ["python", "scripts/validate_dns_v2.py",
             "--dns-file", args.dns_file,
             "--output", "results/dns_validation_v2"],
            "步驟 1/4: 驗證 DNS 數據品質"
        )
    else:
        print("\n⏭️  跳過 DNS 驗證步驟")
    
    # 步驟 2: 生成 QR-Pivot 感測點
    sensor_file = f"data/qr_sensors_re100_K{args.K}_v2.npz"
    run_command(
        ["python", "scripts/generate_2d_slice_qr_sensors_fixed_v2.py",
         "--dns-file", args.dns_file,
         "--K", str(args.K),
         "--snapshot-key", args.snapshot,
         "--output", sensor_file],
        "步驟 2/4: 生成 QR-Pivot 感測點"
    )
    
    # 步驟 3: 視覺化感測點品質
    run_command(
        ["python", "scripts/visualize_qr_sensors.py",
         "--input", sensor_file,
         "--output", "results/qr_validation_v2"],
        "步驟 3/4: 視覺化感測點品質"
    )
    
    # 步驟 4: 準備訓練配置
    print("\n" + "=" * 60)
    print("📝 步驟 4/4: 訓練配置準備")
    print("=" * 60)
    
    config_template = "configs/kolmogorov_experiments/kolmogorov_2d_re100_qr_adaptive.yml"
    
    if not Path(config_template).exists():
        print(f"⚠️  警告: 配置模板不存在: {config_template}")
        print("   請手動創建配置文件並更新以下欄位:")
        print(f"     - dns_file: {args.dns_file}")
        print(f"     - sensor_file: {sensor_file}")
        print(f"     - K: {args.K}")
    else:
        print(f"✅ 配置模板: {config_template}")
        print("\n⚠️  請手動檢查並更新以下欄位:")
        print(f"   - dns_file: {args.dns_file}")
        print(f"   - sensor_file: {sensor_file}")
        print(f"   - K: {args.K}")
    
    # 步驟 5: 啟動訓練（可選）
    if not args.skip_training:
        print("\n" + "=" * 60)
        print("🚀 步驟 5/5: 啟動 PINN 訓練")
        print("=" * 60)
        
        train_cmd = [
            "nohup", "python", "scripts/train.py",
            "--cfg", config_template,
            "--epochs", "2000"
        ]
        
        print(f"執行: {' '.join(train_cmd)}")
        print("   日誌輸出: log/train_re100_v2.log")
        
        response = input("\n是否立即啟動訓練？[y/N]: ")
        if response.lower() == 'y':
            with open("log/train_re100_v2.log", "w") as log_file:
                subprocess.Popen(
                    train_cmd[1:],  # 去掉 nohup
                    stdout=log_file,
                    stderr=subprocess.STDOUT
                )
            print("✅ 訓練已在後台啟動！")
            print("📝 監控訓練: tail -f log/train_re100_v2.log")
        else:
            print("⏭️  跳過訓練步驟")
            print("\n手動啟動訓練:")
            print(f"   python scripts/train.py --cfg {config_template} --epochs 2000")
    
    # 完成總結
    print("\n" + "=" * 60)
    print("🎉 自動化處理完成！")
    print("=" * 60)
    print("\n📂 生成的文件:")
    print(f"   - DNS 驗證: results/dns_validation_v2/")
    print(f"   - 感測點: {sensor_file}")
    print(f"   - 感測點視覺化: results/qr_validation_v2/")
    print("\n📋 下一步:")
    print(f"   1. 檢查驗證結果: ls -lh results/dns_validation_v2/")
    print(f"   2. 檢查感測點品質: ls -lh results/qr_validation_v2/")
    print(f"   3. 更新配置: nano {config_template}")
    print(f"   4. 啟動訓練: python scripts/train.py --cfg {config_template}")
    print("=" * 60)

if __name__ == "__main__":
    main()
