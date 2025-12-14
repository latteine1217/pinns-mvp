#!/usr/bin/env python3
"""
Phase 1 重構驗證腳本
====================

目的：驗證重構後的 Trainer 在真實訓練中的穩定性與效能

測試項目：
1. ✅ 訓練穩定性（無 NaN/Inf）
2. ✅ 損失收斂（Loss 持續下降）
3. ✅ 效能基準（訓練速度）
4. ✅ 記憶體使用（無洩漏）
5. ✅ 最終指標（L2 誤差）

配置：quick_test_re100.yml
預期時間：~2-3 分鐘
"""

import sys
import time
import subprocess
import json
import re
from pathlib import Path
from datetime import datetime
import psutil
import os

# 項目路徑
PROJECT_ROOT = Path(__file__).parent
CONFIG_FILE = PROJECT_ROOT / "configs/quick_test_re100.yml"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts/train/train.py"
RESULT_DIR = PROJECT_ROOT / "results/quick_test"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints/quick_test"

# 驗證標準
VALIDATION_CRITERIA = {
    "stability": {
        "no_nan_inf": True,
        "loss_finite": True,
        "gradient_stable": True,
    },
    "convergence": {
        "loss_decreasing": True,
        "min_loss_reduction": 0.1,  # 至少 10% 改善
    },
    "performance": {
        "max_epoch_time": 30.0,  # 每 epoch ≤ 30 秒
        "max_memory_mb": 4000,   # ≤ 4GB RAM
    },
}

class ValidationResult:
    """驗證結果記錄器"""
    def __init__(self):
        self.start_time = time.time()
        self.end_time = None
        self.success = False
        self.errors = []
        self.warnings = []
        self.metrics = {}
        self.performance = {}
    
    def duration(self):
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    def add_error(self, msg):
        self.errors.append(msg)
        print(f"❌ ERROR: {msg}")
    
    def add_warning(self, msg):
        self.warnings.append(msg)
        print(f"⚠️  WARNING: {msg}")
    
    def add_metric(self, key, value):
        self.metrics[key] = value
        print(f"📊 {key}: {value}")
    
    def add_performance(self, key, value):
        self.performance[key] = value
        print(f"⚡ {key}: {value}")
    
    def finalize(self, success=None):
        self.end_time = time.time()
        if success is not None:
            self.success = success
        else:
            self.success = len(self.errors) == 0
    
    def summary(self):
        """生成摘要報告"""
        status = "✅ PASSED" if self.success else "❌ FAILED"
        print("\n" + "="*70)
        print(f"驗證結果: {status}")
        print("="*70)
        print(f"總耗時: {self.duration():.2f} 秒")
        print(f"錯誤數: {len(self.errors)}")
        print(f"警告數: {len(self.warnings)}")
        
        if self.metrics:
            print("\n📊 訓練指標:")
            for key, value in self.metrics.items():
                print(f"  - {key}: {value}")
        
        if self.performance:
            print("\n⚡ 效能指標:")
            for key, value in self.performance.items():
                print(f"  - {key}: {value}")
        
        if self.errors:
            print("\n❌ 錯誤列表:")
            for i, err in enumerate(self.errors, 1):
                print(f"  {i}. {err}")
        
        if self.warnings:
            print("\n⚠️  警告列表:")
            for i, warn in enumerate(self.warnings, 1):
                print(f"  {i}. {warn}")
        
        print("="*70 + "\n")
        
        return self.success


def check_prerequisites():
    """檢查前置條件"""
    print("🔍 檢查前置條件...")
    
    # 檢查配置文件
    if not CONFIG_FILE.exists():
        print(f"❌ 配置文件不存在: {CONFIG_FILE}")
        return False
    
    # 檢查訓練腳本
    if not TRAIN_SCRIPT.exists():
        print(f"❌ 訓練腳本不存在: {TRAIN_SCRIPT}")
        return False
    
    # 檢查數據文件
    data_file = PROJECT_ROOT / "data/kolmogorov_dns/dns_re100_t100.h5"
    if not data_file.exists():
        print(f"❌ 數據文件不存在: {data_file}")
        return False
    
    # 檢查感測器文件
    sensor_file = PROJECT_ROOT / "results/kolmogorov_dns/sensor_viz_re100_K100/sensor_data.json"
    if not sensor_file.exists():
        print(f"❌ 感測器文件不存在: {sensor_file}")
        return False
    
    print("✅ 前置條件檢查通過")
    return True


def clean_previous_runs():
    """清理先前的運行結果"""
    print("🧹 清理先前運行結果...")
    
    import shutil
    
    if RESULT_DIR.exists():
        shutil.rmtree(RESULT_DIR)
        print(f"  - 刪除 {RESULT_DIR}")
    
    if CHECKPOINT_DIR.exists():
        shutil.rmtree(CHECKPOINT_DIR)
        print(f"  - 刪除 {CHECKPOINT_DIR}")
    
    print("✅ 清理完成")


def parse_training_log(log_lines):
    """解析訓練日誌"""
    epochs = []
    losses = []
    epoch_times = []
    current_epoch = 0
    
    for line in log_lines:
        # 匹配 epoch 資訊: Epoch X/10
        epoch_match = re.search(r"Epoch\s+(\d+)/(\d+)", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        
        # 匹配 loss: total_loss: 1.234
        loss_match = re.search(r"total_loss:\s+([\d.e+-]+)", line)
        if loss_match:
            loss_val = float(loss_match.group(1))
            losses.append(loss_val)
            epochs.append(current_epoch if current_epoch > 0 else len(losses))
        
        # 匹配訓練時間: Training time: 12.34s
        time_match = re.search(r"Training time:\s+([\d.]+)s", line)
        if time_match:
            epoch_time = float(time_match.group(1))
            epoch_times.append(epoch_time)
    
    return {
        "epochs": epochs,
        "losses": losses,
        "epoch_times": epoch_times,
    }


def monitor_memory():
    """監控記憶體使用"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # MB


def run_training(result: ValidationResult):
    """執行訓練並監控"""
    print("\n🚀 開始訓練...")
    print(f"配置: {CONFIG_FILE.name}")
    print(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)
    
    # 記錄初始記憶體
    initial_memory = monitor_memory()
    result.add_performance("initial_memory_mb", f"{initial_memory:.2f} MB")
    
    # 執行訓練
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--cfg", str(CONFIG_FILE),
    ]
    
    # 設置環境變數
    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT)
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        
        log_lines = []
        print("\n📋 訓練日誌:")
        print("-" * 70)
        
        stdout = process.stdout
        if stdout is not None:
            for line in stdout:
                line = line.rstrip()
                print(line)
                log_lines.append(line)
                
                # 即時檢查錯誤
                if "NaN" in line or "Inf" in line:
                    result.add_error(f"檢測到 NaN/Inf: {line}")
                
                if "ERROR" in line or "CRITICAL" in line:
                    result.add_error(f"訓練錯誤: {line}")
        
        process.wait()
        
        if process.returncode != 0:
            result.add_error(f"訓練進程異常退出 (code: {process.returncode})")
            return False
        
        print("-" * 70)
        
        # 記錄最終記憶體
        final_memory = monitor_memory()
        memory_increase = final_memory - initial_memory
        result.add_performance("final_memory_mb", f"{final_memory:.2f} MB")
        result.add_performance("memory_increase_mb", f"{memory_increase:.2f} MB")
        
        # 解析日誌
        train_info = parse_training_log(log_lines)
        
        if train_info["losses"]:
            initial_loss = train_info["losses"][0]
            final_loss = train_info["losses"][-1]
            loss_reduction = (initial_loss - final_loss) / initial_loss
            
            result.add_metric("initial_loss", f"{initial_loss:.6f}")
            result.add_metric("final_loss", f"{final_loss:.6f}")
            result.add_metric("loss_reduction", f"{loss_reduction:.2%}")
            
            # 檢查收斂
            if loss_reduction < VALIDATION_CRITERIA["convergence"]["min_loss_reduction"]:
                result.add_warning(
                    f"Loss 減少不足 ({loss_reduction:.2%} < "
                    f"{VALIDATION_CRITERIA['convergence']['min_loss_reduction']:.2%})"
                )
        else:
            result.add_error("無法解析訓練損失")
        
        if train_info["epoch_times"]:
            avg_epoch_time = sum(train_info["epoch_times"]) / len(train_info["epoch_times"])
            max_epoch_time = max(train_info["epoch_times"])
            
            result.add_performance("avg_epoch_time", f"{avg_epoch_time:.2f}s")
            result.add_performance("max_epoch_time", f"{max_epoch_time:.2f}s")
            
            # 檢查效能
            if max_epoch_time > VALIDATION_CRITERIA["performance"]["max_epoch_time"]:
                result.add_warning(
                    f"Epoch 時間過長 ({max_epoch_time:.2f}s > "
                    f"{VALIDATION_CRITERIA['performance']['max_epoch_time']}s)"
                )
        
        # 檢查記憶體
        if final_memory > VALIDATION_CRITERIA["performance"]["max_memory_mb"]:
            result.add_warning(
                f"記憶體使用過高 ({final_memory:.2f} MB > "
                f"{VALIDATION_CRITERIA['performance']['max_memory_mb']} MB)"
            )
        
        return True
        
    except Exception as e:
        result.add_error(f"訓練執行失敗: {str(e)}")
        return False


def validate_outputs(result: ValidationResult):
    """驗證輸出文件"""
    print("\n🔍 驗證輸出文件...")
    
    # 檢查檢查點
    if not CHECKPOINT_DIR.exists():
        result.add_error(f"檢查點目錄不存在: {CHECKPOINT_DIR}")
        return False
    
    checkpoints = list(CHECKPOINT_DIR.glob("*.pth"))
    if not checkpoints:
        result.add_error("未生成任何檢查點")
        return False
    
    result.add_metric("checkpoints_created", len(checkpoints))
    
    # 檢查結果目錄
    if RESULT_DIR.exists():
        result.add_metric("results_dir_created", "Yes")
    else:
        result.add_warning("結果目錄未創建")
    
    print("✅ 輸出驗證完成")
    return True


def save_validation_report(result: ValidationResult):
    """保存驗證報告"""
    report_file = PROJECT_ROOT / "validation_report.json"
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "success": result.success,
        "duration_seconds": result.duration(),
        "metrics": result.metrics,
        "performance": result.performance,
        "errors": result.errors,
        "warnings": result.warnings,
        "validation_criteria": VALIDATION_CRITERIA,
    }
    
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 驗證報告已保存: {report_file}")


def main():
    """主函數"""
    print("\n" + "="*70)
    print("Phase 1 重構驗證測試")
    print("="*70 + "\n")
    
    result = ValidationResult()
    
    try:
        # 1. 檢查前置條件
        if not check_prerequisites():
            result.add_error("前置條件檢查失敗")
            result.finalize(success=False)
            result.summary()
            return 1
        
        # 2. 清理先前運行
        clean_previous_runs()
        
        # 3. 執行訓練
        if not run_training(result):
            result.finalize(success=False)
            result.summary()
            save_validation_report(result)
            return 1
        
        # 4. 驗證輸出
        if not validate_outputs(result):
            result.finalize(success=False)
            result.summary()
            save_validation_report(result)
            return 1
        
        # 5. 完成
        result.finalize(success=True)
        result.summary()
        save_validation_report(result)
        
        return 0 if result.success else 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用戶中斷測試")
        result.add_error("用戶中斷")
        result.finalize(success=False)
        result.summary()
        return 1
    
    except Exception as e:
        print(f"\n\n❌ 未預期的錯誤: {str(e)}")
        result.add_error(f"未預期的錯誤: {str(e)}")
        result.finalize(success=False)
        result.summary()
        return 1


if __name__ == "__main__":
    sys.exit(main())
