#!/bin/bash
# ===========================
# PINNs-SparseFlow 伺服器環境檢查腳本
# 用途：驗證訓練環境是否正確配置
# ===========================

echo "========================================"
echo "PINNs-SparseFlow 環境檢查"
echo "========================================"
echo "執行時間: $(date)"
echo "執行用戶: $(whoami)"
echo "主機名稱: $(hostname)"
echo ""

# ===========================
# 1. 檢查專案目錄
# ===========================
echo "----------------------------------------"
echo "1. 專案目錄檢查"
echo "----------------------------------------"

PROJECT_DIR="${HOME}/pinns-sparse-flow"

if [ -d "${PROJECT_DIR}" ]; then
    echo "✅ 專案目錄存在: ${PROJECT_DIR}"
    cd ${PROJECT_DIR}
    
    # 檢查關鍵目錄
    for dir in "scripts/train" "configs" "pinnx" "data"; do
        if [ -d "${dir}" ]; then
            echo "  ✅ ${dir}/"
        else
            echo "  ❌ ${dir}/ (缺失)"
        fi
    done
else
    echo "❌ 專案目錄不存在: ${PROJECT_DIR}"
    echo "   請先上傳專案或克隆 git repository"
    exit 1
fi

echo ""

# ===========================
# 2. Python 環境檢查
# ===========================
echo "----------------------------------------"
echo "2. Python 環境"
echo "----------------------------------------"

# 檢查 Python 版本
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✅ Python: ${PYTHON_VERSION}"
else
    echo "❌ Python3 未安裝"
    exit 1
fi

# 檢查 Conda 環境
if command -v conda &> /dev/null; then
    echo "✅ Conda 已安裝"
    CONDA_ENV=$(conda env list | grep "pinns-sparse-flow" || echo "")
    if [ -z "${CONDA_ENV}" ]; then
        echo "  ⚠️  pinns-sparse-flow 環境未創建"
        echo "  建議執行: conda env create -f environment.yml"
    else
        echo "  ✅ pinns-sparse-flow 環境已創建"
    fi
else
    echo "⚠️  Conda 未安裝，使用系統 Python"
fi

echo ""

# ===========================
# 3. PyTorch 與 CUDA 檢查
# ===========================
echo "----------------------------------------"
echo "3. PyTorch & CUDA"
echo "----------------------------------------"

python3 << 'EOF'
import sys
try:
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
    
    # CUDA 檢查
    if torch.cuda.is_available():
        print(f"✅ CUDA available: True")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"           Memory: {mem_total:.1f} GB")
    else:
        print("❌ CUDA not available")
        print("   請檢查 CUDA 驅動安裝")
except ImportError as e:
    print(f"❌ PyTorch 未安裝: {e}")
    sys.exit(1)
EOF

echo ""

# ===========================
# 4. 關鍵依賴檢查
# ===========================
echo "----------------------------------------"
echo "4. 關鍵依賴套件"
echo "----------------------------------------"

python3 << 'EOF'
import sys

packages = {
    'numpy': 'NumPy',
    'scipy': 'SciPy',
    'h5py': 'HDF5 處理',
    'yaml': 'YAML 配置',
    'wandb': 'WandB 實驗追蹤',
    'matplotlib': '可視化',
}

missing = []
for pkg, desc in packages.items():
    try:
        __import__(pkg)
        print(f"✅ {desc} ({pkg})")
    except ImportError:
        print(f"❌ {desc} ({pkg}) - 未安裝")
        missing.append(pkg)

if missing:
    print(f"\n⚠️  缺少套件: {', '.join(missing)}")
    print("   請執行: pip install " + ' '.join(missing))
    sys.exit(1)
EOF

echo ""

# ===========================
# 5. GPU 狀態檢查
# ===========================
echo "----------------------------------------"
echo "5. GPU 狀態 (nvidia-smi)"
echo "----------------------------------------"

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv
else
    echo "❌ nvidia-smi 未找到，請檢查 NVIDIA 驅動"
fi

echo ""

# ===========================
# 6. 資料檔案檢查
# ===========================
echo "----------------------------------------"
echo "6. 訓練資料檢查"
echo "----------------------------------------"

DATA_DIR="${PROJECT_DIR}/data"

# 檢查 DNS 資料
DNS_FILE="${DATA_DIR}/kolmogorov_dns/dns_re50_t100.h5"
if [ -f "${DNS_FILE}" ]; then
    echo "✅ DNS 資料: ${DNS_FILE}"
    SIZE=$(du -h "${DNS_FILE}" | cut -f1)
    echo "   大小: ${SIZE}"
else
    echo "❌ DNS 資料缺失: ${DNS_FILE}"
fi

# 檢查 Leith 先驗資料
LEITH_FILE="${DATA_DIR}/lowfi/kolmogorov_leith/rans_re50_kf4_leith.h5"
if [ -f "${LEITH_FILE}" ]; then
    echo "✅ Leith 先驗: ${LEITH_FILE}"
    SIZE=$(du -h "${LEITH_FILE}" | cut -f1)
    echo "   大小: ${SIZE}"
else
    echo "⚠️  Leith 先驗缺失: ${LEITH_FILE}"
fi

# 檢查感測器配置
SENSOR_FILE="${DATA_DIR}/sensors/kolmogorov/sensors_K100_re50_256x256.json"
if [ -f "${SENSOR_FILE}" ]; then
    echo "✅ 感測器配置: ${SENSOR_FILE}"
else
    echo "⚠️  感測器配置缺失: ${SENSOR_FILE}"
fi

echo ""

# ===========================
# 7. WandB 配置檢查
# ===========================
echo "----------------------------------------"
echo "7. WandB 配置"
echo "----------------------------------------"

WANDB_CONFIG="${PROJECT_DIR}/.wandb_config"
if [ -f "${WANDB_CONFIG}" ]; then
    echo "✅ WandB 配置存在: ${WANDB_CONFIG}"
    # 檢查是否包含 API key（不顯示完整 key）
    if grep -q "WANDB_API_KEY" "${WANDB_CONFIG}"; then
        echo "   ✅ WANDB_API_KEY 已設置"
    else
        echo "   ⚠️  WANDB_API_KEY 未設置"
    fi
else
    echo "⚠️  WandB 配置缺失: ${WANDB_CONFIG}"
    echo "   建議創建: echo 'export WANDB_API_KEY=your_key' > .wandb_config"
fi

# 測試 WandB 登入
python3 -c "import wandb; print('✅ WandB 套件可用')" 2>/dev/null || echo "❌ WandB 套件測試失敗"

echo ""

# ===========================
# 8. 配置檔案驗證
# ===========================
echo "----------------------------------------"
echo "8. 配置檔案驗證"
echo "----------------------------------------"

CONFIG_FILE="${PROJECT_DIR}/configs/kolmogorov_re50_kf4_K100.yml"
if [ -f "${CONFIG_FILE}" ]; then
    echo "✅ 主配置檔案存在: ${CONFIG_FILE}"
    
    # 驗證 YAML 語法
    python3 << EOF
import yaml
try:
    with open('${CONFIG_FILE}', 'r') as f:
        config = yaml.safe_load(f)
    print("   ✅ YAML 語法正確")
    print(f"   實驗名稱: {config.get('experiment', {}).get('name', 'N/A')}")
    print(f"   模型類型: {config.get('model', {}).get('type', 'N/A')}")
    print(f"   訓練 Epochs: {config.get('training', {}).get('epochs', 'N/A')}")
except Exception as e:
    print(f"   ❌ YAML 解析錯誤: {e}")
EOF
else
    echo "❌ 主配置檔案缺失: ${CONFIG_FILE}"
fi

echo ""

# ===========================
# 9. 日誌目錄檢查
# ===========================
echo "----------------------------------------"
echo "9. 輸出目錄"
echo "----------------------------------------"

LOG_DIR="${HOME}/logs"
if [ -d "${LOG_DIR}" ]; then
    echo "✅ 日誌目錄: ${LOG_DIR}"
else
    echo "⚠️  日誌目錄不存在，正在創建..."
    mkdir -p ${LOG_DIR}
    echo "   ✅ 已創建: ${LOG_DIR}"
fi

for dir in "checkpoints" "results"; do
    TARGET_DIR="${PROJECT_DIR}/${dir}"
    if [ -d "${TARGET_DIR}" ]; then
        echo "✅ ${dir}/ 目錄存在"
    else
        echo "⚠️  ${dir}/ 不存在，正在創建..."
        mkdir -p ${TARGET_DIR}
        echo "   ✅ 已創建"
    fi
done

echo ""

# ===========================
# 10. SLURM 環境檢查
# ===========================
echo "----------------------------------------"
echo "10. SLURM 環境"
echo "----------------------------------------"

if command -v sbatch &> /dev/null; then
    echo "✅ SLURM 已安裝"
    
    # 檢查可用分區
    PARTITIONS=$(sinfo -h -o "%P" | head -5)
    echo "   可用分區:"
    echo "${PARTITIONS}" | while read -r partition; do
        echo "     - ${partition}"
    done
    
    # 檢查 GPU 配額
    echo ""
    echo "   GPU 節點狀態:"
    sinfo -p r740 -o "%.10P %.5a %.10l %.6D %.6t %.14C %.8G" 2>/dev/null || echo "     (無法查詢)"
else
    echo "❌ SLURM 未安裝"
fi

echo ""

# ===========================
# 總結
# ===========================
echo "========================================"
echo "環境檢查完成"
echo "========================================"
echo ""
echo "下一步操作："
echo "1. 如有缺失依賴，請先安裝"
echo "2. 確保資料檔案已上傳到正確位置"
echo "3. 設置 WandB API key: echo 'export WANDB_API_KEY=your_key' > .wandb_config"
echo "4. 提交訓練作業: sbatch slurm_train.sh"
echo ""
echo "========================================"
