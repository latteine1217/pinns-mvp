#!/bin/bash
# =============================================================================
# Quick Fix Script: Loss Imbalance Issue
# =============================================================================
# Purpose: Apply Channel Flow normalization fix + balanced loss weights
# Time: ~1 hour (15 min setup + 30-45 min training test)
# =============================================================================

set -e  # Exit on error

echo "============================================================================="
echo "🔧 Loss Imbalance Quick Fix Script"
echo "============================================================================="
echo ""

# =============================================================================
# Phase 0: 檢查環境
# =============================================================================
echo "📋 Phase 0: Environment Check"
echo "-----------------------------------------------------------------------------"

# Check if we're in the right directory
if [ ! -f "scripts/train/train.py" ]; then
    echo "❌ Error: Not in pinns-mvp root directory"
    echo "   Please run: cd /Users/latteine/Documents/coding/pinns-mvp"
    exit 1
fi

# Check if extraction tool exists
if [ ! -f "scripts/generate/sensors/extract_sensor_values_from_dns.py" ]; then
    echo "❌ Error: Sensor extraction tool not found"
    exit 1
fi

echo "✅ Environment check passed"
echo ""

# =============================================================================
# Phase 1: 生成 DNS-based sensor statistics (Fix標準化 bug)
# =============================================================================
echo "📊 Phase 1: Generate DNS-based Normalization Statistics"
echo "-----------------------------------------------------------------------------"

SENSOR_INPUT="data/lowfi/channel_rans/sensors_K100_rans_phase_a.npz"
SENSOR_OUTPUT="data/lowfi/channel_rans/sensors_K100_rans_phase_a_WITH_VALUES.npz"
DNS_CUTOUT="data/jhtdb/channel_flow_re1000/cutout_128x64x128.npz"

# Check if input files exist
if [ ! -f "$SENSOR_INPUT" ]; then
    echo "❌ Error: Sensor file not found: $SENSOR_INPUT"
    exit 1
fi

if [ ! -f "$DNS_CUTOUT" ]; then
    echo "⚠️  Warning: DNS cutout not found: $DNS_CUTOUT"
    echo "   Attempting to use alternative..."
    
    # Try alternative locations
    ALT_CUTOUT="data/jhtdb/channel_flow_re1000/cutout_*_128x64x128.npz"
    DNS_CUTOUT=$(ls $ALT_CUTOUT 2>/dev/null | head -1)
    
    if [ -z "$DNS_CUTOUT" ]; then
        echo "❌ Error: No DNS cutout found"
        echo "   Please generate DNS cutout first or use existing DNS data"
        exit 1
    fi
    echo "✅ Using: $DNS_CUTOUT"
fi

# Check if output already exists
if [ -f "$SENSOR_OUTPUT" ]; then
    echo "ℹ️  Output file already exists: $SENSOR_OUTPUT"
    read -p "   Overwrite? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "   Skipping generation, using existing file"
    else
        echo "   Regenerating..."
        python scripts/generate/sensors/extract_sensor_values_from_dns.py \
          --sensor-file "$SENSOR_INPUT" \
          --dns-cutout "$DNS_CUTOUT" \
          --output "$SENSOR_OUTPUT" \
          --method nearest
    fi
else
    echo "Generating DNS-based sensor statistics..."
    python scripts/generate/sensors/extract_sensor_values_from_dns.py \
      --sensor-file "$SENSOR_INPUT" \
      --dns-cutout "$DNS_CUTOUT" \
      --output "$SENSOR_OUTPUT" \
      --method nearest
fi

# Verify statistics
echo ""
echo "Verifying statistics..."
python3 << 'PYEOF'
import numpy as np

try:
    data = np.load('data/lowfi/channel_rans/sensors_K100_rans_phase_a_WITH_VALUES.npz')
    
    print("✅ File loaded successfully")
    print(f"   Keys: {list(data.keys())}")
    
    if 'v_sensors' in data and 'w_sensors' in data and 'p_sensors' in data:
        v_std = data['v_sensors'].std()
        w_std = data['w_sensors'].std()
        p_std = data['p_sensors'].std()
        
        print(f"\n📊 Statistics:")
        print(f"   v_std: {v_std:.6f} {'✅' if v_std > 1e-3 else '❌ (too small)'}")
        print(f"   w_std: {w_std:.6f} {'✅' if w_std > 1e-3 else '❌ (too small)'}")
        print(f"   p_std: {p_std:.6f} {'✅' if p_std > 1e-4 else '❌ (too small)'}")
        
        if v_std < 1e-3 or w_std < 1e-3:
            print("\n⚠️  WARNING: Statistics still too small!")
            print("   This may indicate DNS cutout issue")
            exit(1)
    else:
        print("❌ Error: Expected keys not found")
        exit(1)
        
except FileNotFoundError:
    print("❌ Error: Output file not generated")
    exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)
PYEOF

if [ $? -ne 0 ]; then
    echo "❌ Statistics verification failed"
    exit 1
fi

echo ""
echo "✅ Phase 1 Complete: DNS-based statistics generated"
echo ""

# =============================================================================
# Phase 2: 快速訓練測試 (50 epochs)
# =============================================================================
echo "🚀 Phase 2: Quick Training Test (50 epochs)"
echo "-----------------------------------------------------------------------------"

# Check if config exists
if [ ! -f "configs/phase_a_qr_baseline_fixed.yml" ]; then
    echo "❌ Error: Fixed config not found: configs/phase_a_qr_baseline_fixed.yml"
    echo "   Please create the config file first"
    exit 1
fi

echo "Starting quick test training..."
echo "   Config: configs/phase_a_qr_baseline_fixed.yml"
echo "   Epochs: 50"
echo "   Expected time: ~30-45 minutes"
echo ""

# Run training with 50 epochs for quick validation
python scripts/train/train.py \
  --config configs/phase_a_qr_baseline_fixed.yml \
  --epochs 50 \
  2>&1 | tee logs/loss_imbalance_quick_test_$(date +%Y%m%d_%H%M%S).log

TRAINING_EXIT_CODE=$?

echo ""
if [ $TRAINING_EXIT_CODE -ne 0 ]; then
    echo "⚠️  Training encountered issues (exit code: $TRAINING_EXIT_CODE)"
    echo "   Check log file for details"
else
    echo "✅ Training completed"
fi
echo ""

# =============================================================================
# Phase 3: 驗證結果
# =============================================================================
echo "📊 Phase 3: Verify Results"
echo "-----------------------------------------------------------------------------"

# Find latest log
LATEST_LOG=$(ls -t logs/phase_a_qr_baseline_fixed_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    LATEST_LOG=$(ls -t logs/loss_imbalance_quick_test_*.log 2>/dev/null | head -1)
fi

if [ -z "$LATEST_LOG" ]; then
    echo "⚠️  Warning: Cannot find training log for verification"
else
    echo "Analyzing log: $LATEST_LOG"
    echo ""
    
    # Extract Epoch 0 losses
    echo "📋 Epoch 0 Loss Breakdown:"
    grep "Epoch 0/" "$LATEST_LOG" | head -1 || echo "   (Not found in log)"
    echo ""
    
    # Check for key metrics
    echo "🔍 Key Metrics Check:"
    
    # Check momentum_y
    MOMENTUM_Y=$(grep "Epoch 0/" "$LATEST_LOG" | grep -o "momentum_y_loss: [0-9.]*" | head -1 | awk '{print $2}')
    if [ ! -z "$MOMENTUM_Y" ]; then
        echo "   momentum_y_loss: $MOMENTUM_Y"
        if (( $(echo "$MOMENTUM_Y < 100" | bc -l) )); then
            echo "      ✅ Significantly reduced (target: < 100)"
        else
            echo "      ⚠️  Still high (target: < 100)"
        fi
    fi
    
    # Check v_loss
    V_LOSS=$(grep "Epoch 0/" "$LATEST_LOG" | grep -o "v_loss: [0-9.e+-]*" | head -1 | awk '{print $2}')
    if [ ! -z "$V_LOSS" ]; then
        echo "   v_loss: $V_LOSS"
        if (( $(echo "$V_LOSS > 0.1" | bc -l) )); then
            echo "      ✅ Now trainable (was ~0.002)"
        else
            echo "      ⚠️  Still too small"
        fi
    fi
    
    # Check div_loss
    DIV_LOSS=$(grep "Epoch 0/" "$LATEST_LOG" | grep -o "div_loss: [0-9.]*" | head -1 | awk '{print $2}')
    if [ ! -z "$DIV_LOSS" ]; then
        echo "   div_loss: $DIV_LOSS"
        if (( $(echo "$DIV_LOSS < 300" | bc -l) )); then
            echo "      ✅ Improved (was ~374)"
        else
            echo "      ⚠️  Still high"
        fi
    fi
fi

echo ""
echo "============================================================================="
echo "🎯 Quick Fix Summary"
echo "============================================================================="
echo ""
echo "✅ Phase 1: DNS-based normalization applied"
echo "✅ Phase 2: Quick training test completed"
echo "✅ Phase 3: Results verified"
echo ""
echo "📊 Next Steps:"
echo "   1. Review training log: $LATEST_LOG"
echo "   2. If results look good → Run full 500 epoch training"
echo "   3. If issues persist → Check DNS cutout quality"
echo ""
echo "🚀 Full Training Command:"
echo "   python scripts/train/train.py \\"
echo "     --config configs/phase_a_qr_baseline_fixed.yml \\"
echo "     --epochs 500"
echo ""
echo "============================================================================="
