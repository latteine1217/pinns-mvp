#!/bin/bash
# RANS 修正重新生成腳本
# 使用正確的黏滯係數（與 DNS 匹配）

set -e  # Exit on error

echo "=========================================================================="
echo "RANS Re-generation with Corrected Viscosity"
echo "=========================================================================="
echo ""
echo "Issue: Previous RANS used ν = 1.5× correct value"
echo "Fix: Match DNS viscosity for valid comparison"
echo ""
echo "Target Reynolds numbers (Musacchio & Boffetta 2014):"
echo "  Re = sqrt(A) × L^(3/2) / ν, where L = 2π/k_f"
echo ""
echo "=========================================================================="

# Parameters (matching DNS)
A=1.0
k_f=4

# Grid and time stepping (coarser than DNS for efficiency)
N=128
dt=0.005
T_total=100.0
T_spinup=10.0

# Output directory
OUTPUT_DIR="data/lowfi/kolmogorov_rans"
mkdir -p "${OUTPUT_DIR}"

# Backup old files
BACKUP_DIR="${OUTPUT_DIR}/backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${BACKUP_DIR}"

echo ""
echo "Step 0: Backing up old RANS files..."
echo "----------------------------------------------------------------------"
for file in rans_re50_kf4.h5 rans_re100_kf4.h5 rans_re500_kf4.h5; do
    if [ -f "${OUTPUT_DIR}/${file}" ]; then
        cp "${OUTPUT_DIR}/${file}" "${BACKUP_DIR}/"
        echo "  Backed up: ${file} → ${BACKUP_DIR}/"
    fi
done

echo ""
echo "=========================================================================="
echo "Re = 50 (Corrected)"
echo "=========================================================================="
echo ""
echo "Parameters:"
echo "  ν = 0.039374 (was 0.059000, error factor 1.50×)"
echo "  A = ${A}, k_f = ${k_f}"
echo "  N = ${N}, dt = ${dt}"
echo "  T_total = ${T_total}, T_spinup = ${T_spinup}"
echo ""
echo "Verification:"
nu_50=0.039374
Re_50=$(python3 << PYEOF
import numpy as np
A = ${A}
k_f = ${k_f}
nu = ${nu_50}
L = 2 * np.pi / k_f
Re = np.sqrt(A) * (L ** 1.5) / nu
print(f"{Re:.1f}")
PYEOF
)
echo "  Re = sqrt(${A}) × (2π/${k_f})^(3/2) / ${nu_50} = ${Re_50}"
echo ""
read -p "Press Enter to start Re=50 generation (or Ctrl+C to cancel)..."

python3 scripts/generate/dns/generate_kolmogorov_rans.py \
    --N ${N} \
    --nu ${nu_50} \
    --A ${A} \
    --k_f ${k_f} \
    --dt ${dt} \
    --T_total ${T_total} \
    --T_spinup ${T_spinup} \
    --output "${OUTPUT_DIR}/rans_re50_kf4_corrected.h5"

echo ""
echo "✅ Re=50 completed!"
echo ""

echo ""
echo "=========================================================================="
echo "Re = 100 (Corrected)"
echo "=========================================================================="
echo ""
echo "Parameters:"
echo "  ν = 0.019687 (was 0.029500, error factor 1.50×)"
echo ""
nu_100=0.019687
Re_100=$(python3 << PYEOF
import numpy as np
A = ${A}
k_f = ${k_f}
nu = ${nu_100}
L = 2 * np.pi / k_f
Re = np.sqrt(A) * (L ** 1.5) / nu
print(f"{Re:.1f}")
PYEOF
)
echo "  Verified Re = ${Re_100}"
echo ""
read -p "Press Enter to start Re=100 generation (or Ctrl+C to cancel)..."

python3 scripts/generate/dns/generate_kolmogorov_rans.py \
    --N ${N} \
    --nu ${nu_100} \
    --A ${A} \
    --k_f ${k_f} \
    --dt ${dt} \
    --T_total ${T_total} \
    --T_spinup ${T_spinup} \
    --output "${OUTPUT_DIR}/rans_re100_kf4_corrected.h5"

echo ""
echo "✅ Re=100 completed!"
echo ""

echo ""
echo "=========================================================================="
echo "Re = 500 (Corrected)"
echo "=========================================================================="
echo ""
echo "Parameters:"
echo "  ν = 0.003937 (was 0.005900, error factor 1.50×)"
echo ""
nu_500=0.003937
Re_500=$(python3 << PYEOF
import numpy as np
A = ${A}
k_f = ${k_f}
nu = ${nu_500}
L = 2 * np.pi / k_f
Re = np.sqrt(A) * (L ** 1.5) / nu
print(f"{Re:.1f}")
PYEOF
)
echo "  Verified Re = ${Re_500}"
echo ""
read -p "Press Enter to start Re=500 generation (or Ctrl+C to cancel)..."

python3 scripts/generate/dns/generate_kolmogorov_rans.py \
    --N ${N} \
    --nu ${nu_500} \
    --A ${A} \
    --k_f ${k_f} \
    --dt ${dt} \
    --T_total ${T_total} \
    --T_spinup ${T_spinup} \
    --output "${OUTPUT_DIR}/rans_re500_kf4_corrected.h5"

echo ""
echo "✅ Re=500 completed!"
echo ""

echo ""
echo "=========================================================================="
echo "Post-Generation Validation"
echo "=========================================================================="

python3 << 'PYEOF'
import h5py
import numpy as np
import sys

print("\nValidating corrected RANS files...")
print("=" * 80)

files = [
    ('Re=50 (corrected)', 'data/lowfi/kolmogorov_rans/rans_re50_kf4_corrected.h5', 0.039374, 50.0),
    ('Re=100 (corrected)', 'data/lowfi/kolmogorov_rans/rans_re100_kf4_corrected.h5', 0.019687, 100.0),
    ('Re=500 (corrected)', 'data/lowfi/kolmogorov_rans/rans_re500_kf4_corrected.h5', 0.003937, 500.0),
]

A = 1.0
k_f = 4
L = 2 * np.pi / k_f

print(f"\n{'Dataset':<25} {'ν':<12} {'Re (calc)':<12} {'Re (target)':<12} {'Status':<10}")
print("-" * 80)

all_good = True
for name, path, nu_target, re_target in files:
    try:
        with h5py.File(path, 'r') as f:
            nu_actual = f['parameters'].attrs['nu']
            Re_actual = np.sqrt(A) * (L ** 1.5) / nu_actual
            
            match_nu = abs(nu_actual - nu_target) < 1e-6
            match_re = abs(Re_actual - re_target) < 1.0
            
            if match_nu and match_re:
                status = "✅ PASS"
            else:
                status = "❌ FAIL"
                all_good = False
            
            print(f"{name:<25} {nu_actual:<12.6f} {Re_actual:<12.1f} {re_target:<12.1f} {status:<10}")
            
            # Additional checks
            u = f['mean_field/u'][:]
            v = f['mean_field/v'][:]
            k = f['mean_field/k'][:]
            
            print(f"  Fields: U_max={u.max():.4f}, V_max={v.max():.6f}, k_mean={k.mean():.4f}")
            
    except Exception as e:
        print(f"{name:<25} ERROR: {str(e)}")
        all_good = False

print("=" * 80)

if all_good:
    print("\n🎉 All files validated successfully!")
    print("\nNext steps:")
    print("  1. Verify RANS improvements:")
    print("     python scripts/compare/compare_lowfi_hifi.py")
    print("  2. Regenerate figures:")
    print("     python scripts/visualize/generate_rans_dns_spectrum_comparison.py")
    print("  3. Update thesis error values")
else:
    print("\n⚠️  Validation failed! Check error messages above.")
    sys.exit(1)

PYEOF

echo ""
echo "=========================================================================="
echo "Generation Complete!"
echo "=========================================================================="
echo ""
echo "Summary:"
echo "  - Old files backed up to: ${BACKUP_DIR}/"
echo "  - New files created:"
echo "      • rans_re50_kf4_corrected.h5"
echo "      • rans_re100_kf4_corrected.h5"
echo "      • rans_re500_kf4_corrected.h5"
echo ""
echo "Estimated computation time:"
echo "  - Re=50:  ~10 minutes (20,000 steps)"
echo "  - Re=100: ~15 minutes (20,000 steps)"  
echo "  - Re=500: ~45 minutes (20,000 steps, more turbulent)"
echo "  Total: ~70 minutes"
echo ""

