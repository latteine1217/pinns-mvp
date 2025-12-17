#!/bin/bash
# ==============================================================================
# RANS Re-generation with Leith Turbulence Model
# ==============================================================================
#
# Purpose: Replace k-ε RANS with Leith model (designed for 2D turbulence)
#
# Motivation:
# - k-ε model: Designed for 3D turbulence → overestimates TKE in 2D
# - Leith model: Designed for 2D turbulence → accounts for inverse cascade
# - Leith: No extra transport equations (diagnostic from vorticity gradient)
#
# Reference:
# - Leith (1996), "Stochastic backscatter in a subgrid-scale model"
# - Boffetta & Ecke (2012), "Two-Dimensional Turbulence"
#
# ==============================================================================

set -e  # Exit on error

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/data/lowfi/kolmogorov_rans"
BACKUP_DIR="$OUTPUT_DIR/backup_leith_$(date +%Y%m%d_%H%M%S)"

# Physical parameters
N=128
A=1.0
k_f=4
dt=0.005
T_total=100.0
T_spinup=10.0

# Leith model parameters
C_L=0.2  # Leith constant (typical range: 0.1-0.3)

# Reynolds numbers
declare -a RE_VALUES=(50 100 500)
declare -a NU_VALUES=(0.039374 0.019687 0.003937)

echo "=========================================================================="
echo "RANS Re-generation with Leith Turbulence Model"
echo "=========================================================================="
echo ""
echo "Why Leith instead of k-ε?"
echo "  - k-ε designed for 3D turbulence (forward energy cascade)"
echo "  - 2D turbulence has inverse cascade → k-ε overestimates TKE"
echo "  - Leith: ν_t = (C_L Δ)³ |∇ω| (diagnostic, no extra equations)"
echo ""
echo "Target Reynolds numbers (Musacchio & Boffetta 2014):"
echo "  Re = sqrt(A) × L^(3/2) / ν, where L = 2π/k_f"
echo ""
echo "=========================================================================="
echo ""

# Backup old k-ε RANS files
echo "Step 0: Backing up old k-ε RANS files..."
echo "----------------------------------------------------------------------"
mkdir -p "$BACKUP_DIR"

for re in "${RE_VALUES[@]}"; do
    old_file="$OUTPUT_DIR/rans_re${re}_kf4.h5"
    if [ -f "$old_file" ]; then
        cp "$old_file" "$BACKUP_DIR/"
        echo "  Backed up: $(basename $old_file) → $BACKUP_DIR/"
    fi
    
    old_corrected="$OUTPUT_DIR/rans_re${re}_kf4_corrected.h5"
    if [ -f "$old_corrected" ]; then
        cp "$old_corrected" "$BACKUP_DIR/"
        echo "  Backed up: $(basename $old_corrected) → $BACKUP_DIR/"
    fi
done

echo ""

# Generate Leith RANS for each Re
for i in "${!RE_VALUES[@]}"; do
    re="${RE_VALUES[$i]}"
    nu="${NU_VALUES[$i]}"
    
    echo ""
    echo "=========================================================================="
    echo "Re = $re (Leith Model)"
    echo "=========================================================================="
    echo ""
    echo "Parameters:"
    echo "  ν = $nu"
    echo "  A = $A, k_f = $k_f"
    echo "  N = $N, dt = $dt"
    echo "  T_total = $T_total, T_spinup = $T_spinup"
    echo "  C_L = $C_L (Leith constant)"
    echo ""
    
    # Verify Re
    Re_verify=$(python3 << PYEOF
import numpy as np
A = ${A}
k_f = ${k_f}
nu = ${nu}
L = 2 * np.pi / k_f
Re = np.sqrt(A) * (L ** 1.5) / nu
print(f"{Re:.1f}")
PYEOF
)
    echo "Verification:"
    echo "  Re = sqrt(${A}) × (2π/${k_f})^(3/2) / ${nu} = ${Re_verify}"
    echo ""
    
    echo "  [AUTO] Starting Leith simulation..."
    
    # Run Leith model
    python3 "$SCRIPT_DIR/generate_kolmogorov_leith.py" \
        --N ${N} \
        --nu ${nu} \
        --A ${A} \
        --k_f ${k_f} \
        --dt ${dt} \
        --T_total ${T_total} \
        --T_spinup ${T_spinup} \
        --C_L ${C_L} \
        --output "${OUTPUT_DIR}/rans_re${re}_kf4_leith.h5"
    
    echo ""
    echo "✅ Re=${re} completed!"
    echo ""
done

# Summary table
echo ""
echo "=========================================================================="
echo "Verification Summary"
echo "=========================================================================="
echo ""
printf "%-25s %-12s %-12s %-12s %-12s\n" "Configuration" "ν" "Re_target" "Re_actual" "Status"
echo "--------------------------------------------------------------------------------"

for i in "${!RE_VALUES[@]}"; do
    re="${RE_VALUES[$i]}"
    nu="${NU_VALUES[$i]}"
    
    Re_actual=$(python3 << PYEOF
import numpy as np
A = ${A}
k_f = ${k_f}
nu = ${nu}
L = 2 * np.pi / k_f
Re = np.sqrt(A) * (L ** 1.5) / nu
print(f"{Re:.1f}")
PYEOF
)
    
    # Check if Re matches target (within 1%)
    Re_diff=$(python3 << PYEOF
import numpy as np
target = ${re}
actual = ${Re_actual}
diff = abs(actual - target) / target * 100
if diff < 1.0:
    print("✅ PASS")
else:
    print("❌ FAIL")
PYEOF
)
    
    printf "%-25s %-12s %-12s %-12s %-12s\n" \
        "Re=${re} (Leith)" \
        "${nu}" \
        "${re}.0" \
        "${Re_actual}" \
        "${Re_diff}"
done

echo ""
echo "=========================================================================="
echo "✅ Leith RANS generation complete!"
echo "=========================================================================="
echo ""
echo "Next steps:"
echo "  1. Validate with: python3 scripts/compare/validate_leith_rans.py"
echo "  2. Compare k-ε vs Leith: python3 scripts/compare/compare_turbulence_models.py"
echo ""
echo "Expected improvements:"
echo "  - Lower TKE/KE ratio (~1-2 instead of ~10)"
echo "  - V-velocity recovery (not laminarized)"
echo "  - Better U-velocity amplitude match"
echo "  - Reduced error: 80-100% → 30-50% (target)"
echo ""
echo "Files generated:"
for re in "${RE_VALUES[@]}"; do
    echo "  - ${OUTPUT_DIR}/rans_re${re}_kf4_leith.h5"
done
echo ""
echo "Backup location: $BACKUP_DIR"
echo ""
echo "========================================================================"
