#!/bin/bash
# DNS Pœëås,

RESULT_DIR="results/dns_re1000_visualization"

echo "======================================"
echo "DNS Re=1000 ï–Pœ"
echo "======================================"
echo ""
echo "=Ê òåï–"
echo ""
ls -lh "$RESULT_DIR"/*.png | awk '{printf "  - %-40s %8s\n", $9, $5}'
echo ""
echo "=Ä 1J‡ö"
echo "  - $RESULT_DIR/REPORT.md"
echo ""
echo "======================================"
echo "å¹"
echo "  1. macOS: open $RESULT_DIR/"
echo "  2. 1J: cat $RESULT_DIR/REPORT.md"
echo "======================================"
