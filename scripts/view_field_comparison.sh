#!/bin/bash
# 快速查看場對比圖表的腳本

echo "============================================================"
echo "📊 場對比可視化圖表"
echo "============================================================"
echo ""
echo "可視化結果位於: results/field_comparison/"
echo ""
echo "生成的圖表 (24 張):"
echo "-----------------------------------------------------------"
echo ""
echo "【U 速度場】"
echo "  1. u_baseline_slice_z.png    - Baseline 2D 切片對比"
echo "  2. u_fourier_slice_z.png     - Fourier 2D 切片對比"
echo "  3. u_baseline_statistics.png - Baseline 統計分佈"
echo "  4. u_fourier_statistics.png  - Fourier 統計分佈"
echo "  5. u_baseline_profile_y.png  - Baseline Y 剖面"
echo "  6. u_fourier_profile_y.png   - Fourier Y 剖面"
echo ""
echo "【V 速度場】"
echo "  7-12. 同上格式 (v_*.png)"
echo ""
echo "【W 速度場】"
echo "  13-18. 同上格式 (w_*.png)"
echo ""
echo "【壓力場】"
echo "  19-24. 同上格式 (p_*.png)"
echo ""
echo "-----------------------------------------------------------"
echo "💡 提示："
echo "  - 切片圖：真實 | 預測 | 絕對誤差（三列對比）"
echo "  - 統計圖：直方圖 + 散點圖 + 誤差分佈"
echo "  - 剖面圖：沿通道高度 (Y) 的平均剖面"
echo ""
echo "🔍 快速診斷檢查清單："
echo "  1. 切片圖是否顯示流場結構相似性？"
echo "  2. 散點圖是否沿對角線 (y=x) 分佈？"
echo "  3. 誤差分佈是否以零為中心？"
echo "  4. Y 剖面是否捕捉到邊界層特徵？"
echo ""
echo "============================================================"

# 如果在 macOS，可以用 open 命令打開
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo ""
    read -p "是否在預覽應用中打開所有圖表？(y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        open results/field_comparison/*.png
        echo "✅ 已在預覽應用中打開所有圖表"
    fi
fi
