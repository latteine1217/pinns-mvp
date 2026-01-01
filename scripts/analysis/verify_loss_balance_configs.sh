#!/bin/bash
# Loss Balance 實驗配置驗證腳本
# 檢查 A1/A2/A3 三個實驗的配置差異

set -e

echo "============================================================"
echo "Loss Balance 實驗配置驗證"
echo "============================================================"
echo ""

# 配置文件路徑
CONFIG_DIR="configs/experiments"
A1="loss_balance_A1_baseline.yml"
A2="loss_balance_A2_normalize_only.yml"
A3="loss_balance_A3_manual_reweight.yml"

# 檢查文件是否存在
echo "📂 檢查配置文件..."
for cfg in "$A1" "$A2" "$A3"; do
  if [ -f "$CONFIG_DIR/$cfg" ]; then
    echo "  ✅ $cfg 存在"
  else
    echo "  ❌ $cfg 不存在"
    echo ""
    echo "⚠️  配置文件缺失！實驗可能使用命令行參數或默認配置運行。"
    echo ""
    echo "建議行動："
    echo "1. 檢查訓練腳本調用: grep 'loss_balance' scripts/train/*.py"
    echo "2. 搜索配置文件: fd 'loss_balance' configs/"
    echo "3. 查看運行歷史: history | grep train.py"
    exit 1
  fi
done
echo ""

# 提取關鍵配置
echo "============================================================"
echo "📊 關鍵配置對比"
echo "============================================================"
echo ""

echo "--- 實驗名稱 ---"
for cfg in "$A1" "$A2" "$A3"; do
  exp_name=$(grep "^experiment:" "$CONFIG_DIR/$cfg" | awk '{print $2}')
  echo "  $cfg: $exp_name"
done
echo ""

echo "--- Loss 權重 ---"
echo "配置項            A1 Baseline    A2 Normalize   A3 Reweight"
echo "---------------------------------------------------------------"

# Data weight
for cfg in "$A1" "$A2" "$A3"; do
  val=$(grep "data_weight:" "$CONFIG_DIR/$cfg" | awk '{print $2}' || echo "N/A")
  printf "%-20s %s\n" "$cfg" "$val"
done | paste - - - | awk '{printf "%-17s %-15s %-15s %s\n", "data_weight:", $2, $4, $6}'

# Momentum X weight
for cfg in "$A1" "$A2" "$A3"; do
  val=$(grep "momentum_x_weight:" "$CONFIG_DIR/$cfg" | awk '{print $2}' || echo "N/A")
  printf "%-20s %s\n" "$cfg" "$val"
done | paste - - - | awk '{printf "%-17s %-15s %-15s %s\n", "momentum_x_weight:", $2, $4, $6}'

# Momentum Y weight
for cfg in "$A1" "$A2" "$A3"; do
  val=$(grep "momentum_y_weight:" "$CONFIG_DIR/$cfg" | awk '{print $2}' || echo "N/A")
  printf "%-20s %s\n" "$cfg" "$val"
done | paste - - - | awk '{printf "%-17s %-15s %-15s %s\n", "momentum_y_weight:", $2, $4, $6}'

# Continuity weight
for cfg in "$A1" "$A2" "$A3"; do
  val=$(grep "continuity_weight:" "$CONFIG_DIR/$cfg" | awk '{print $2}' || echo "N/A")
  printf "%-20s %s\n" "$cfg" "$val"
done | paste - - - | awk '{printf "%-17s %-15s %-15s %s\n", "continuity_weight:", $2, $4, $6}'

echo ""
echo "--- Prior Loss 權重 ---"
for cfg in "$A1" "$A2" "$A3"; do
  val=$(grep "consistency_weight:" "$CONFIG_DIR/$cfg" | awk '{print $2}' || echo "N/A")
  printf "%-20s %s\n" "$cfg" "$val"
done | paste - - - | awk '{printf "%-17s %-15s %-15s %s\n", "consistency_weight:", $2, $4, $6}'

echo ""
echo "--- 隨機種子 ---"
for cfg in "$A1" "$A2" "$A3"; do
  val=$(grep "^seed:" "$CONFIG_DIR/$cfg" | awk '{print $2}' || echo "N/A")
  printf "%-20s %s\n" "$cfg" "$val"
done | paste - - - | awk '{printf "%-17s %-15s %-15s %s\n", "seed:", $2, $4, $6}'

echo ""
echo "============================================================"
echo "📝 完整配置差異 (A1 vs A2)"
echo "============================================================"
if diff -u "$CONFIG_DIR/$A1" "$CONFIG_DIR/$A2" > /tmp/diff_A1_A2.txt; then
  echo "⚠️  A1 與 A2 配置完全相同！"
else
  echo "✅ A1 與 A2 有以下差異："
  cat /tmp/diff_A1_A2.txt | head -50
fi
echo ""

echo "============================================================"
echo "📝 完整配置差異 (A1 vs A3)"
echo "============================================================"
if diff -u "$CONFIG_DIR/$A1" "$CONFIG_DIR/$A3" > /tmp/diff_A1_A3.txt; then
  echo "⚠️  A1 與 A3 配置完全相同！"
else
  echo "✅ A1 與 A3 有以下差異："
  cat /tmp/diff_A1_A3.txt | head -50
fi
echo ""

echo "============================================================"
echo "💡 結論與建議"
echo "============================================================"
echo ""

# 檢查是否有差異
if diff -q "$CONFIG_DIR/$A1" "$CONFIG_DIR/$A2" > /dev/null 2>&1; then
  echo "⚠️  A1 與 A2 配置相同 → 解釋為何實驗結果相同"
else
  echo "✅ A1 與 A2 有差異"
fi

if diff -q "$CONFIG_DIR/$A1" "$CONFIG_DIR/$A3" > /dev/null 2>&1; then
  echo "⚠️  A1 與 A3 配置相同 → 無法測試 prior weight 影響"
else
  echo "✅ A1 與 A3 有差異"
  
  # 檢查 consistency_weight 差異
  a1_prior=$(grep "consistency_weight:" "$CONFIG_DIR/$A1" | awk '{print $2}')
  a3_prior=$(grep "consistency_weight:" "$CONFIG_DIR/$A3" | awk '{print $2}')
  
  if [ "$a1_prior" != "$a3_prior" ]; then
    echo ""
    echo "📊 Prior Weight 變化: $a1_prior → $a3_prior"
    
    # 計算預期影響
    prior_loss=1.39  # Epoch 0
    data_loss=191.6  # Epoch 0
    
    echo "   根據 Epoch 0 數據:"
    echo "   - Prior loss: $prior_loss (已加權)"
    echo "   - Data loss: $data_loss (已加權)"
    echo "   - Prior 佔比: 0.72%"
    echo ""
    echo "   即使 prior weight 降低 50% ($a1_prior → $a3_prior),"
    echo "   total loss 僅變化 ~0.36%，遠小於數值誤差。"
    echo ""
    echo "💡 建議: 需要更激進的權重調整 (如 data_weight: 100→10)"
  fi
fi

echo ""
echo "下一步:"
echo "1. 若配置相同 → 需確認實驗是如何運行的"
echo "2. 若差異太小 → 設計新實驗 (B 系列)，大幅調整權重"
echo "3. 參考報告: context/loss_balance_experiment_report.md"
echo ""
