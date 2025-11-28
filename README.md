# 🌊 PINNs-MVP: 基於物理資訊神經網路的湍流場重建

**少量資料 × 物理先驗：基於 Kolmogorov Flow DNS 的 PINNs 逆重建**

[![研究](https://img.shields.io/badge/研究-PINNs逆問題-blue)](https://github.com/latteine/pinns-mvp)
[![資料來源](https://img.shields.io/badge/資料-Kolmogorov_Flow_DNS-green)](docs/KOLMOGOROV_DNS_GUIDE.md)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-orange)](https://pytorch.org/)
[![硬體支援](https://img.shields.io/badge/硬體-CUDA%20%7C%20MPS%20%7C%20CPU-yellow)](A100_DEPLOYMENT_GUIDE.md)
[![狀態](https://img.shields.io/badge/狀態-積極開發中-success)](README.md)

> **使命**：從極度稀疏的感測器觀測中，重建高保真 2D/3D 湍流場；所有研究基於自建 **Kolmogorov Flow DNS** 並經過雷諾數校準與物理驗證。

---

## 📌 項目速覽
- 針對稀疏觀測的湍流逆問題，結合物理約束與神經網路實現穩健重建。
- Fourier-SIREN MLP + VS-PINN + GradNorm/因果訓練，兼顧高頻細節與梯度穩定。
- 配置驅動：所有實驗由 YAML 控制；自動裝置選擇支援 CUDA/MPS/CPU。
- 感測器佈局離線優化（QR-Pivot/DEIM）；DNS 生成與 Re 校準工具內建。

---

## 🚀 快速開始

```bash
# 1) 取得程式碼與環境
git clone https://github.com/latteine/pinns-mvp.git
cd pinns-mvp
conda env create -f environment.yml
conda activate pinns-mvp

# 2) 生成或下載 Kolmogorov Flow DNS（參見 docs/KOLMOGOROV_DNS_GUIDE.md）
python scripts/generate_kolmogorov_dns.py \
  --N 512 --nu 0.0125 --k_f 8 --T_end 40.0 \
  --output data/kolmogorov_dns_re56_512x512_kf8_midway.h5

# 3) 執行訓練（配置驅動）
python scripts/train.py --cfg configs/kolmogorov_re100_kf4_K100.yml
```

更多硬體與部署建議：`A100_DEPLOYMENT_GUIDE.md`。配置模板：`configs/templates/`。

---

## 核心技術深度解析

本專案結合多項技術，解決湍流重建的高頻、多尺度與梯度剛性挑戰。

### 1. 模型架構: Fourier-SIREN MLP
- **傅立葉特徵**：先將 `(t, x, y, z)` 映射到高維頻域，消除 MLP 的頻譜偏差。
- **正弦激活 (SIREN)**：`sin(ωx)` 使高階導數平滑，適合 PDE 殘差計算。
- **整體效果**：同時捕捉宏觀結構與微觀渦旋，並保持梯度穩定。

### PINNs 內部運作流 (Internal Workflow)
1. **輸入準備**：原始座標與監測點物理量 → `UnifiedNormalizer` 標準化；VS-PINN 依各向異性縮放座標。
2. **神經網路**：`fourier_mlp.PINNNet` 進行 Fourier 特徵映射與 SIREN MLP；內建 `fourier_normalize_input` 確保頻率穩定。
3. **輸出後處理**：`OutputTransform` 將標準化輸出還原至物理量 (u, v, w, p)。
4. **損失與權重**：`residuals.py` 計算 PDE/邊界/數據損失，`priors.py` 提供均值約束；`weighting.py` 中的 GradNorm/因果/自適應排程平衡權重。
5. **優化流程**：彙總總損失 → 反向傳播 → `Adam/LBFGS/SOAP` 更新；學習率由 `WarmupCosineScheduler` 等策略調度。
6. **檢查點與物理驗證**：保存模型與標準化參數前執行物理一致性檢查。

### 2. 物理引擎: 變數縮放 PINN (VS-PINN)
- **非等向縮放**：`(X, Y, Z) = (N_x·x, N_y·y, N_z·z)`，以大幅放大壁法向梯度 (例 `N_y=12`)，平衡剛性。
- **鏈式法則修正**：PDE 殘差計算時對導數加入縮放係數，如 `∂u/∂x = N_x · ∂u/∂X`。
- **效果**：反向傳播時各向梯度均衡，提升收斂性與穩定度。

### 3. 數據策略: QR 分解感測器佈局
- **QR-Pivoting**：對 DNS 快照矩陣進行列主元 QR，挑選資訊量最高的行作為感測器位置。
- **離線生成**：訓練前產生感測器位置文件，訓練僅讀取這些點作監督信號。

### 4. 訓練策略: 自適應權重、課程學習、因果訓練
- **GradNorm**：動態平衡各損失項的梯度範數，避免單一損失主導。
- **因果權重**：`w(t) = exp(-ε × ∫₀ᵗ Loss(τ)dτ)`，強化早期時間點的物理約束，配置中啟用 `losses.causal_weighting`。
- **課程學習**：多階段調整學習率與損失權重，先擬合數據再強化物理，再精修細節。

---

## 總體工作流程

```mermaid
graph TD
    A[JHTDB 高保真數據] --> B{QR-Pivot 離線分析};
    B --> C[生成最優感測器位置文件];
    C --> D[訓練數據載入器];
    A --> D;
    D --> E{模型訓練};
    subgraph E [訓練循環]
        direction LR
        E1[座標輸入] --> E2(Fourier-SIREN MLP);
        E2 --> E3[預測流場 u,v,w,p];
        E3 --> E4[損失計算];
        subgraph E4
            direction TB
            L1[數據損失]
            L2[物理殘差 (VS-PINN)]
            L3[邊界條件]
        end
        E4 --> E5{GradNorm 動態加權};
        E5 --> E6[總損失];
        E6 --> E7[反向傳播與優化];
    end
    F[課程學習調度器] --> E;
    E --> G[重建的完整流場];
```

---

## 相關文件與資源
- DNS 生成與校準：`docs/KOLMOGOROV_DNS_GUIDE.md`, `scripts/README_REYNOLDS_CALCULATOR.md`
- A100 部署：`A100_DEPLOYMENT_GUIDE.md`
- 感測器與 QR-Pivot：`scripts/README.md`, `docs/QR_SENSOR_VISUALIZATION_GUIDE.md`
- 配置模板：`configs/templates/`，更多設定見 `configs/README.md`

## 貢獻與授權
- 歡迎 Issue/PR，遵循一般 Fork & PR 流程。
- 授權：MIT。若在研究中引用，請註明本專案網址。
