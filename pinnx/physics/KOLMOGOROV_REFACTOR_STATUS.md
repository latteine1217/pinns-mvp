## Kolmogorov Flow 2D Status

更新日期：2026-01-02  

### 現況
- 已採用 `NavierStokesBase` 作為共用基類
- 使用標準介面計算殘差、渦度、enstrophy
- 物理參數以標準型別管理（`nu`, `rho`, `amplitude`, `wavenumber`）

### 介面基準
- 主要入口：`KolmogorovFlow2D.residual()`
- 連續性：`compute_continuity_residual(coords, [u, v])`
- 動能：`compute_kinetic_energy([u, v])`
- 物理資訊：`get_physics_info()` 提供標準鍵
