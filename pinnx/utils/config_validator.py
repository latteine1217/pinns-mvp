"""
配置驗證工具

目標：
1. 檢測常見的配置錯誤（鍵名拼寫、類型錯誤等）
2. 提供清晰的錯誤訊息，避免 Silent Failure
3. 在訓練開始前盡早發現問題

設計原則：
- Fail Fast: 發現錯誤立即停止，不要等到訓練中途
- Clear Errors: 錯誤訊息要明確指出問題和修復方法
- Zero Performance Cost: 只在初始化時執行，不影響訓練速度
"""

import logging
from typing import Dict, Any, List, Tuple, Optional


class ConfigValidator:
    """
    配置驗證器
    
    檢測常見的配置錯誤並提供修復建議。
    """
    
    # 常見錯誤鍵名映射（錯誤 -> 正確）
    COMMON_TYPOS = {
        'loss': 'losses',  # 最常見的錯誤！
        'optimizer': 'training.optimizer',
        'lr': 'training.optimizer.lr',
        'epoch': 'training.epochs',
    }
    
    # 必填頂層鍵
    REQUIRED_TOP_LEVEL_KEYS = [
        'experiment',
        'physics',
        'model',
        'training',
        'losses',
    ]
    
    # 必填子鍵（路徑 -> 鍵列表）
    REQUIRED_NESTED_KEYS = {
        'physics': ['nu', 'domain'],
        'training': ['epochs', 'optimizer'],
        'model': ['in_dim', 'out_dim'],
    }
    
    def __init__(self, strict_mode: bool = False):
        """
        初始化驗證器
        
        Args:
            strict_mode: 嚴格模式（True: 任何警告都視為錯誤）
        """
        self.strict_mode = strict_mode
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """
        驗證配置
        
        Args:
            config: 配置字典
        
        Returns:
            (是否通過, 錯誤列表, 警告列表)
        """
        self.errors = []
        self.warnings = []
        
        # 檢查 1: 常見錯誤鍵名
        self._check_common_typos(config)
        
        # 檢查 2: 必填鍵
        self._check_required_keys(config)
        
        # 檢查 3: 值類型與範圍
        self._check_value_types(config)
        
        # 檢查 4: 邏輯一致性
        self._check_logical_consistency(config)
        
        # 嚴格模式：警告視為錯誤
        if self.strict_mode and self.warnings:
            self.errors.extend(self.warnings)
            self.warnings = []
        
        passed = len(self.errors) == 0
        return passed, self.errors, self.warnings
    
    def _check_common_typos(self, config: Dict[str, Any]) -> None:
        """檢查常見錯誤鍵名"""
        # 檢查頂層錯誤鍵
        for wrong_key, correct_key in self.COMMON_TYPOS.items():
            if wrong_key in config:
                # 檢查是否已有正確的鍵
                if '.' in correct_key:
                    # 嵌套鍵（如 training.optimizer）
                    keys = correct_key.split('.')
                    parent = config
                    for k in keys[:-1]:
                        parent = parent.get(k, {})
                    has_correct_key = keys[-1] in parent
                else:
                    # 頂層鍵
                    has_correct_key = correct_key in config
                
                if not has_correct_key:
                    self.errors.append(
                        f"❌ 配置鍵名錯誤: '{wrong_key}'\n"
                        f"   正確的鍵名應為: '{correct_key}'\n"
                        f"   修復方法: 將 config['{wrong_key}'] 改為 config['{correct_key}']"
                    )
                else:
                    # 同時存在兩個鍵，可能是重複定義
                    self.warnings.append(
                        f"⚠️  配置中同時存在 '{wrong_key}' 和 '{correct_key}'\n"
                        f"   程式會使用 '{correct_key}'，'{wrong_key}' 將被忽略\n"
                        f"   建議: 移除 '{wrong_key}' 避免混淆"
                    )
    
    def _check_required_keys(self, config: Dict[str, Any]) -> None:
        """檢查必填鍵"""
        # 檢查頂層必填鍵
        for key in self.REQUIRED_TOP_LEVEL_KEYS:
            if key not in config:
                self.errors.append(
                    f"❌ 缺少必填配置段: '{key}'\n"
                    f"   修復方法: 在 config 中添加 '{key}' 段落\n"
                    f"   參考: configs/standard_config_template.yml"
                )
        
        # 檢查嵌套必填鍵
        for parent_key, required_keys in self.REQUIRED_NESTED_KEYS.items():
            if parent_key not in config:
                continue  # 已在頂層檢查報告
            
            parent_config = config[parent_key]
            for key in required_keys:
                if key not in parent_config:
                    self.errors.append(
                        f"❌ 缺少必填配置: '{parent_key}.{key}'\n"
                        f"   修復方法: 在 config['{parent_key}'] 中添加 '{key}'\n"
                        f"   參考: configs/standard_config_template.yml"
                    )
    
    def _check_value_types(self, config: Dict[str, Any]) -> None:
        """檢查值類型與範圍"""
        # 檢查學習率
        if 'training' in config and 'optimizer' in config['training']:
            optimizer_cfg = config['training']['optimizer']
            if isinstance(optimizer_cfg, dict):
                lr = optimizer_cfg.get('lr')
                if lr is not None:
                    if not isinstance(lr, (int, float)):
                        self.errors.append(
                            f"❌ 學習率類型錯誤: {type(lr).__name__}\n"
                            f"   期望類型: float\n"
                            f"   當前值: {lr}"
                        )
                    elif lr <= 0 or lr > 1.0:
                        self.warnings.append(
                            f"⚠️  學習率範圍異常: {lr}\n"
                            f"   建議範圍: 1e-5 ~ 1e-2\n"
                            f"   當前值看起來{'過大' if lr > 1.0 else '過小'}"
                        )
        
        # 檢查 epochs
        if 'training' in config:
            epochs = config['training'].get('epochs')
            if epochs is not None:
                if not isinstance(epochs, int):
                    self.errors.append(
                        f"❌ Epochs 類型錯誤: {type(epochs).__name__}\n"
                        f"   期望類型: int\n"
                        f"   當前值: {epochs}"
                    )
                elif epochs <= 0:
                    self.errors.append(
                        f"❌ Epochs 必須為正整數\n"
                        f"   當前值: {epochs}"
                    )
        
        # 檢查模型維度
        if 'model' in config:
            in_dim = config['model'].get('in_dim')
            out_dim = config['model'].get('out_dim')
            
            if in_dim is not None and in_dim not in [2, 3]:
                self.warnings.append(
                    f"⚠️  輸入維度異常: {in_dim}\n"
                    f"   本專案僅支援 2D 或 3D 問題\n"
                    f"   如確認正確，可忽略此警告"
                )
            
            if out_dim is not None and out_dim < 2:
                self.errors.append(
                    f"❌ 輸出維度過小: {out_dim}\n"
                    f"   NS 方程至少需要 [u, v] 兩個速度分量"
                )
    
    def _check_logical_consistency(self, config: Dict[str, Any]) -> None:
        """檢查邏輯一致性"""
        # 檢查 1: 低保真先驗啟用但未提供路徑
        lowfi_cfg = config.get('lowfi_prior', {})
        if lowfi_cfg.get('enabled', False):
            data_path = lowfi_cfg.get('data_path', '')
            if not data_path or data_path == '':
                self.errors.append(
                    "❌ lowfi_prior.enabled=true 但未提供 data_path\n"
                    "   修復方法: 設置 lowfi_prior.data_path 指向 RANS 數據文件\n"
                    "   或者設置 lowfi_prior.enabled=false"
                )
        
        # 檢查 2: 模型輸入維度與物理域不匹配
        if 'model' in config and 'physics' in config:
            in_dim = config['model'].get('in_dim')
            domain = config['physics'].get('domain', {})
            has_z = 'z_range' in domain
            
            if in_dim == 2 and has_z:
                self.warnings.append(
                    "⚠️  模型輸入維度 (in_dim=2) 與物理域 (3D) 不匹配\n"
                    "   可能原因:\n"
                    "   1. 2D 切片訓練（正常，可忽略）\n"
                    "   2. 配置錯誤（應設置 in_dim=3）"
                )
            elif in_dim == 3 and not has_z:
                self.errors.append(
                    "❌ 模型輸入維度 (in_dim=3) 與物理域 (2D) 不匹配\n"
                    "   修復方法: 設置 in_dim=2 或在 physics.domain 中添加 z_range"
                )
        
        # 檢查 3: VS-PINN 配置不完整
        physics_type = config.get('physics', {}).get('type', '')
        if 'vs_pinn' in physics_type.lower():
            vs_pinn_cfg = config.get('physics', {}).get('vs_pinn', {})
            if not vs_pinn_cfg:
                self.errors.append(
                    "❌ physics.type='vs_pinn_*' 但未提供 physics.vs_pinn 配置\n"
                    "   修復方法: 添加 physics.vs_pinn.scaling_factors\n"
                    "   參考: configs/standard_config_template.yml"
                )


def validate_config_file(config: Dict[str, Any], strict_mode: bool = False) -> None:
    """
    驗證配置文件（主入口函數）
    
    Args:
        config: 配置字典
        strict_mode: 嚴格模式（警告視為錯誤）
    
    Raises:
        ValueError: 配置驗證失敗
    """
    validator = ConfigValidator(strict_mode=strict_mode)
    passed, errors, warnings = validator.validate(config)
    
    # 打印警告
    if warnings:
        logging.warning("=" * 80)
        logging.warning("⚠️  配置驗證發現 %d 個警告:", len(warnings))
        for i, warning in enumerate(warnings, 1):
            logging.warning(f"\n警告 {i}:\n{warning}")
        logging.warning("=" * 80)
    
    # 打印錯誤並拋出異常
    if not passed:
        error_msg = f"\n{'=' * 80}\n"
        error_msg += f"❌ 配置驗證失敗，發現 {len(errors)} 個錯誤:\n"
        for i, error in enumerate(errors, 1):
            error_msg += f"\n錯誤 {i}:\n{error}\n"
        error_msg += f"{'=' * 80}\n"
        error_msg += "\n💡 提示: 參考正確的配置模板:\n"
        error_msg += "   configs/standard_config_template.yml\n"
        raise ValueError(error_msg)
    
    logging.info("✅ 配置驗證通過")


def quick_check_common_errors(config: Dict[str, Any]) -> Optional[str]:
    """
    快速檢查最常見的錯誤（用於內部調用）
    
    Args:
        config: 配置字典
    
    Returns:
        錯誤訊息（None 表示無錯誤）
    """
    # 檢查最常見的錯誤：loss vs losses
    if 'loss' in config and 'losses' not in config:
        return (
            "配置鍵名錯誤: 使用了 'loss'（單數）\n"
            "正確的鍵名應為 'losses'（複數）\n"
            "請參考 configs/standard_config_template.yml 修正"
        )
    
    return None
