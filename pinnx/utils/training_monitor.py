"""
通用訓練監控系統

提供自動檢測、日誌解析、趨勢分析等功能，支持多實驗並行監控。
"""

import re
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import yaml


@dataclass
class ProcessInfo:
    """訓練進程資訊"""
    pid: int
    config_name: str
    config_path: str
    cpu_percent: float
    memory_mb: float
    start_time: Optional[datetime] = None


@dataclass
class EpochMetrics:
    """單個 Epoch 的指標"""
    epoch: int
    total_epochs: int
    timestamp: datetime
    losses: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def progress_percent(self) -> float:
        """訓練進度百分比"""
        return (self.epoch / self.total_epochs) * 100 if self.total_epochs > 0 else 0.0


@dataclass
class TrainingStatus:
    """訓練狀態摘要"""
    config_name: str
    process_info: Optional[ProcessInfo]
    current_metrics: Optional[EpochMetrics]
    recent_metrics: List[EpochMetrics] = field(default_factory=list)
    log_file: Optional[Path] = None
    checkpoint_dir: Optional[Path] = None
    
    @property
    def is_active(self) -> bool:
        """是否為活躍訓練"""
        return self.process_info is not None
    
    @property
    def eta_seconds(self) -> Optional[float]:
        """預計剩餘時間（秒）"""
        if not self.current_metrics or len(self.recent_metrics) < 2:
            return None
        
        # 計算平均 epoch 時間
        time_diffs = []
        for i in range(1, len(self.recent_metrics)):
            dt = (self.recent_metrics[i].timestamp - 
                  self.recent_metrics[i-1].timestamp).total_seconds()
            time_diffs.append(dt)
        
        if not time_diffs:
            return None
        
        avg_epoch_time = sum(time_diffs) / len(time_diffs)
        remaining_epochs = (self.current_metrics.total_epochs - 
                           self.current_metrics.epoch)
        
        return avg_epoch_time * remaining_epochs


class ProcessDetector:
    """訓練進程自動檢測器"""
    
    @staticmethod
    def detect_active_trainings() -> List[ProcessInfo]:
        """
        檢測所有活躍的訓練進程
        
        Returns:
            List[ProcessInfo]: 活躍訓練進程列表
        """
        try:
            # 使用 ps 命令查找 train.py 進程
            cmd = ["ps", "aux"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            processes = []
            for line in result.stdout.split('\n'):
                if 'train.py' in line and '--cfg' in line and 'grep' not in line:
                    proc_info = ProcessDetector._parse_process_line(line)
                    if proc_info:
                        processes.append(proc_info)
            
            return processes
        
        except Exception as e:
            print(f"⚠️  檢測進程時發生錯誤: {e}")
            return []
    
    @staticmethod
    def _parse_process_line(line: str) -> Optional[ProcessInfo]:
        """
        解析 ps 命令輸出行
        
        Args:
            line: ps 命令輸出的一行
        
        Returns:
            ProcessInfo 或 None
        """
        try:
            # 解析格式: USER PID %CPU %MEM ... COMMAND
            parts = line.split()
            if len(parts) < 11:
                return None
            
            pid = int(parts[1])
            cpu_percent = float(parts[2])
            memory_mb = float(parts[3]) * 10  # 粗略估算
            
            # 提取配置文件路徑
            cfg_match = re.search(r'--cfg\s+(\S+)', line)
            if not cfg_match:
                return None
            
            config_path = cfg_match.group(1)
            config_name = Path(config_path).stem
            
            return ProcessInfo(
                pid=pid,
                config_name=config_name,
                config_path=config_path,
                cpu_percent=cpu_percent,
                memory_mb=memory_mb
            )
        
        except Exception as e:
            print(f"⚠️  解析進程行時發生錯誤: {e}")
            return None


class LogParser:
    """訓練日誌解析器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化日誌解析器
        
        Args:
            config: 監控配置字典
        """
        self.config = config
        self.epoch_pattern = re.compile(config['log_parsing']['epoch_pattern'])
        self.loss_patterns = [
            re.compile(pattern) 
            for pattern in config['log_parsing']['loss_patterns']
        ]
    
    def parse_log_file(self, log_path: Path, tail_lines: int = 100) -> List[EpochMetrics]:
        """
        解析日誌文件
        
        Args:
            log_path: 日誌文件路徑
            tail_lines: 只解析最後 N 行
        
        Returns:
            List[EpochMetrics]: 解析出的 epoch 指標列表
        """
        if not log_path.exists():
            return []
        
        try:
            # 讀取最後 N 行
            with open(log_path, 'r') as f:
                lines = f.readlines()
            
            lines = lines[-tail_lines:] if len(lines) > tail_lines else lines
            
            metrics_list = []
            current_epoch = None
            current_losses = {}
            
            for line in lines:
                # 檢測 epoch 行
                epoch_match = self.epoch_pattern.search(line)
                if epoch_match:
                    # 保存前一個 epoch 的數據
                    if current_epoch is not None:
                        metrics_list.append(EpochMetrics(
                            epoch=current_epoch,
                            total_epochs=int(epoch_match.group(2)),
                            timestamp=datetime.now(),
                            losses=current_losses.copy()
                        ))
                    
                    # 開始新 epoch
                    current_epoch = int(epoch_match.group(1))
                    current_losses = {}
                
                # 解析 loss 指標
                if current_epoch is not None:
                    for pattern in self.loss_patterns:
                        for match in pattern.finditer(line):
                            key = match.group(1)
                            try:
                                value = float(match.group(2))
                                current_losses[key] = value
                            except ValueError:
                                pass
            
            # 保存最後一個 epoch
            if current_epoch is not None:
                metrics_list.append(EpochMetrics(
                    epoch=current_epoch,
                    total_epochs=current_epoch,  # 會被更新
                    timestamp=datetime.now(),
                    losses=current_losses
                ))
            
            return metrics_list
        
        except Exception as e:
            print(f"⚠️  解析日誌文件時發生錯誤: {e}")
            return []
    
    def detect_anomalies(self, metrics: EpochMetrics) -> List[str]:
        """
        檢測異常值（NaN, Inf, 梯度爆炸等）
        
        Args:
            metrics: Epoch 指標
        
        Returns:
            List[str]: 警告訊息列表
        """
        warnings = []
        
        for key, value in metrics.losses.items():
            # 檢測 NaN/Inf
            if not isinstance(value, (int, float)):
                continue
            
            if value != value:  # NaN check
                warnings.append(f"⚠️  {key} = NaN")
            elif value == float('inf'):
                warnings.append(f"⚠️  {key} = +Inf")
            elif value == float('-inf'):
                warnings.append(f"⚠️  {key} = -Inf")
            
            # 檢測異常大的值
            thresholds = self._get_thresholds(key)
            if thresholds:
                error_threshold = thresholds.get('error')
                warning_threshold = thresholds.get('warning')
                
                if error_threshold is not None and value > error_threshold:
                    warnings.append(f"🔥 {key} = {value:.2e} (超過錯誤閾值)")
                elif warning_threshold is not None and value > warning_threshold:
                    warnings.append(f"⚠️  {key} = {value:.2e} (超過警告閾值)")
        
        return warnings
    
    def _get_thresholds(self, metric_name: str) -> Optional[Dict[str, float]]:
        """取得指標閾值"""
        for category in ['primary', 'rans', 'constraints']:
            metrics = self.config['metrics'].get(category, [])
            for m in metrics:
                if m['name'] == metric_name:
                    return {
                        'warning': m.get('threshold_warning'),
                        'error': m.get('threshold_error')
                    }
        return None


class TrainingMonitor:
    """訓練監控協調器"""
    
    def __init__(self, config_path: str = "configs/monitoring.yml"):
        """
        初始化訓練監控器
        
        Args:
            config_path: 監控配置文件路徑
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.log_parser = LogParser(self.config)
        self.base_dir = Path.cwd()
    
    def _load_config(self) -> Dict[str, Any]:
        """載入監控配置"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_all_training_status(self) -> List[TrainingStatus]:
        """
        取得所有訓練狀態
        
        Returns:
            List[TrainingStatus]: 訓練狀態列表
        """
        # 檢測活躍進程
        active_processes = ProcessDetector.detect_active_trainings()
        
        status_list = []
        for proc in active_processes:
            status = self._build_training_status(proc)
            if status:
                status_list.append(status)
        
        return status_list
    
    def get_training_status(self, config_name: str) -> Optional[TrainingStatus]:
        """
        取得特定訓練的狀態
        
        Args:
            config_name: 配置名稱（不含 .yml 副檔名）
        
        Returns:
            TrainingStatus 或 None
        """
        # 檢測該訓練是否活躍
        active_processes = ProcessDetector.detect_active_trainings()
        proc_info = None
        
        for proc in active_processes:
            if proc.config_name == config_name:
                proc_info = proc
                break
        
        # 構建狀態（即使進程不活躍，也可以解析日誌）
        return self._build_training_status(proc_info, config_name)
    
    def _build_training_status(
        self, 
        proc_info: Optional[ProcessInfo],
        config_name: Optional[str] = None
    ) -> Optional[TrainingStatus]:
        """
        構建訓練狀態
        
        Args:
            proc_info: 進程資訊（可為 None）
            config_name: 配置名稱（當 proc_info 為 None 時使用）
        
        Returns:
            TrainingStatus 或 None
        """
        if proc_info:
            cfg_name = proc_info.config_name
        elif config_name:
            cfg_name = config_name
        else:
            return None
        
        # 查找日誌文件
        log_dir = self.base_dir / self.config['paths']['log_base_dir'] / cfg_name
        log_file = log_dir / "training_stdout.log"
        
        if not log_file.exists():
            # 日誌文件不存在，返回基本狀態
            return TrainingStatus(
                config_name=cfg_name,
                process_info=proc_info,
                current_metrics=None,
                log_file=None
            )
        
        # 解析日誌
        metrics_list = self.log_parser.parse_log_file(log_file, tail_lines=200)
        
        if not metrics_list:
            return TrainingStatus(
                config_name=cfg_name,
                process_info=proc_info,
                current_metrics=None,
                log_file=log_file
            )
        
        # 取得最新指標
        current = metrics_list[-1]
        recent = metrics_list[-20:] if len(metrics_list) >= 20 else metrics_list
        
        # 檢測異常
        current.warnings = self.log_parser.detect_anomalies(current)
        
        # 查找檢查點目錄
        checkpoint_dir = (self.base_dir / 
                         self.config['paths']['checkpoint_base_dir'] / 
                         cfg_name)
        
        return TrainingStatus(
            config_name=cfg_name,
            process_info=proc_info,
            current_metrics=current,
            recent_metrics=recent,
            log_file=log_file,
            checkpoint_dir=checkpoint_dir if checkpoint_dir.exists() else None
        )
    
    def format_status_report(self, status: TrainingStatus) -> str:
        """
        格式化狀態報告
        
        Args:
            status: 訓練狀態
        
        Returns:
            str: 格式化的報告文本
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"📊 Training Status: {status.config_name}")
        lines.append("=" * 80)
        
        # 進程資訊
        if status.process_info:
            proc = status.process_info
            lines.append(f"🟢 Active Process:")
            lines.append(f"   PID: {proc.pid}")
            lines.append(f"   CPU: {proc.cpu_percent:.1f}%")
            lines.append(f"   Memory: {proc.memory_mb:.0f} MB")
        else:
            lines.append("🔴 No Active Process")
        
        lines.append("")
        
        # 訓練進度
        if status.current_metrics:
            metrics = status.current_metrics
            lines.append(f"📈 Progress:")
            lines.append(f"   Epoch: {metrics.epoch}/{metrics.total_epochs} "
                        f"({metrics.progress_percent:.1f}%)")
            
            # ETA
            if status.eta_seconds:
                eta_time = timedelta(seconds=int(status.eta_seconds))
                lines.append(f"   ETA: {eta_time}")
            
            lines.append("")
            
            # 關鍵指標
            lines.append(f"📉 Key Metrics:")
            key_metrics = ['total_loss', 'data_loss', 'pde_loss', 
                          'wall_loss', 'mean_constraint']
            
            for key in key_metrics:
                if key in metrics.losses:
                    value = metrics.losses[key]
                    # 計算趨勢
                    trend = self._calculate_trend(status.recent_metrics, key)
                    trend_icon = "→" if trend == 0 else ("↓" if trend < 0 else "↑")
                    
                    lines.append(f"   {key:20s}: {value:10.4f} {trend_icon}")
            
            # 警告
            if metrics.warnings:
                lines.append("")
                lines.append("⚠️  Warnings:")
                for warning in metrics.warnings:
                    lines.append(f"   {warning}")
        else:
            lines.append("⚠️  No metrics available")
        
        lines.append("=" * 80)
        return "\n".join(lines)
    
    def _calculate_trend(self, recent_metrics: List[EpochMetrics], key: str) -> int:
        """
        計算指標趨勢
        
        Args:
            recent_metrics: 最近的 epoch 指標
            key: 指標名稱
        
        Returns:
            int: -1 (下降), 0 (持平), +1 (上升)
        """
        if len(recent_metrics) < 2:
            return 0
        
        values = [m.losses[key] for m in recent_metrics if key in m.losses]
        if len(values) < 2:
            return 0
        
        # 簡單線性趨勢
        first_half = sum(values[:len(values)//2], 0.0) / (len(values)//2)
        second_half = sum(values[len(values)//2:], 0.0) / (len(values) - len(values)//2)
        
        ratio = (second_half - first_half) / (abs(first_half) + 1e-8)
        
        if ratio < -0.05:
            return -1
        elif ratio > 0.05:
            return 1
        else:
            return 0
