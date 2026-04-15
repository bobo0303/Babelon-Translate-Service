"""
統一日誌管理模組 (Unified Logging Manager)
提供單例模式的日誌管理器，統一管理整個應用程式的日誌配置
"""

import logging
import logging.handlers
import os
from typing import Optional, Dict


class LogManager:
    """
    單例日誌管理器
    統一管理所有模組的日誌配置，避免重複配置
    """
    _instance = None
    _initialized = False
    _loggers: Dict[str, logging.Logger] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LogManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化日誌管理器（只執行一次）"""
        if LogManager._initialized:
            return
            
        LogManager._initialized = True
        
        # 預設配置
        self.default_log_level = logging.INFO
        self.default_log_format = "%(asctime)s - %(message)s"
        self.default_log_file = "logs/app.log"
        self.default_max_bytes = 10 * 1024 * 1024  # 10MB
        self.default_backup_count = 5
        self.console_output = True
        
        # 確保日誌目錄存在
        os.makedirs("logs", exist_ok=True)
        
        # 配置第三方套件的日誌等級
        self._configure_third_party_loggers()
        
        # 配置根日誌器
        self._configure_root_logger()
    
    def _configure_third_party_loggers(self):
        """配置第三方套件的日誌等級，減少冗長的輸出"""
        third_party_loggers = [
            'azure', 'azure.storage', 'azure.core', 'azure.storage.blob',
            'openai', 'openai.api_requestor', 'openai._client', 'openai._base_client',
            'httpx', 'urllib3', 'requests'
        ]
        
        for logger_name in third_party_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    def _configure_root_logger(self):
        """配置根日誌器"""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.default_log_level)
        
        # 清除現有處理器
        root_logger.handlers.clear()
        
        # 添加文件處理器
        file_handler = logging.handlers.RotatingFileHandler(
            self.default_log_file,
            maxBytes=self.default_max_bytes,
            backupCount=self.default_backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(self.default_log_format))
        root_logger.addHandler(file_handler)
        
        # 添加控制台處理器
        if self.console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(self.default_log_format))
            root_logger.addHandler(console_handler)
    
    def get_logger(self, name: str = None) -> logging.Logger:
        """
        獲取指定名稱的日誌器
        
        Args:
            name: 日誌器名稱，通常使用 __name__
            
        Returns:
            配置好的日誌器
        """
        if name is None:
            name = "root"
        
        # 如果已經創建過，直接返回
        if name in self._loggers:
            return self._loggers[name]
        
        # 創建新的日誌器
        logger = logging.getLogger(name)
        logger.setLevel(self.default_log_level)
        
        # 不需要重複添加處理器，繼承根日誌器的處理器即可
        logger.propagate = True
        
        # 快取日誌器
        self._loggers[name] = logger
        
        return logger
    
    def create_logger(
        self,
        name: str,
        log_file: Optional[str] = None,
        log_level: int = None,
        console_output: Optional[bool] = None
    ) -> logging.Logger:
        """
        創建自定義配置的日誌器（用於特殊需求）
        
        Args:
            name: 日誌器名稱
            log_file: 日誌文件路徑（可選）
            log_level: 日誌等級（可選）
            console_output: 是否輸出到控制台（可選）
            
        Returns:
            配置好的日誌器
        """
        logger = logging.getLogger(name)
        logger.setLevel(log_level or self.default_log_level)
        logger.handlers.clear()
        logger.propagate = False
        
        # 添加文件處理器
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.default_max_bytes,
                backupCount=self.default_backup_count,
                encoding='utf-8'
            )
            file_handler.setFormatter(logging.Formatter(self.default_log_format))
            logger.addHandler(file_handler)
        
        # 添加控制台處理器
        if console_output or (console_output is None and self.console_output):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(self.default_log_format))
            logger.addHandler(console_handler)
        
        self._loggers[name] = logger
        return logger
    
    def set_level(self, level: int):
        """
        設置所有日誌器的等級
        
        Args:
            level: 日誌等級 (logging.DEBUG, logging.INFO, etc.)
        """
        self.default_log_level = level
        logging.getLogger().setLevel(level)
        
        for logger in self._loggers.values():
            logger.setLevel(level)
    
    def set_console_output(self, enabled: bool):
        """
        啟用或禁用控制台輸出
        
        Args:
            enabled: True 啟用, False 禁用
        """
        self.console_output = enabled


# 全局日誌管理器實例
_log_manager = LogManager()


def get_logger(name: str = None) -> logging.Logger:
    """
    獲取日誌器的便捷函數
    
    Args:
        name: 日誌器名稱，建議使用 __name__
        
    Returns:
        配置好的日誌器
        
    Examples:
        >>> logger = get_logger(__name__)
        >>> logger.info("This is a log message")
    """
    return _log_manager.get_logger(name)


def create_logger(
    name: str,
    log_file: Optional[str] = None,
    log_level: int = None,
    console_output: Optional[bool] = None
) -> logging.Logger:
    """
    創建自定義配置的日誌器
    
    Args:
        name: 日誌器名稱
        log_file: 獨立的日誌文件路徑
        log_level: 日誌等級
        console_output: 是否輸出到控制台
        
    Returns:
        配置好的日誌器
    """
    return _log_manager.create_logger(name, log_file, log_level, console_output)


def set_log_level(level: int):
    """設置全局日誌等級"""
    _log_manager.set_level(level)


def set_console_output(enabled: bool):
    """啟用或禁用控制台輸出"""
    _log_manager.set_console_output(enabled)


# ===== 向後兼容的函數 =====
def setup_application_logger(
    logger_name: str = None,
    log_level: int = logging.INFO,
    log_format: str = None,
    log_file: str = "logs/app.log",
    max_bytes: int = 10*1024*1024,
    backup_count: int = 5,
    console_output: bool = True
):
    """
    向後兼容的函數（已棄用）
    建議使用 get_logger(__name__) 代替
    """
    if logger_name:
        return get_logger(logger_name)
    return get_logger()


def get_configured_logger(name: str = None):
    """
    向後兼容的函數（已棄用）
    建議使用 get_logger(__name__) 代替
    """
    return get_logger(name)


def configure_third_party_loggers():
    """向後兼容的函數（已棄用）"""
    pass  # 已在 LogManager 初始化時自動配置


def quick_setup():
    """向後兼容的函數（已棄用）"""
    return get_logger()


# 導出主要函數
__all__ = [
    'LogManager',
    'get_logger',
    'create_logger',
    'set_log_level',
    'set_console_output',
    # 向後兼容
    'setup_application_logger',
    'get_configured_logger',
    'configure_third_party_loggers',
    'quick_setup'
]