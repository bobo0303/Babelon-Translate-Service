"""
統一日誌配置模組
用於配置應用程式和第三方套件的日誌等級
"""

import logging
import logging.handlers
import os


def configure_third_party_loggers():
    """配置第三方套件的日誌等級，減少冗長的輸出"""
    
    # Azure SDK 相關日誌
    logging.getLogger('azure').setLevel(logging.WARNING)
    logging.getLogger('azure.storage').setLevel(logging.WARNING)
    logging.getLogger('azure.core').setLevel(logging.WARNING)
    logging.getLogger('azure.storage.blob').setLevel(logging.WARNING)
    
    # OpenAI Azure 相關日誌
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('openai.api_requestor').setLevel(logging.WARNING)
    logging.getLogger('openai._client').setLevel(logging.WARNING)
    logging.getLogger('openai._base_client').setLevel(logging.WARNING)
    
    # HTTP 客戶端日誌
    logging.getLogger('httpx').setLevel(logging.WARNING)
    
    # 其他可能的第三方套件
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)


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
    設置應用程式日誌器
    
    Args:
        logger_name: 日誌器名稱，None 表示根日誌器
        log_level: 日誌等級
        log_format: 日誌格式
        log_file: 日誌文件路徑
        max_bytes: 單個日誌文件最大大小
        backup_count: 保留的日誌文件數量
        console_output: 是否輸出到控制台
    
    Returns:
        配置好的日誌器
    """
    
    # 設置預設格式
    if log_format is None:
        log_format = "%(asctime)s - %(message)s"
    
    # 確保日誌目錄存在
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # 配置基本設置
    logging.basicConfig(level=log_level, format=log_format)
    
    # 配置第三方日誌器
    configure_third_party_loggers()
    
    # 獲取或創建日誌器
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    
    # 清除現有的處理器（避免重複添加）
    if logger.handlers:
        logger.handlers.clear()
    
    # 創建文件處理器
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    # 創建控制台處理器（如果需要）
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(console_handler)
    
    # 防止日誌向上傳播（避免重複輸出）
    logger.propagate = False
    
    return logger


def get_configured_logger(name: str = None):
    """
    獲取已配置的日誌器
    如果日誌器尚未配置，會使用預設設置進行配置
    
    Args:
        name: 日誌器名稱
        
    Returns:
        配置好的日誌器
    """
    logger = logging.getLogger(name)
    
    # 如果日誌器沒有處理器，使用預設配置
    if not logger.handlers:
        return setup_application_logger(name)
    
    return logger


# 便捷函數：快速設置應用程式日誌
def quick_setup():
    """快速設置應用程式日誌，使用預設配置"""
    configure_third_party_loggers()
    return setup_application_logger()


# 導出主要函數
__all__ = [
    'configure_third_party_loggers',
    'setup_application_logger', 
    'get_configured_logger',
    'quick_setup'
]