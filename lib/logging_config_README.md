# 日誌配置使用指南

## 概述

`lib/logging_config.py` 提供了統一的日誌配置管理，包括：
- 應用程式日誌設置
- 第三方套件日誌等級控制（Azure SDK, OpenAI, HTTPX等）
- 避免重複配置代碼

## 使用方法

### 1. 快速設置（推薦）

```python
from lib.logging_config import setup_application_logger

# 使用預設配置設置應用程式日誌
logger = setup_application_logger(__name__)
logger.info(" | 應用程式啟動 | ")
```

### 2. 自定義配置

```python
from lib.logging_config import setup_application_logger

logger = setup_application_logger(
    logger_name=__name__,
    log_level=logging.DEBUG,
    log_file="logs/custom.log",
    max_bytes=20*1024*1024,  # 20MB
    backup_count=10,
    console_output=True
)
```

### 3. 獲取已配置的日誌器

```python
from lib.logging_config import get_configured_logger

# 如果日誌器已配置則直接返回，否則使用預設配置
logger = get_configured_logger(__name__)
```

### 4. 僅配置第三方套件日誌等級

```python
from lib.logging_config import configure_third_party_loggers

# 僅設置第三方套件日誌等級，不影響應用程式日誌
configure_third_party_loggers()
```

## 配置的第三方套件

該模組會自動設置以下套件的日誌等級為 WARNING：

- **Azure SDK**: `azure`, `azure.storage`, `azure.core`, `azure.storage.blob`
- **OpenAI**: `openai`, `openai.api_requestor`, `openai._client`, `openai._base_client`
- **HTTP 客戶端**: `httpx`, `urllib3`, `requests`

## 日誌格式

預設格式：`"%(asctime)s - %(message)s"`

輸出範例：
```
2025-11-11 06:23:31,233 -  | 應用程式啟動 | 
```

## 文件管理

- 預設日誌文件：`logs/app.log`
- 自動輪轉：單文件最大 10MB，保留 5 個備份
- 自動創建日誌目錄

## 遷移指南

### 舊代碼
```python
import logging
import logging.handlers

# 大量重複的配置代碼...
logging.getLogger('azure').setLevel(logging.WARNING)
logging.getLogger('azure.storage').setLevel(logging.WARNING)
# ... 更多重複代碼

logger = logging.getLogger(__name__)
file_handler = logging.handlers.RotatingFileHandler(...)
# ... 更多配置
```

### 新代碼
```python
from lib.logging_config import setup_application_logger

logger = setup_application_logger(__name__)
```

## 優點

1. **統一管理**：所有日誌配置集中在一個文件
2. **避免重複**：無需在每個文件中重複配置
3. **第三方套件控制**：自動減少冗長的第三方日誌
4. **靈活配置**：支持自定義參數
5. **易於維護**：修改配置只需更新一個文件