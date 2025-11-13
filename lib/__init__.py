"""
Library module with organized structure and backward compatibility
重新組織後仍保持原有的 import 路徑
"""

# 導入子模組
from . import config, core, storage

# 為了向後兼容，將子模組的內容暴露在頂級命名空間
from .config.constant import *
from .core.logging_config import *
from .core.response_manager import *
from .storage.azure_blob_service import *
from .storage.database_service import *
from .storage.meeting_record import *

# 將子模組也作為頂級屬性暴露
import sys
sys.modules[__name__ + '.constant'] = config.constant
sys.modules[__name__ + '.logging_config'] = core.logging_config
sys.modules[__name__ + '.response_manager'] = core.response_manager
sys.modules[__name__ + '.azure_blob_service'] = storage.azure_blob_service
sys.modules[__name__ + '.database_service'] = storage.database_service
sys.modules[__name__ + '.meeting_record'] = storage.meeting_record
