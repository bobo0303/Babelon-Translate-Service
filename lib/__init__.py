"""
Library module with organized structure and backward compatibility
重新組織後仍保持原有的 import 路徑
"""

# 導入子模組
from . import config, core

# 為了向後兼容，將子模組的內容暴露在頂級命名空間
from .config.constant import *
from .core.logging_config import *

# 將子模組也作為頂級屬性暴露
import sys
sys.modules[__name__ + '.constant'] = config.constant
sys.modules[__name__ + '.logging_config'] = core.logging_config
