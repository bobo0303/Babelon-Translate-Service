"""
API module with organized structure and backward compatibility
重新組織後仍保持原有的 import 路徑
"""

# 導入子模組
from . import core, translation, audio, websocket

# 導入 websocket router 為了向後兼容
from .websocket.websocket import router as websocket_router

# 為了向後兼容，將子模組的內容暴露在頂級命名空間
from .core.model import *
from .core.threading_api import *
from .core.utils import *
from .core.post_process import *
from .translation.gemma_translate import *
from .translation.gpt_translate import *
from .translation.ollama_translate import *
from .audio.audio_process import *
from .audio.audio_utils import *
from .audio.vad_manager import *
from .websocket.websocket import *
from .websocket.websocket_manager import *
from .websocket.websocket_stt_manager import *

# 將子模組也作為頂級屬性暴露以支持 from api.xxx import
import sys
sys.modules[__name__ + '.model'] = core.model
sys.modules[__name__ + '.threading_api'] = core.threading_api
sys.modules[__name__ + '.utils'] = core.utils
sys.modules[__name__ + '.post_process'] = core.post_process
sys.modules[__name__ + '.gemma_translate'] = translation.gemma_translate
sys.modules[__name__ + '.gpt_translate'] = translation.gpt_translate
sys.modules[__name__ + '.ollama_translate'] = translation.ollama_translate
sys.modules[__name__ + '.audio_process'] = audio.audio_process
sys.modules[__name__ + '.audio_utils'] = audio.audio_utils
sys.modules[__name__ + '.vad_manager'] = audio.vad_manager
sys.modules[__name__ + '.websocket'] = websocket.websocket
sys.modules[__name__ + '.websocket_manager'] = websocket.websocket_manager
sys.modules[__name__ + '.websocket_stt_manager'] = websocket.websocket_stt_manager
