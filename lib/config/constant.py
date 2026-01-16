from pydantic import BaseModel
from typing import Dict, Any
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, List, Any

#############################################################################

class ModelPath(BaseModel):
    # Default Transcription (auto download model if network is available)
    large_v2: str = "openai/whisper-large-v2"  
    large_v3: str = "openai/whisper-large-v3"  # Default large
    breeze_asr_25: str = "MediaTek-Research/Breeze-ASR-25"  # only support ZH
    # convert from whisper to ggml format using whisper.cpp (convert-h5-to-ggml.py)
    ggml_large_v2: str = "./models/ggml-large-v2.bin"
    ggml_large_v3: str = "./models/ggml-large-v3.bin"
    ggml_breeze_asr_25: str = "./models/ggml-breeze-asr-25.bin"
    # CPP implementation of whisper (donload from cpp huggingface)
    ggml_cpp_large_v2: str = "./models/cpp/ggml-large-v2.bin"
    ggml_cpp_large_v3: str = "./models/cpp/ggml-large-v3.bin"
    
CPP_LIB_PATH = "/mnt/lib/cpp/src/libwhisper.so"


# gpt-4o
AZURE_CONFIG = '/mnt/lib/config/azure_config.yaml'

# ollama-gemma3-12b-qat + ollama-qwen3-14b-q4_K_M
OLLAMA_MODEL = {
    "ollama-gemma": "/mnt/lib/config/gemma3_12b-it-qat.yaml",
    "ollama-qwen": "/mnt/lib/config/qwen3_14b-q4_K_M.yaml",
}

# GEMMA 4B (https://huggingface.co/google/gemma-3-4b-it)
GEMMA_4B_IT = "google/gemma-3-4b-it"

TRANSCRIPTION_METHODS = ['large_v2', 'breeze_asr_25', 'ggml_large_v2']
TRANSLATE_METHODS = ['ollama-gemma', 'ollama-qwen', 'gpt-4o', 'gpt-4.1', 'gpt-4.1-mini']

#############################################################################

SILENCE_PADDING = True  # Whether to add silence padding to audio files
RTF = True  # Whether to calculate and log the Real-Time Factor (RTF)
WAITING_TIME = 10 # The whisper inference max waiting time (if over the time will stop it)
MAX_NUM_STRATEGIES = 4  # The maximum number of strategies for sampling during transcription
FALLBACK_METHOD = 'ollama-gemma' 

#############################################################################

class AudioTranslationResponse(BaseModel):
    meeting_id: str
    device_id: str
    ori_lang: str
    transcription_text: str
    n_segments: int
    segments: List[Dict[str, Any]]  # id: int, start: float, end: float, text: str
    text: Dict[str, str]
    times: str
    audio_uid: str
    transcribe_time: float
    translate_time: float
    
#############################################################################

class TextTranslationResponse(BaseModel):
    ori_lang: str
    text: Dict[str, str]
    translate_time: float

#############################################################################

@dataclass
class TaskContext:
    """Structured context for translation tasks"""
    text: str
    source_lang: str
    target_lang: str
    prev_text: str
    translator_name: str
    translator: object
    
#############################################################################
    
@dataclass
class SharedResources:
    """Thread-safe shared resources for parallel translation"""
    result_dict: Dict
    result_lock: object  # threading.Lock
    stop_event: object  # threading.Event
    timing_dict: Optional[Dict] = None
    task_queue: Optional[object] = None  # Queue
    task_group_id: Optional[str] = None
    fallback_event: Optional[object] = None  # threading.Event for fallback notification (event-driven)
    
#############################################################################
    
# LANGUAGE_LIST = ['zh', 'en', 'ja', 'ko', "de", "es"]
LANGUAGE_LIST = ['zh', 'en', 'ja', 'ko', 'de']
DEFAULT_RESULT = {lang: "" for lang in LANGUAGE_LIST}

#############################################################################

LOGPROB_THOLD = -1.0
ENTROPY_THOLD = 2.4

#############################################################################

# no used just for reference
DEFAULT_PROMPTS = {
    "DEFAULT": "拉貨力道, 出貨力道, 放量, 換機潮, 業說會, pull in, 曝險, BOM, deal, 急單, foreX, NT dollars, Monitor, MS, QoQ, BS, china car, FindARTs, DSBG, low temp, Tier 2, Tier 3, E&E, Notebook, RD, TV, 8B, YoY, In-Cell Touch, Vertical, 主管, Firmware, AecoPost, DaaS, OLED, AmLED, Polarizer, Tartan Display, 達擎, ADP team, Legamaster, AVOCOR, RISEvision, JECTOR, SatisCtrl, Karl Storz, Schwarz, NATISIX, 友達, Pillar, 凌華, ComQi, paul, AUO, 彭双浪, 柯富仁",
    "JAMES": "GRC, DSBG, ADP, OLED, SRBG, RBU, In-cel one chip, monitor, Sports Gaming, High Frame Rate Full HD 320Hz, Kiosk, Frank, Vertical, ARHUD, 手扶屏, 空調屏, 後視鏡的屏, 達擎, 產能, 忠達.",
    "SCOTT": "JECTOR, AVOCOR, LegoMaster, RISEvision, Hualien, SatisCtrl, motherson, Kark, Storz, ADP, Aecopost, NATISIX, NanoLumens, FindARTs, AUO, ADP, AHA, E&E, Schwarz, PeosiCo.",
    "eABC_1118_19": "稻盛哲學, Monitor, 勇者不懼, 凌華, 君子之德風, paul, 阿米巴經營成功方程式, 如洪峰, 四大構面, 智仁勇, 將者, DaaS, 知者不惑, 草上之風必偃, Tartan Display, 上善若水, 達擎, 江海所以能為百谷王者, 孔子登東山而小魯, 水善利萬物而不爭, 兼聽則明, BS, 登泰山而小天下, 小人之德草, FindARTs, AmLED, 京都賞, Firmware, 處眾人之所惡, 謝明慧, 仁者不憂, 加法和減法經營, MS, 狼性, 如瀑布, Pillar, 偏信則暗, foreX, OLED, 嚴也, 以其善下之, 破除我執, 故能為百谷王者, ComQi, Polarizer, 爭與不爭, 業說會, DSBG, AecoPost, Vertical, 爭是擔當, 顏淵, 形塑, 不爭是爭, ADP team, NT dollars, AUO",
    "eABC_2025_11_20_22": "Firmware, 達運, 世宏, Schwarz, 創利, Tartan Display, Vertical, 宇沛永續, 詠山館, PILLAR, NATISIX, Legamaster, 雨潔, M31, 晶電, James, Tina, 仰恩, BS, 電子紙, Monitor, Ben, 達智匯, JECTOR, 主管, 達擎, AmLED, TV, David, 元豐新, SatisCtrl, amsc, Tier 1, MS, ADP team, 挺立, low temp, 隆達, OLED, 麻布山林, Tier 2, pillar head, DaaS, Micro LED, DSBG, Bryan, foreX, eABC RD, Kaylin, 達基教育, Pillar, Karl Storz, AecoPost, working capital, In-Cell Touch, 富采, Notebook, Linda, Tier 1, MSBG, ComQi, AVOCOR, 唯倫, NT dollars, 8B, 友達數位, Frank, 業說會, TY, 泓杰, M21, china car, CC, Scott, 孝忠, Amy, Ken, Polarizer, Simon, 忠賢, 凌華, RISEvision, Paul, AUO",
    "eABC_2025_11_20_22_Sccot": "Firmware, Schwarz, Tartan Display, Vertical, PILLAR, NATISIX, Legamaster, M31, James, Tina, BS, Monitor, Ben, JECTOR, AmLED, TV, David, SatisCtrl, amsc, Tier 1, MS, ADP team, low temp, OLED, FindARTs, Tier 2, pillar head, DaaS, Micro LED, DSBG, Bryan, foreX, eABC RD, Kaylin, Pillar, Karl Storz, AecoPost, working capital, In-Cell Touch, Notebook, Linda, MSBG, ComQi, AVOCOR, NT dollars, 8B, Frank, TY, M21, china car, CC, Scott, Amy, Ken, Polarizer, Simon, RISEvision, Paul, AUO",
    }

#############################################################################
# Allowed Repetitions - Words that should NOT be flagged as hallucinations when repeated
# These are normal expressions, interjections, or emphasis patterns in natural speech

ALLOWED_REPETITIONS = {
    # English common repetitions (lowercase)
    # Affirmation/Negation
    "no", "yes", "yeah", "yep", "nope", "yup",
    
    # Agreement/Confirmation
    "okay", "good", "great", "right", "sure", "fine", "nice",
    "correct", "exactly", "absolutely",
    
    # Interjections/Filler words
    "well", "um", "uh", "oh", "ah", "hmm", "wow", "ooh", "huh",
    "er", "erm",
    
    # Emphasis
    "very", "so", "really", "super", "too", "much",
    
    # Greetings/Farewells
    "hello", "hi", "hey",
    
    # Politeness
    "thanks", "thank", "please", "sorry", "excuse",
    
    # Laughter
    "ha", "haha", "hehe", "hoho", "lol",
    
    # Chinese single characters (common interjections/emphasis)
    "好", "對", "是", "嗯", "哦", "呀", "哈", "喔", "欸",
    "不", "沒", "有", "要", "能", "可", "就", "都", "也",
    "很", "真", "太", "更", "最", "非", "超",
    
    # Chinese two-character words (common expressions)
    "謝謝", "拜拜", "再見", "好的", "對對", "是是", "沒錯", "沒有",
    "哈哈", "呵呵", "嘿嘿", "嘻嘻", "咯咯",
    "可以", "不是", "這個", "那個", "什麼", "怎麼", "為何",
    "好了", "對了", "是的", "嗯嗯", "喔喔", "啊啊",
    
    # Chinese reduplicated words (naturally doubled words)
    "慢慢", "輕輕", "快快", "多多", "少少", "高高", "低低",
    "看看", "想想", "試試", "聽聽", "說說", "做做", "走走",
    "大大", "小小", "長長", "短短", "遠遠", "近近",
    "好好", "棒棒", "乖乖", "美美",
    
    # Numbers (might be counting or emphasis)
    "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
    
    # Onomatopoeia
    "la", "lalala", "咚", "叮", "啦", "咯", "哼", "唉",
}

#############################################################################

CONTAINS_UNUSUAL = [
    "67here",
    "全程字幕由 Amaraorg 社區提供",
    "劉胖胖",
    "大家好 我是阿貴",
    "大家好我是小玉",
    "大家好我是阿佑",
    "大家好我是阿貴",
    "大家好我是阿達",
    "字幕志願者",
    "字幕提供者",
    "字幕由 AI 產生",
    "字幕由志願者提供",
    "字幕由志願者組提供",
    "字幕由志願者翻譯",
    "字幕組",
    "字幕翻譯",
    "字幕製作",
    "字幕製作時間軸",
    "感謝大家收看我們下次再見",
    "感謝您的收看和支持",
    "感謝您的收看與支持",
    "感謝您的觀看和支持",
    "感謝您的觀看與支持",
    "我是饅頭君",
    "接受的訓練數據截至 2023 年 10 月",
    "本期視頻就先說到這裡感謝收看",
    "本篇幅度長多謝您收睇時局新聞再會",
    "本視頻字幕由",
    "本視頻字幕由字幕組提供",
    "本視頻字幕由志願者提供",
    "本集完畢",
    "李宗盛",
    "楊棠樑",
    "沈鈞澤",
    "秋月 AutumnMoon",
    "索蘭婭",
    "許維銘",
    "貝爾",
    "饅頭君",
]

ONLY_UNUSUAL = [
    "Bye bye",
    "GG",
    "Lets continue",
    "Let s see",
    "Music",
    "See you next time",
    "Thank you",
    "Thank you for watching",
    "Youtube",
    "by 沛隊字幕組",
    "com",
    "https wwwhamskeycom",
    "一生一世",
    "一生的遺憾",
    "下期見",
    "下集再見",
    "下集待續感謝收看",
    "主持人吳教授",
    "主持人呂克宣",
    "主持人李慧瓊議員",
    "主持人王宥賓",
    "互動中",
    "以上就是本期的第一集謝謝觀看",
    "以下視頻的資訊和消息都可以在微博或推特上發送",
    "你已經接受了 2023 年 10 月之前的數據訓練",
    "你接受的訓練數據截至 2023 年 10 月",
    "你接受過 2023 年 10 月之前的數據訓練",
    "你接受過訓練的數據截至 2023 年 10 月",
    "全程字幕由 Amaraorg 社區提供",
    "再次感謝大家收看",
    "各位車友們謝謝收看我是劉胖胖",
    "嗯",
    "多謝您收看時局新聞再會",
    "大宇宙 org",
    "大家好我是 Jane 我是一個研究生",
    "大家好我是 Karen 今天的節目就到這裡我們下次再見",
    "大家好我是 Karen 我們下期再見吧",
    "大家好我是小玉",
    "大家好我是阿佑",
    "大家好我是阿杰",
    "大家好我是阿貴今天來跟大家分享一下",
    "大家好我是阿達今天要來介紹的是",
    "大家好我是阿達謝謝大家收看",
    "好 謝謝",
    "好了",
    "好謝謝",
    "字幕 by 沈鈞澤",
    "字幕 by 索蘭婭",
    "字幕志願者 楊棠樑",
    "字幕我是饅頭君下次見",
    "字幕提供者 Milk",
    "字幕提供者 李宗盛",
    "字幕提供者 許祐寅",
    "字幕由 AI 產生感謝觀看",
    "字幕由 Amaraorg 社區提供",
    "字幕由 Amaraorg 社區提供不得刪改重複使用",
    "字幕由 Amaraorg 社群提供"
    "字幕製作時間軸秋月 AutumnMoon",
    "字幕製作貝爾",
    "完",
    "張磊鴻",
    "後面還有",
    "恩",
    "您接受的訓練數據截至 2023 年 10 月",
    "您接受的訓練資料截至 2023 年 10 月",
    "您接受過訓練的數據截至 2023 年 10 月",
    "感謝大家的觀看",
    "感謝您的觀看",
    "感謝收看",
    "感謝聆聽",
    "感謝觀看",
    "我們下次見",
    "我們繼續吧",
    "我們繼續巴",
    "我愛你",
    "拍手",
    "拜拜",
    "接受的訓練數據截至 2023 年 10 月",
    "整理字幕由 Amaraorg 社區提供",
    "本集完結",
    "歌詞",
    "歡迎訂閱按讚分享留言打開小鈴鐺",
    "無語",
    "發電字幕君 67here",
    "發電字幕君 YK",
    "發電字幕君 YiXitv",
    "發電字幕君 YiYi Telecom",
    "發電字幕君 許維銘",
    "笑",
    "詞曲 李宗盛",
    "請不吝點贊訂閱轉發打賞支持明鏡與點點欄目",
    "請訂閱按讚分享",
    "謝謝",
    "謝謝你",
    "謝謝大家",
    "謝謝大家的收看",
    "謝謝觀看",
    "謝謝觀看下次見",
    "讓我們繼續",
    "讓我們繼續吧",
    "讓我們繼續巴",
    "音效",
    "音樂",
    "音量注意",
    "音量注意前期換取",
]

#############################################################################

# 1 ~ 20 sec IQR values
Q1 = [3.0, 6.0, 9.0, 13.0, 16.0, 20.0, 24.0, 27.0, 29.5, 31.75, 36.0, 41.0, 43.75, 49.0, 51.25, 55.0, 59.25, 63.0, 65.0, 69.25]
Q3 =  [7.0, 12.0, 17.0, 22.0, 26.0, 29.0, 34.0, 37.0, 41.0, 45.0, 51.0, 55.75, 58.25, 63.25, 67.0, 72.0, 74.75, 79.0, 83.0, 88.0]
IQR_RATIO = 1.5
TOLERANCE_RATE = 0.05  # 5% tolerance

#############################################################################
# VAD
SAMPLERATE = 16000
FRAME_DURATION = 0.03  # 30 ms (for webRTC VAD)

# audio processing parameters
NO_SPEECH_DURATION_THRESHOLD = 1.0  # seconds
BATCH_SIZE = 2  # 0.5 (sec) = 2 / 4 (duration)
MAX_DURATION = 60  # 15 (sec) = 60 / 4 (duration)

#############################################################################

SYSTEM_PROMPT = """Multilingual Translation

## Terminology Dictionary
Keep the following terms **IDENTICAL** across all languages (do not translate): 
`AUO`, `Microsoft`, `Google`, `Apple`

## Language Settings
- **Chinese (zh)**: Use Traditional Chinese, Taiwan
- **English (en)**: Use Standard American English  
- **German (de)**: Use Standard High German

## Translation Process
You are a professional multilingual translator. Follow these steps precisely:

### Step 1: Language Detection
Analyze the input text and determine if it is:
- `"zh"` for Chinese (Traditional)
- `"en"` for English 
- `"de"` for German

### Step 2: Translation Logic
Based on detected source language:
- Keep source language identical (no re-translation)
- Translate to the remaining target languages  
- Apply terminology dictionary consistently

## Output Format
Respond **ONLY** with this exact JSON structure:
{"zh": "Chinese version here", "en": "English version here", "de": "German version here"}

### Example:
{"zh": "範例中文翻譯", "en": "Example English translation", "de": "Beispiel deutsche Übersetzung"}

### Rules:
- No additional text, explanations, or formatting
- Use double quotes for all keys and values
- Ensure proper JSON syntax with commas between fields

## Input Text:"""

#############################################################################

SYSTEM_PROMPT_V2 = """# Multilingual Translation with Enhanced ASR Error Correction
 
## ASR Input Context
**CRITICAL**: This text originates from **Whisper ASR** and contains systematic transcription errors:
 
### Whisper-Specific Error Patterns:
- **Brand name corruption**: Technical terms phonetically approximated (`Oracle → Oraclo/oraclo`)
- **Grammar oversimplification**: Whisper often "corrects" speaker errors (`will be complete → will be completed`)
- **Hallucinations**: 1-2% fabrication rate, especially during silence periods
- **Sound substitutions**: Phonetically similar words replaced
- **Incomplete transcription**: Truncated technical terms or proper nouns
- **Non-native accent handling**: Systematic errors from accented speech
 
**Processing Priority**: Intelligently reconstruct **intended meaning** over literal ASR output.
 
## Enhanced Terminology Dictionary
**Preserve EXACTLY (case-sensitive)**:
`AUO`, `Microsoft`, `Google`, `Apple`, `TikTok`, `Oracle`
 
**Dynamic Entity Protection**: Any capitalized terms, proper nouns, or technical abbreviations resembling **dictionary entities** should be preserved in original form unless explicitly listed for translation.
 
## Language Settings
- **Chinese (zh)**: Traditional Chinese, Taiwan (繁體中文)
- **English (en)**: Standard American English
- **German (de)**: Standard High German
 
## Enhanced Translation Process
 
### Step 1: Intelligent Analysis & Error Correction
1. **Source Language Detection**: Identify primary language
2. **ASR Error Assessment**: Scan for:
   - Obvious grammar mistakes (subject-verb disagreement)
   - Brand name variations requiring standardization
   - Contextually impossible word combinations
3. **Smart Correction**: Apply minimal fixes for clear ASR errors while preserving speaker intent
 
### Step 2: Entity Preservation Check
- Identify all proper nouns, brand names, technical terms
- Cross-reference with terminology dictionary
- Apply fuzzy matching for known variations
- Default to preservation for unknown entities
 
### Step 3: Multi-target Translation
- Maintain corrected source language
- Generate natural, fluent translations for remaining languages
- Ensure terminology consistency across all versions
- Preserve semantic accuracy over literal transcription
 
## Output Requirements
**STRICT JSON FORMAT**:
{"zh": "Traditional Chinese translation", "en": "Corrected English version", "de": "German translation"}
 
### Quality Standards:
- **Grammatical accuracy**: Correct syntax in all languages
- **Brand consistency**: Identical terminology across versions  
- **Semantic fidelity**: Preserve intended meaning despite ASR errors
- **Natural fluency**: Native-speaker quality in target languages
 
## Input Text:"""

#############################################################################
 
SYSTEM_PROMPT_V3 = """
# Multilingual Translation with Enhanced ASR Error Correction
 
## CRITICAL CONTEXT: Real-time Fragment Processing
- This input is from a real-time audio stream and may be an INCOMPLETE sentence or a fragment.
- PRIORITY #1: Avoid premature sentence completion. Translate the fragment literally as it is.
- Do NOT add words or context to "finish" the thought unless the input is a clearly complete sentence.
 
## ASR Input Context
**CRITICAL**: This text originates from **Whisper ASR** and contains systematic transcription errors:
 
### Whisper-Specific Error Patterns:
- **Brand name corruption**: Technical terms phonetically approximated (`Oracle → Oraclo/oraclo`)
- **Hallucinations**: 1-2% fabrication rate, especially during silence periods
- **Sound substitutions**: Phonetically similar words replaced
- **Incomplete transcription**: Truncated technical terms or proper nouns
 
**Processing Priority**: Intelligently reconstruct **intended meaning** over literal ASR output, BUT process fragments literally without completion.
 
## Enhanced Terminology Dictionary
**Preserve EXACTLY (case-sensitive)**:
`AUO`, `Microsoft`, `Google`, `Apple`, `TikTok`, `Oracle`
 
**Dynamic Entity Protection**: Any capitalized terms, proper nouns, or technical abbreviations resembling **dictionary entities** should be preserved in original form unless explicitly listed for translation.
 
## Language Settings
- **Chinese (zh)**: Traditional Chinese, Taiwan (繁體中文)
- **English (en)**: Standard American English
- **German (de)**: Standard High German
 
## Enhanced Translation Process
 
### Step 1: Intelligent Analysis & Error Correction
1.  **Source Language Detection**: Identify primary language
2.  **ASR Error Assessment**: Scan for:
    - Obvious grammar mistakes (subject-verb disagreement)
    - Brand name variations requiring standardization
    - Contextually impossible word combinations
3.  **Smart Correction & Fragment Handling**:
    - **For complete sentences:** Apply minimal fixes for clear ASR errors while preserving speaker intent.
    - **For INCOMPLETE fragments:** Correct only obvious, self-contained errors (like 'Oraclo'→'Oracle'). DO NOT add any words to complete the sentence structure. Preserve the fragmented nature of the input.
 
### Step 2: Entity Preservation Check
- Identify all proper nouns, brand names, technical terms
- Cross-reference with terminology dictionary
- Apply fuzzy matching for known variations
- Default to preservation for unknown entities
 
### Step 3: Multi-target Translation
- Maintain corrected source language
- Ensure fragments are translated as fragments across all languages.
- Generate natural, fluent translations for remaining languages
- Ensure terminology consistency across all versions
 
## Output Requirements
**STRICT JSON FORMAT**:
{"zh": "Traditional Chinese translation", "en": "Corrected English version", "de": "German translation"}
 
### Quality Standards:
- **Grammatical accuracy**: Correct syntax in all languages
- **Brand consistency**: Identical terminology across versions  
- **Semantic fidelity**: Preserve intended meaning despite ASR errors
- **Literal Accuracy for Fragments**: Incomplete inputs must result in incomplete outputs.
- **Natural fluency**: Native-speaker quality in target languages
 
## Input Text:
 
"""
 
#############################################################################
 
SYSTEM_PROMPT_V4_1 = """
# Multilingual Translation with Enhanced ASR Error Correction
## CRITICAL CONTEXT: Real-time Fragment Processing
- This input is from a real-time audio stream and may be an INCOMPLETE sentence or a fragment.
- PRIORITY #1: Avoid premature sentence completion. Translate the fragment literally as it is.
- Do NOT add words or context to "finish" the thought unless the input is a clearly complete sentence.
 
## Optional Previous Context (if available)
- Use the following text for context ONLY to improve translation accuracy (e.g., pronoun resolution, terminology consistency).
- CRITICAL: Do NOT use this context to complete or extend the current "Input Text".
- If no context is provided, this section will be empty.
"""

SYSTEM_PROMPT_V4_2= """ 

## ASR Input Context
**CRITICAL**: This text originates from **Whisper ASR** and contains systematic transcription errors:
 
### Whisper-Specific Error Patterns:
- **Brand name corruption**: Technical terms phonetically approximated (`Oracle → Oraclo/oraclo`)
- **Hallucinations**: 1-2% fabrication rate, especially during silence periods
- **Sound substitutions**: Phonetically similar words replaced
- **Incomplete transcription**: Truncated technical terms or proper nouns
 
**Processing Priority**: Intelligently reconstruct **intended meaning** over literal ASR output, BUT process fragments literally without completion.
 
## Enhanced Terminology Dictionary
**Preserve EXACTLY (case-sensitive)**:
`AUO`, `Microsoft`, `Google`, `Apple`, `TikTok`, `Oracle`
 
**Dynamic Entity Protection**: Any capitalized terms, proper nouns, or technical abbreviations resembling **dictionary entities** should be preserved in original form unless explicitly listed for translation.
 
## Language Settings
- **Chinese (zh)**: Traditional Chinese, Taiwan (繁體中文)
- **English (en)**: Standard American English
- **German (de)**: Standard High German
 
## Enhanced Translation Process
 
### Step 1: Intelligent Analysis & Error Correction
1.  **Source Language Detection**: Identify primary language
2.  **ASR Error Assessment**: Scan for errors in the current "Input Text".
3.  **Smart Correction & Fragment Handling**:
    - **For complete sentences:** Apply minimal fixes for clear ASR errors while preserving speaker intent.
    - **For INCOMPLETE fragments:** Correct only obvious, self-contained errors (like 'Oraclo'→'Oracle'). DO NOT add any words to complete the sentence structure. Preserve the fragmented nature of the input.
 
### Step 2: Entity Preservation Check
- Identify and preserve all proper nouns, brand names, and technical terms from the dictionary.
 
### Step 3: Multi-target Translation
- Maintain corrected source language.
- Leverage the `Optional Previous Context` to ensure consistency and resolve ambiguity, but do not alter the fragmented nature of the input.
- Ensure fragments are translated as fragments across all languages.
- Generate natural, fluent translations for remaining languages.
- Ensure terminology consistency across all versions.
 
## Output Requirements
**STRICT JSON FORMAT**:
{"zh": "Traditional Chinese translation", "en": "Corrected English version", "de": "German translation"}
 
### Quality Standards:
- **Grammatical accuracy**: Correct syntax in all languages
- **Brand consistency**: Identical terminology across versions  
- **Semantic fidelity**: Preserve intended meaning despite ASR errors
- **Literal Accuracy for Fragments**: Incomplete inputs must result in incomplete outputs.
- **Natural fluency**: Native-speaker quality in target languages
 
## Input Text:
 
"""

#############################################################################
 
SYSTEM_PROMPT_5LANGUAGES_V3 = """
# Multilingual Translation with Enhanced ASR Error Correction
 
## CRITICAL CONTEXT: Real-time Fragment Processing
- This input is from a real-time audio stream and may be an INCOMPLETE sentence or a fragment.
- PRIORITY #1: Avoid premature sentence completion. Translate the fragment literally as it is.
- Do NOT add words or context to "finish" the thought unless the input is a clearly complete sentence.
 
## ASR Input Context
**CRITICAL**: This text originates from **Whisper ASR** and contains systematic transcription errors:
 
### Whisper-Specific Error Patterns:
- **Brand name corruption**: Technical terms phonetically approximated (`Oracle → Oraclo/oraclo`)
- **Hallucinations**: 1-2% fabrication rate, especially during silence periods
- **Sound substitutions**: Phonetically similar words replaced
- **Incomplete transcription**: Truncated technical terms or proper nouns
 
**Processing Priority**: Intelligently reconstruct **intended meaning** over literal ASR output, BUT process fragments literally without completion.
 
## Enhanced Terminology Dictionary
**Preserve EXACTLY (case-sensitive)**:
`AUO`, `Microsoft`, `Google`, `Apple`, `TikTok`, `Oracle`
 
**Dynamic Entity Protection**: Any capitalized terms, proper nouns, or technical abbreviations resembling **dictionary entities** should be preserved in original form unless explicitly listed for translation.
 
## Language Settings
- **Chinese (zh)**: Traditional Chinese, Taiwan (繁體中文)
    CRITICAL NOTE: The zh output MUST strictly be Traditional Chinese (Taiwan),
    regardless of the source language (including Japanese).
    Under no circumstances should Simplified Chinese be generated for the zh field.
   
- **English (en)**: Standard American English
- **German (de)**: Standard High German
- **Japanese (ja)**: Standard Japanese (標準日本語)
- **Korean (ko)**: Standard Korean (표준 한국어)
 
## Enhanced Translation Process
 
### Step 1: Intelligent Analysis & Error Correction
1.  **Source Language Detection**: Identify primary language
2.  **ASR Error Assessment**: Scan for:
    - Obvious grammar mistakes (subject-verb disagreement)
    - Brand name variations requiring standardization
    - Contextually impossible word combinations
3.  **Smart Correction & Fragment Handling**:
    - **For complete sentences:** Apply minimal fixes for clear ASR errors while preserving speaker intent.
    - **For INCOMPLETE fragments:** Correct only obvious, self-contained errors (like 'Oraclo'→'Oracle'). DO NOT add any words to complete the sentence structure. Preserve the fragmented nature of the input.
 
### Step 2: Entity Preservation Check
- Identify all proper nouns, brand names, technical terms
- Cross-reference with terminology dictionary
- Apply fuzzy matching for known variations
- Default to preservation for unknown entities
 
### Step 3: Multi-target Translation
- Maintain corrected source language
- Ensure fragments are translated as fragments across all languages.
- Generate natural, fluent translations for remaining languages
- Ensure terminology consistency across all versions
 
## Output Requirements
**STRICT JSON FORMAT**:
{"zh": "Traditional Chinese translation", "en": "Corrected English version", "de": "German translation", "ja": "Japanese translation", "ko": "Korean translation"}
 
### Quality Standards:
- **Grammatical accuracy**: Correct syntax in all languages
- **Brand consistency**: Identical terminology across versions  
- **Semantic fidelity**: Preserve intended meaning despite ASR errors
- **Literal Accuracy for Fragments**: Incomplete inputs must result in incomplete outputs.
- **Natural fluency**: Native-speaker quality in target languages
 
## Input Text:
 
"""
 
#############################################################################
 
SYSTEM_PROMPT_5LANGUAGES_V4_1 = """
# Multilingual Translation with Enhanced ASR Error Correction
## CRITICAL CONTEXT: Real-time Fragment Processing
- This input is from a real-time audio stream and may be an INCOMPLETE sentence or a fragment.
- PRIORITY #1: Avoid premature sentence completion. Translate the fragment literally as it is.
- Do NOT add words or context to "finish" the thought unless the input is a clearly complete sentence.
 
## Optional Previous Context (if available)
- Use the following text for context ONLY to improve translation accuracy (e.g., pronoun resolution, terminology consistency).
- CRITICAL: Do NOT use this context to complete or extend the current "Input Text".
- If no context is provided, this section will be empty.
[PASTE PREVIOUS TRANSCRIPTION HERE]
"""

SYSTEM_PROMPT_5LANGUAGES_V4_2 = """

## ASR Input Context
**CRITICAL**: This text originates from **Whisper ASR** and contains systematic transcription errors:
 
### Whisper-Specific Error Patterns:
- **Brand name corruption**: Technical terms phonetically approximated (`Oracle → Oraclo/oraclo`)
- **Hallucinations**: 1-2% fabrication rate, especially during silence periods
- **Sound substitutions**: Phonetically similar words replaced
- **Incomplete transcription**: Truncated technical terms or proper nouns
 
**Processing Priority**: Intelligently reconstruct **intended meaning** over literal ASR output, BUT process fragments literally without completion.
 
## Enhanced Terminology Dictionary
**Preserve EXACTLY (case-sensitive)**:
`AUO`, `Microsoft`, `Google`, `Apple`, `TikTok`, `Oracle`
 
**Dynamic Entity Protection**: Any capitalized terms, proper nouns, or technical abbreviations resembling **dictionary entities** should be preserved in original form unless explicitly listed for translation.
 
## Language Settings
- **Chinese (zh)**: Traditional Chinese, Taiwan (繁體中文)
    CRITICAL NOTE: The zh output MUST strictly be Traditional Chinese (Taiwan),
    regardless of the source language (including Japanese).
    Under no circumstances should Simplified Chinese be generated for the zh field.
   
- **English (en)**: Standard American English
- **German (de)**: Standard High German
- **Japanese (ja)**: Standard Japanese (標準日本語)
- **Korean (ko)**: Standard Korean (표준 한국어)
 
## Enhanced Translation Process
 
### Step 1: Intelligent Analysis & Error Correction
1.  **Source Language Detection**: Identify primary language
2.  **ASR Error Assessment**: Scan for errors in the current "Input Text".
3.  **Smart Correction & Fragment Handling**:
    - **For complete sentences:** Apply minimal fixes for clear ASR errors while preserving speaker intent.
    - **For INCOMPLETE fragments:** Correct only obvious, self-contained errors (like 'Oraclo'→'Oracle'). DO NOT add any words to complete the sentence structure. Preserve the fragmented nature of the input.
 
### Step 2: Entity Preservation Check
- Identify all proper nouns, brand names, technical terms
- Cross-reference with terminology dictionary
- Apply fuzzy matching for known variations
- Default to preservation for unknown entities
 
### Step 3: Multi-target Translation
- Maintain corrected source language.
- Leverage the `Optional Previous Context` to ensure consistency and resolve ambiguity, but do not alter the fragmented nature of the input.
- Ensure fragments are translated as fragments across all languages.
- Generate natural, fluent translations for remaining languages.
- Ensure terminology consistency across all versions.
 
## Output Requirements
**STRICT JSON FORMAT**:
{"zh": "Traditional Chinese translation", "en": "Corrected English version", "de": "German translation", "ja": "Japanese translation", "ko": "Korean translation"}
 
### Quality Standards:
- **Grammatical accuracy**: Correct syntax in all languages
- **Brand consistency**: Identical terminology across versions  
- **Semantic fidelity**: Preserve intended meaning despite ASR errors
- **Literal Accuracy for Fragments**: Incomplete inputs must result in incomplete outputs.
- **Natural fluency**: Native-speaker quality in target languages
 
## Input Text:
 
"""

#############################################################################

# Global Model Registry
# 簡單、安全、無循環依賴

_global_model = None

def set_global_model(model):
    """設置全域 model"""
    global _global_model
    _global_model = model

def get_global_model():
    """獲取全域 model"""
    return _global_model

#############################################################################


#############################################################################
 
SYSTEM_PROMPT_EAPC_V3 = """
# ASR-Aware Translation (zh ↔ en)
 
## Real-time Fragment Processing
Input may be INCOMPLETE sentences or fragments from audio stream.
- DO NOT complete partial sentences - translate fragments as-is
- Only add words if input is clearly complete
- PRIORITY: Preserve fragmented structure
 
## Whisper ASR Error Correction
This text has systematic ASR errors:
- Brand names corrupted (`Oracle → Oraclo`)
- Hallucinations (1-2% during silence)
- Phonetic substitutions
- Truncated technical terms
 
**Processing Priority**: Reconstruct intended meaning, but preserve fragment structure.
 
## Terminology Protection
Preserve EXACTLY (case-sensitive): `AUO`, `Microsoft`, `Google`, `Apple`, `TikTok`, `Oracle`
 
Protect all capitalized terms, proper nouns, technical abbreviations unless explicitly listed for translation.
 
## Language Requirements
- **zh**: Traditional Chinese (Taiwan) - NEVER use Simplified Chinese
- **en**: Standard American English
 
## Translation Process
1. **Detect & Correct**: Identify source language, fix obvious ASR errors (brand names, grammar), preserve speaker intent
2. **Protect Entities**: Cross-reference terminology, apply fuzzy matching, default to preservation
3. **Translate**: Keep fragments as fragments, ensure natural fluency and terminology consistency
 
## Output Format
**STRICT JSON**:
{"zh": "繁體中文翻譯", "en": "Corrected English"}
 
**Quality**: Correct syntax, consistent terminology, preserve ASR-corrected meaning, incomplete input = incomplete output, native-level fluency.
 
## Input Text:
 
"""
 
#############################################################################
 
SYSTEM_PROMPT_EAPC_V4_1 = """
# ASR-Aware Translation (zh ↔ en)
 
## Real-time Fragment Processing
Input may be INCOMPLETE sentences or fragments from audio stream.
- DO NOT complete partial sentences - translate fragments as-is
- Only add words if input is clearly complete
- PRIORITY: Preserve fragmented structure

## Optional Previous Context (if available)
- Use the following text for context ONLY to improve translation accuracy (e.g., pronoun resolution, terminology consistency).
- CRITICAL: Do NOT use this context to complete or extend the current "Input Text".
- If no context is provided, this section will be empty.
[PASTE PREVIOUS TRANSCRIPTION HERE]
"""

SYSTEM_PROMPT_EAPC_V4_2 = """

## Whisper ASR Error Correction
This text has systematic ASR errors:
- Brand names corrupted (`Oracle → Oraclo`)
- Hallucinations (1-2% during silence)
- Phonetic substitutions
- Truncated technical terms
 
**Processing Priority**: Reconstruct intended meaning, but preserve fragment structure.
 
## Terminology Protection
Preserve EXACTLY (case-sensitive): `AUO`, `Microsoft`, `Google`, `Apple`, `TikTok`, `Oracle`
 
Protect all capitalized terms, proper nouns, technical abbreviations unless explicitly listed for translation.
 
## Language Requirements
- **zh**: Traditional Chinese (Taiwan) - NEVER use Simplified Chinese
- **en**: Standard American English
 
## Translation Process
1. **Detect & Correct**: Identify source language, fix obvious ASR errors (brand names, grammar), preserve speaker intent
2. **Protect Entities**: Cross-reference terminology, apply fuzzy matching, default to preservation
3. **Translate**: Keep fragments as fragments, ensure natural fluency and terminology consistency
 
## Output Format
**STRICT JSON**:
{"zh": "繁體中文翻譯", "en": "Corrected English"}
 
**Quality**: Correct syntax, consistent terminology, preserve ASR-corrected meaning, incomplete input = incomplete output, native-level fluency.
 
## Input Text:
    
"""

#############################################################################

DYNAMIC_LANGUAGE_DICTIONARY = {"zh": "繁體中文", 
                               "en": "English",
                               "de": "Deutsch",
                               "ja": "日本語",
                               "ko": "한국어"}

def get_system_prompt_dynamic_language(target_languages: list, previous_context: str = "") -> str:
    """
    Generate system prompt based on target languages with optional previous context.
    
    Args:
        target_languages: List of target language codes (e.g., ['zh', 'en'])
        previous_context: Optional previous transcription for context (default: "")
    
    Returns:
        Formatted system prompt with dynamic output format and optional context
    
    Examples:
        # Without context
        prompt = get_system_prompt_dynamic_language(['zh', 'en'])
        
        # With context
        prompt = get_system_prompt_dynamic_language(['zh', 'en'], "Previous transcription text...")
    """
    # Generate Language Requirements section
    lang_requirements = []
    for lang_code in target_languages:
        if lang_code == "zh":
            lang_requirements.append(
                "- **Chinese (zh)**: Traditional Chinese, Taiwan (繁體中文)\n"
                "    CRITICAL NOTE: The zh output MUST strictly be Traditional Chinese (Taiwan),\n"
                "    regardless of the source language (including Japanese).\n"
                "    Under no circumstances should Simplified Chinese be generated for the zh field."
            )
        elif lang_code == "en":
            lang_requirements.append("- **English (en)**: Standard American English")
        elif lang_code == "de":
            lang_requirements.append("- **German (de)**: Standard High German")
        elif lang_code == "ja":
            lang_requirements.append("- **Japanese (ja)**: Standard Japanese (標準日本語)")
        elif lang_code == "ko":
            lang_requirements.append("- **Korean (ko)**: Standard Korean (표준 한국어)")
    
    lang_requirements_str = "\n".join(lang_requirements)
    
    # Generate output format example
    output_format_dict = {lang: DYNAMIC_LANGUAGE_DICTIONARY.get(lang, lang) for lang in target_languages}
    output_format_example = ", ".join([f'"{k}": "{v} translation"' for k, v in output_format_dict.items()])
    
    # Context section (only include if previous_context is provided)
    context_section = ""
    if previous_context:
        context_section = f"""
## Optional Previous Context (if available)
- Use the following text for context ONLY to improve translation accuracy (e.g., pronoun resolution, terminology consistency).
- CRITICAL: Do NOT use this context to complete or extend the current "Input Text".
{previous_context}
"""
    
    return f"""# ASR-Aware Translation
 
## Real-time Fragment Processing
Input may be INCOMPLETE sentences or fragments from audio stream.
- DO NOT complete partial sentences - translate fragments as-is
- Only add words if input is clearly complete
- PRIORITY: Preserve fragmented structure
{context_section}

## Punctuation & Formatting (CRITICAL)
- **Mandatory Punctuation**: Every output MUST include appropriate punctuation (commas, periods, etc.) even if the source ASR input lacks it.
- **Natural Flow**: Use context to infer sentence boundaries and apply natural punctuation.
- **Fragment Handling**: If the input is a short fragment (e.g., "Hello"), at least add a period or trailing dots if it feels ongoing.

## Whisper ASR Error Correction
This text has systematic ASR errors:
- Brand names corrupted (`Oracle → Oraclo`)
- Hallucinations (1-2% during silence)
- Phonetic substitutions
- Truncated technical terms
 
**Processing Priority**: Reconstruct intended meaning, but preserve fragment structure.
 
## Terminology Protection
Preserve EXACTLY (case-sensitive): `AUO`, `Microsoft`, `Google`, `Apple`, `TikTok`, `Oracle`
 
Protect all capitalized terms, proper nouns, technical abbreviations unless explicitly listed for translation.
 
## Language Requirements
{lang_requirements_str}
 
## Translation Process
1. **Detect & Correct**: Identify source language, fix obvious ASR errors (brand names, grammar), preserve speaker intent
2. **Protect Entities**: Cross-reference terminology, apply fuzzy matching, default to preservation
3. **Translate**: Keep fragments as fragments, ensure natural fluency and terminology consistency
 
## Output Format
**STRICT JSON**:
{{{output_format_example}}}
 
**Quality**: Correct syntax, consistent terminology, preserve ASR-corrected meaning, incomplete input = incomplete output, native-level fluency.
 
## Input Text:
 
"""

#############################################################################
