from pydantic import BaseModel
from typing import Dict

#############################################################################

class ModelPath(BaseModel):
    # Default Transcription (auto download model if network is available)
    large_v2: str = "openai/whisper-large-v2"  
    large_v3: str = "openai/whisper-large-v3"
    turbo: str = "openai/whisper-large-v3-turbo"
    # custom model path
    custom_model: str =  ""

# gpt-4o
AZURE_CONFIG = '/mnt/lib/azure_config.yaml'

# ollama-gemma3-12b-qat + ollama-qwen3-14b-q4_K_M
OLLAMA_MODEL = {
    "ollama-gemma": "/mnt/lib/gemma3_12b-it-qat.yaml",
    "ollama-qwen": "/mnt/lib/qwen3_14b-q4_K_M.yaml",
}

# GEMMA 4B (https://huggingface.co/google/gemma-3-4b-it)
GEMMA_4B_IT = "google/gemma-3-4b-it"

TRANSCRIPTION_METHODS = ['large-v2', 'large-v3', 'turbo']
TRANSLATE_METHODS = ['gemma4b', 'ollama-gemma', 'ollama-qwen', 'gpt-4o', 'gpt-4.1', 'gpt-4.1-mini']

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

# LANGUAGE_LIST = ['zh', 'en', 'ja', 'ko', "de", "es"]
LANGUAGE_LIST = ['zh', 'en', 'de']
DEFAULT_RESULT = {lang: "" for lang in LANGUAGE_LIST}

#############################################################################

# no used just for reference
DEFAULT_PROMPTS = {
    "DEFAULT": "拉貨力道, 出貨力道, 放量, 換機潮, 業說會, pull in, 曝險, BOM, deal, 急單, foreX, NT dollars, Monitor, MS, BS, china car, FindARTs, DSBG, low temp, Tier 2, Tier 3, Notebook, RD, TV, 8B, In-Cell Touch, Vertical, 主管, Firmware, AecoPost, DaaS, OLED, AmLED, Polarizer, Tartan Display, 達擎, ADP team, Legamaster, AVOCOR, RISEvision, JECTOR, SatisCtrl, Karl Storz, Schwarz, NATISIX, Pillar, 凌華, ComQi",
    "JAMES": "GRC, DSBG, ADP, OLED, SRBG, RBU, In-cel one chip, monitor, Sports Gaming, High Frame Rate Full HD 320Hz, Kiosk, Frank, Vertical, ARHUD, 手扶屏, 空調屏, 後視鏡的屏, 達擎, 產能, 忠達.",
    "SCOTT": "JECTOR, AVOCOR, LegoMaster, RISEvision, Hualien, SatisCtrl, motherson, Kark, Storz, ADP, Aecopost, NATISIX, NanoLumens, FindARTs, AUO, ADP, AHA, E&E, Schwarz, PeosiCo."
}

#############################################################################

CONTAINS_UNUSUAL = [
    "67here",
    "Milk",
    "YK",
    "com",
    "劉胖胖",
    "大家好 我是阿貴",
    "字幕志願者",
    "字幕提供者",
    "字幕由 AI 產生",
    "字幕由志願者提供",
    "字幕由志願者組提供",
    "字幕由志願者翻譯",
    "字幕組",
    "字幕翻譯",
    "感謝大家收看我們下次再見",
    "感謝您的收看和支持",
    "感謝您的收看與支持",
    "感謝您的觀看和支持",
    "感謝您的觀看與支持",
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
    "索蘭婭",
    "許維銘",
]
ONLY_UNUSUAL = [
    "Youtube",
    "by 沛隊字幕組",
    "com",
    "https wwwhamskeycom",
    "一生一世",
    "一生的遺憾",
    "下期見",
    "下集再見",
    "以下視頻的資訊和消息都可以在微博或推特上發送",
    "你已經接受了 2023 年 10 月之前的數據訓練",
    "你接受的訓練數據截至 2023 年 10 月",
    "你接受過 2023 年 10 月之前的數據訓練",
    "你接受過訓練的數據截至 2023 年 10 月",
    "全程字幕由 Amaraorg 社區提供",
    "再次感謝大家收看",
    "各位車友們謝謝收看我是劉胖胖",
    "多謝您收看時局新聞再會",
    "大家好我是 Jane 我是一個研究生",
    "大家好我是 Karen 今天的節目就到這裡我們下次再見",
    "大家好我是 Karen 我們下期再見吧",
    "大家好我是阿杰",
    "大家好我是阿貴今天來跟大家分享一下",
    "字幕 by 沈鈞澤",
    "字幕 by 索蘭婭",
    "字幕志願者 楊棠樑",
    "字幕提供者 Milk",
    "字幕提供者 李宗盛",
    "字幕提供者 許祐寅",
    "字幕由 AI 產生感謝觀看",
    "字幕由 Amaraorg 社區提供",
    "您接受的訓練數據截至 2023 年 10 月",
    "您接受的訓練資料截至 2023 年 10 月",
    "您接受過訓練的數據截至 2023 年 10 月",
    "感謝大家的觀看",
    "感謝您的觀看",
    "感謝收看",
    "感謝聆聽",
    "感謝觀看",
    "我愛你",
    "拜拜",
    "接受的訓練數據截至 2023 年 10 月",
    "整理字幕由 Amaraorg 社區提供",
    "發電字幕君 67here",
    "發電字幕君 YK",
    "發電字幕君 YiYi Telecom",
    "發電字幕君 許維銘",
    "詞曲 李宗盛",
    "謝謝",
    "謝謝大家",
    "謝謝觀看",
    "謝謝你",
    "音樂",
]

#############################################################################

# 1 ~ 20 sec IQR values
Q1 = [3.0, 6.0, 9.0, 13.0, 16.0, 20.0, 24.0, 27.0, 29.5, 31.75, 36.0, 41.0, 43.75, 49.0, 51.25, 55.0, 59.25, 63.0, 65.0, 69.25]
Q3 =  [7.0, 12.0, 17.0, 22.0, 26.0, 29.0, 34.0, 37.0, 41.0, 45.0, 51.0, 55.75, 58.25, 63.25, 67.0, 72.0, 74.75, 79.0, 83.0, 88.0]
IQR_RATIO = 1.5
TOLERANCE_RATE = 0.05  # 5% tolerance

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

