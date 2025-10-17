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
WAITING_TIME = 60 # The whisper inference max waiting time (if over the time will stop it)
MAX_NUM_STRATEGIES = 4  # The maximum number of strategies for sampling during transcription

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
    "DEFAULT": "拉貨力道, 出貨力道, 放量, 換機潮, pull in, 曝險, BOM, deal, 急單, foreX, NT dollars, Monitor, china car, DSBG, low temp, Tier 2, Tier 3, Notebook, RD, TV, 8B, In-Cell Touch, Vertical, 主管, Firmware, AecoPost, DaaS, OLED, AmLED, Polarizer, Tartan Display, 達擎, ADP team, Legamaster, AVOCOR, FindARTs, RISEvision, JECTOR, SatisCtrl, Karl Storz, Schwarz, NATISIX",
    "JAMES": "GRC, DSBG, ADP, OLED, SRBG, RBU, In-cel one chip, monitor, Sports Gaming, High Frame Rate Full HD 320Hz, Kiosk, Frank, Vertical, ARHUD, 手扶屏, 空調屏, 後視鏡的屏, 達擎, 產能, 忠達.",
    "SCOTT": "JECTOR, AVOCOR, LegoMaster, RISEvision, Hualien, SatisCtrl, motherson, Kark, Storz, ADP, Aecopost, NATISIX, NanoLumens, FindARTs, AUO, ADP, AHA, E&E, Schwarz, PeosiCo."
}

CONTAINS_UNUSUAL = ["字幕志願者", "字幕組", "字幕翻譯"]
ONLY_UNUSUAL = ["謝謝", "謝謝觀看", "感謝聆聽", "感謝收看", "感謝觀看", "以下視頻的資訊和消息都可以在微博或推特上發送", "字幕由 Amara.org 社區提供", "多謝您收看時局新聞，再會！", "大家好，我是 Karen，我們下期再見吧！", "下集再見", "各位車友們，謝謝收看，我是劉胖胖。",]

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

