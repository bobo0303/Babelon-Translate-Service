<div align="center">

# 🌐 Babelon Translation Service

**Multi-language Audio Transcription & Translation Platform**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Integrating advanced ASR & AI translation technology to provide professional solutions for meeting transcription and real-time translation*

</div>

---

## 📝 Project Overview

**Babelon** is a FastAPI-based multi-language audio transcription and translation service. It integrates ASR (Automatic Speech Recognition) with multiple AI translation engines (GPT-4o/4.1, Claude Sonnet/Haiku, Ollama) to provide high-precision speech-to-text and multi-language translation capabilities.

### 🎯 Use Cases
- 📋 **Meeting Transcription** — Automatically generate multi-language meeting minutes
- 🎙️ **Real-time Voice Translation** — WebSocket streaming with VAD for live audio
- 📝 **Text Translation** — Translate text directly to 5 languages
- 🌍 **Multi-language Content** — One request, 5 language outputs

---

## ✨ Core Features

| Category | Features |
|----------|----------|
| **🎵 Transcription** | Whisper large-v2/v3, BreezeASR (GGML quantized), GGML C++ optimized models |
| **🌍 Translation** | GPT-4o, GPT-4.1, GPT-4.1-mini, Claude Sonnet 4.5, Claude Haiku 4.5 |
| **🔄 Pipeline** | Queue-based transcription + parallel multi-thread translation |
| **⚡ Streaming** | WebSocket real-time audio with VAD (Voice Activity Detection) |
| **🛡️ Fault Tolerance** | Auto-fallback to local Ollama on 403/failure, duplicate request cancellation |
| **🎯 Multi-strategy** | Up to 4 transcription sampling strategies for optimal accuracy |
| **🔍 Auto-detect** | Language auto-detection via Whisper or Azure Speech |
| **📊 Monitoring** | Health check (active + passive), RTF metrics, log viewer API |

---

## 🌐 Supported Languages

| Code | Language | Azure Locale |
|------|----------|--------------|
| `zh` | 🇹🇼 Traditional Chinese | zh-TW |
| `en` | 🇺🇸 English | en-US |
| `ja` | 🇯🇵 Japanese | ja-JP |
| `ko` | 🇰🇷 Korean | ko-KR |
| `de` | 🇩🇪 German | de-DE |
| `auto` | 🔍 Auto-detect | — |

---

## 🏗 System Architecture

```
🌐 Babelon Translation Service
│
├── 🚀 main.py                          # FastAPI application & all endpoints
│
├── 📂 api/
│   ├── 📂 core/
│   │   ├── 🧠 transcribe_manager.py    # ASR model management & inference
│   │   ├── 🔤 translate_manager.py     # Translation orchestration & fallback
│   │   ├── 🔄 threading_api.py         # Pipeline coordinator & threading
│   │   └── 🛠️ utils.py                 # Post-processing, formatting, response tracking
│   │
│   ├── 📂 translation/
│   │   ├── 💬 gpt_translate.py          # Azure OpenAI GPT translation
│   │   ├── 🤖 claude_translate.py       # Azure Claude translation
│   │   └── 🦙 ollama_translate.py       # Local Ollama fallback translation
│   │
│   ├── 📂 transcraption/
│   │   └── ⚙️ post_process.py           # ASR post-processing & error correction
│   │
│   ├── 📂 azure_sdk/
│   │   └── 🎤 speech_lid.py             # Azure Speech language detection
│   │
│   └── 📂 websocket/
│       └── 🔌 vad_translate_stream.py   # WebSocket real-time VAD streaming
│
├── 📚 lib/
│   ├── 📂 config/
│   │   └── ⚙️ constant.py               # Languages, models, prompts, settings
│   └── 📂 core/
│       ├── 📋 logging_config.py          # Logging configuration
│       ├── 💓 health_check.py            # Health check service
│       └── 🏗️ base_object.py             # BaseResponse, Status enums
│
├── 🐳 Dockerfile / docker-compose.yml   # Container deployment
├── 📦 requirements.txt                  # Python dependencies
└── 📋 logs/app.log                      # Application logs
```

---

## 📋 API Endpoints

### Health & Status

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Basic health check |
| `GET` | `/health_check` | Detailed status (port, uptime, processing state) |
| `GET` | `/list_optional_items` | List available transcription & translation models |
| `GET` | `/get_current_model` | Currently loaded transcription & translation model |
| `GET` | `/get_prompt` | Current transcription prompt |
| `GET` | `/get_pretext_status` | Previous text context enabled/disabled |
| `GET` | `/get_logs?lines=100` | Latest N lines from app.log |

### Configuration

| Method | Endpoint | Parameters | Description |
|--------|----------|------------|-------------|
| `POST` | `/change_transcription_model` | `model_name` | Switch ASR model |
| `POST` | `/change_translator` | `model_name` | Switch translation engine |
| `POST` | `/set_prompt` | `prompts` | Set/clear custom transcription prompt |
| `POST` | `/change_pretext_usage` | `enable` (bool) | Toggle previous text context |

### Language Detection

| Method | Endpoint | Parameters | Description |
|--------|----------|------------|-------------|
| `POST` | `/language_detect` | `file`, `method` ("whisper"\|"azure_speech") | Detect audio language |

### Translation

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/translate_pipeline` | ✅ **Recommended** — Queue-based pipeline with parallel translation |
| `POST` | `/translate` | Legacy — Thread-based transcription + translation |
| `POST` | `/text_translate` | Text-only translation (no audio) |
| `WebSocket` | `/S2TT/vad_translate_stream` | Real-time audio streaming with VAD |

---

### `POST /translate_pipeline` — Main Endpoint

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | UploadFile | *required* | Audio file (.wav, .mp3, .m4a) |
| `meeting_id` | str | `123` | Meeting identifier |
| `device_id` | str | `123` | Device identifier |
| `audio_uid` | str | `123` | Audio unique ID |
| `times` | datetime | *required* | Timestamp |
| `o_lang` | str | `zh` | Source language (zh\|en\|ja\|ko\|de\|auto) |
| `t_lang` | str | `zh,en,ja,ko,de` | Target languages (comma-separated) |
| `prev_text` | str | `""` | Previous context for better accuracy |
| `multi_strategy_transcription` | int | `4` | Strategies count (1-4) |
| `transcription_post_processing` | bool | `True` | Enable ASR post-processing |
| `multi_translate` | bool | `True` | Parallel multi-thread translation |

**Response**:
```json
{
  "status": "OK",
  "message": " | Transcription: ... | ZH: ... | EN: ... | ",
  "data": {
    "meeting_id": "123",
    "device_id": "456",
    "ori_lang": "zh",
    "detected_lang": "zh",
    "transcription_text": "原始辨識文字",
    "n_segments": 3,
    "segments": [{"id": 0, "start": 0.0, "end": 2.5, "text": "..."}],
    "text": {
      "zh": "中文翻譯",
      "en": "English translation",
      "ja": "日本語翻訳",
      "ko": "한국어 번역",
      "de": "Deutsche Übersetzung"
    },
    "transcribe_time": 1.25,
    "translate_time": 1.91,
    "stable_text": "",
    "unstable_text": "",
    "trim_duration": 0.0,
    "trim_updated": false
  }
}
```

### `POST /text_translate` — Text Only

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | str | *required* | Text to translate |
| `source_language` | str | `zh` | Source language |
| `target_language` | str | `zh,en,ja,ko,de` | Target languages (comma-separated) |
| `multi_translate` | bool | `True` | Parallel translation |

---

## 🎙️ Transcription Models

| Model | Type | Default | Description |
|-------|------|---------|-------------|
| **`ggml_breeze_asr_25`** | GGML | ✅ | Chinese-optimized, quantized, fast |
| `ggml_large_v3` | GGML | | Memory-efficient Whisper large-v3 |
| `large_v2` | PyTorch | | OpenAI Whisper large-v2 |
| `large_v3` | PyTorch | | OpenAI Whisper large-v3 |
| `breeze_asr_25` | PyTorch | | MediaTek BreezeASR (Chinese-only) |
| `ggml_cpp_*` | C++ GGML | | Native optimized (requires C++ libs) |

Default model can be set via `DEFAULT_MODEL` environment variable.

## 🤖 Translation Models

| Model | Provider | Notes |
|-------|----------|-------|
| **`gpt-4o`** | Azure OpenAI | ✅ Default |
| `gpt-4.1` | Azure OpenAI | Highest quality |
| `gpt-4.1-mini` | Azure OpenAI | Best speed/quality balance |
| `claude-sonnet-4-5` | Azure Anthropic | High quality |
| `claude-haiku-4-5` | Azure Anthropic | Fast |

**Fallback**: All models auto-fallback to local `ollama-gemma` on 403 or failure.

**Translation modes**:
- `multi_translate: True` → Each target language gets its own translator thread (parallel)
- `multi_translate: False` → Single LLM translates all languages in one request

---

## 🎯 Multi-strategy Transcription

| Strategy | Prompt | Context | Temperature | Use Case |
|----------|--------|---------|-------------|----------|
| **1** | Custom prompt | + Previous text | Standard | Continuous dialogue |
| **2** | Custom prompt | — | Standard | Professional terminology |
| **3** | None | — | Low | General speech, high accuracy |
| **4** | None | — | High | Unclear audio, heavy accents |

Set `multi_strategy_transcription=4` (default) to use all strategies and select the best result.

---

## 🚀 Quick Start

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.9+ | 3.11+ |
| **GPU** | GTX 1660 (6GB) | RTX 3080+ (10GB+) |
| **RAM** | 8GB | 16GB+ |
| **Docker** | 20.10+ | Latest |

### Docker Deployment (Recommended)

```bash
# Clone
git clone https://github.com/bobo0303/Babelon-Translate-Service.git
cd Babelon-Translate-Service

# Build and start
docker-compose up -d

# Check logs
docker logs -f babelon
```

### Local Installation

```bash
pip install -r requirements.txt
python main.py
```

Service starts on port `80` (configurable via `PORT` env var).

---

## ⚙️ Environment Variables

### Required — Translation APIs

```bash
# GPT-4o (Azure OpenAI)
GPT_4O_API_KEY=your_key
GPT_4O_API_VERSION=2024-02-15-preview
GPT_4O_ENDPOINT=https://your-endpoint.openai.azure.com
GPT_4O_DEPLOYMENT=your-deployment

# GPT-4.1
GPT_41_API_KEY=your_key
GPT_41_API_VERSION=2025-01-01-preview
GPT_41_ENDPOINT=https://your-endpoint.openai.azure.com/
GPT_41_DEPLOYMENT=gpt-4.1

# GPT-4.1-mini
GPT_41_MINI_API_KEY=your_key
GPT_41_MINI_API_VERSION=2025-01-01-preview
GPT_41_MINI_ENDPOINT=https://your-endpoint.openai.azure.com/
GPT_41_MINI_DEPLOYMENT=gpt-4.1-mini

# Claude Sonnet 4.5 (Azure)
CLAUDE_SONNET_4_5_API_KEY=your_key
CLAUDE_SONNET_4_5_ENDPOINT=https://your-endpoint.openai.azure.com
CLAUDE_SONNET_4_5_DEPLOYMENT=claude-sonnet-4-5

# Claude Haiku 4.5 (Azure)
CLAUDE_HAIKU_4_5_API_KEY=your_key
CLAUDE_HAIKU_4_5_ENDPOINT=https://your-endpoint.openai.azure.com
CLAUDE_HAIKU_4_5_DEPLOYMENT=claude-haiku-4-5

# Azure Speech (for language detection)
AZURE_SPEECH_SUBSCRIPTION_KEY=your_key
AZURE_SPEECH_SERVICE_REGION=your_region
```

### Optional

```bash
# Ollama fallback
OLLAMA_HOST=http://172.17.0.1:52013/
OLLAMA_GEMMA_MODEL=gemma3:12b-it-qat

# Service
PORT=80
DEFAULT_MODEL=ggml_breeze_asr_25
BACKEND_DOMAIN=http://your-backend
HEALTH_CHECK_CYCLE_SEC=30
```

---

## 🎯 Usage Examples

### Python

```python
import requests
import datetime

# Audio transcription + translation
with open("audio.wav", "rb") as f:
    resp = requests.post("http://localhost:80/translate_pipeline", 
        files={"file": ("audio.wav", f, "audio/wav")},
        data={
            "meeting_id": "mtg_001",
            "device_id": "mic_01",
            "audio_uid": "uid_001",
            "times": datetime.datetime.now().isoformat(),
            "o_lang": "zh",
            "t_lang": "zh,en,ja,ko,de",
            "multi_strategy_transcription": 4,
        })
print(resp.json())

# Text-only translation
resp = requests.post("http://localhost:80/text_translate",
    data={
        "text": "今天天氣很好",
        "source_language": "zh",
        "target_language": "en,ja,ko,de",
    })
print(resp.json())

# Switch translation model
requests.post("http://localhost:80/change_translator",
    data={"model_name": "claude-haiku-4-5"})

# View logs
resp = requests.get("http://localhost:80/get_logs?lines=50")
print(resp.json())
```

### curl

```bash
# Audio translation
curl -X POST "http://localhost:80/translate_pipeline" \
  -F "file=@audio.wav" \
  -F "meeting_id=mtg_001" \
  -F "audio_uid=uid_001" \
  -F "times=2026-04-27T10:00:00" \
  -F "o_lang=zh" \
  -F "t_lang=zh,en,ja,ko,de"

# Text translation
curl -X POST "http://localhost:80/text_translate" \
  -F "text=Hello World" \
  -F "source_language=en" \
  -F "target_language=zh,ja,ko,de"

# Switch model
curl -X POST "http://localhost:80/change_translator" \
  -F "model_name=gpt-4.1-mini"

# Check status
curl http://localhost:80/get_current_model
```

---

## 🐛 Troubleshooting

<details>
<summary><strong>GPU Memory Insufficient</strong></summary>

```bash
nvidia-smi  # Check GPU usage

# Switch to smaller/quantized model
curl -X POST "http://localhost:80/change_transcription_model" \
  -F "model_name=ggml_breeze_asr_25"
```

</details>

<details>
<summary><strong>Translation API 403 Error</strong></summary>

The system auto-fallbacks to local Ollama. Check logs:
```bash
curl "http://localhost:80/get_logs?lines=50"
```
Verify API keys in environment variables are correct.

</details>

<details>
<summary><strong>Translation Timeout</strong></summary>

Default timeout is 10 seconds (`WAITING_TIME`). For longer audio, the pipeline will force-terminate and return partial results. Adjust in `lib/config/constant.py`.

</details>

---

## 📊 Performance Reference

| Metric | Target |
|--------|--------|
| **RTF** (Real-Time Factor) | < 0.3 excellent, 0.3-0.5 good |
| **Translation (GPT-4o)** | ~1.9s avg per request |
| **Translation (Claude Haiku)** | ~3.8s avg per request |
| **Translation (Claude Sonnet)** | ~5.6s avg per request |

---

## 📄 License

[MIT License](LICENSE)

---

<div align="center">

[![Made with ❤️](https://img.shields.io/badge/Made%20with-❤️-red.svg)](https://github.com/bobo0303/Babelon-Translate-Service)
[![Powered by FastAPI](https://img.shields.io/badge/Powered%20by-FastAPI-009688.svg)](https://fastapi.tiangolo.com)

</div>