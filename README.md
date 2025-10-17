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

**Babelon** is a FastAPI-based multi-language audio transcription and translation service platform. It integrates the latest ASR (Automatic Speech Recognition) and AI translation technologies to provide high-precision speech-to-text and multi-language translation capabilities.

### 🎯 Use Cases
- 📋 **Meeting Transcription** - Automatically generate multi-language meeting minutes
- 🎙️ **Real-time Voice Translation** - Instant multi-language support for online meetings
- 📝 **Voice Memos** - Quickly convert voice to editable text
- 🌍 **Multi-language Content Creation** - One-click generation of multi-language versions

## ✨ Core Features

<table>
<tr>
<td width="50%">

### 🎯 **Core Functionality**
- 🎵 **High-precision Audio Transcription** - Support for Whisper large-v2/v3/turbo models
- 🌍 **Multi-language Translation** - Chinese (Traditional), English, German translation
- ⚡ **Real-time Streaming Processing** - SSE technology for instant translation feedback
- 🔄 **Multi-strategy Transcription** - Up to 4 strategies ensuring optimal accuracy
- 🛠️ **Intelligent Post-processing** - Automatic correction of common ASR errors

</td>
<td width="50%">

### 🛠 **Technical Highlights**
- 🚀 **Multi AI Model Integration** - Whisper, Gemma, Ollama, GPT-4o/4.1
- ⚡ **GPU Acceleration** - CUDA support for dramatically improved processing speed
- 🐳 **Containerized Deployment** - One-click deployment with Docker/Docker Compose
- 🔧 **Intelligent Resource Management** - Auto cleanup, memory optimization
- 🛡️ **Multi-strategy Fault Tolerance** - Ensuring service stability

</td>
</tr>
</table>

## 🏗 System Architecture

```
🌐 Babelon Translation Service
│
├── 🚀 main.py                    # FastAPI main application
│
├── 📂 api/                       # API core modules
│   ├── 🧠 model.py               # Model management center
│   ├── 🔤 gemma_translate.py     # Gemma translation engine
│   ├── 💬 gpt_translate.py       # GPT-4o/4.1 translation engine
│   ├── 🦙 ollama_translate.py    # Ollama translation engine
│   ├── ⚙️ post_process.py        # Intelligent post-processing module
│   ├── 🔄 threading_api.py       # Multi-threading API
│   └── 🛠️ utils.py               # Utility functions
│
├── 📚 lib/                       # Shared libraries
│   ├── 🏗️ base_object.py         # Base object definitions
│   ├── ⚙️ constant.py            # System constants
│   └── ☁️ azure_config.yaml      # Azure API configuration
│
├── 🛠️ tools/                     # Tool programs
│   └── 🔊 audio_splitter.py      # Audio splitting tool
│
├── 🎵 audio/                     # Audio temporary storage
└── 📋 logs/                      # System logs
```

## 🚀 Quick Start

### 📋 System Requirements

<table>
<tr>
<td><strong>🐍 Python</strong></td>
<td>3.9 or higher</td>
</tr>
<tr>
<td><strong>🔥 GPU</strong></td>
<td>NVIDIA CUDA support (4GB+ VRAM recommended)</td>
</tr>
<tr>
<td><strong>🐳 Container</strong></td>
<td>Docker & Docker Compose</td>
</tr>
<tr>
<td><strong>💾 Memory</strong></td>
<td>16GB+ RAM (recommended)</td>
</tr>
</table>

### 🛠️ Installation & Deployment

<details>
<summary><strong>🐳 Method 1: Docker Deployment (Recommended)</strong></summary>

```bash
# 📥 Clone the project
git clone https://github.com/bobo0303/Babelon-Translate-Service.git
cd Babelon-Translate-Service

# 🏗️ Build and start the service
docker build -t babelon .
# Or use Docker Compose
docker-compose up -d

# 🔧 Enter the container
docker exec -it babelon bash

# 🚀 Start the service
python main.py
```

</details>

<details>
<summary><strong>💻 Method 2: Local Installation</strong></summary>

```bash
# 📦 Install dependencies
pip install -r requirements.txt

# ⚙️ Set environment variables
export HUGGINGFACE_HUB_TOKEN="your-hf-token"

# 🚀 Start the service
python main.py
```

</details>

### ⚙️ Initial Setup

<table>
<tr>
<td>⚠️</td>
<td><strong>Important Note</strong>: <code>azure_config.yaml</code> needs to be prepared separately and is not included in the repository</td>
</tr>
</table>

<details>
<summary><strong>🔧 Step 1: Azure OpenAI Configuration</strong></summary>

Create an `azure_config.yaml` file and fill in the following content:

```yaml
# GPT version configuration - unified variable naming
gpt_models:
  # GPT-4o
  "gpt-4o":
    API_KEY: "your_azure_api_key"  
    API_VERSION: "2024-02-15-preview"  
    ENDPOINT: "https://your-endpoint.openai.azure.com"  
    DEPLOYMENT: "your-deployment-name"
  
  # GPT-4.1
  "gpt-4.1":
    API_KEY: "your_azure_api_key"  
    API_VERSION: "2025-01-01-preview"
    ENDPOINT: "https://your-endpoint.openai.azure.com/"
    DEPLOYMENT: "gpt-4.1"
  
  # GPT-4.1 Mini
  "gpt-4.1-mini":
    API_KEY: "your_azure_api_key"
    API_VERSION: "2025-01-01-preview"
    ENDPOINT: "https://your-endpoint.openai.azure.com/"
    DEPLOYMENT: "gpt-4.1-mini"
```

</details>

<details>
<summary><strong>🤗 Step 2: Hugging Face Login (Gemma Model)</strong></summary>

```bash
# First need to agree to terms of use on Hugging Face
# Visit: https://huggingface.co/google/gemma-3-4b-it

# Login using Token
huggingface-cli login --token your_hf_token
```

</details>

<details>
<summary><strong>🦙 Step 3: Ollama Setup (Optional)</strong></summary>

```bash
# 🐳 Build Ollama Docker container
docker run -d -it --gpus all --shm-size 32G --runtime nvidia \
  --device=/dev/nvidia-uvm --device=/dev/nvidia-uvm-tools \
  --device=/dev/nvidiactl --device=/dev/nvidia0 \
  -v ./ollama:/root/.ollama -p 52013:11434 \
  --name ollama ollama/ollama

# 🧪 Test Ollama model
docker exec -it ollama ollama run gemma3:12b-it-qat --verbose
```

</details>

## 📋 API Documentation

### Basic Information
- **Service Address**: `http://localhost:80`
- **API Documentation**: `http://localhost:80/docs`
- **Health Check**: `GET /`

### Main API Endpoints

#### 🎵 Audio Transcription & Translation
**Endpoint**: `POST /translate`

Perform speech recognition on audio files and translate into multiple languages, supporting meeting transcription, voice memos, and other applications.

```http
POST /translate
Content-Type: multipart/form-data

file: audio file (.wav, .mp3, .m4a)
meeting_id: string
device_id: string  
audio_uid: string
times: datetime
o_lang: string (zh|en|de)
prev_text: string (optional, previous context)
multi_strategy_transcription: int (1-4, default 1)
transcription_post_processing: bool (default true)
use_translate: bool (default true)
```

**Response Format**:
```json
{
  "status": "OK",
  "message": "Translation result summary",
  "data": {
    "meeting_id": "123",
    "device_id": "456", 
    "ori_lang": "zh",
    "transcription_text": "Original transcription",
    "text": {
      "zh": "Chinese translation",
      "en": "English translation",
      "de": "German translation"
    },
    "times": "2024-01-01T10:00:00",
    "audio_uid": "789",
    "transcribe_time": 2.5,
    "translate_time": 1.2
  }
}
```

#### 📝 Text Translation
**Endpoint**: `POST /text_translate`

Directly translate existing text content to quickly obtain multi-language versions.

```http
POST /text_translate
Content-Type: application/x-www-form-urlencoded

text: Text to be translated
language: Source language (zh|en|de)
```

#### ⚡ Real-time Streaming Translation (SSE)
**Endpoint**: `POST/GET /sse_audio_translate`

Support real-time audio processing, suitable for online meetings, live broadcasts, and other scenarios requiring instant feedback.

```http
# Submit audio to processing queue
POST /sse_audio_translate

# Establish Server-Sent Events connection to receive results
GET /sse_audio_translate
Accept: text/event-stream

# Stop streaming connection
POST /stop_sse
```

#### ⚙️ System Management
**Endpoint**: Multiple management endpoints

Provide model switching, parameter adjustment, system status query, and other management functions.

```http
GET /get_current_model          # View currently used model
GET /list_optional_items        # List all available models and translation options
POST /change_transcription_model # Switch speech recognition model
POST /change_translation_method  # Switch translation engine
POST /set_prompt                # Set custom prompt
```

## ⚙️ System Configuration

### 🌍 Supported Languages

<table>
<tr>
<th>Language Code</th>
<th>Language Name</th>
<th>Description</th>
</tr>
<tr>
<td><code>zh</code></td>
<td>🇹🇼 Traditional Chinese</td>
<td>Taiwan standard Chinese</td>
</tr>
<tr>
<td><code>en</code></td>
<td>🇺🇸 English</td>
<td>American English</td>
</tr>
<tr>
<td><code>de</code></td>
<td>🇩🇪 German</td>
<td>Standard German</td>
</tr>
</table>

### 🎙️ Transcription Model Options

<table>
<tr>
<th>Model Name</th>
<th>Description</th>
<th>Recommended Use</th>
</tr>
<tr>
<td><code>large_v2</code></td>
<td>OpenAI Whisper Large v2 (default)</td>
<td>✅ Good stability</td>
</tr>
<tr>
<td><code>large_v3</code></td>
<td>OpenAI Whisper Large v3</td>
<td>🎯 Higher accuracy</td>
</tr>
<tr>
<td><code>turbo</code></td>
<td>OpenAI Whisper Large v3 Turbo</td>
<td>⚡ Speed priority</td>
</tr>
<tr>
<td><code>TCM</code></td>
<td>Custom model path</td>
<td>🔧 Customization needs</td>
</tr>
</table>

### 🤖 Translation Engine Options

<table>
<tr>
<th>Engine Name</th>
<th>Description</th>
<th>Deployment</th>
<th>Features</th>
</tr>
<tr>
<td><code>gpt-4.1-mini</code></td>
<td>GPT-4.1 Mini (default)</td>
<td>☁️ Azure OpenAI API</td>
<td>🏆 Best quality & speed</td>
</tr>
<tr>
<td><code>gpt-4.1</code></td>
<td>GPT-4.1</td>
<td>☁️ Azure OpenAI API</td>
<td>🏆 Highest quality</td>
</tr>
<tr>
<td><code>gpt-4o</code></td>
<td>GPT-4o</td>
<td>☁️ Azure OpenAI API</td>
<td>🏆 Excellent quality</td>
</tr>
<tr>
<td><code>gemma4b</code></td>
<td>Google Gemma 4B</td>
<td>💻 Local execution</td>
<td>🔒 Privacy protection</td>
</tr>
<tr>
<td><code>ollama-gemma</code></td>
<td>Ollama Gemma</td>
<td>🐳 Container deployment</td>
<td>⚡ Quick deployment</td>
</tr>
<tr>
<td><code>ollama-qwen</code></td>
<td>Ollama Qwen</td>
<td>🐳 Container deployment</td>
<td>🌏 Chinese optimized</td>
</tr>
</table>

### 🔧 Environment Variable Configuration

```bash
# 🤗 Hugging Face Token (for Gemma model usage)
# Please visit and agree to terms first: https://huggingface.co/google/gemma-3-4b-it
export HUGGINGFACE_HUB_TOKEN="your_hf_token"

# 🔥 GPU Configuration
export NVIDIA_VISIBLE_DEVICES=all
export CUDA_VISIBLE_DEVICES=0
```

## 🔧 Advanced Features

### 🎯 Multi-strategy Transcription

<table>
<tr>
<th>Strategy</th>
<th>Description</th>
<th>Use Case</th>
</tr>
<tr>
<td>🔹 <strong>Strategy 1</strong></td>
<td>Custom prompt + previous context</td>
<td>Continuous dialogue, meeting transcription</td>
</tr>
<tr>
<td>🔸 <strong>Strategy 2</strong></td>
<td>Custom prompt only</td>
<td>Professional terminology, specific domains</td>
</tr>
<tr>
<td>🔹 <strong>Strategy 3</strong></td>
<td>No prompt, low temperature sampling</td>
<td>General speech, high accuracy requirements</td>
</tr>
<tr>
<td>🔸 <strong>Strategy 4</strong></td>
<td>High temperature diversified sampling</td>
<td>Unclear audio, heavy accents</td>
</tr>
</table>

### 🛠️ Intelligent Post-processing

<table>
<tr>
<td width="50%">

#### 🔍 **Error Detection**
- 🔤 Automatic identification of common ASR errors
- 🏢 Brand name standardization
- 📝 Intelligent grammar correction
- ⚠️ Hallucination content detection and filtering

</td>
<td width="50%">

#### ⚡ **Performance Optimization**
- 🧹 24-hour automatic audio file cleanup
- 💾 Intelligent GPU memory recycling
- 🔄 Zero-downtime model hot switching
- 📊 Real-time performance monitoring (RTF)

</td>
</tr>
</table>

## 🎯 Usage Examples

### Python Client Example
```python
import requests

# Audio translation
files = {'file': open('audio.wav', 'rb')}
data = {
    'meeting_id': '001',
    'device_id': 'mic_01', 
    'audio_uid': 'audio_001',
    'times': '2024-01-01T10:00:00',
    'o_lang': 'zh',
    'multi_strategy_transcription': 2,
    'use_translate': True
}

response = requests.post(
    'http://localhost:80/translate',
    files=files,
    data=data
)
print(response.json())

# Text translation
data = {'text': 'Hello World', 'language': 'en'}
response = requests.post(
    'http://localhost:80/text_translate',
    data=data
)
print(response.json())
```

### curl Example
```bash
# Audio translation
curl -X POST "http://localhost:80/translate" \
  -F "file=@audio.wav" \
  -F "meeting_id=001" \
  -F "device_id=mic_01" \
  -F "audio_uid=audio_001" \
  -F "times=2024-01-01T10:00:00" \
  -F "o_lang=zh" \
  -F "use_translate=true"

# Text translation  
curl -X POST "http://localhost:80/text_translate" \
  -F "text=Hello World" \
  -F "language=en"
```

## 📊 Performance Optimization

### 💻 Recommended Hardware Configuration

<table>
<tr>
<th>Component</th>
<th>Minimum Requirement</th>
<th>Recommended Configuration</th>
<th>Optimal Performance</th>
</tr>
<tr>
<td><strong>🔥 GPU</strong></td>
<td>GTX 1660 (6GB)</td>
<td>RTX 3080 (10GB)</td>
<td>RTX 4090 (24GB)</td>
</tr>
<tr>
<td><strong>💾 Memory</strong></td>
<td>8GB RAM</td>
<td>16GB RAM</td>
<td>32GB+ RAM</td>
</tr>
<tr>
<td><strong>💿 Storage</strong></td>
<td>HDD</td>
<td>SATA SSD</td>
<td>NVMe SSD</td>
</tr>
<tr>
<td><strong>🌐 Network</strong></td>
<td>10 Mbps</td>
<td>100 Mbps</td>
<td>1 Gbps</td>
</tr>
</table>

### ⚙️ Performance Tuning Parameters

<details>
<summary><strong>🔧 Adjust the following parameters in constant.py</strong></summary>

```python
# ⏱️ Transcription timeout setting
WAITING_TIME = 60           # Unit: seconds, recommend 30-120

# 🎯 Strategy count setting
MAX_NUM_STRATEGIES = 4      # Maximum 4, can reduce to 1-2 for speed

# 🔇 Silence padding
SILENCE_PADDING = True      # Improve boundary word recognition, slight processing time increase

# 📈 Real-time factor calculation
RTF = True                  # Enable performance monitoring, slight performance impact
```

</details>

### 📈 Performance Monitoring Metrics

<table>
<tr>
<td><strong>🎯 RTF (Real-Time Factor)</strong></td>
<td>< 0.3 Excellent | 0.3-0.5 Good | > 0.5 Needs optimization</td>
</tr>
<tr>
<td><strong>💾 GPU Memory Usage</strong></td>
<td>< 80% Safe | 80-90% Caution | > 90% Dangerous</td>
</tr>
<tr>
<td><strong>⚡ Processing Speed</strong></td>
<td>1-minute audio < 20 seconds processing is optimal</td>
</tr>
</table>

## 🛡️ Security Considerations

### API Security
- Recommended to use reverse proxy (nginx)
- Configure API rate limiting
- Enable HTTPS for production environment

### Data Privacy
- Automatic audio file cleanup
- Sensitive information log masking
- Support for local deployment, data stays on-premises

## 🐛 Troubleshooting

### ❓ Common Issues

<details>
<summary><strong>🔥 Issue 1: GPU Memory Insufficient</strong></summary>

**Symptoms**: CUDA out of memory error

**Solutions**:
```bash
# 🔍 Check GPU usage
nvidia-smi

# 🔧 Solutions
# 1. Use smaller model
POST /change_transcription_model
model_name: turbo

# 2. Clear GPU cache
docker restart babelon

# 3. Reduce concurrent processing count
```

</details>

<details>
<summary><strong>🤗 Issue 2: Model Loading Failed</strong></summary>

**Symptoms**: Hugging Face model download failed

**Solutions**:
```bash
# 🔐 Check login status
huggingface-cli whoami

# 🧹 Clear cache and re-download
rm -rf ~/.cache/huggingface/

# 🔄 Re-login
huggingface-cli login --token your_token
```

</details>

<details>
<summary><strong>☁️ Issue 3: Translation API Error</strong></summary>

**Symptoms**: Translation function unresponsive or error

**Solutions**:
```bash
# ⚙️ Check Azure configuration
cat lib/azure_config.yaml

# 🔍 Verify API connectivity
curl -X POST "https://your-endpoint.openai.azure.com/openai/deployments/your-deployment/chat/completions?api-version=2024-02-15-preview" \
  -H "Content-Type: application/json" \
  -H "api-key: your-api-key" \
  -d '{"messages":[{"role":"user","content":"test"}]}'

# 📋 View detailed logs
tail -f logs/app.log
```

</details>

### 📋 Debugging Tools

<table>
<tr>
<th>Tool</th>
<th>Purpose</th>
<th>Command</th>
</tr>
<tr>
<td><strong>📊 Real-time Logs</strong></td>
<td>Monitor system running status</td>
<td><code>tail -f logs/app.log</code></td>
</tr>
<tr>
<td><strong>🔍 Error Search</strong></td>
<td>Quickly locate error messages</td>
<td><code>grep "error" logs/app.log</code></td>
</tr>
<tr>
<td><strong>🔧 Debug Mode</strong></td>
<td>Enable detailed logging</td>
<td>Set <code>logging.DEBUG</code> in main.py</td>
</tr>
<tr>
<td><strong>🔥 GPU Monitoring</strong></td>
<td>Check GPU usage status</td>
<td><code>watch -n 1 nvidia-smi</code></td>
</tr>
</table>

## 🤝 Development Participation

### Development Environment Setup
```bash
# Clone project
git clone https://github.com/bobo0303/Babelon-Translate-Service.git
cd Babelon-Translate-Service

# Install dependencies
pip install -r requirements.txt

# Configure necessary files
# 1. Create azure_config.yaml (contains Azure API secrets)
# 2. Set Hugging Face Token
```

### Development Notes
- Please ensure Azure OpenAI API is configured
- GPU environment recommended to use Docker deployment
- Please confirm all dependent models are downloaded before testing
- Secret files (azure_config.yaml) should not be committed to repository

## 📄 License

This project is licensed under [MIT License](LICENSE)

## 📞 Contact Information

<table>
<tr>
<td>👤 <strong>Project Maintainer</strong></td>
<td>Bobo</td>
</tr>
<tr>
<td>🐛 <strong>Issue Reporting</strong></td>
<td><a href="https://github.com/bobo0303/Babelon-Translate-Service/issues">GitHub Issues</a></td>
</tr>
<tr>
<td>⭐ <strong>If you find this useful</strong></td>
<td>Please give us a Star ⭐</td>
</tr>
</table>

## 🔄 Change Log

### 📅 v1.0.0 (2025-10-17)
- 🎉 **Major version release**
- 🎵 Support for multi-language audio transcription and translation
- 🤖 Integration of multiple AI models (Whisper, Gemma, Ollama, GPT-4o/4.1)
- ⚡ Implementation of SSE real-time streaming translation
- 🐳 Docker containerized deployment support
- 🛠️ Complete English documentation and comments
- 🔧 Enhanced multi-strategy transcription capabilities
- 📊 Real-time performance monitoring (RTF)

---

<div align="center">

### 🌟 Thank you for using Babelon Translation Service!

**⚠️ Disclaimer**: This service is still under continuous development, and some features may be subject to adjustment.  
Please test thoroughly before production use and comply with relevant AI service terms of use.

<br>

[![Made with ❤️](https://img.shields.io/badge/Made%20with-❤️-red.svg)](https://github.com/bobo0303/Babelon-Translate-Service)
[![Powered by FastAPI](https://img.shields.io/badge/Powered%20by-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![AI Translation](https://img.shields.io/badge/AI-Translation-blue.svg)](https://github.com/bobo0303/Babelon-Translate-Service)

</div>