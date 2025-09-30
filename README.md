<div align="center">

# 🌐 Babelon 翻譯服務

**多語言音頻轉錄與翻譯平台**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*整合先進 ASR 與 AI 翻譯技術，為會議記錄與即時翻譯提供專業解決方案*

</div>

---

## 📝 項目概述

**Babelon** 是一個基於 FastAPI 的多語言音頻轉錄與翻譯服務平台。融合了最新的 ASR（自動語音識別）和 AI 翻譯技術，提供高精度的語音轉文字和多語言翻譯功能。

### 🎯 適用場景
- 📋 **會議記錄** - 自動生成多語言會議紀錄
- 🎙️ **即時語音翻譯** - 線上會議即時多語言支援
- 📝 **語音備忘錄** - 將語音快速轉換為可編輯文字
- 🌍 **多語言內容創作** - 一鍵生成多語言版本內容

## ✨ 核心特色

<table>
<tr>
<td width="50%">

### 🎯 **核心功能**
- 🎵 **高精度音頻轉錄** - 支援 Whisper large-v2/v3/turbo 模型
- 🌍 **多語言翻譯** - 中文（繁體）、英文、德文互譯
- ⚡ **即時串流處理** - SSE 技術實現即時翻譯回饋
- 🔄 **多重策略轉錄** - 最多 4 種策略確保最佳準確度
- 🛠️ **智能後處理** - 自動修正 ASR 常見錯誤

</td>
<td width="50%">

### 🛠 **技術亮點**
- 🚀 **多 AI 模型整合** - Whisper、Gemma、Ollama、GPT-4o
- ⚡ **GPU 加速運算** - CUDA 支援，大幅提升處理速度
- 🐳 **容器化部署** - Docker/Docker Compose 一鍵部署
- 🔧 **智能資源管理** - 自動清理、記憶體優化
- 🛡️ **多策略容錯** - 確保服務穩定性

</td>
</tr>
</table>

## 🏗 系統架構

## 🏗 系統架構

```
🌐 Babelon 翻譯服務
│
├── 🚀 main.py                    # FastAPI 主應用程式
│
├── 📂 api/                       # API 核心模組
│   ├── 🧠 model.py               # 模型管理中心
│   ├── 🔤 gemma_translate.py     # Gemma 翻譯引擎
│   ├── 💬 gpt_translate.py       # GPT-4o 翻譯引擎
│   ├── 🦙 ollama_translate.py    # Ollama 翻譯引擎
│   ├── ⚙️ post_process.py        # 智能後處理模組
│   └── 🔒 threading_api.py       # 多執行緒 API (機密)
│
├── 📚 lib/                       # 共用函式庫
│   ├── 🏗️ base_object.py         # 基礎物件定義
│   ├── ⚙️ constant.py            # 系統常數設定
│   └── ☁️ azure_config.yaml      # Azure API 設定
│
├── 🛠️ tools/                     # 工具程式集
│   └── 🔊 audio_splitter.py      # 音頻分割工具
│
├── 🎵 audio/                     # 音頻暫存區
└── 📋 logs/                      # 系統日誌
```

## 🚀 快速開始

### 📋 環境需求

<table>
<tr>
<td><strong>🐍 Python</strong></td>
<td>3.9 或以上版本</td>
</tr>
<tr>
<td><strong>🔥 GPU</strong></td>
<td>NVIDIA CUDA 支援 (建議 4GB+ VRAM)</td>
</tr>
<tr>
<td><strong>🐳 容器</strong></td>
<td>Docker & Docker Compose</td>
</tr>
<tr>
<td><strong>💾 記憶體</strong></td>
<td>16GB+ RAM (建議)</td>
</tr>
</table>

### 🛠️ 安裝部署

<details>
<summary><strong>🐳 方法一：Docker 部署（推薦）</strong></summary>

```bash
# 📥 複製項目
git clone https://github.com/bobo0303/Babelon-Translate-Service.git
cd Babelon-Translate-Service

# 🏗️ 建置並啟動服務
docker build -t babelon .
# 或使用 Docker Compose
docker-compose up -d

# 🔧 進入容器
docker exec -it babelon bash

# 🚀 啟動服務
python main.py
```

</details>

<details>
<summary><strong>💻 方法二：本地安裝</strong></summary>

```bash
# 📦 安裝相依套件
pip install -r requirements.txt

# ⚙️ 設定環境變數
export HUGGINGFACE_HUB_TOKEN="your-hf-token"

# 🚀 啟動服務
python main.py
```

</details>

### ⚙️ 初次使用設定

<table>
<tr>
<td>⚠️</td>
<td><strong>重要提醒</strong>：以下檔案需要自行準備，包含機密資訊未包含在 repository 中</td>
</tr>
</table>

<details>
<summary><strong>🔧 步驟 1：Azure OpenAI 配置</strong></summary>

建立 `azure_config.yaml` 檔案並填入以下內容：

```yaml
API_KEY: "your_azure_api_key"
AZURE_API_VERSION: "xxxx-xx-xx-preview"
AZURE_ENDPOINT: "https://your-endpoint.openai.azure.com"
AZURE_DEPLOYMENT: "your-deployment-name"
```

</details>

<details>
<summary><strong>🤗 步驟 2：Hugging Face 登入（Gemma 模型）</strong></summary>

```bash
# 首先需要在 Hugging Face 同意使用條款
# 訪問：https://huggingface.co/google/gemma-3-4b-it

# 使用 Token 登入
huggingface-cli login --token your_hf_token
```

</details>

<details>
<summary><strong>🦙 步驟 3：Ollama 設定（可選）</strong></summary>

```bash
# 🐳 建置 Ollama Docker 容器
docker run -d -it --gpus all --shm-size 32G --runtime nvidia \
  --device=/dev/nvidia-uvm --device=/dev/nvidia-uvm-tools \
  --device=/dev/nvidiactl --device=/dev/nvidia0 \
  -v ./ollama:/root/.ollama -p 52013:11434 \
  --name ollama ollama/ollama

# 🧪 測試 Ollama 模型
docker exec -it ollama ollama run gemma3:12b-it-qat --verbose
```

</details>

## 📋 API 文檔

### 基本資訊
- **服務地址**：`http://localhost:80`
- **API 文檔**：`http://localhost:80/docs`
- **健康檢查**：`GET /`

### 主要 API 端點

#### 🎵 音頻轉錄翻譯
**端點**：`POST /translate`

將音頻檔案進行語音識別並翻譯成多國語言，支援會議記錄、語音備忘錄等應用場景。

```http
POST /translate
Content-Type: multipart/form-data

file: audio file (.wav, .mp3, .m4a)
meeting_id: string
device_id: string  
audio_uid: string
times: datetime
o_lang: string (zh|en|de)
prev_text: string (可選，前文語境)
multi_strategy_transcription: int (1-4，預設1)
transcription_post_processing: bool (預設true)
```

**回應格式**：
```json
{
  "status": "OK",
  "message": "翻譯結果摘要",
  "data": {
    "meeting_id": "123",
    "device_id": "456", 
    "ori_lang": "zh",
    "text": {
      "zh": "中文翻譯",
      "en": "English translation",
      "de": "Deutsche Übersetzung"
    },
    "times": "2024-01-01T10:00:00",
    "audio_uid": "789",
    "transcribe_time": 2.5,
    "translate_time": 1.2
  }
}
```

#### 📝 純文字翻譯
**端點**：`POST /text_translate`

直接翻譯已有的文字內容，快速獲得多語言版本。

```http
POST /text_translate
Content-Type: application/x-www-form-urlencoded

text: 要翻譯的文字
language: 來源語言 (zh|en|de)
```

#### ⚡ 即時串流翻譯（SSE）
**端點**：`POST/GET /sse_audio_translate`

支援即時音頻處理，適用於線上會議、直播等需要即時回饋的場景。

```http
# 提交音頻到處理佇列
POST /sse_audio_translate

# 建立 Server-Sent Events 連線接收結果
GET /sse_audio_translate
Accept: text/event-stream

# 停止串流連線
POST /stop_sse
```

#### ⚙️ 系統管理
**端點**：多個管理端點

提供模型切換、參數調整、系統狀態查詢等管理功能。

```http
GET /get_current_model          # 查看當前使用的模型
GET /list_optional_items        # 列出所有可用的模型和翻譯選項
POST /change_transcription_model # 切換語音識別模型
POST /change_translation_method  # 切換翻譯引擎
POST /set_prompt                # 設定自定義提示詞
```

## ⚙️ 系統配置

### 🌍 支援語言

<table>
<tr>
<th>語言代碼</th>
<th>語言名稱</th>
<th>說明</th>
</tr>
<tr>
<td><code>zh</code></td>
<td>🇹🇼 繁體中文</td>
<td>台灣地區標準中文</td>
</tr>
<tr>
<td><code>en</code></td>
<td>🇺🇸 英文</td>
<td>美式英語</td>
</tr>
<tr>
<td><code>de</code></td>
<td>🇩🇪 德文</td>
<td>標準德語</td>
</tr>
</table>

### 🎙️ 轉錄模型選項

<table>
<tr>
<th>模型名稱</th>
<th>說明</th>
<th>推薦使用</th>
</tr>
<tr>
<td><code>large_v2</code></td>
<td>OpenAI Whisper Large v2（預設）</td>
<td>✅ 穩定性佳</td>
</tr>
<tr>
<td><code>large_v3</code></td>
<td>OpenAI Whisper Large v3</td>
<td>🎯 精度更高</td>
</tr>
<tr>
<td><code>turbo</code></td>
<td>OpenAI Whisper Large v3 Turbo</td>
<td>⚡ 速度優先</td>
</tr>
<tr>
<td><code>TCM</code></td>
<td>自定義模型路徑</td>
<td>🔧 客製化需求</td>
</tr>
</table>

### 🤖 翻譯引擎選項

<table>
<tr>
<th>引擎名稱</th>
<th>說明</th>
<th>部署方式</th>
<th>特色</th>
</tr>
<tr>
<td><code>gpt4o</code></td>
<td>GPT-4o（預設）</td>
<td>☁️ Azure OpenAI API</td>
<td>🏆 最高品質</td>
</tr>
<tr>
<td><code>gemma4b</code></td>
<td>Google Gemma 4B</td>
<td>💻 本地運行</td>
<td>🔒 隱私保護</td>
</tr>
<tr>
<td><code>ollama-gemma</code></td>
<td>Ollama Gemma</td>
<td>🐳 容器部署</td>
<td>⚡ 快速部署</td>
</tr>
<tr>
<td><code>ollama-qwen</code></td>
<td>Ollama Qwen</td>
<td>🐳 容器部署</td>
<td>🌏 中文優化</td>
</tr>
</table>

### 🔧 環境變數設定

```bash
# 🤗 Hugging Face Token（Gemma 模型使用）
# 請先訪問並同意使用條款：https://huggingface.co/google/gemma-3-4b-it
export HUGGINGFACE_HUB_TOKEN="your_hf_token"

# 🔥 GPU 設定
export NVIDIA_VISIBLE_DEVICES=all
export CUDA_VISIBLE_DEVICES=0
```

## 🔧 進階功能

### 🎯 多重策略轉錄

<table>
<tr>
<th>策略</th>
<th>描述</th>
<th>適用場景</th>
</tr>
<tr>
<td>🔹 <strong>策略 1</strong></td>
<td>自定義提示詞 + 前文語境</td>
<td>連續對話、會議記錄</td>
</tr>
<tr>
<td>🔸 <strong>策略 2</strong></td>
<td>僅使用自定義提示詞</td>
<td>專業術語、特定領域</td>
</tr>
<tr>
<td>🔹 <strong>策略 3</strong></td>
<td>無提示詞，低溫度採樣</td>
<td>一般語音、高準確度需求</td>
</tr>
<tr>
<td>🔸 <strong>策略 4</strong></td>
<td>高溫度多樣化採樣</td>
<td>不清晰音頻、口音較重</td>
</tr>
</table>

### 🛠️ 智能後處理

<table>
<tr>
<td width="50%">

#### 🔍 **錯誤檢測**
- 🔤 ASR 常見錯誤自動識別
- 🏢 品牌名稱標準化處理
- 📝 語法錯誤智能修正
- ⚠️ 幻覺內容檢測過濾

</td>
<td width="50%">

#### ⚡ **效能優化**
- 🧹 24小時自動清理音頻檔案
- 💾 GPU 記憶體智能回收
- 🔄 模型熱切換零中斷
- 📊 即時性能監控 (RTF)

</td>
</tr>
</table>

## 🎯 使用範例

### Python 客戶端範例
```python
import requests

# 音頻翻譯
files = {'file': open('audio.wav', 'rb')}
data = {
    'meeting_id': '001',
    'device_id': 'mic_01', 
    'audio_uid': 'audio_001',
    'times': '2024-01-01T10:00:00',
    'o_lang': 'zh',
    'multi_strategy_transcription': 2
}

response = requests.post(
    'http://localhost:80/translate',
    files=files,
    data=data
)
print(response.json())

# 文字翻譯
data = {'text': '你好世界', 'language': 'zh'}
response = requests.post(
    'http://localhost:80/text_translate',
    data=data
)
print(response.json())
```

### curl 範例
```bash
# 音頻翻譯
curl -X POST "http://localhost:80/translate" \
  -F "file=@audio.wav" \
  -F "meeting_id=001" \
  -F "device_id=mic_01" \
  -F "audio_uid=audio_001" \
  -F "times=2024-01-01T10:00:00" \
  -F "o_lang=zh"

# 文字翻譯  
curl -X POST "http://localhost:80/text_translate" \
  -F "text=Hello World" \
  -F "language=en"
```

## 📊 效能最佳化

### 💻 建議硬體配置

<table>
<tr>
<th>組件</th>
<th>最低需求</th>
<th>建議配置</th>
<th>最佳效能</th>
</tr>
<tr>
<td><strong>🔥 GPU</strong></td>
<td>GTX 1660 (6GB)</td>
<td>RTX 3080 (10GB)</td>
<td>RTX 4090 (24GB)</td>
</tr>
<tr>
<td><strong>💾 記憶體</strong></td>
<td>8GB RAM</td>
<td>16GB RAM</td>
<td>32GB+ RAM</td>
</tr>
<tr>
<td><strong>💿 儲存</strong></td>
<td>HDD</td>
<td>SATA SSD</td>
<td>NVMe SSD</td>
</tr>
<tr>
<td><strong>🌐 網路</strong></td>
<td>10 Mbps</td>
<td>100 Mbps</td>
<td>1 Gbps</td>
</tr>
</table>

### ⚙️ 效能調校參數

<details>
<summary><strong>🔧 在 constant.py 中調整以下參數</strong></summary>

```python
# ⏱️ 轉錄超時設定
WAITING_TIME = 60           # 單位：秒，建議 30-120

# 🎯 策略數量設定
MAX_NUM_STRATEGIES = 4      # 最大 4 種，可降低至 1-2 提升速度

# 🔇 靜音填充
SILENCE_PADDING = True      # 提升邊界詞識別，輕微增加處理時間

# 📈 即時係數計算
RTF = True                  # 啟用效能監控，輕微影響效能
```

</details>

### 📈 效能監控指標

<table>
<tr>
<td><strong>🎯 RTF (Real-Time Factor)</strong></td>
<td>< 0.3 優秀 | 0.3-0.5 良好 | > 0.5 需優化</td>
</tr>
<tr>
<td><strong>💾 GPU 記憶體使用率</strong></td>
<td>< 80% 安全 | 80-90% 注意 | > 90% 危險</td>
</tr>
<tr>
<td><strong>⚡ 處理速度</strong></td>
<td>1分鐘音頻 < 20秒處理為佳</td>
</tr>
</table>

## 🛡️ 安全性考慮

### API 安全
- 建議使用反向代理（nginx）
- 設定 API 速率限制
- 生產環境啟用 HTTPS

### 資料隱私
- 音頻檔案自動清理
- 敏感資訊日誌遮蔽
- 支援本地部署，資料不出境

## 🐛 故障排除

### ❓ 常見問題

<details>
<summary><strong>🔥 問題 1：GPU 記憶體不足</strong></summary>

**症狀**：CUDA out of memory 錯誤

**解決方案**：
```bash
# 🔍 檢查 GPU 使用狀況
nvidia-smi

# 🔧 解決方案
# 1. 使用較小的模型
POST /change_transcription_model
model_name: turbo

# 2. 清理 GPU 快取
docker restart babelon

# 3. 降低並發處理數量
```

</details>

<details>
<summary><strong>🤗 問題 2：模型載入失敗</strong></summary>

**症狀**：Hugging Face 模型下載失敗

**解決方案**：
```bash
# 🔐 檢查登入狀態
huggingface-cli whoami

# 🧹 清除快取重新下載
rm -rf ~/.cache/huggingface/

# 🔄 重新登入
huggingface-cli login --token your_token
```

</details>

<details>
<summary><strong>☁️ 問題 3：翻譯 API 錯誤</strong></summary>

**症狀**：翻譯功能無回應或錯誤

**解決方案**：
```bash
# ⚙️ 檢查 Azure 配置
cat lib/azure_config.yaml

# 🔍 驗證 API 連通性
curl -X POST "https://your-endpoint.openai.azure.com/openai/deployments/your-deployment/chat/completions?api-version=2024-02-15-preview" \
  -H "Content-Type: application/json" \
  -H "api-key: your-api-key" \
  -d '{"messages":[{"role":"user","content":"test"}]}'

# 📋 查看詳細日誌
tail -f logs/app.log
```

</details>

### 📋 除錯工具

<table>
<tr>
<th>工具</th>
<th>用途</th>
<th>指令</th>
</tr>
<tr>
<td><strong>📊 即時日誌</strong></td>
<td>監控系統運行狀態</td>
<td><code>tail -f logs/app.log</code></td>
</tr>
<tr>
<td><strong>🔍 錯誤搜尋</strong></td>
<td>快速定位錯誤訊息</td>
<td><code>grep "error" logs/app.log</code></td>
</tr>
<tr>
<td><strong>🔧 除錯模式</strong></td>
<td>開啟詳細日誌</td>
<td>在 main.py 中設定 <code>logging.DEBUG</code></td>
</tr>
<tr>
<td><strong>🔥 GPU 監控</strong></td>
<td>檢查 GPU 使用狀況</td>
<td><code>watch -n 1 nvidia-smi</code></td>
</tr>
</table>

## 🤝 開發參與

### 開發環境設定
```bash
# 複製項目
git clone https://github.com/bobo0303/Babelon-Translate-Service.git
cd Babelon-Translate-Service

# 安裝依賴
pip install -r requirements.txt

# 配置必要檔案
# 1. 創建 azure_config.yaml
# 2. 準備 threading_api.py（包含機密資訊）
# 3. 設定 Hugging Face Token
```

### 開發注意事項
- 請確保已配置 Azure OpenAI API
- GPU 環境建議使用 Docker 部署
- 測試前請確認所有依賴模型已下載
- 機密檔案請勿提交到 repository

## 📄 授權條款

本項目採用 [MIT License](LICENSE) 開源授權

## 📞 聯絡資訊

<table>
<tr>
<td>👤 <strong>項目維護者</strong></td>
<td>Bobo</td>
</tr>
<tr>
<td>🐛 <strong>問題回報</strong></td>
<td><a href="https://github.com/bobo0303/Babelon-Translate-Service/issues">GitHub Issues</a></td>
</tr>
<tr>
<td>⭐ <strong>如果覺得有用</strong></td>
<td>歡迎給個 Star ⭐</td>
</tr>
</table>

## 🔄 更新日誌

### 📅 v1.0.0 (2024-01-01)
- 🎉 **初始版本發布**
- 🎵 支援多語言音頻轉錄翻譯
- 🤖 整合多種 AI 模型 (Whisper, Gemma, Ollama, GPT-4o)
- ⚡ 實現 SSE 即時串流翻譯
- 🐳 Docker 容器化部署支援

---

<div align="center">

### 🌟 感謝使用 Babelon 翻譯服務！

**⚠️ 免責聲明**：本服務仍在持續開發中，部分功能可能會有調整。  
生產環境使用前請充分測試，並遵守相關 AI 服務的使用條款。

<br>

[![Made with ❤️](https://img.shields.io/badge/Made%20with-❤️-red.svg)](https://github.com/bobo0303/Babelon-Translate-Service)
[![Powered by FastAPI](https://img.shields.io/badge/Powered%20by-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![AI Translation](https://img.shields.io/badge/AI-Translation-blue.svg)](https://github.com/bobo0303/Babelon-Translate-Service)

</div>