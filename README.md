# Babelon 翻譯服務

## 📝 項目概述

Babelon 是一個基於 FastAPI 的多語言音頻轉錄與翻譯服務，整合了先進的 ASR（自動語音識別）和 AI 翻譯技術。該服務能夠將音頻文件轉錄為文字，並翻譯成多種語言，特別適用於會議記錄、即時翻譯等場景。

## ✨ 主要功能

### 🎯 核心功能
- **音頻轉錄**：支援多種 Whisper 模型（large-v2, large-v3, turbo）
- **多語言翻譯**：支援中文（繁體）、英文、德文之間的互相翻譯
- **即時串流**：支援 Server-Sent Events (SSE) 即時翻譯
- **多重策略轉錄**：最多 4 種策略提升轉錄準確度
- **智能後處理**：自動修正 ASR 常見錯誤

### 🛠 技術特色
- **多模型支援**：整合 OpenAI Whisper、Gemma、Ollama、GPT-4o
- **GPU 加速**：支援 CUDA 加速運算
- **容器化部署**：Docker + Docker Compose 快速部署
- **資源管理**：自動清理舊音頻文件，記憶體優化
- **錯誤重試**：多策略容錯機制

## 🏗 系統架構

```
Babelon/
├── main.py                 # FastAPI 主應用程式
├── api/                    # API 模組
│   ├── model.py           # 核心模型管理
│   ├── gemma_translate.py # Gemma 翻譯引擎
│   ├── gpt_translate.py   # GPT-4o 翻譯引擎
│   ├── ollama_translate.py # Ollama 翻譯引擎
│   ├── post_process.py    # 後處理模組
│   └── threading_api.py   # 多執行緒 API（需自行準備，含機密資訊）
├── lib/                   # 共用函式庫
│   ├── base_object.py     # 基礎物件定義
│   ├── constant.py        # 常數與設定
│   └── azure_config.yaml  # Azure 設定
├── tools/                 # 工具程式
│   └── audio_splitter.py  # 音頻分割工具
├── audio/                 # 音頻檔案暫存
└── logs/                  # 日誌檔案
```

## 🚀 快速開始

### 環境需求
- Python 3.9+
- CUDA 支援的 GPU（建議）
- Docker & Docker Compose
- 4GB+ GPU 記憶體

### 安裝部署

#### 方法一：Docker 部署（推薦）
```bash
# 複製項目
git clone <https://github.com/bobo0303/Babelon-Translate-Service.git>
cd Babelon

# 啟動服務
docker build -t babelon .
    or
docker-compose up -d

# 進入容器
docker exec -it babelon bash

# 在容器內啟動服務
python main.py
```

#### 方法二：本地安裝
```bash
# 安裝依賴
pip install -r requirements.txt

# 設定環境變數
export HUGGINGFACE_HUB_TOKEN="your-hf-token"

# 啟動服務
python main.py
```

### 初次使用設定

⚠️ **重要提醒**：`azure_config.yaml` 需要自行準備並配置 Azure OpenAI API。

1. **azure_config.yaml**（檔案必備內容）
```bash
API_KEY:  
AZURE_API_VERSION: 
AZURE_ENDPOINT: 
AZURE_DEPLOYMENT: 
```

2. **Hugging Face 登入**（使用 Gemma 模型需要）
```bash
huggingface-cli login --token your_hf_token
```

3. **Ollama 設定**（使用 Ollama 需要）
```bash
# 建置 ollama docker 
docker run -d -it --gpus all --shm-size 32G --runtime nvidia --device=/dev/nvidia-uvm --device=/dev/nvidia-uvm-tools --device=/dev/nvidiactl --device=/dev/nvidia0 -v ./ollama:/root/.ollama -p 52013:11434 --name ollama ollama/ollama
# 啟動 Ollama 測試
docker exec -it ollama ollama run gemma3:12b-it-qat --verbose
```

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

## ⚙️ 配置說明

### 支援的語言
- `zh`：繁體中文（台灣）
- `en`：英文（美式）
- `de`：德文（標準德語）

### 轉錄模型選項
- `large_v2`：OpenAI Whisper Large v2（預設）
- `large_v3`：OpenAI Whisper Large v3
- `turbo`：OpenAI Whisper Large v3 Turbo
- `TCM`：自定義模型路徑

### 翻譯引擎選項
- `gpt4o`：GPT-4o（預設，需要 Azure OpenAI API）
- `gemma4b`：Google Gemma 4B（本地運行）
- `ollama-gemma`：Ollama Gemma 模型
- `ollama-qwen`：Ollama Qwen 模型

### 環境變數設定
```bash
# Hugging Face Token（Gemma 模型使用）
HUGGINGFACE_HUB_TOKEN=your_hf_token

# GPU 設定
NVIDIA_VISIBLE_DEVICES=all
CUDA_VISIBLE_DEVICES=0
```

## 🔧 進階功能

### 多重策略轉錄
系統支援最多 4 種轉錄策略以提升準確度：
1. **策略 1**：使用自定義提示詞 + 前文語境
2. **策略 2**：僅使用自定義提示詞
3. **策略 3**：無提示詞，低溫度採樣
4. **策略 4**：高溫度多樣化採樣

### 智能後處理
- 自動修正 ASR 常見錯誤
- 品牌名稱標準化
- 語法錯誤修正
- 幻覺檢測與過濾

### 自動資源管理
- 定時清理 24 小時前的音頻檔案
- GPU 記憶體自動回收
- 模型熱切換不中斷服務

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

### 建議的硬體配置
- **GPU**：NVIDIA RTX 3080 以上
- **記憶體**：16GB+ RAM
- **儲存**：SSD 推薦
- **網路**：穩定的網路連線（API 調用）

### 效能調校
```python
# 在 constant.py 中調整參數
WAITING_TIME = 60           # 轉錄超時時間
MAX_NUM_STRATEGIES = 4      # 最大策略數
SILENCE_PADDING = True      # 靜音填充
RTF = True                  # 計算即時係數
```

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

### 常見問題

#### 1. GPU 記憶體不足
```bash
# 檢查 GPU 使用狀況
nvidia-smi

# 降低模型規模
# 改用 turbo 模型替代 large_v3
```

#### 2. 模型載入失敗
```bash
# 檢查 Hugging Face 登入狀態
huggingface-cli whoami

# 清除模型快取
rm -rf ~/.cache/huggingface/
```

#### 3. 翻譯 API 錯誤
```bash
# 檢查 Azure API 金鑰設定
echo $AZURE_OPENAI_API_KEY

# 查看詳細錯誤日誌
tail -f logs/app.log
```

### 日誌除錯
```bash
# 查看即時日誌
tail -f logs/app.log

# 搜尋特定錯誤
grep "error" logs/app.log

# 調整日誌等級（在 main.py 中）
logging.basicConfig(level=logging.DEBUG)
```

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

本項目採用 [MIT License](LICENSE)

## 📞 聯絡資訊

- **項目維護者**：[Bobo]
- **問題回報**：[GitHub Issues]

## 🔄 更新日誌

### v1.0.0 (2024-01-01)
- 初始版本發布
- 支援多語言音頻轉錄翻譯
- 整合多種 AI 模型
- 實現 SSE 即時串流

---

**注意**：本服務仍在持續開發中，部分功能可能會有調整。生產環境使用前請充分測試。