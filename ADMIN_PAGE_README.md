# WebSocket Admin Console 使用說明

## 概述

這是一個專為 WebSocket 服務設計的管理控制台，提供即時監控和測試功能。

## 訪問方式

啟動服務後，在瀏覽器中訪問：
```
http://localhost/admin
```
或
```
http://your-server-ip:port/admin
```

## 主要功能

### 1. 🔌 連線設定
- **Meeting ID**: 設定會議 ID
- **Speaker ID**: 設定講者 ID
- **Speaker Name**: 設定講者名稱
- **Device ID**: 設定設備 ID
- **連線按鈕**: 建立 WebSocket 連線
- **斷線按鈕**: 關閉當前連線
- **重新連線**: 快速重新連線

### 2. 💬 訊息測試
支援所有 WebSocket 控制訊息類型：

- `set_translate`: 設定翻譯開關 (true/false)
- `set_prev_text`: 設定前文內容和使用開關
- `set_post_processing`: 設定後處理開關
- `set_language`: 設定來源語言 (zh/en/de/ja/ko)
- `set_meeting_id`: 動態更改會議 ID
- `set_recording_id`: 設定錄音 ID
- `set_speaker`: 設定講者 ID 和名稱
- `set_device_id`: 設定設備 ID
- `clear_stt_queue`: 清除 STT 佇列
- `set_pre_buffer`: 設定預緩衝參數
- `set_silent_duration`: 設定靜音持續時間
- `get_prams`: 查詢當前所有參數
- `ping`: 連線測試

**使用方式**：
1. 選擇訊息類型
2. 根據訊息類型填寫相應參數
3. 點擊「發送訊息」按鈕

### 3. 🎤 音訊串流
直接從瀏覽器錄製並串流音訊到 WebSocket 服務：

- **選擇麥克風**: 自動列出可用的麥克風設備
- **開始錄音**: 開始捕獲麥克風音訊並串流
- **停止錄音**: 停止音訊串流
- **即時統計**: 顯示已發送的音訊區塊數量和錄音時長
- **音訊視覺化**: 即時顯示音訊波形

**音訊格式**：
- 採樣率: 16kHz
- 聲道: 單聲道 (Mono)
- 區塊大小: 512 samples (32ms)
- 數據格式: Float32 PCM

### 4. 📋 訊息日誌
即時顯示所有 WebSocket 通訊：

- **發送訊息** (藍色): 客戶端發送到伺服器的訊息
- **接收訊息** (綠色): 伺服器回傳的訊息
- **錯誤訊息** (紅色): 錯誤和異常
- **系統訊息** (灰色): 連線狀態等系統資訊

**功能**：
- 自動捲動到最新日誌
- 清除日誌按鈕
- 時間戳記顯示

### 5. 📊 統計資訊
即時顯示：
- **已發送**: 發送的文字訊息數量
- **已接收**: 接收的訊息數量
- **音訊區塊**: 發送的音訊區塊總數

## 使用流程

### 基本測試流程
1. 填寫連線參數 (Meeting ID, Speaker ID 等)
2. 點擊「連線」按鈕建立 WebSocket 連線
3. 選擇要測試的訊息類型
4. 填寫參數並發送
5. 在日誌區查看伺服器回應

### 音訊串流測試流程
1. 建立 WebSocket 連線
2. 在「選擇麥克風」下拉選單中選擇麥克風
3. 點擊「開始錄音」
4. 說話或播放音訊
5. 觀察音訊視覺化和統計資訊
6. 在日誌區查看 STT 轉譯結果
7. 點擊「停止錄音」結束

## 技術細節

### WebSocket 端點
```
ws://host:port/S2TT/vad_translate_stream?payload={json_payload}
```

### Payload 格式
```json
{
  "meeting_id": "string",
  "speaker_id": "string",
  "speaker_name": "string",
  "recording_id": "string",
  "device_id": "string"
}
```

### 音訊資料格式
- 使用 Web Audio API 捕獲麥克風
- 重新採樣至 16kHz
- 每 32ms (512 samples) 發送一個區塊
- 以 Float32Array binary data 發送

### 控制訊息格式
所有控制訊息都是 JSON 格式的文字訊息：
```json
{
  "type": "message_type",
  "param1": "value1",
  "param2": "value2"
}
```

## 注意事項

1. **瀏覽器權限**: 使用麥克風功能需要授予瀏覽器麥克風訪問權限
2. **HTTPS**: 在生產環境中使用麥克風功能需要 HTTPS
3. **瀏覽器兼容性**: 建議使用 Chrome、Edge 或 Firefox 最新版本
4. **網絡延遲**: 日誌中的時間戳為客戶端時間
5. **音訊格式**: 自動處理音訊重新採樣和格式轉換

## 故障排除

### 無法連線
- 檢查服務是否正在運行
- 確認 WebSocket 端點正確
- 檢查防火牆設置

### 麥克風無法使用
- 授予瀏覽器麥克風權限
- 確認已選擇正確的麥克風設備
- 檢查麥克風是否被其他應用占用

### 沒有收到訊息
- 確認已建立連線（狀態指示器為綠色）
- 檢查伺服器日誌
- 確認訊息格式正確

## 開發資訊

- **文件位置**: `/mnt/static/admin.html`
- **路由**: `/admin` (在 `main.py` 中定義)
- **靜態文件**: 掛載在 `/static`
- **無需登入**: 直接訪問（生產環境建議添加認證）

## 更新記錄

### v1.0.0 (2025-11-17)
- 初始版本
- 支援所有 WebSocket 訊息類型
- 麥克風音訊串流功能
- 即時日誌顯示
- 統計資訊面板
