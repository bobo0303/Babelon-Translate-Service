#!/usr/bin/env python3
"""
Simple Audio WebSocket Test Client
================================

這是一個簡化的客戶端，專門用於測試 Paul 的音檔檔案。
會自動載入音檔並以正確的格式發送到你的 WebSocket 服務。

使用方式：
    python test_paul_audio.py
"""

import asyncio
import json
import logging
import numpy as np
import librosa
import websockets
from datetime import datetime

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class SimpleAudioClient:
    def __init__(self):
        # 設定參數
        self.audio_file = "/mnt/old/2025 Q2業說會_Paul.wav"
        self.server_url = "ws://localhost:80"
        self.target_sample_rate = 16000
        self.chunk_duration = 0.032  # 32ms
        self.chunk_size = int(self.target_sample_rate * self.chunk_duration)  # 512 samples
        
        # 音檔處理參數
        self.gain_db = 10.0  # 增益 5dB
        self.gain_linear = 10 ** (self.gain_db / 20.0)  # 將 dB 轉換為線性增益
        
        # 連線資訊
        self.payload_data = {
            "meeting_id": "paul_test_meeting",
            "speaker_id": "paul_speaker",
            "speaker_name": "Paul",
            "recording_id": f"paul_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
    async def load_audio(self):
        """載入並預處理音檔"""
        logger.info(f" | 🎵 載入音檔: {self.audio_file} | ")
        
        try:
            # 載入音檔並轉換為 16kHz
            audio_data, original_sr = librosa.load(
                self.audio_file, 
                sr=self.target_sample_rate, 
                dtype=np.float32
            )
            
            # 確保是單聲道
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            duration = len(audio_data) / self.target_sample_rate
            num_chunks = len(audio_data) // self.chunk_size
            
            logger.info(f" | 📊 音檔資訊: | ")
            logger.info(f" |    - 長度: {len(audio_data)} samples | ")
            logger.info(f" |    - 時長: {duration:.2f} 秒 | ") 
            logger.info(f" |    - 採樣率: {self.target_sample_rate} Hz | ")
            logger.info(f" |    - 區塊數量: {num_chunks} 個 (每個 {self.chunk_size} samples) | ")
            logger.info(f" |    - 音頻增益: +{self.gain_db} dB (線性增益: {self.gain_linear:.2f}x) | ")
            
            return audio_data
            
        except Exception as e:
            logger.error(f" | ❌ 載入音檔失敗: {e} | ")
            raise
    
    def apply_gain_to_chunk(self, chunk):
        """
        對音檔區塊應用增益
        
        Args:
            chunk: 音檔區塊 (numpy array)
            
        Returns:
            處理後的音檔區塊
        """
        # 應用線性增益
        gained_chunk = chunk * self.gain_linear
        
        # 防止音檔削波 (clipping) - 限制在 [-1, 1] 範圍內
        gained_chunk = np.clip(gained_chunk, -1.0, 1.0)
        
        return gained_chunk
            
    async def connect_websocket(self):
        """連接到 WebSocket"""
        try:
            # 建立連線 URL
            import urllib.parse
            payload_json = json.dumps(self.payload_data)
            encoded_payload = urllib.parse.quote(payload_json)
            ws_url = f"{self.server_url}/S2TT/vad_translate_stream?payload={encoded_payload}"
            
            logger.info(f" | 🔗 連接到: {ws_url} | ")
            
            # 建立 WebSocket 連線
            websocket = await websockets.connect(ws_url)
            logger.info(f" | ✅ WebSocket 連線建立成功 | ")
            
            return websocket
            
        except Exception as e:
            logger.error(f" | ❌ WebSocket 連線失敗: {e} | ")
            raise
            
    async def send_audio_stream(self, websocket, audio_data):
        """發送音檔資料流"""
        logger.info(f" | 🚀 開始發送音檔資料... | ")
        
        # 開始接收訊息的任務
        async def message_receiver():
            try:
                while True:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    try:
                        response = json.loads(message)
                        
                        # 處理不同類型的回應
                        if "connection_id" in response:
                            logger.info(f" | 🎯 連線 ID: {response['connection_id']} | ")
                        elif "chunk_length" in response:
                            state = response.get("audio_state", "unknown")
                            if state == "success":
                                logger.debug(f" | ✅ 區塊處理成功: {response['chunk_length']} bytes | ")
                            else:
                                logger.warning(f" | ⚠️ 區塊狀態: {state} | ")
                        elif "error_code" in response:
                            logger.error(f" | ❌ 伺服器錯誤: {response} | ")
                        else:
                            logger.info(f" | 📨 收到訊息: {response} | ")
                            
                    except json.JSONDecodeError:
                        logger.info(f" | 📨 非 JSON 訊息: {message} | ")
                        
            except asyncio.TimeoutError:
                pass
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f" | 🔌 WebSocket 連線已關閉 | ")
            except Exception as e:
                logger.error(f" | ❌ 接收訊息錯誤: {e} | ")
                
        # 啟動訊息接收器
        receiver_task = asyncio.create_task(message_receiver())
        
        try:
            total_chunks = len(audio_data) // self.chunk_size
            sent_chunks = 0
            
            # 分割音檔並發送
            for i in range(0, len(audio_data), self.chunk_size):
                # 取得音檔區塊
                end_idx = min(i + self.chunk_size, len(audio_data))
                chunk = audio_data[i:end_idx]
                
                # 如果最後一個區塊不足，填充零
                if len(chunk) < self.chunk_size:
                    padding = np.zeros(self.chunk_size - len(chunk), dtype=np.float32)
                    chunk = np.concatenate([chunk, padding])
                
                # 應用 5dB 增益到音檔區塊
                chunk = self.apply_gain_to_chunk(chunk)
                
                # 轉換為二進位資料並發送
                chunk_bytes = chunk.astype(np.float32).tobytes()
                await websocket.send(chunk_bytes)
                
                sent_chunks += 1
                
                # 記錄進度
                if sent_chunks % 100 == 0 or sent_chunks <= 10:
                    progress = (sent_chunks / total_chunks) * 100
                    elapsed_time = sent_chunks * self.chunk_duration
                    logger.info(f" | 📈 進度: {sent_chunks}/{total_chunks} ({progress:.1f}%) - {elapsed_time:.1f}s | ")
                
                # 等待下一個區塊間隔 (32ms)
                await asyncio.sleep(self.chunk_duration)
                
            logger.info(f" | ✅ 音檔發送完成! 總共發送 {sent_chunks} 個區塊 | ")
            
            # 等待一點時間讓伺服器處理完剩餘資料
            await asyncio.sleep(999999.0)
            
        finally:
            # 取消訊息接收器
            receiver_task.cancel()
            
    async def run(self):
        """執行完整的音檔串流流程"""
        try:
            # 載入音檔
            audio_data = await self.load_audio()
            
            # 連接 WebSocket
            websocket = await self.connect_websocket()
            
            try:
                # 發送音檔資料流
                await self.send_audio_stream(websocket, audio_data)
                
            finally:
                # 關閉連線
                await websocket.close()
                logger.info(f" | 🔌 WebSocket 連線已關閉 | ")
                
            logger.info(f" | 🎉 音檔串流測試完成! | ")
            return True
            
        except KeyboardInterrupt:
            logger.info(f" | 🛑 使用者中斷測試 | ")
            return False
        except Exception as e:
            logger.error(f" | ❌ 測試執行錯誤: {e} | ")
            return False

async def main():
    """主要執行函數"""
    print("=" * 60)
    print("🎵 Paul 音檔 WebSocket 串流測試")
    print("=" * 60)
    print("檔案:", "/mnt/old/2025_Q1業_Paul.wav")
    print("伺服器:", "ws://localhost:80")
    print("格式:", "16kHz, 32ms chunks (512 samples)")
    print("增益:", "+5 dB")
    print("=" * 60)
    
    client = SimpleAudioClient()
    success = await client.run()
    
    if success:
        print("\n🎉 測試成功完成!")
    else:
        print("\n❌ 測試失敗!")

if __name__ == "__main__":
    asyncio.run(main())