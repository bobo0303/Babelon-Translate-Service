import os
import sys
import json
import threading
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Set

from fastapi import WebSocket
from api.audio_process import AudioProcessor
from lib.constant import SAMPLERATE, FRAME_DURATION


class ConnectionManager:
    def __init__(self, logger):

        self.connections: Dict[str, WebSocket] = {}
        self.audio_processors: Dict[str, AudioProcessor] = {}
        self.logger = logger

    async def connect(self, websocket: WebSocket, connection_id: str, payload_data: dict):
        meeting_id = payload_data.get("meeting_id", "default_meeting_id")

        await websocket.accept()
        self.logger.info(f"🔗 WebSocket 連線已建立: {connection_id}, meeting_id: {meeting_id}")
        
        self.connections[connection_id] = websocket

        processor = AudioProcessor(self.logger, payload_data, self.connections, connection_id)
        
        self.audio_processors[connection_id] = processor
    
        status_data = {
            "connection_id": connection_id,
            "meeting_id": meeting_id,
            "speaker_id": payload_data.get("speaker_id", "default_speaker_id"),
            "speaker_name": payload_data.get("speaker_name", "default_speaker_name"),
            "recording_id": payload_data.get("recording_id", "default_recording_id"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
            "samplerate": SAMPLERATE,
            "frame_duration_for_webrtc": FRAME_DURATION,
        }   
        
        await self._send_message(
            connection_id, status_data
        )

    async def handle_message(self, connection_id: str, message: str):
        self.logger.info(f" | Get message: {message} | ")
        return_message = "Babelon-Translate-service websocket can't handle message. If need we can implement this feature."
        self.logger.info(f" | {return_message} | ")
        
        message_data = {"get_message": message,
                        "return_message": return_message,
        }
        
        await self._send_message(
            connection_id, message_data
        )

    async def handle_binary_data(self, connection_id: str, data: bytes):
        """處理接收到的二進位音訊資料."""
        if connection_id not in self.connections:
            return
        
        processor: Optional[AudioProcessor] = self.audio_processors.get(connection_id)
        
        if processor:
            try:
                chunk_length, audio_state = await processor.preprocess_chunk(data)
                
                message = {
                    "chunk_length": chunk_length,
                    "audio_state": audio_state,
                }
                
                await self._send_message(
                    connection_id, message
                )
    
            except Exception as e:
                self.logger.error(f"❌ 音訊處理錯誤: {connection_id}, {str(e)}")
                audio_state = "error"

                await self._send_error(
                    self.connections[connection_id],
                    "PROCESSING_ERROR",
                    f"音訊處理失敗: {str(e)}",
                )

    async def disconnect(self, connection_id: str):
        """斷開 WebSocket 連線."""
        if connection_id not in self.connections:
            return

        # 清理連線
        websocket = self.connections.pop(connection_id, None)
        self.audio_processors.pop(connection_id, None)

        # 關閉 WebSocket
        if websocket:
            try:
                await websocket.close()
            except Exception:
                pass  # 忽略關閉錯誤

    async def _send_message(self, connection_id: str, message: dict):

        websocket = self.connections.get(connection_id)
        if not websocket:
            return
        
        try:
            await websocket.send_text(json.dumps(message))
        except Exception:
            # 連線可能已斷開，清理資源
            await self.disconnect(connection_id)
            
    async def _send_error(
        self,
        websocket: WebSocket,
        error_code: str,
        message: str,
        details: Optional[Dict] = None,
    ):
        """發送錯誤訊息."""
        error_message = {
            "error_code": error_code,
            "message": message,
            "details": details
        }

        try:
            await websocket.send_text(json.dumps(error_message))
        except Exception:
            pass  # 忽略發送錯誤
