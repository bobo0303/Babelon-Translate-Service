import os
import sys
import json
import threading
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Set

from fastapi import WebSocket
from api.audio.audio_process import AudioProcessor
from lib.config.constant import SAMPLERATE, FRAME_DURATION


class ConnectionManager:
    def __init__(self, logger):

        self.connections: Dict[str, WebSocket] = {}
        self.audio_processors: Dict[str, AudioProcessor] = {}
        self.logger = logger

    async def connect(self, websocket: WebSocket, connection_id: str, payload_data: dict):
        meeting_id = payload_data.get("meeting_id", "default_meeting_id")

        await websocket.accept()
        self.logger.info(f" | 🔗 WebSocket 連線已建立: {connection_id}, meeting_id: {meeting_id} | ")
        
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

    async def handle_message(self, connection_id: str, message: str, ws: WebSocket):
        """處理接收到的文字訊息."""
        self.logger.info(f" | Get message: {message} | ")
        
        connection_state = self.connections.get(connection_id)
        if not connection_state:
            return
        
        audio_processor = self.audio_processors[connection_id]

        try:
            # 解析訊息
            message_data = json.loads(message)
            message_type = message_data.get("type")
            
            if message_type == "set_translate":
                use_translate = message_data.get("use_translate", True)
                audio_processor.stt_processor.set_translation(use_translate)
                self.logger.info(f" | Set use_translate to {use_translate} | ")
                await ws.send_text(
                    json.dumps({"type": "info", "message": f" | Set use_translate to {use_translate} | "})
                )
            
            elif message_type == "set_prev_text":
                use_prev_text = message_data.get("use_prev_text", None)
                if use_prev_text is not None:
                    audio_processor.stt_processor.set_prev_text_usage(use_prev_text)
                    self.logger.info(f" | Set use_prev_text to {use_prev_text} | ")
                    await ws.send_text(
                        json.dumps({"type": "info", "message": f" | Set use_prev_text to {use_prev_text} | "})
                    )
                
                prev_text = message_data.get("prev_text", None)
                if prev_text is not None:
                    audio_processor.stt_processor.set_prev_text(prev_text)
                    self.logger.info(f" | Set prev_text to {prev_text} | ")
                    await ws.send_text(
                        json.dumps({"type": "info", "message": f" | Set prev_text to {prev_text} | "})
                    )
                
            elif message_type == "set_post_processing":
                post_processing = message_data.get("use_post_processing", True)
                audio_processor.stt_processor.set_post_processing_usage(post_processing)
                self.logger.info(f" | Set post_processing to {post_processing} | ")
                await ws.send_text(
                    json.dumps({"type": "info", "message": f" | Set post_processing to {post_processing} | "})
                )
                
            elif message_type == "set_language":
                language = message_data.get("language", "zh")
                audio_processor.stt_processor.set_language(language)
                self.logger.info(f" | Set source language to {language} | ")
                await ws.send_text(
                    json.dumps({"type": "info", "message": f" | Set source language to {language} | "})
                )
            
            elif message_type == "set_meeting_id":
                new_meeting_id = message_data.get("meeting_id", "default_meeting_id")
                audio_processor.set_output_directory(new_meeting_id)
                audio_processor.stt_processor.set_meeting_id(new_meeting_id)
                self.logger.info(f" | Set meeting_id to {new_meeting_id} | ")
                await ws.send_text(
                    json.dumps({"type": "info", "message": f" | Set meeting_id to {new_meeting_id} | "})
                )
            
            elif message_type == "set_recording_id":
                new_recording_id = message_data.get("recording_id", "default_recording_id")
                audio_processor.stt_processor.set_recording_id(new_recording_id)
                await ws.send_text(
                    json.dumps({"type": "info", "message": f" | Set recording_id to {new_recording_id} | "})
                )
            
            elif message_type == "set_speaker":
                new_speaker_id = message_data.get("speaker_id", None)
                if new_speaker_id is not None:
                    audio_processor.stt_processor.set_speaker_id(new_speaker_id)
                    self.logger.info(f" | Set speaker_id to {new_speaker_id} | ")
                    await ws.send_text(
                        json.dumps({"type": "info", "message": f" | Set speaker_id to {new_speaker_id} | "})
                    )
                new_speaker_name = message_data.get("speaker_name", None)
                if new_speaker_name is not None:
                    audio_processor.stt_processor.set_speaker_name(new_speaker_name)
                    self.logger.info(f" | Set speaker_name to {new_speaker_name} | ")
                    await ws.send_text(
                        json.dumps({"type": "info", "message": f" | Set speaker_name to {new_speaker_name} | "})
                    )
                
            elif message_type == "set_device_id":
                new_device_id = message_data.get("device_id", "default_device_id")
                audio_processor.stt_processor.set_device_id(new_device_id)
                self.logger.info(f" | Set device_id to {new_device_id} | ")
                await ws.send_text(
                    json.dumps({"type": "info", "message": f" | Set device_id to {new_device_id} | "})
                )
            
            elif message_type == "clear_stt_queue":
                audio_processor.stt_processor.clear_stt_queue()
                self.logger.info(" | Cleared STT queue | ")
                await ws.send_text(
                    json.dumps({"type": "info", "message": f" | Cleared STT queue | "})
                )
            
            elif message_type == "set_pre_buffer":
                use_pre_buffer = message_data.get("use_pre_buffer", None)
                if use_pre_buffer is not None:
                    audio_processor.set_pre_buffer_usage(use_pre_buffer)
                    self.logger.info(f" | Set use_pre_buffer to {use_pre_buffer} | ")
                    await ws.send_text(
                        json.dumps({"type": "info", "message": f" | Set use_pre_buffer to {use_pre_buffer} | "})
                    )
                pre_buffer_size = message_data.get("pre_buffer_size", None)
                if pre_buffer_size is not None:
                    audio_processor.set_pre_buffer_size(pre_buffer_size)
                    self.logger.info(f" | Set pre_buffer_size to {pre_buffer_size} | ")
                    await ws.send_text(
                        json.dumps({"type": "info", "message": f" | Set pre_buffer_size to {pre_buffer_size} | "})
                    )
            elif message_type == "set_silent_duration":
                silent_duration = message_data.get("silent_duration", 1.0)
                audio_processor.set_silent_duration(silent_duration)
                self.logger.info(f" | Set silent_duration to {silent_duration} | ")
                await ws.send_text(
                    json.dumps({"type": "info", "message": f" | Set silent_duration to {silent_duration} | "})
                )

            elif message_type == "get_prams":
                params = {
                    "meeting_id": audio_processor.stt_processor.meeting_id,
                    "recording_id": audio_processor.stt_processor.recording_id,
                    "speaker_id": audio_processor.stt_processor.speaker_id,
                    "speaker_name": audio_processor.stt_processor.speaker_name,
                    "device_id": audio_processor.stt_processor.device_id,
                    "use_translate": audio_processor.stt_processor.use_translate,
                    "use_prev_text": audio_processor.stt_processor.use_prev_text,
                    "use_post_processing": audio_processor.stt_processor.use_post_processing,
                    "language": audio_processor.stt_processor.language,
                    "pre_buffer_size": audio_processor.pre_buffer_size,
                    "use_pre_buffer": audio_processor.use_pre_buffer,
                    "silent_duration": audio_processor.no_speech_duration_threshold,
                }
                self.logger.info(f" | Get params: {params} | ")
                await ws.send_text(
                    json.dumps({"type": "params", "params": params})
                )   
                
            elif message_type == "ping":
                self.logger.info(" | Ping received | ")
                await ws.send_text(
                    json.dumps({"type": "pong", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), "message": " | 是的我活得很好 ( ˘•ω•˘ ) | "})
                )

            elif message_type == "(́◉◞౪◟◉‵)":
                self.logger.info(" | (́◉◞౪◟◉‵) 有人在搞 (́◉◞౪◟◉‵) | ")
                await ws.send_text(
                    json.dumps({"type": "what's wrong with you", "message": "(́◉◞౪◟◉‵)"})
                )
            
            else:
                self.logger.warning(f"未知的訊息類型: {message_type}")


        except Exception as e:
            self.logger.error(f" | 處理訊息失敗: {str(e)} | ")
            await ws.send_text(
                json.dumps({"type": "error", "message": f" | 處理訊息失敗: {str(e)} | "})
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
                self.logger.error(f" | ❌ 音訊處理錯誤: {connection_id}, {str(e)} | ")
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
