# use webrtcvad and silero vad to detect speech in audio data

from typing import Dict, Optional, Set
from datetime import datetime
import uuid

import logging
import logging.handlers

from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
    Query,
    HTTPException,
    status,
)
from fastapi.responses import HTMLResponse

from api.websocket_manager import ConnectionManager

logger = logging.getLogger(__name__)  
  
# Configure logger settings (if not already configured)  
if not logger.handlers:  
    log_format = "%(asctime)s - %(message)s"  
    log_file = "logs/app.log"  
    logging.basicConfig(level=logging.INFO, format=log_format)  
  
    # Create file handler  
    file_handler = logging.handlers.RotatingFileHandler(  
        log_file, maxBytes=10*1024*1024, backupCount=5  
    )  
    file_handler.setFormatter(logging.Formatter(log_format))  
  
    # Create console handler  
    console_handler = logging.StreamHandler()  
    console_handler.setFormatter(logging.Formatter(log_format))  
  
    logger.addHandler(file_handler)  
    logger.addHandler(console_handler)  
  
logger.setLevel(logging.INFO)  
logger.propagate = False  

router = APIRouter()
connection_manager = ConnectionManager(logger)

@router.websocket("/S2TT/vad_translate_stream")
async def websocket_audio_vad_and_translate(
    websocket: WebSocket,
    payload: Optional[str] = Query("default_connection_info", description="連線資訊"),
    # speaker_id: Optional[str] = Query("default_speaker_id", description="發言者 ID"), # 未來可用於多發言者識別
):
    """
    WebSocket 音訊串流端點。

    支援即時音訊資料傳輸和錄音控制指令。
    認證方式：透過 query parameter 'token' 傳遞 JWT Token。

    Args:
        websocket: WebSocket 連線
        payload: JSON 格式的連線資訊，包含 meeting_id 等資訊
    """
    
    connection_id = f"conn_{uuid.uuid4().hex[:8]}"
    
    # 解析 payload 中的 meeting_id
    try:
        import json
        payload_data = json.loads(payload)
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"解析 payload 失敗: {e}")
        payload_data = {}
        
    try:
        connection_state = await connection_manager.connect(
            websocket, connection_id, payload_data=payload_data
        )

        logger.info(f"🔗 WebSocket 連線已建立: {connection_id}, meeting_id: {meeting_id}")
        await websocket.accept()
        # 訊息處理循環
        while True:
            # 接收訊息（文字或二進位）
            try:
                # 等待訊息
                message = await websocket.receive()

                if "text" in message:
                    # 處理文字訊息（控制指令）
                    await connection_manager.handle_message(
                        connection_id, message["text"]
                    )
                    # 目前不支援僅回覆默認訊息

                elif "bytes" in message:
                    # print(
                    #     f"Received binary data of length {len(message['bytes'])} from {connection_id}"
                    # )
                    # 處理二進位訊息（音訊資料）
                    await connection_manager.handle_binary_data(
                        connection_id, message["bytes"]
                    )

            except WebSocketDisconnect:
                logger.info(f"🔌 WebSocket 連線斷開: {connection_id}")
                break
            except Exception as e:
                logger.error(f"❌ WebSocket 訊息處理錯誤: {connection_id}, {str(e)}")
                break

    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket 連線在認證階段斷開: {connection_id}")
    except Exception as e:
        logger.error(f"❌ WebSocket 連線錯誤: {connection_id}, {str(e)}")
        try:
            await websocket.close(code=1011, reason=f"Server error: {str(e)}")
        except Exception:
            pass

    finally:
        # 清理連線
        await connection_manager.disconnect(connection_id)