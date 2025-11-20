# use webrtcvad and silero vad to detect speech in audio data

import uuid
from typing import Optional

from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
    Query,
)

from api.websocket.websocket_manager import ConnectionManager
from lib.core.logging_config import get_configured_logger

# 獲取配置好的日誌器
logger = get_configured_logger(__name__)

router = APIRouter()
connection_manager = ConnectionManager(logger)

def set_model(model):
    """設置 model 到 connection_manager"""
    connection_manager.model = model

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
    
    try:
        import json
        payload_data = json.loads(payload)
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f" | 解析 payload 失敗: {e} | ")
        payload_data = {}
        
    try:
        connection_state = await connection_manager.connect(
            websocket, connection_id, payload_data=payload_data
        )

        # 訊息處理循環
        while True:
            # 接收訊息（文字或二進位）
            try:
                # 等待訊息
                message = await websocket.receive()

                # 檢查是否收到斷線訊息
                if message.get("type") == "websocket.disconnect":
                    logger.info(f" | 🔌 WebSocket 收到斷線訊息: {connection_id} | ")
                    break

                if "text" in message:
                    # 處理文字訊息（控制指令）
                    await connection_manager.handle_message(
                        connection_id, message["text"], websocket
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
                logger.info(f" | 🔌 WebSocket 連線斷開: {connection_id} | ")
                break
            except RuntimeError as e:
                if "Cannot call" in str(e) and "disconnect message" in str(e):
                    logger.info(f" | 🔌 WebSocket 已斷線，停止接收訊息: {connection_id} | ")
                    break
                else:
                    logger.error(f" | ❌ WebSocket 運行時錯誤: {connection_id}, {str(e)} | ")
                    break
            except Exception as e:
                logger.error(f" | ❌ WebSocket 訊息處理錯誤: {connection_id}, {str(e)} | ")
                break

    except WebSocketDisconnect:
        logger.info(f" | 🔌 WebSocket 連線在認證階段斷開: {connection_id} | ")
    except Exception as e:
        logger.error(f" | ❌ WebSocket 連線錯誤: {connection_id}, {str(e)} | ")
        try:
            await websocket.close(code=1011, reason=f"Server error: {str(e)}")
        except Exception:
            pass

    finally:
        # 清理連線
        await connection_manager.disconnect(connection_id)