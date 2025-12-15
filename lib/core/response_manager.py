

import json
import asyncio
import threading

from lib.config.constant import AudioTranslationResponse
from wjy3 import BaseResponse, Status
from lib.storage.meeting_record import MeetingRecord, MeetingRecordSql
from lib.storage.azure_blob_service import get_azure_blob_service

def storage_upload(logger, response_data: AudioTranslationResponse, other_info: dict):
    """
    處理儲存和上傳相關的任務
    """
    
    # try:
    #     _save_trans_history_to_db(logger, response_data, other_info)
    # except Exception as e:
    #     logger.error(f"| 儲存轉錄歷史到資料庫錯誤: {e} | ")
    
    # try:
    #     _upload_audio_to_azure_blob(logger, response_data.meeting_id, other_info.get("audio_file_name", ""))
    # except Exception as e:
    #     logger.error(f"| 上傳音訊檔案到 Azure Blob 錯誤: {e} | ")
        
    # save transcription history to db in background thread
    db_thread = threading.Thread(
        target=_save_trans_history_to_db,
        args=(logger, response_data, other_info),
        daemon=True
    )
    db_thread.start()
    
    # upload audio to azure blob in background thread
    upload_thread = threading.Thread(
        target=_upload_audio_to_azure_blob,
        args=(logger, response_data.meeting_id, other_info.get("audio_file_name", "")),
        daemon=True
    )
    upload_thread.start()

def process_stt_response(logger, response_data: AudioTranslationResponse, other_info: dict):
    connections = other_info.get("connection")  # 這是 Dict[str, WebSocket]
    connection_id = other_info.get("connection_id")  # 這是 str
    
    # response stt result to admin page
    # TODO: implement admin if needed 
    
    # response stt result to websocket
    if connections and connection_id:
        websocket = connections.get(connection_id)
        if websocket:
            _response_websocket(logger, response_data, websocket)
        else:
            logger.warning(f" | WebSocket connection not found for {connection_id} | ")
    
    # save to db and upload to blob using unified function
    storage_upload(logger, response_data, other_info)

def _response_websocket(logger, response_data: AudioTranslationResponse, websocket):
    """回應 WebSocket 的函數"""
    state = Status.OK if response_data.transcription_text else Status.FAILED

    try:
        return_info = BaseResponse(
            status=state,
            message=f" | Transcription: {response_data.transcription_text} | ZH: {response_data.text.get('zh')} | EN: {response_data.text.get('en')} | DE: {response_data.text.get('de')} | ",
            data=response_data
        )
        
        asyncio.run(websocket.send_text(json.dumps(return_info.model_dump())))
        logger.debug(f" | Sent STT result to websocket: {return_info.model_dump_json()} | ")
        
    except Exception as e:
        logger.error(f"| 發送 STT 結果錯誤: {e} | ")



def _save_trans_history_to_db(
    logger,
    response_data: AudioTranslationResponse, 
    other_info: dict,
):
    """
    儲存轉錄歷史至資料庫
    """
    logger.debug(
        f" | before save to db,\naudio_uid: {response_data.audio_uid}, \ntimestamp:{response_data.times}  | \n"
    )

    # 解析時間戳記
    from datetime import datetime
    try:
        # 假設 response_data.times 是字串格式的時間戳記
        audio_frame_timestamp = datetime.fromisoformat(response_data.times.replace('Z', '+00:00')) if 'T' in response_data.times else datetime.now()
    except:
        audio_frame_timestamp = datetime.now()
    
    # 建構 STT 專用資訊
    stt_data_dict = {
        "use_translate": other_info.get("use_translate", False),
        "use_prev_text": other_info.get("use_prev_text", False),
        "post_processing": other_info.get("post_processing", False),
        "process_method": other_info.get("process_method", "unknown")
    }
    
    stt_data = json.dumps(stt_data_dict, ensure_ascii=False)

    MeetingRecordSql().create(
        MeetingRecord(
            meeting_id=response_data.meeting_id,
            device_id=response_data.device_id,
            task_id=other_info.get("task_id", ""),
            audio_id=response_data.audio_uid,
            audio_frame_timestamp=audio_frame_timestamp,
            audio_file_name=other_info.get("audio_file_name") or "unknown.wav",
            source_lang=response_data.ori_lang,
            transcription_text=response_data.transcription_text,
            translation=json.dumps(response_data.text, ensure_ascii=False),
            transcribe_time=response_data.transcribe_time,
            translate_time=response_data.translate_time,
            audio_length=other_info.get("audio_length", 0.0),
            rtf=other_info.get("rtf", 1.0),
            audio_tags=other_info.get("audio_tags", ""),
            strategy=other_info.get("strategy", "unknown"),
            prev_text=other_info.get("prev_text", ""),
            stt_data=stt_data,
        )
    )

    logger.debug(
        f' | after save to db | audio UID: {response_data.audio_uid} | timestamp: {response_data.times} | '
    )
    
    logger.debug(
        f' | Save to db | audio UID: {response_data.audio_uid} | timestamp:{response_data.times} | ')
    
def _upload_audio_to_azure_blob(logger, meeting_id: str, audio_file_name: str):
    """
    上傳音訊檔案到 Azure Blob Storage
    """
    try:
        local_file_path = f"audio/{meeting_id}/{audio_file_name}"

        logger.debug(
            f" | Uploading audio file to Azure Blob | audio file name: {audio_file_name} | meeting ID: {meeting_id} | "
        )

        # 取得 Azure Blob Service 實例
        azure_blob_service = get_azure_blob_service()

        # 上傳檔案到 Azure Blob Storage
        blob_url = azure_blob_service.upload_file(
            local_file_path=local_file_path,
            blob_name=audio_file_name,
            meeting_id=meeting_id,
        )
        
        if blob_url:
            logger.debug(f" | Successfully uploaded audio file to Azure Blob | audio file name: {audio_file_name} | meeting ID: {meeting_id} | blob URL: {blob_url} | ")
        else:
            logger.warning(
                f" | Failed to upload audio file to Azure Blob | audio file name: {audio_file_name} | meeting ID: {meeting_id} | "
            )

    except Exception as e:
        logger.error(f" | Error uploading audio file to Azure Blob: {e} | ")

