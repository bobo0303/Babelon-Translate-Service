
import os
import uuid
import random
import numpy as np
import soundfile as sf
from datetime import datetime

from api.vad_manager import VADProcessors
from api.websocket_stt_manager import WebSocketSttManager
from lib.constant import SAMPLERATE, NO_SPEECH_DURATION_THRESHOLD, BATCH_SIZE, MAX_DURATION

class AudioProcessor:
    def __init__(self, logger, payload_data: dict, connection, connection_id):
        self.samplerate = SAMPLERATE
        self.logger = logger
        
        self.save_file = True 
        self.recording_data = []
        self.is_speech = False
        self.last_speech_time = None
        self.batch_size = BATCH_SIZE
        self.max_duration = MAX_DURATION
        self.batch_list = []
        self.audio_uid = ""
        self.output_directory = f"audio/{payload_data.get('meeting_id', 'default_meeting_id')}/"

        # 添加緩衝區設定
        self.pre_buffer = []  # 存放語音開始前的音檔片段
        self.pre_buffer_size = 5  # 緩衝區大小（5個chunk）
        
        self.no_speech_duration_threshold = NO_SPEECH_DURATION_THRESHOLD
        self.vad_processor = VADProcessors(self.logger)
        self.stt_processor = WebSocketSttManager(self.logger, payload_data, connection, connection_id)

    async def preprocess_chunk(self, audio_bytes: bytes):
        """處理音訊資料塊."""

        if len(audio_bytes) > 1:
            audio_np_array = np.frombuffer(audio_bytes, dtype=np.float32)

            self._recording_audio_stream_block(audio_np_array)

        return len(audio_bytes), "success"

    def _recording_audio_stream_block(self, audio_stream_block: np.ndarray):
        frame_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        self.logger.debug(
            f"new sound stream block, audio length:{round(len(audio_stream_block)/self.samplerate ,3)} sec, frame_timestamp:{frame_timestamp}"
        )
        
        try:
            audio_data = (audio_stream_block[:]).copy()
            
            is_speaking = self._handle_speech_detection(audio_data)
            
            self._process_recording_data(audio_data, is_speaking, frame_timestamp)
            
        except Exception as e:
            self.logger.error(
                f"recording_audio_stream_block exception, frame_timestamp:{frame_timestamp}, error:{e}"
            )
            
    def _handle_speech_detection(self, audio_data: np.array) -> bool:
        is_speaking = self.vad_processor.is_speech(
            audio_data, samplerate=self.samplerate
        )
        return is_speaking
    
    def _process_recording_data(self, audio_data: np.array, is_speaking: bool, frame_timestamp: str):
        """
        is_speaking : 用來判斷是否為語音  = True | False
        self.speech : 用來判斷語音的開始與結束  = True | False
        """
        # 如果還沒開始語音，先存到緩衝區
        if not self.is_speech:
            # 維護緩衝區大小
            self.pre_buffer.append(audio_data.copy())
            if len(self.pre_buffer) > self.pre_buffer_size:
                self.pre_buffer.pop(0)  # 移除最舊的
 
        # 計算目前錄音資料的長度 (duration=1，約0.25秒)
        duration = int(((len(self.recording_data) * 4) / self.samplerate))

        # 語音開始
        if is_speaking:
            # 第一次檢測到語音時，將緩衝區的音檔加到錄音資料前面
            if not self.is_speech:
                # 將緩衝區的所有音檔合併到 recording_data 開頭
                for buffered_chunk in self.pre_buffer:
                    self.recording_data.extend(buffered_chunk)
                    
            self.is_speech = True
            self.last_speech_time = None
            self.recording_data.extend(audio_data)
                

        if (
            self.is_speech
            and duration >= self.batch_size
            and duration < self.max_duration
            and duration % self.batch_size == 0
            and duration not in self.batch_list
        ):
            # 超過batch_size，透過小batch 來提升UI的即時性
            self.logger.info(
                f"length: {len(self.recording_data)}, 到達 batch_size:{self.batch_list}, frame_timestamp:{frame_timestamp}"
            )

            if self.audio_uid == "":
                self.audio_uid = self._generate_uid()
                self.vad_processor.create_silero_vad_step(
                    self.audio_uid, 
                    self.recording_data, 
                    frame_timestamp, 
                    audio_tags="audio_start",
                    callback=self._save_recording_data
                )

            else:
                self.vad_processor.create_silero_vad_step(
                    self.audio_uid, 
                    self.recording_data, 
                    frame_timestamp,
                    callback=self._save_recording_data
                )

            self.batch_list.append(duration)
            self.batch_size = 12

        elif self.is_speech and duration >= self.max_duration:
            # 超過最大錄音時間
            self.logger.info(f" | 超過最大錄音時間, frame_timestamp:{frame_timestamp} | ")
            self.vad_processor.create_silero_vad_step(
                self.audio_uid, 
                self.recording_data, 
                frame_timestamp, 
                audio_tags="audio_end",
                callback=self._save_recording_data
            )
            self._clear_pre_buffer()
            self.recording_data = self.recording_data[-self.samplerate :]
            self.batch_list = []
            self.audio_uid = ""

        elif not is_speaking and self.is_speech:
            # 持續 no_speech_duration_threshold 秒沒有語音就視為斷句
            if self.last_speech_time:
                time_diff = (datetime.now() - self.last_speech_time).total_seconds()
                if time_diff < self.no_speech_duration_threshold:
                    if self.audio_uid == "":
                        self.recording_data.extend(audio_data)
                    return
                self.last_speech_time = None
            else:
                self.last_speech_time = datetime.now()
                if self.audio_uid == "":
                   self.recording_data.extend(audio_data)
                return

            # 語音結束
            if self.audio_uid == "":
                self.logger.warning(f" | audio lower limit not over 0.5s, skip this audio | frame_timestamp:{frame_timestamp} | ")
            else:
                self.logger.info(f" | Silent over limit time | batch_size:{self.batch_list} | frame_timestamp:{frame_timestamp} | ")
                self.vad_processor.create_silero_vad_step(
                    self.audio_uid, 
                    self.recording_data, 
                    frame_timestamp, 
                    audio_tags="audio_end",
                    callback=self._save_recording_data
                )
            self.is_speech = False
            self._clear_recording_data()
            self._clear_pre_buffer()
            self.batch_list = []
            self.audio_uid = ""
            self.batch_size = 2

    
    def _save_recording_data(
        self,
        recording_data,
        audio_uid: str,
        frame_timestamp: str,
        audio_tags: str,
    ):
        recording_data = np.copy(recording_data)
        # 保存錄音檔案
        if self.save_file:
            self.logger.debug(
                f"before save file, audio_uid:{audio_uid}, frame_timestamp:{frame_timestamp}"
            )
            self._save_audio_file(recording_data, audio_uid, frame_timestamp, audio_tags)
            self.logger.debug(
                f"after save file, audio_uid:{audio_uid}, frame_timestamp:{frame_timestamp}"
            )

    def _save_audio_file(self, recording_np, audio_uid: str, frame_timestamp: str, audio_tags: str):
        # 生成文件名
        filename = (
            f"{audio_uid}_{frame_timestamp.replace(':', ';').replace(' ', '_')}.wav"
        )

        self.logger.debug(" | 保存文件開始 | ")
        # 檢查並創建輸出目錄
        os.makedirs(self.output_directory, exist_ok=True)

        # 完整文件路徑
        full_path = os.path.join(self.output_directory, filename)

        # 確保 recording_np 是正確的 numpy array 格式
        if isinstance(recording_np, list):
            # 如果是 list，轉換為 numpy array
            audio_array = np.array(recording_np, dtype=np.float32)
        else:
            # 如果已經是 numpy array，確保格式正確
            audio_array = np.array(recording_np, dtype=np.float32)
        
        self.logger.debug(f" | 音頻數據格式: type={type(audio_array)}, shape={audio_array.shape}, dtype={audio_array.dtype} | ")

        # 保存文件
        sf.write(full_path, audio_array, SAMPLERATE)
        self.logger.debug(f" | Saved file: {full_path} | ")        
        self.logger.debug(" | 保存文件結束 | ")

        self.stt_processor.send_to_stt(audio_array, audio_uid, frame_timestamp, audio_tags=audio_tags)

    def _generate_uid(self):
        return str(
            uuid.uuid3(
                uuid.NAMESPACE_DNS,
                str(datetime.now().timestamp()) + str(random.random()),
            )
        ).replace("-", "")
        
    def _clear_recording_data(self):
        # self.logger.debug(f" | 清除錄音資料 | ")
        self.recording_data.clear()
        
    def _clear_pre_buffer(self):
        self.pre_buffer.clear()