import webrtcvad
import silero_vad
import silero_vad.utils_vad
import threading
import numpy as np

from lib.constant import SAMPLERATE, FRAME_DURATION

import logging
import logging.handlers

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

class WebrtcVAD:
    def __init__(
        self,
        mode=1,
    ):
        self.vad = webrtcvad.Vad(mode)

    def is_speech(self, audio_data, samplerate, frame_duration=0.03):
        frame_samples = int(samplerate * frame_duration)
        frame = (audio_data[-frame_samples:] * np.iinfo(np.int16).max).astype(np.int16)
        return self.vad.is_speech(frame.tobytes(), sample_rate=samplerate)
    
class SileroVAD:
    def __init__(self):
        model_name = "vad.jit"
        package_path = "vad.data"

        from importlib import resources as impresources

        try:
            # Use the new recommended API
            model_file_path = str(impresources.files(package_path).joinpath(model_name))
        except:
            # Fallback to deprecated API if needed
            with impresources.path(package_path, model_name) as f:
                model_file_path = f

        self.silero_vad_model = silero_vad.utils_vad.init_jit_model(model_file_path)
        self.silero_vad_lock = threading.Lock()

    def create_silero_vad_step(self, audio_uid, audio_data: np.array, frame_timestamp, audio_tags=""):

        silero_vad_thread = threading.Thread(
            target=self._silero_vad_step,
            kwargs={
                "audio_uid": audio_uid,
                "audio_data": np.copy(audio_data),
                "frame_timestamp": frame_timestamp,
                "audio_tags": audio_tags,
            },
            daemon=True,
        )
        silero_vad_thread.start()

    def _silero_vad_step(
        self,
        audio_uid: str,
        audio_data: np.array,
        frame_timestamp: str,
        audio_tags: str,
    ):
        try:
            with self.silero_vad_lock:
                speech_timestamps = silero_vad.get_speech_timestamps(
                    audio_data, self.silero_vad_model
                )

            # 如果 silero 判斷有人聲 speech_timestamps 就有值
            if len(speech_timestamps) or self.has_voice_audio_dict.get(audio_uid):
                self.logger.info(
                    f"after silero vad, --- is voice ---, audio_uid:{audio_uid}, frame_timestamp:{frame_timestamp}"
                )
                self.has_voice_audio_dict[audio_uid] = True
                self._save_recording_data(
                    audio_data, audio_uid, frame_timestamp, audio_tags
                )
            else:
                self.logger.info(
                    f"after silero vad, --- not voice ---, audio_uid:{audio_uid}, frame_timestamp:{frame_timestamp}"
                )
        except Exception as e:
            self.logger.error(
                f"silero_vad_step exception, audio_uid:{audio_uid}, frame_timestamp:{frame_timestamp}, error:{e}"
            )
    
    def _save_recording_data(
        self,
        recording_data,
        audio_uid: str,
        frame_timestamp: str,
        audio_tags: str,
    ):
        # 保存錄音檔案
        if self.save_file:
            self.logger.info(
                f"before save file, audio_uid:{audio_uid}, frame_timestamp:{frame_timestamp}"
            )
            self._save_audio_file(recording_data, audio_uid, frame_timestamp)
            self.logger.info(
                f"after save file, audio_uid:{audio_uid}, frame_timestamp:{frame_timestamp}"
            )

        self.logger.info(
            f"before _invoke_callback, audio_uid:{audio_uid}, frame_timestamp:{frame_timestamp}"
        )
        # 回調處理
        self._invoke_callback(
            self.stt_callback,
            recording_data,
            audio_uid,
            frame_timestamp,
            audio_tags,
        )
        self.logger.info(
            f"after _invoke_callback, audio_uid:{audio_uid}, frame_timestamp:{frame_timestamp}"
        )

    
class VADProcessors:
    def __init__(self, mode=3):
        self.webrtc_vad = WebrtcVAD(mode=mode)
        self.silero_vad = SileroVAD()
        
    def is_speech(self, audio_data, samplerate=SAMPLERATE, frame_duration=FRAME_DURATION):
        return self.webrtc_vad.is_speech(audio_data, samplerate, frame_duration)

    def create_silero_vad_step(self, audio_uid, audio_data: np.array, frame_timestamp, audio_tags=""):
        return self.silero_vad.create_silero_vad_step(audio_uid, audio_data, frame_timestamp, audio_tags)