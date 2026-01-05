import webrtcvad
import silero_vad
import silero_vad.utils_vad
import threading
import numpy as np

from lib.config.constant import SAMPLERATE, FRAME_DURATION

class WebrtcVAD:
    def __init__(
        self,
        logger,
        mode=1,
    ):
        self.vad = webrtcvad.Vad(mode)
        self.logger = logger

    def is_speech(self, audio_data, samplerate, frame_duration=0.03):
        # Ensure audio_data is exactly frame_samples long
        frame_samples = int(samplerate * frame_duration)
        if len(audio_data) != frame_samples:
            self.logger.warning(f"WebrtcVAD: Expected {frame_samples} samples, got {len(audio_data)}")
            # Pad or trim to exact size
            if len(audio_data) < frame_samples:
                audio_data = np.pad(audio_data, (0, frame_samples - len(audio_data)), mode='constant')
            else:
                audio_data = audio_data[:frame_samples]
        
        # Convert float32 [-1.0, 1.0] to int16 [-32768, 32767]
        # Clip to prevent overflow
        frame = np.clip(audio_data, -1.0, 1.0)
        frame = (frame * 32767).astype(np.int16)
        
        return self.vad.is_speech(frame.tobytes(), sample_rate=samplerate)
    
class SileroVAD:
    def __init__(self, logger):
        self.logger = logger
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

    def create_silero_vad_step(self, audio_uid, audio_data: np.array, frame_timestamp, audio_tags="", callback=None):

        silero_vad_thread = threading.Thread(
            target=self._silero_vad_step,
            kwargs={
                "audio_uid": audio_uid,
                "audio_data": np.copy(audio_data),
                "frame_timestamp": frame_timestamp,
                "audio_tags": audio_tags,
                "callback": callback,
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
        callback=None,
    ):
        try:
            with self.silero_vad_lock:
                speech_timestamps = silero_vad.get_speech_timestamps(
                    audio_data, self.silero_vad_model
                )

            # 如果 silero 判斷有人聲 speech_timestamps 就有值
            if len(speech_timestamps):
                self.logger.debug(
                    f" | after silero vad, --- is voice ---, audio_uid:{audio_uid}, frame_timestamp:{frame_timestamp} | "
                )
                
                # 使用回調函數來處理儲存
                if callback:
                    callback(audio_data, audio_uid, frame_timestamp, audio_tags)
                else:
                    self.logger.warning(f" | No callback provided for audio_uid:{audio_uid} | ")
            else:
                self.logger.debug(
                    f" | after silero vad, --- not voice ---, audio_uid:{audio_uid}, frame_timestamp:{frame_timestamp} | "
                )
                self.logger.info(
                    f" | Silero VAD detect no voice (skip to stt), audio_uid:{audio_uid}, frame_timestamp:{frame_timestamp} | "
                )
        except Exception as e:
            self.logger.error(
                f" | silero_vad_step exception, audio_uid:{audio_uid}, frame_timestamp:{frame_timestamp}, error:{e} | "
            )
    
    
class VADProcessors:
    def __init__(self, logger, mode=3):
        self.logger = logger
        self.webrtc_vad = WebrtcVAD(self.logger, mode=mode)
        self.silero_vad = SileroVAD(self.logger)
        
    def is_speech(self, audio_data, samplerate=SAMPLERATE, frame_duration=FRAME_DURATION):
        return self.webrtc_vad.is_speech(audio_data, samplerate, frame_duration)

    def create_silero_vad_step(self, audio_uid, audio_data: np.array, frame_timestamp, audio_tags="", callback=None):
        return self.silero_vad.create_silero_vad_step(audio_uid, audio_data, frame_timestamp, audio_tags, callback)