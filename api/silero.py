import sys
import os

# 添加项目根目录到 Python 路径，确保可以找到 vad 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import silero_vad
import silero_vad.utils_vad
import threading
import numpy as np
from datetime import datetime

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

    def create_silero_vad_step(self, frame_timestamp):

        silero_vad_thread = threading.Thread(
            target=self._silero_vad_step,
            kwargs={
                "audio_data": np.copy(self.recording_data),
                "frame_timestamp": frame_timestamp,
            },
            daemon=True,
        )
        silero_vad_thread.start()
        
    def _silero_vad_step(
            self,
            audio_data: np.array,
            frame_timestamp: str,   # datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        ):
            try:
                # self.logger.info(
                #     f"before silero vad, audio_uid:{audio_uid}, audio length:{round(len(audio_data)/self.samplerate ,1)}, frame_timestamp:{frame_timestamp}"
                # )
                # self.logger.info(
                #     f"before silero vad, time:{datetime.now().strftime('%H:%M:%S.%f')}"
                # )
                with self.silero_vad_lock:
                    speech_timestamps = silero_vad.get_speech_timestamps(
                        audio_data, self.silero_vad_model
                    )

                # self.logger.info(
                #     f"after silero vad, time:{datetime.now().strftime('%H:%M:%S.%f')}"
                # )

                # if not len(speech_timestamps) and self.has_voice_audio_dict.get(audio_uid):
                #     self.logger.info(
                #         f"wrong test, audio_uid:{audio_uid}, frame_timestamp:{frame_timestamp}"
                #     )

                # 如果 silero 判斷有人聲 speech_timestamps 就有值
                if len(speech_timestamps):
                    logger.info(
                        f"after silero vad, --- is voice ---, frame_timestamp:{frame_timestamp}"
                    )
                    return True
                else:
                    logger.info(
                        f"after silero vad, --- not voice ---, frame_timestamp:{frame_timestamp}"
                    )
                    return False
            except Exception as e:
                logger.error(
                    f"silero_vad_step exception, frame_timestamp:{frame_timestamp}, error:{e}"
                )
                return False


if __name__ == "__main__":
    # 簡單測試 SileroVAD
    import time
    import wave
    import librosa
    
    print("Initializing SileroVAD...")
    vad = SileroVAD()
    print("SileroVAD initialized successfully!")
    
    # 測試真實音頻文件
    audio_file_path = "/mnt/audio/test.wav"
    
    try:
        print(f"\nLoading audio file: {audio_file_path}")
        
        # 使用 librosa 讀取音頻文件
        audio_data, sample_rate = librosa.load(audio_file_path, sr=16000)  # 重採樣到 16kHz
        audio_data = audio_data.astype(np.float32)
        
        print(f"Audio loaded successfully!")
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Duration: {len(audio_data) / sample_rate:.2f} seconds")
        print(f"Audio shape: {audio_data.shape}")
        print(f"Audio min/max: {audio_data.min():.4f} / {audio_data.max():.4f}")
        
        # 設置必要的屬性進行測試
        vad.recording_data = audio_data
        
        print("\nTesting with real audio file...")
        result = vad._silero_vad_step(audio_data, datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
        print(f"Real audio file result: {result}")
        
    except Exception as e:
        print(f"Error loading audio file: {e}")
        print("Falling back to synthetic audio tests...")
        
        # 生成測試音頻數據 (16kHz, 1秒)
        samplerate = 16000
        duration = 1.0
        samples = int(samplerate * duration)
        
        # 模擬靜音音頻
        silent_audio = np.zeros(samples, dtype=np.float32)
        
        # 模擬語音音頻 (混合頻率的正弦波)
        t = np.linspace(0, duration, samples)
        speech_audio = (0.1 * np.sin(2 * np.pi * 300 * t) + 
                       0.1 * np.sin(2 * np.pi * 800 * t) + 
                       0.05 * np.sin(2 * np.pi * 1200 * t))
        
        # 模擬雜訊音頻
        noise_audio = np.random.normal(0, 0.02, samples).astype(np.float32)
        
        # 設置必要的屬性進行測試
        vad.recording_data = silent_audio
        
        print("\nTesting with silent audio...")
        result = vad._silero_vad_step(silent_audio, datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
        print(f"Silent audio result: {result}")
        
        print("\nTesting with speech-like audio...")
        result = vad._silero_vad_step(speech_audio, datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
        print(f"Speech-like audio result: {result}")
        
        print("\nTesting with noise audio...")
        result = vad._silero_vad_step(noise_audio, datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
        print(f"Noise audio result: {result}")
    
    print("\nSileroVAD test completed!")