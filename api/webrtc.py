import webrtcvad
import numpy as np

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


""" 
samplerate = 16000
speech_detector_ins = SpeechDetector(mode=3)

def _handle_speech_detection(audio_data):
        is_speech = speech_detector_ins.is_speech(audio_data, samplerate)
        return is_speech
"""


if __name__ == "__main__":
    # 簡單測試 WebrtcVAD
    import time
    
    # 初始化 VAD
    vad = WebrtcVAD(mode=3)  # mode 0-3, 3 is most aggressive
    
    # 生成測試音頻數據 (16kHz, 30ms)
    samplerate = 16000
    frame_duration = 0.03
    frame_samples = int(samplerate * frame_duration)
    
    # 測試靜音音頻
    silent_audio = np.zeros(frame_samples, dtype=np.float32)
    print(f"Silent audio is_speech: {vad.is_speech(silent_audio, samplerate)}")
    
    # 測試有聲音的音頻 (模擬語音信號)
    t = np.linspace(0, frame_duration, frame_samples)
    speech_audio = 0.1 * np.sin(2 * np.pi * 1000 * t)  # 1kHz sine wave
    print(f"Speech-like audio is_speech: {vad.is_speech(speech_audio, samplerate)}")
    
    # 測試雜訊音頻
    noise_audio = np.random.normal(0, 0.01, frame_samples).astype(np.float32)
    print(f"Noise audio is_speech: {vad.is_speech(noise_audio, samplerate)}")
    
    print("WebrtcVAD test completed!")
    