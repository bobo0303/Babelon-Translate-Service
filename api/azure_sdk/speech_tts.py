"""
Azure Speech TTS (Text-to-Speech)

使用 Azure Speech SDK (WebSocket v2 + TextStream) 將文字合成語音，
支援多語言並行合成，記錄 first byte latency 與 total synthesis time。
"""
import os
import sys
import time
import base64
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import azure.cognitiveservices.speech as speechsdk

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.config.constant import SPEECH_CONFIG, LANG_TO_AZURE_LOCALE
from lib.core.logging_config import get_logger

logger = get_logger(__name__)

# Azure TTS voice mapping (language -> voice name)
LANG_TO_VOICE = {
    "zh": "zh-TW-HsiaoChenNeural",
    "en": "en-US-AvaMultilingualNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "de": "de-DE-KatjaNeural",
}


@dataclass
class TTSResult:
    """單一語言的 TTS 合成結果"""
    language: str
    audio_base64: str = ""           # base64 encoded audio (wav)
    audio_duration: float = 0.0      # 音檔播放時長 (秒)
    first_byte_latency: float = 0.0  # 從發送請求到收到第一個 audio byte 的時間 (秒)
    synthesis_time: float = 0.0      # 完整合成時間 (秒)
    text_length: int = 0             # 合成文字長度
    success: bool = False
    error: str = ""


@dataclass
class TTSBatchResult:
    """多語言 TTS 批次結果"""
    results: Dict[str, TTSResult] = field(default_factory=dict)
    total_time: float = 0.0          # 全部語言合成的總時間 (wall clock)


class AzureSpeechTTS:
    def __init__(self):
        self._azure_config = SPEECH_CONFIG
        self._region = self._azure_config["ServiceRegion"]
        self._key = self._azure_config["SubscriptionKey"]
    
    def _create_speech_config(self, voice: str) -> speechsdk.SpeechConfig:
        """建立使用 WebSocket v2 endpoint 的 SpeechConfig"""
        endpoint = f"wss://{self._region}.tts.speech.microsoft.com/cognitiveservices/websocket/v2"
        speech_config = speechsdk.SpeechConfig(endpoint=endpoint, subscription=self._key)
        speech_config.speech_synthesis_voice_name = voice
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
        )
        speech_config.set_property(speechsdk.PropertyId.SpeechSynthesis_FrameTimeoutInterval, "100000000")
        speech_config.set_property(speechsdk.PropertyId.SpeechSynthesis_RtfTimeoutThreshold, "10")
        return speech_config
    
    def synthesize(self, text: str, language: str) -> TTSResult:
        """
        合成單一語言的 TTS 音檔 (WebSocket v2 + TextStream)。
        
        Args:
            text: 要合成的文字
            language: 語言代碼 (zh, en, ja, ko, de)
        
        Returns:
            TTSResult: 包含 base64 音檔和延遲數據
        """
        result = TTSResult(language=language, text_length=len(text))
        
        if not text or not text.strip():
            result.error = "Empty text"
            return result
        
        voice = LANG_TO_VOICE.get(language)
        if not voice:
            result.error = f"Unsupported language: {language}"
            return result
        
        speech_config = self._create_speech_config(voice)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
        
        first_byte_time = None
        
        def on_synthesizing(evt):
            nonlocal first_byte_time
            if first_byte_time is None and evt.result.audio_data:
                first_byte_time = time.time()
        
        synthesizer.synthesizing.connect(on_synthesizing)
        
        try:
            # 使用 TextStream 方式
            tts_request = speechsdk.SpeechSynthesisRequest(
                input_type=speechsdk.SpeechSynthesisRequestInputType.TextStream
            )
            
            start_time = time.time()
            tts_task = synthesizer.speak_async(tts_request)
            
            # 一次餵入全部文字
            tts_request.input_stream.write(text)
            tts_request.input_stream.close()
            
            # 等待合成完成
            synthesis_result = tts_task.get()
            end_time = time.time()
            
            if synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                audio_data = synthesis_result.audio_data
                result.audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                result.synthesis_time = end_time - start_time
                result.first_byte_latency = (first_byte_time - start_time) if first_byte_time else result.synthesis_time
                result.audio_duration = (len(audio_data) - 44) / (24000 * 2)  # 24kHz 16bit mono, minus WAV header
                result.success = True
                
                logger.info(
                    f" | TTS [{language}] | voice: {voice} | "
                    f"first_byte: {result.first_byte_latency:.3f}s | "
                    f"total: {result.synthesis_time:.3f}s | "
                    f"audio_dur: {result.audio_duration:.2f}s | "
                    f"text_len: {result.text_length} | "
                )
            elif synthesis_result.reason == speechsdk.ResultReason.Canceled:
                cancellation = synthesis_result.cancellation_details
                result.error = f"Canceled: {cancellation.reason} - {cancellation.error_details}"
                result.synthesis_time = end_time - start_time
                logger.error(f" | TTS [{language}] canceled: {result.error} | ")
            else:
                result.error = f"Unexpected reason: {synthesis_result.reason}"
                result.synthesis_time = end_time - start_time
                logger.error(f" | TTS [{language}] unexpected: {result.error} | ")
                
        except Exception as e:
            result.error = str(e)
            result.synthesis_time = time.time() - start_time
            logger.error(f" | TTS [{language}] error: {e} | ")
        finally:
            synthesizer.synthesizing.disconnect_all()
        
        return result
    
    def synthesize_batch(self, texts: Dict[str, str]) -> TTSBatchResult:
        """
        對多個語言並行做 TTS 合成。
        
        Args:
            texts: {language: text} 字典，例如 {"zh": "你好", "en": "Hello"}
        
        Returns:
            TTSBatchResult: 所有語言的合成結果
        """
        batch_result = TTSBatchResult()
        batch_start = time.time()
        
        # 過濾空文字
        valid_texts = {lang: text for lang, text in texts.items() 
                       if text and text.strip() and lang in LANG_TO_VOICE}
        
        if not valid_texts:
            logger.warning(" | TTS batch: no valid texts to synthesize | ")
            return batch_result
        
        threads: List[threading.Thread] = []
        results_lock = threading.Lock()
        
        def _synth_worker(lang: str, text: str):
            tts_result = self.synthesize(text, lang)
            with results_lock:
                batch_result.results[lang] = tts_result
        
        # 並行合成所有語言
        for lang, text in valid_texts.items():
            t = threading.Thread(target=_synth_worker, args=(lang, text))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        batch_result.total_time = time.time() - batch_start
        
        # 記錄批次統計
        success_count = sum(1 for r in batch_result.results.values() if r.success)
        logger.info(
            f" | TTS batch complete | "
            f"languages: {list(valid_texts.keys())} | "
            f"success: {success_count}/{len(valid_texts)} | "
            f"total_time: {batch_result.total_time:.3f}s | "
        )
        
        return batch_result
