"""
Azure Speech Language Identification (LID)

使用 Continuous LID 模式偵測音檔語言，回傳出現最多的語言、耗時與信心度
"""
import os
import sys
import asyncio
import time
from collections import Counter
from dataclasses import dataclass
from typing import Optional
import yaml
import azure.cognitiveservices.speech as speechsdk

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.config.constant import AZURE_CONFIG, LANGUAGE_LIST, LANG_TO_AZURE_LOCALE
from lib.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class LIDResult:
    """語言偵測結果"""
    language: str                    # 出現最多的語言
    elapsed_time: float              # 處理耗時（秒）
    confidence: Optional[float]      # 平均信心度（若有）


class AzureSpeechLID:
    def __init__(self):
        # 載入設定
        with open(AZURE_CONFIG, 'r') as file:
            config = yaml.safe_load(file)
        
        self._azure_config = config['speech_models']['azure_speech']
        
        # 排除 'auto'，保留有效語言
        self.language_list = [lang for lang in LANGUAGE_LIST if lang != 'auto']
        
        # 轉換為 Azure locale 格式
        self._azure_locales = [
            LANG_TO_AZURE_LOCALE.get(lang, lang) for lang in self.language_list
        ]
        
        # 預先建立 SpeechConfig（可重用）
        self._speech_config = speechsdk.SpeechConfig(
            subscription=self._azure_config["SubscriptionKey"],
            region=self._azure_config["ServiceRegion"]
        )
        
        # 檢查 SDK 支援的 LID 模式設定方式
        self._lid_mode_type = self._detect_lid_mode_support()
    
    @staticmethod
    def _normalize_language(azure_locale: str) -> str:
        """將 Azure locale（如 zh-TW）轉換為簡短標籤（如 zh）"""
        if not azure_locale:
            return "unknown"
        return azure_locale.split("-")[0].lower()
    
    def _detect_lid_mode_support(self) -> str:
        """檢測 SDK 支援的 LanguageIdMode 設定方式"""
        if hasattr(speechsdk, 'LanguageIdMode'):
            return 'root'
        elif hasattr(speechsdk, 'languageconfig') and hasattr(speechsdk.languageconfig, 'LanguageIdMode'):
            return 'languageconfig'
        else:
            return 'property'
    
    def _create_auto_detect_config(self, languages: list) -> speechsdk.AutoDetectSourceLanguageConfig:
        """建立 AutoDetectSourceLanguageConfig"""
        if self._lid_mode_type == 'root':
            return speechsdk.AutoDetectSourceLanguageConfig(
                languages=languages,
                mode=speechsdk.LanguageIdMode.Continuous
            )
        elif self._lid_mode_type == 'languageconfig':
            return speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
                languages=languages,
                mode=speechsdk.languageconfig.LanguageIdMode.Continuous
            )
        else:
            # 使用 property 方式設定
            self._speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode, "Continuous"
            )
            return speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=languages)

    async def detect_language(self, audio_file_path: str, languages: list = None) -> LIDResult:
        """
        偵測音檔語言（Continuous LID 模式，非阻塞）
        
        Args:
            audio_file_path: 音檔路徑
            languages: 候選語言列表（Azure locale 格式），預設使用初始化的 _azure_locales
            
        Returns:
            LIDResult: 包含最常見語言、耗時、信心度等資訊
        """
        if languages is None:
            languages = self._azure_locales
        
        auto_detect_config = self._create_auto_detect_config(languages)
        audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
        
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self._speech_config,
            auto_detect_source_language_config=auto_detect_config,
            audio_config=audio_config
        )
        
        detected_segments = []
        loop = asyncio.get_event_loop()
        done_event = asyncio.Event()
        
        def on_recognized(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                lang = evt.result.properties.get(
                    speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
                )
                
                # 嘗試取得信心度
                confidence = 0.0
                try:
                    auto_detect_result = speechsdk.AutoDetectSourceLanguageResult(evt.result)
                    confidence = auto_detect_result.confidence if hasattr(auto_detect_result, 'confidence') else 0.0
                except Exception:
                    pass
                
                if lang:
                    detected_segments.append({
                        "language": lang,
                        "confidence": confidence
                    })
        
        def on_session_stopped(evt):
            loop.call_soon_threadsafe(done_event.set)
        
        def on_canceled(evt):
            if evt.result.reason == speechsdk.ResultReason.Canceled:
                cancellation = speechsdk.CancellationDetails.from_result(evt.result)
                logger.warning(f"Recognition canceled: {cancellation.reason}, {cancellation.error_details}")
            loop.call_soon_threadsafe(done_event.set)
        
        recognizer.recognized.connect(on_recognized)
        recognizer.session_stopped.connect(on_session_stopped)
        recognizer.canceled.connect(on_canceled)
        
        start_time = time.time()
        recognizer.start_continuous_recognition()
        
        # 非阻塞等待
        await done_event.wait()
        
        recognizer.stop_continuous_recognition()
        elapsed_time = time.time() - start_time
        
        return self._aggregate_results(detected_segments, elapsed_time)
    
    def _aggregate_results(self, segments: list, elapsed_time: float) -> LIDResult:
        """彙整偵測結果"""
        if not segments:
            return LIDResult(
                language="unknown",
                elapsed_time=elapsed_time,
                confidence=None,
            )
        
        # 先將所有語言標籤正規化為簡短格式
        for seg in segments:
            seg["language"] = self._normalize_language(seg["language"])
        
        # 統計各語言出現次數
        language_counts = Counter(seg["language"] for seg in segments)
        most_common_lang = language_counts.most_common(1)[0][0]
        
        # 計算該語言的平均信心度
        confidences = [
            seg["confidence"] for seg in segments 
            if seg["language"] == most_common_lang and seg["confidence"] is not None
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return LIDResult(
            language=most_common_lang,
            elapsed_time=elapsed_time,
            confidence=avg_confidence,
        )

