"""
Benchmark Helper

提供給 main.py 使用的便捷函數，用於記錄 benchmark 數據
最小化對主程式的影響

使用方式:
    from api.core.benchmark_helper import record_pipeline_result
    
    # 在 translate_pipeline 返回前調用
    record_pipeline_result(
        audio_uid=audio_uid,
        times=times,
        start_time=request_start_time,
        result=result,
        other_info=other_info,
        response_data=response_data,
        is_cancelled=is_cancelled,
        cancel_type=cancel_type,
        is_final=(multi_strategy_transcription == 4)
    )
"""

import time
import logging
from typing import Optional, Any

from api.core.trim_benchmark_recorder import (
    get_benchmark_recorder,
    TrimBenchmarkRecorder
)

logger = logging.getLogger(__name__)


def record_pipeline_result(
    audio_uid: str,
    times: str,
    start_time: float,
    result: Optional[tuple] = None,
    other_info: Optional[dict] = None,
    response_data: Any = None,
    is_cancelled: bool = False,
    cancel_type: Optional[str] = None,
    is_final: bool = False
) -> None:
    """
    記錄 pipeline 結果到 benchmark recorder
    
    這個函數會檢查 recorder 是否啟用，如果未啟用則直接返回，不做任何事。
    這確保了對主程式的最小影響。
    
    Args:
        audio_uid: 音訊 UID
        times: 請求時間戳
        start_time: 請求開始時間 (time.time())
        result: pipeline 返回的結果 tuple
        other_info: 其他資訊 dict
        response_data: AudioTranslationResponse 對象
        is_cancelled: 是否被取消
        cancel_type: 取消類型 ("transcribe_cancel" or "translate_cancel")
        is_final: 是否為該 UID 的最後一筆（通常是 multi_strategy_transcription == 4）
    """
    recorder = get_benchmark_recorder()
    
    # Debug log
    print(f"[BENCHMARK_HELPER] Called: audio_uid={audio_uid}, recorder.enabled={recorder.is_enabled()}")
    logger.info(f" | [BENCHMARK_HELPER] Called: audio_uid={audio_uid}, recorder.enabled={recorder.is_enabled()} | ")
    
    # 如果 benchmark 未啟用，直接返回
    if not recorder.is_enabled():
        print(f"[BENCHMARK_HELPER] Skipped - not enabled")
        return
    
    try:
        # 計算響應時間
        response_time = time.time() - start_time
        
        print(f"[BENCHMARK_HELPER] Recording: response_time={response_time:.3f}s")
        
        # 從 result 和 other_info 提取資訊
        transcription_text = ""
        transcribe_time = 0.0
        translate_time = 0.0
        trim_duration = 0.0
        trim_updated = False
        stable_text = ""
        unstable_text = ""
        
        if result and result[0] is not None:
            # result = (ori_pred, n_segments, segments, translated_result, transcription_time, translate_time, translate_method, timing_dict)
            transcription_text = result[0] if result[0] else ""
            transcribe_time = result[4] if len(result) > 4 else 0.0
            translate_time = result[5] if len(result) > 5 else 0.0
        
        if other_info:
            trim_duration = other_info.get('trim_duration', 0.0)
            trim_updated = other_info.get('trim_updated', False)
            stable_text = other_info.get('stable_text', '')
            unstable_text = other_info.get('unstable_text', '')
            
            # 檢查是否因為 cancelled_by_times 而取消
            if 'cancelled_by_times' in other_info:
                is_cancelled = True
                cancel_type = "translate_cancel"  # 翻譯後被取消
        
        if response_data:
            # 如果有 response_data，優先使用
            if hasattr(response_data, 'transcription_text') and response_data.transcription_text:
                transcription_text = response_data.transcription_text
            if hasattr(response_data, 'transcribe_time'):
                transcribe_time = response_data.transcribe_time
            if hasattr(response_data, 'translate_time'):
                translate_time = response_data.translate_time
        
        # 記錄到 benchmark recorder
        recorder.record_request(
            audio_uid=audio_uid,
            times=times,
            response_time=response_time,
            is_cancelled=is_cancelled,
            cancel_type=cancel_type,
            transcription_text=transcription_text,
            is_final=is_final,
            transcribe_time=transcribe_time,
            translate_time=translate_time,
            trim_duration=trim_duration,
            trim_updated=trim_updated,
            stable_text=stable_text,
            unstable_text=unstable_text
        )
        
        # 如果是最後一筆，標記最終文本
        if is_final and transcription_text:
            recorder.mark_uid_final(audio_uid, transcription_text)
            
    except Exception as e:
        # 捕獲所有錯誤，確保不影響主程式
        logger.warning(f" | Benchmark recording error (ignored): {e} | ")


def is_benchmark_enabled() -> bool:
    """檢查 benchmark 是否正在記錄"""
    return get_benchmark_recorder().is_enabled()
