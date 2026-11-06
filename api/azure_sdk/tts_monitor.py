"""
TTS Pipeline 監控 Tracker

記錄每段 pipeline 各階段的「絕對時間起止點」，輸出 Gantt-style CSV。
X 軸是 session 開始後的真實時間 (秒)，每段在時間軸上顯示：
  收音(recording) → 轉譯(transcribe) → 翻譯(translate) → TTS

CSV 格式適合後續畫 Gantt / timeline 圖。
"""
import os
import csv
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from lib.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SegmentTiming:
    """單段 pipeline 的完整時間記錄。
    
    時間線是嚴格串接的:
    recording_start → recording_end(=request_arrival) → transcribe_end → translate_end → tts_end
    """
    segment_idx: int = 0
    audio_uid: str = ""
    times: str = ""                     # 前端傳來的 datetime 字串
    recording_start: float = 0.0        # 收音開始的 unix timestamp (前端 times)
    recording_end: float = 0.0          # 收音結束 = request 到達 (unix timestamp)
    transcribe_end: float = 0.0         # 轉錄結束 (unix timestamp)
    translate_end: float = 0.0          # 翻譯結束 (unix timestamp)
    tts_end: float = 0.0               # TTS 結束 (unix timestamp)
    input_audio_duration: float = 0.0   # 輸入音檔長度 (秒)
    transcribe_time: float = 0.0        # 轉錄耗時 (秒)
    translate_time: float = 0.0         # 翻譯耗時 (秒)
    tts_wall_time: float = 0.0          # TTS 整批 wall clock (秒)
    tts_per_lang: Dict[str, Dict] = field(default_factory=dict)
    max_tts_audio_duration: float = 0.0 # 最長語言的 TTS 輸出音檔長度 (秒)


class TTSMonitor:
    """TTS Pipeline 即時監控器，輸出 Gantt-style timeline CSV"""
    
    CSV_PATH = "./static/tts_monitor.csv"
    
    def __init__(self, max_segments: int = 500):
        self._lock = threading.Lock()
        self._segments: List[SegmentTiming] = []
        self._max_segments = max_segments
        self._session_start: Optional[float] = None
        self._segment_counter = 0
        os.makedirs(os.path.dirname(self.CSV_PATH), exist_ok=True)
    
    def reset(self):
        """清除所有監控數據"""
        with self._lock:
            self._segments.clear()
            self._segment_counter = 0
            self._session_start = None
            if os.path.exists(self.CSV_PATH):
                os.remove(self.CSV_PATH)
            logger.info(" | TTS Monitor: reset | ")
    
    def record(self, audio_uid: str, times: str,
               input_audio_duration: float,
               transcribe_time: float, translate_time: float,
               tts_wall_time: float, tts_per_lang: Dict[str, Dict],
               recording_start: Optional[float] = None,
               request_arrival: Optional[float] = None):
        """記錄一段 pipeline 的時間數據並寫入 CSV。
        
        時間線嚴格串接:
          recording_start → recording_end (= request_arrival) → +transcribe → +translate → +tts
        
        recording_start: 前端 times.timestamp() — 收音開始時間
        request_arrival: time.time() at request — 收音結束/pipeline 開始
        """
        with self._lock:
            req_time = request_arrival if request_arrival is not None else time.time()
            # 收音開始 = request到達 - 音檔長度
            rec_start_time = req_time - input_audio_duration
            
            if self._session_start is None:
                self._session_start = rec_start_time
            
            self._segment_counter += 1
            
            # 嚴格串接: 每個階段的結束 = 上一個結束 + 該階段耗時
            rec_end = req_time  # 收音結束 = request 到達
            transcribe_end = rec_end + transcribe_time
            translate_end = transcribe_end + translate_time
            tts_end = translate_end + tts_wall_time            
            # 最長語言的 TTS 輸出音檔長度
            max_audio_dur = 0.0
            for lang_data in tts_per_lang.values():
                ad = lang_data.get("audio_duration", 0)
                if ad > max_audio_dur:
                    max_audio_dur = ad
            
            seg = SegmentTiming(
                segment_idx=self._segment_counter,
                audio_uid=audio_uid,
                times=times,
                recording_start=rec_start_time,
                recording_end=rec_end,
                transcribe_end=transcribe_end,
                translate_end=translate_end,
                tts_end=tts_end,
                input_audio_duration=input_audio_duration,
                transcribe_time=transcribe_time,
                translate_time=translate_time,
                tts_wall_time=tts_wall_time,
                tts_per_lang=tts_per_lang,
                max_tts_audio_duration=max_audio_dur,
            )
            self._segments.append(seg)
            
            if len(self._segments) > self._max_segments:
                self._segments.pop(0)
            
            # log
            total_pipeline = transcribe_time + translate_time + tts_wall_time
            logger.info(
                f" | TTS Monitor | seg #{self._segment_counter} | "
                f"input: {input_audio_duration:.2f}s | "
                f"pipeline: {total_pipeline:.2f}s "
                f"(transcribe={transcribe_time:.2f} + translate={translate_time:.2f} + tts={tts_wall_time:.2f}) | "
                f"max_tts_audio: {max_audio_dur:.2f}s | "
            )
            
            self._write_csv()
    
    def _write_csv(self):
        """
        輸出 Gantt-style CSV。每段嚴格串接: recording → transcribe → translate → tts → playback。
        
        所有 start/end 都是相對於 session_start 的秒數。
        每個階段的 end 就是下一個階段的 start，絕不重疊。
        """
        if not self._segments or self._session_start is None:
            return
        
        headers = [
            "segment_idx",
            "stage",
            "start",
            "end",
            "duration",
            "language",
            "tts_output_audio_duration",
            "accumulated_delay",
            "recording_time",
        ]
        
        rows = []
        
        for seg in self._segments:
            # 所有時間轉為相對 session_start 的秒數
            rec_start = seg.recording_start - self._session_start
            rec_end = seg.recording_end - self._session_start
            trans_end = seg.transcribe_end - self._session_start
            translate_end = seg.translate_end - self._session_start
            tts_end = seg.tts_end - self._session_start
            playback_end = tts_end + seg.max_tts_audio_duration
            
            # 累積延遲 = TTS 結束時間 - 收音結束時間 (越過多少時間才能播放)
            delay = tts_end - rec_end
            
            # recording: rec_start → rec_end
            rows.append([seg.segment_idx, "recording", f"{rec_start:.3f}", f"{rec_end:.3f}",
                         f"{seg.input_audio_duration:.3f}", "", "", f"{delay:.3f}",
                         seg.times])
            # transcribe: rec_end → trans_end
            rows.append([seg.segment_idx, "transcribe", f"{rec_end:.3f}", f"{trans_end:.3f}",
                         f"{seg.transcribe_time:.3f}", "", "", "", ""])
            # translate: trans_end → translate_end
            rows.append([seg.segment_idx, "translate", f"{trans_end:.3f}", f"{translate_end:.3f}",
                         f"{seg.translate_time:.3f}", "", "", "", ""])
            # tts_batch: translate_end → tts_end
            rows.append([seg.segment_idx, "tts_batch", f"{translate_end:.3f}", f"{tts_end:.3f}",
                         f"{seg.tts_wall_time:.3f}", "", "", "", ""])
            # playback: tts_end → tts_end + max_audio_duration
            rows.append([seg.segment_idx, "playback", f"{tts_end:.3f}", f"{playback_end:.3f}",
                         f"{seg.max_tts_audio_duration:.3f}", "", f"{seg.max_tts_audio_duration:.3f}", "", ""])
            
            # 每語言 TTS 詳細 (都從 translate_end 開始，各自結束時間不同)
            for lang, lang_data in seg.tts_per_lang.items():
                synth_time = lang_data.get("synthesis_time", 0)
                audio_dur = lang_data.get("audio_duration", 0)
                rows.append([seg.segment_idx, f"tts_{lang}", f"{translate_end:.3f}",
                             f"{translate_end + synth_time:.3f}", f"{synth_time:.3f}",
                             lang, f"{audio_dur:.3f}", "", ""])
        
        with open(self.CSV_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
    
    def get_csv_path(self) -> str:
        """回傳 CSV 檔案路徑"""
        return self.CSV_PATH if os.path.exists(self.CSV_PATH) else ""
    
    def get_data(self) -> Dict:
        """取得所有監控數據 (用於 API 回傳)"""
        with self._lock:
            if not self._segments or self._session_start is None:
                return {"session_start": None, "total_segments": 0, "accumulated_delay": 0, "segments": []}
            
            segments_list = []
            for seg in self._segments:
                rec_start = seg.recording_start - self._session_start
                rec_end = seg.recording_end - self._session_start
                trans_end = seg.transcribe_end - self._session_start
                translate_end = seg.translate_end - self._session_start
                tts_end = seg.tts_end - self._session_start
                delay = tts_end - rec_end
                segments_list.append({
                    "segment_idx": seg.segment_idx,
                    "audio_uid": seg.audio_uid,
                    "times": seg.times,
                    "recording": [round(rec_start, 3), round(rec_end, 3)],
                    "transcribe": [round(rec_end, 3), round(trans_end, 3)],
                    "translate": [round(trans_end, 3), round(translate_end, 3)],
                    "tts": [round(translate_end, 3), round(tts_end, 3)],
                    "playback": [round(tts_end, 3), round(tts_end + seg.max_tts_audio_duration, 3)],
                    "input_audio_duration": seg.input_audio_duration,
                    "tts_per_lang": seg.tts_per_lang,
                    "delay": round(delay, 3),
                })
            
            return {
                "session_start": self._session_start,
                "total_segments": self._segment_counter,
                "total_elapsed": round(self._segments[-1].tts_end - self._session_start, 3),
                "csv_path": self.CSV_PATH,
                "segments": segments_list,
            }
