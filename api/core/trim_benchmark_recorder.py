"""
Trim Benchmark Recorder

獨立模組：記錄和分析 Trim 功能的測試數據

功能：
1. API 響應速度統計（每筆 + 平均）
2. Cancel 統計（比例 + 連續 cancel 分布）
3. 最終文本記錄（每個 UID 最後一筆）
4. 報告生成（JSON + 可讀格式）

使用方式：
    from api.core.trim_benchmark_recorder import get_benchmark_recorder
    
    # 開始新的測試
    recorder = get_benchmark_recorder()
    recorder.start_test("test_with_trim", trim_enabled=True)
    
    # 記錄每一筆請求結果
    recorder.record_request(
        audio_uid="xxx",
        times="2026-01-21 10:00:00",
        response_time=0.5,
        is_cancelled=False,
        cancel_type=None,  # "transcribe_cancel" or "translate_cancel"
        transcription_text="你好世界",
        is_final=False
    )
    
    # 測試結束，生成報告
    recorder.end_test()
    recorder.export_report("benchmark_results/test_with_trim.json")
"""

import os
import json
import time
import logging
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from collections import defaultdict

# ============================================================================
# Logger
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class RequestRecord:
    """單筆請求的記錄"""
    audio_uid: str
    times: str  # timestamp string
    sequence: int  # 該 UID 的第幾筆請求（從 1 開始）
    
    # 時間相關
    response_time: float  # API 響應時間（秒）
    transcribe_time: float = 0.0
    translate_time: float = 0.0
    
    # 狀態
    is_cancelled: bool = False
    cancel_type: Optional[str] = None  # "transcribe_cancel", "translate_cancel", None
    
    # 結果
    transcription_text: str = ""
    is_final: bool = False  # 是否為該 UID 的最後一筆
    
    # Trim 相關
    trim_duration: float = 0.0
    trim_updated: bool = False
    stable_text: str = ""
    unstable_text: str = ""
    
    # 時間戳
    recorded_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class UIDStatistics:
    """單個 UID 的統計"""
    audio_uid: str
    total_requests: int = 0
    cancelled_requests: int = 0
    successful_requests: int = 0
    
    # 響應時間（只計算成功的）
    response_times: List[float] = field(default_factory=list)
    avg_response_time: float = 0.0
    
    # 連續 cancel 追蹤
    consecutive_cancels: List[int] = field(default_factory=list)  # 每段連續 cancel 的長度
    current_cancel_streak: int = 0
    
    # 最終文本
    final_text: str = ""
    final_sequence: int = 0


@dataclass
class TestStatistics:
    """整個測試的統計"""
    test_name: str
    trim_enabled: bool
    start_time: str
    end_time: str = ""
    
    # 總計
    total_requests: int = 0
    total_cancelled: int = 0
    total_successful: int = 0
    cancel_rate: float = 0.0
    
    # 響應時間統計
    avg_response_time: float = 0.0
    min_response_time: float = 0.0
    max_response_time: float = 0.0
    total_response_times: List[float] = field(default_factory=list)
    
    # 連續 cancel 分布 {長度: 次數}
    consecutive_cancel_distribution: Dict[int, int] = field(default_factory=dict)
    
    # UID 數量
    total_uids: int = 0
    
    # 每個 UID 的統計
    uid_statistics: Dict[str, dict] = field(default_factory=dict)


# ============================================================================
# Benchmark Recorder (Singleton)
# ============================================================================

class TrimBenchmarkRecorder:
    """
    Trim 測試的 Benchmark 記錄器
    
    Thread-safe, Singleton 模式
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.enabled = False  # 是否啟用記錄
        self.current_test: Optional[str] = None
        self.trim_enabled: bool = False
        
        # 數據存儲
        self.records: Dict[str, List[RequestRecord]] = defaultdict(list)  # {audio_uid: [records]}
        self.uid_stats: Dict[str, UIDStatistics] = {}
        self.test_stats: Optional[TestStatistics] = None
        
        # 線程鎖
        self.data_lock = threading.Lock()
        
        logger.info(" | TrimBenchmarkRecorder initialized | ")
    
    # ========== 測試控制 ==========
    
    def start_test(self, test_name: str, trim_enabled: bool = True) -> None:
        """
        開始新的測試
        
        Args:
            test_name: 測試名稱（用於報告文件名）
            trim_enabled: 是否啟用 trim 功能
        """
        with self.data_lock:
            # 清空之前的數據
            self.records.clear()
            self.uid_stats.clear()
            
            self.current_test = test_name
            self.trim_enabled = trim_enabled
            self.enabled = True
            
            self.test_stats = TestStatistics(
                test_name=test_name,
                trim_enabled=trim_enabled,
                start_time=datetime.now().isoformat()
            )
            
            logger.info(f" | Benchmark test started: {test_name} (trim_enabled={trim_enabled}) | ")
    
    def end_test(self) -> TestStatistics:
        """
        結束測試並計算統計
        
        Returns:
            TestStatistics: 測試統計結果
        """
        with self.data_lock:
            if not self.enabled or self.test_stats is None:
                logger.warning(" | No active test to end | ")
                return None
            
            self.enabled = False
            self.test_stats.end_time = datetime.now().isoformat()
            
            # 計算統計
            self._calculate_statistics()
            
            logger.info(f" | Benchmark test ended: {self.current_test} | ")
            logger.info(f" | Total requests: {self.test_stats.total_requests}, "
                       f"Cancelled: {self.test_stats.total_cancelled} ({self.test_stats.cancel_rate:.1%}), "
                       f"Avg response time: {self.test_stats.avg_response_time:.3f}s | ")
            
            return self.test_stats
    
    def is_enabled(self) -> bool:
        """檢查是否正在記錄"""
        return self.enabled
    
    # ========== 記錄 API ==========
    
    def record_request(
        self,
        audio_uid: str,
        times: str,
        response_time: float,
        is_cancelled: bool = False,
        cancel_type: Optional[str] = None,
        transcription_text: str = "",
        is_final: bool = False,
        transcribe_time: float = 0.0,
        translate_time: float = 0.0,
        trim_duration: float = 0.0,
        trim_updated: bool = False,
        stable_text: str = "",
        unstable_text: str = ""
    ) -> None:
        """
        記錄單筆請求結果
        
        Args:
            audio_uid: 音訊 UID
            times: 請求時間戳（字串格式）
            response_time: API 響應時間（秒）
            is_cancelled: 是否被取消
            cancel_type: 取消類型 ("transcribe_cancel" 或 "translate_cancel")
            transcription_text: 轉錄文本
            is_final: 是否為該 UID 的最後一筆
            transcribe_time: 轉錄時間
            translate_time: 翻譯時間
            trim_duration: Trim 時長
            trim_updated: 是否觸發 trim 更新
            stable_text: 穩定文本
            unstable_text: 不穩定文本
        """
        print(f"[RECORDER] record_request called: enabled={self.enabled}")
        if not self.enabled:
            print(f"[RECORDER] Skipped - not enabled")
            return
        
        print(f"[RECORDER] Recording uid={audio_uid}, response_time={response_time:.3f}s")
        with self.data_lock:
            # 計算 sequence
            sequence = len(self.records[audio_uid]) + 1
            
            # 創建記錄
            record = RequestRecord(
                audio_uid=audio_uid,
                times=times,
                sequence=sequence,
                response_time=response_time,
                transcribe_time=transcribe_time,
                translate_time=translate_time,
                is_cancelled=is_cancelled,
                cancel_type=cancel_type,
                transcription_text=transcription_text,
                is_final=is_final,
                trim_duration=trim_duration,
                trim_updated=trim_updated,
                stable_text=stable_text,
                unstable_text=unstable_text
            )
            
            # 存儲記錄
            self.records[audio_uid].append(record)
            
            # 更新 UID 統計
            self._update_uid_stats(audio_uid, record)
            
            # 日誌
            status = "CANCELLED" if is_cancelled else "SUCCESS"
            logger.debug(
                f" | [BENCHMARK] {audio_uid} seq={sequence} {status} "
                f"time={response_time:.3f}s text='{transcription_text[:30]}...' | "
            )
    
    def mark_uid_final(self, audio_uid: str, final_text: str) -> None:
        """
        標記 UID 的最終文本
        
        當 UID 結束時調用，記錄最終完整的文本
        """
        if not self.enabled:
            return
        
        with self.data_lock:
            if audio_uid in self.uid_stats:
                self.uid_stats[audio_uid].final_text = final_text
                self.uid_stats[audio_uid].final_sequence = len(self.records[audio_uid])
                logger.debug(f" | [BENCHMARK] UID {audio_uid} final text: '{final_text[:50]}...' | ")
    
    # ========== 統計計算 ==========
    
    def _update_uid_stats(self, audio_uid: str, record: RequestRecord) -> None:
        """更新單個 UID 的統計（內部使用，需持有鎖）"""
        if audio_uid not in self.uid_stats:
            self.uid_stats[audio_uid] = UIDStatistics(audio_uid=audio_uid)
        
        stats = self.uid_stats[audio_uid]
        stats.total_requests += 1
        
        if record.is_cancelled:
            stats.cancelled_requests += 1
            stats.current_cancel_streak += 1
        else:
            stats.successful_requests += 1
            stats.response_times.append(record.response_time)
            
            # 如果之前有連續 cancel，記錄下來
            if stats.current_cancel_streak > 0:
                stats.consecutive_cancels.append(stats.current_cancel_streak)
                stats.current_cancel_streak = 0
        
        # 更新最終文本（每次成功都更新，最後一次就是 final）
        if not record.is_cancelled and record.transcription_text:
            stats.final_text = record.transcription_text
            stats.final_sequence = record.sequence
    
    def _calculate_statistics(self) -> None:
        """計算整體統計（內部使用，需持有鎖）"""
        if self.test_stats is None:
            return
        
        all_response_times = []
        total_cancelled = 0
        total_requests = 0
        consecutive_cancel_dist = defaultdict(int)
        
        for audio_uid, stats in self.uid_stats.items():
            total_requests += stats.total_requests
            total_cancelled += stats.cancelled_requests
            all_response_times.extend(stats.response_times)
            
            # 計算該 UID 的平均響應時間
            if stats.response_times:
                stats.avg_response_time = sum(stats.response_times) / len(stats.response_times)
            
            # 處理最後可能還在進行的連續 cancel
            if stats.current_cancel_streak > 0:
                stats.consecutive_cancels.append(stats.current_cancel_streak)
            
            # 統計連續 cancel 分布
            for streak in stats.consecutive_cancels:
                consecutive_cancel_dist[streak] += 1
            
            # 存儲 UID 統計
            self.test_stats.uid_statistics[audio_uid] = {
                "total_requests": stats.total_requests,
                "cancelled_requests": stats.cancelled_requests,
                "successful_requests": stats.successful_requests,
                "cancel_rate": stats.cancelled_requests / stats.total_requests if stats.total_requests > 0 else 0,
                "avg_response_time": stats.avg_response_time,
                "consecutive_cancels": stats.consecutive_cancels,
                "final_text": stats.final_text,
                "final_sequence": stats.final_sequence
            }
        
        # 更新測試統計
        self.test_stats.total_requests = total_requests
        self.test_stats.total_cancelled = total_cancelled
        self.test_stats.total_successful = total_requests - total_cancelled
        self.test_stats.cancel_rate = total_cancelled / total_requests if total_requests > 0 else 0
        self.test_stats.total_uids = len(self.uid_stats)
        self.test_stats.total_response_times = all_response_times
        self.test_stats.consecutive_cancel_distribution = dict(consecutive_cancel_dist)
        
        if all_response_times:
            self.test_stats.avg_response_time = sum(all_response_times) / len(all_response_times)
            self.test_stats.min_response_time = min(all_response_times)
            self.test_stats.max_response_time = max(all_response_times)
    
    # ========== 報告生成 ==========
    
    def export_report(self, filepath: str = None) -> str:
        """
        導出報告到 JSON 文件
        
        Args:
            filepath: 輸出路徑，默認為 benchmark_results/{test_name}_{timestamp}.json
            
        Returns:
            str: 報告文件路徑
        """
        with self.data_lock:
            if self.test_stats is None:
                logger.warning(" | No test statistics to export | ")
                return None
            
            # 默認路徑
            if filepath is None:
                os.makedirs("benchmark_results", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"benchmark_results/{self.current_test}_{timestamp}.json"
            
            # 構建報告
            report = {
                "summary": {
                    "test_name": self.test_stats.test_name,
                    "trim_enabled": self.test_stats.trim_enabled,
                    "start_time": self.test_stats.start_time,
                    "end_time": self.test_stats.end_time,
                    "total_uids": self.test_stats.total_uids,
                    "total_requests": self.test_stats.total_requests,
                    "total_successful": self.test_stats.total_successful,
                    "total_cancelled": self.test_stats.total_cancelled,
                    "cancel_rate": f"{self.test_stats.cancel_rate:.2%}",
                    "avg_response_time": f"{self.test_stats.avg_response_time:.4f}s",
                    "min_response_time": f"{self.test_stats.min_response_time:.4f}s",
                    "max_response_time": f"{self.test_stats.max_response_time:.4f}s",
                },
                "consecutive_cancel_distribution": {
                    f"連續{k}次取消": v 
                    for k, v in sorted(self.test_stats.consecutive_cancel_distribution.items())
                },
                "uid_statistics": self.test_stats.uid_statistics,
                "all_response_times": self.test_stats.total_response_times,
                "detailed_records": {
                    uid: [asdict(r) for r in records]
                    for uid, records in self.records.items()
                }
            }
            
            # 寫入文件
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f" | Benchmark report exported to: {filepath} | ")
            return filepath
    
    def get_summary(self) -> dict:
        """獲取測試摘要（不含詳細記錄）- 實時計算"""
        with self.data_lock:
            if self.test_stats is None:
                return {}
            
            # 實時計算統計（從 uid_stats 計算，而非 test_stats）
            total_requests = 0
            total_cancelled = 0
            all_response_times = []
            consecutive_cancel_dist = defaultdict(int)
            
            for uid, stats in self.uid_stats.items():
                total_requests += stats.total_requests
                total_cancelled += stats.cancelled_requests
                all_response_times.extend(stats.response_times)
                
                # 連續取消分布
                for streak in stats.consecutive_cancels:
                    consecutive_cancel_dist[streak] += 1
                # 當前未結束的連續取消
                if stats.current_cancel_streak > 0:
                    consecutive_cancel_dist[stats.current_cancel_streak] += 1
            
            cancel_rate = total_cancelled / total_requests if total_requests > 0 else 0
            avg_response_time = sum(all_response_times) / len(all_response_times) if all_response_times else 0
            
            return {
                "test_name": self.test_stats.test_name,
                "trim_enabled": self.test_stats.trim_enabled,
                "total_requests": total_requests,
                "cancel_rate": cancel_rate,
                "avg_response_time": avg_response_time,
                "consecutive_cancel_distribution": dict(consecutive_cancel_dist),
                "total_uids": len(self.uid_stats)
            }
    
    def get_final_texts(self) -> Dict[str, str]:
        """獲取所有 UID 的最終文本"""
        with self.data_lock:
            return {
                uid: stats.final_text 
                for uid, stats in self.uid_stats.items()
            }
    
    def print_summary(self) -> None:
        """打印測試摘要到控制台"""
        with self.data_lock:
            if self.test_stats is None:
                print("No test statistics available.")
                return
            
            print("\n" + "="*70)
            print(f"📊 Benchmark Test Summary: {self.test_stats.test_name}")
            print("="*70)
            print(f"  Trim Enabled: {self.test_stats.trim_enabled}")
            print(f"  Duration: {self.test_stats.start_time} ~ {self.test_stats.end_time}")
            print("-"*70)
            print(f"  Total UIDs: {self.test_stats.total_uids}")
            print(f"  Total Requests: {self.test_stats.total_requests}")
            print(f"  Successful: {self.test_stats.total_successful}")
            print(f"  Cancelled: {self.test_stats.total_cancelled} ({self.test_stats.cancel_rate:.1%})")
            print("-"*70)
            print(f"  Avg Response Time: {self.test_stats.avg_response_time:.4f}s")
            print(f"  Min Response Time: {self.test_stats.min_response_time:.4f}s")
            print(f"  Max Response Time: {self.test_stats.max_response_time:.4f}s")
            print("-"*70)
            print("  Consecutive Cancel Distribution:")
            for k, v in sorted(self.test_stats.consecutive_cancel_distribution.items()):
                print(f"    連續 {k} 次取消: {v} 次")
            print("="*70 + "\n")


# ============================================================================
# 比較工具
# ============================================================================

def compare_results(
    report_with_trim: str,
    report_without_trim: str,
    output_path: str = None
) -> dict:
    """
    比較兩個測試結果
    
    Args:
        report_with_trim: 啟用 trim 的報告 JSON 路徑
        report_without_trim: 未啟用 trim 的報告 JSON 路徑
        output_path: 比較報告輸出路徑
        
    Returns:
        dict: 比較結果
    """
    with open(report_with_trim, 'r', encoding='utf-8') as f:
        data_trim = json.load(f)
    
    with open(report_without_trim, 'r', encoding='utf-8') as f:
        data_no_trim = json.load(f)
    
    # 響應時間比較
    avg_time_trim = float(data_trim['summary']['avg_response_time'].replace('s', ''))
    avg_time_no_trim = float(data_no_trim['summary']['avg_response_time'].replace('s', ''))
    time_improvement = (avg_time_no_trim - avg_time_trim) / avg_time_no_trim if avg_time_no_trim > 0 else 0
    
    # Cancel 率比較
    cancel_rate_trim = float(data_trim['summary']['cancel_rate'].replace('%', '')) / 100
    cancel_rate_no_trim = float(data_no_trim['summary']['cancel_rate'].replace('%', '')) / 100
    
    # 文本相似性比較
    final_texts_trim = {
        uid: stats['final_text'] 
        for uid, stats in data_trim['uid_statistics'].items()
    }
    final_texts_no_trim = {
        uid: stats['final_text'] 
        for uid, stats in data_no_trim['uid_statistics'].items()
    }
    
    text_similarities = {}
    common_uids = set(final_texts_trim.keys()) & set(final_texts_no_trim.keys())
    
    for uid in common_uids:
        text1 = final_texts_trim[uid]
        text2 = final_texts_no_trim[uid]
        similarity = _calculate_text_similarity(text1, text2)
        text_similarities[uid] = {
            "trim_text": text1,
            "no_trim_text": text2,
            "similarity": similarity
        }
    
    avg_similarity = (
        sum(s['similarity'] for s in text_similarities.values()) / len(text_similarities)
        if text_similarities else 0
    )
    
    comparison = {
        "response_time_comparison": {
            "with_trim_avg": f"{avg_time_trim:.4f}s",
            "without_trim_avg": f"{avg_time_no_trim:.4f}s",
            "improvement": f"{time_improvement:.1%}",
            "trim_faster": avg_time_trim < avg_time_no_trim
        },
        "cancel_rate_comparison": {
            "with_trim": f"{cancel_rate_trim:.1%}",
            "without_trim": f"{cancel_rate_no_trim:.1%}",
            "trim_has_lower_cancel_rate": cancel_rate_trim < cancel_rate_no_trim
        },
        "text_similarity": {
            "average_similarity": f"{avg_similarity:.1%}",
            "total_compared_uids": len(text_similarities),
            "details": text_similarities
        },
        "consecutive_cancel_comparison": {
            "with_trim": data_trim.get('consecutive_cancel_distribution', {}),
            "without_trim": data_no_trim.get('consecutive_cancel_distribution', {})
        }
    }
    
    # 輸出比較報告
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        logger.info(f" | Comparison report exported to: {output_path} | ")
    
    return comparison


def _calculate_text_similarity(text1: str, text2: str) -> float:
    """
    計算兩個文本的相似度（使用簡單的 Jaccard 相似度）
    
    可以擴展為更複雜的算法（如 Levenshtein、BLEU 等）
    """
    if not text1 or not text2:
        return 0.0 if text1 != text2 else 1.0
    
    # 簡單的字符級 Jaccard 相似度
    set1 = set(text1)
    set2 = set(text2)
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def print_comparison(comparison: dict) -> None:
    """打印比較結果"""
    print("\n" + "="*70)
    print("📊 Trim vs No-Trim Comparison")
    print("="*70)
    
    rt = comparison['response_time_comparison']
    print("\n⏱️  Response Time:")
    print(f"    With Trim:    {rt['with_trim_avg']}")
    print(f"    Without Trim: {rt['without_trim_avg']}")
    print(f"    Improvement:  {rt['improvement']} {'✅' if rt['trim_faster'] else '❌'}")
    
    cr = comparison['cancel_rate_comparison']
    print("\n🚫 Cancel Rate:")
    print(f"    With Trim:    {cr['with_trim']}")
    print(f"    Without Trim: {cr['without_trim']}")
    print(f"    Lower Cancel: {'With Trim ✅' if cr['trim_has_lower_cancel_rate'] else 'Without Trim'}")
    
    ts = comparison['text_similarity']
    print("\n📝 Text Similarity:")
    print(f"    Average Similarity: {ts['average_similarity']}")
    print(f"    Compared UIDs: {ts['total_compared_uids']}")
    
    print("\n🔄 Consecutive Cancel Distribution:")
    print("    With Trim:", comparison['consecutive_cancel_comparison']['with_trim'])
    print("    Without Trim:", comparison['consecutive_cancel_comparison']['without_trim'])
    
    print("="*70 + "\n")


# ============================================================================
# 全局實例和便利函數
# ============================================================================

# Singleton 實例
_benchmark_recorder = None

def get_benchmark_recorder() -> TrimBenchmarkRecorder:
    """獲取 TrimBenchmarkRecorder 單例實例"""
    global _benchmark_recorder
    if _benchmark_recorder is None:
        _benchmark_recorder = TrimBenchmarkRecorder()
    return _benchmark_recorder


def start_benchmark(test_name: str, trim_enabled: bool = True) -> TrimBenchmarkRecorder:
    """便利函數：開始新的 benchmark 測試"""
    recorder = get_benchmark_recorder()
    recorder.start_test(test_name, trim_enabled)
    return recorder


def end_benchmark() -> TestStatistics:
    """便利函數：結束 benchmark 測試"""
    recorder = get_benchmark_recorder()
    return recorder.end_test()


def record_benchmark(
    audio_uid: str,
    times: str,
    response_time: float,
    is_cancelled: bool = False,
    cancel_type: Optional[str] = None,
    transcription_text: str = "",
    is_final: bool = False,
    **kwargs
) -> None:
    """便利函數：記錄單筆請求"""
    recorder = get_benchmark_recorder()
    recorder.record_request(
        audio_uid=audio_uid,
        times=times,
        response_time=response_time,
        is_cancelled=is_cancelled,
        cancel_type=cancel_type,
        transcription_text=transcription_text,
        is_final=is_final,
        **kwargs
    )
