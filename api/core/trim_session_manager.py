"""
Trim Session Manager

管理基於 Segment 的音訊 Trim 功能

核心概念：
1. 每個 audio_uid 有獨立的 Session，追蹤 trim 狀態和 window buffer
2. 使用前綴鏈穩定性檢查來判斷哪些文本已經確定
3. 確定的文本會被 trim 掉，下次只處理剩餘音訊
4. 支援併發處理，使用引用計數確保 Session 安全清理

使用流程：
1. acquire_session(uid) - 獲取或創建 session，增加引用計數
2. get_trim_info(uid) - 獲取當前 trim 狀態
3. add_result_and_check(uid, result) - 添加結果並檢查穩定性
4. release_session(uid, is_audio_end) - 釋放引用，標記結束
"""

import re
import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from logging.handlers import RotatingFileHandler

from lib.config.constant import ENABLE_TRIM, TRIM_WINDOW_SIZE, TRIM_SESSION_TIMEOUT

# ============================================================================
# 專用 Logger（追蹤 window 變化）
# ============================================================================

def setup_trim_logger():
    """設置專用的 trim session logger"""
    logger = logging.getLogger("trim_session")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    
    if not logger.handlers:
        # 專用 log 檔案
        handler = RotatingFileHandler(
            "logs/trim_session.log",
            maxBytes=10*1024*1024,
            backupCount=3
        )
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s"
        ))
        logger.addHandler(handler)
    
    return logger

trim_logger = setup_trim_logger()

# 主 logger（用於一般訊息）
logger = logging.getLogger(__name__)


# ============================================================================
# 文字處理工具
# ============================================================================

def normalize_text(text: str) -> str:
    """
    標準化文字用於比較
    
    處理步驟：
    1. 在中文和英數之間插入空格
    2. 去除所有空白
    3. 去除標點符號（只保留英數字和中文）
    4. 全部轉小寫
    """
    if not text:
        return ""
    
    # 1. 在中文和英數之間插入空格
    text = re.sub(r'([\u4e00-\u9fff])([a-zA-Z0-9])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z0-9])([\u4e00-\u9fff])', r'\1 \2', text)
    
    # 2. 去除所有空白
    text = re.sub(r'\s+', '', text)
    
    # 3. 去除標點符號
    text = re.sub(r'[^\w\u4e00-\u9fff]', '', text)
    
    # 4. 全小寫
    return text.lower()


# ============================================================================
# 資料結構
# ============================================================================

@dataclass
class TrimSession:
    """單一 audio_uid 的 Trim Session"""
    
    audio_uid: str
    
    # Trim 狀態
    trim_duration: float = 0.0  # 已 trim 的時長（秒）
    trim_text: str = ""  # 已確認的穩定文本
    
    # Window Buffer（儲存最近 N 個 API 結果的 segments）
    window_buffer: deque = field(default_factory=lambda: deque(maxlen=TRIM_WINDOW_SIZE))
    
    # 併發控制
    active_requests: int = 0  # 正在使用此 session 的請求數
    is_ended: bool = False  # 是否收到 audio_end
    cleanup_ready: bool = False  # 是否準備好被清理
    
    # 時間追蹤
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    
    # 統計
    total_requests: int = 0
    trim_updates: int = 0
    
    def __post_init__(self):
        # 確保 window_buffer 有正確的 maxlen
        if not isinstance(self.window_buffer, deque):
            self.window_buffer = deque(maxlen=TRIM_WINDOW_SIZE)


@dataclass 
class TrimResult:
    """Trim 檢查結果"""
    should_update: bool = False
    new_trim_duration: float = 0.0
    new_trim_text: str = ""
    stable_segments: List[dict] = field(default_factory=list)


# ============================================================================
# Trim Session Manager
# ============================================================================

class TrimSessionManager:
    """
    管理所有 audio_uid 的 Trim Session
    
    線程安全，支援併發請求
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton 模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.sessions: Dict[str, TrimSession] = {}
        self.session_lock = threading.Lock()
        self._initialized = True
        
        # 啟動清理線程
        self._start_cleanup_thread()
        
        logger.info(" | TrimSessionManager initialized | ")
    
    def _start_cleanup_thread(self):
        """啟動定期清理過期 session 的線程"""
        def cleanup_loop():
            while True:
                time.sleep(10)  # 每 10 秒檢查一次（加快清理速度）
                self._cleanup_expired_sessions()
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_expired_sessions(self):
        """清理過期的 session 和標記為可清理的 session"""
        current_time = time.time()
        cleanup_uids = []
        
        with self.session_lock:
            for uid, session in self.sessions.items():
                # 情況 1: 已標記為可清理（所有請求完成且收到 audio_end）
                if getattr(session, 'cleanup_ready', False):
                    cleanup_uids.append((uid, 'ready'))
                # 情況 2: 超時且沒有活躍請求
                elif (current_time - session.last_activity > TRIM_SESSION_TIMEOUT 
                      and session.active_requests == 0):
                    cleanup_uids.append((uid, 'expired'))
            
            for uid, reason in cleanup_uids:
                del self.sessions[uid]
                trim_logger.info(f"[CLEANUP] Session {reason}: uid={uid}")
        
        if cleanup_uids:
            logger.info(f" | Cleaned up {len(cleanup_uids)} trim sessions | ")
    
    # ========== 公開 API ==========
    
    def is_enabled(self) -> bool:
        """檢查 trim 功能是否啟用"""
        return ENABLE_TRIM
    
    def acquire_session(self, audio_uid: str) -> TrimSession:
        """
        獲取或創建 session，增加引用計數
        
        在請求開始時調用
        """
        with self.session_lock:
            if audio_uid not in self.sessions:
                self.sessions[audio_uid] = TrimSession(audio_uid=audio_uid)
                trim_logger.info(f" | [SESSION] Created: uid={audio_uid} | ")
            
            session = self.sessions[audio_uid]
            session.active_requests += 1
            session.total_requests += 1
            session.last_activity = time.time()
            
            trim_logger.debug(
                f" | [ACQUIRE] uid={audio_uid}, active={session.active_requests}, "
                f"trim_duration={session.trim_duration:.3f}s | "
            )
            
            return session
    
    def release_session(self, audio_uid: str, is_audio_end: bool = False):
        """
        釋放 session 引用，可選標記為結束
        
        在請求結束時調用
        
        注意：即使 is_audio_end=True，也不會立即刪除 session，
        而是標記為結束，等待 cleanup thread 或所有請求完成後再刪除。
        這樣可以避免並行請求場景下 session 被過早刪除的問題。
        """
        with self.session_lock:
            if audio_uid not in self.sessions:
                return
            
            session = self.sessions[audio_uid]
            session.active_requests = max(0, session.active_requests - 1)
            session.last_activity = time.time()
            
            if is_audio_end:
                session.is_ended = True
                trim_logger.info(
                    f"[END] uid={audio_uid}, active={session.active_requests}, "
                    f"total_requests={session.total_requests}, trim_updates={session.trim_updates}, "
                    f"final_trim={session.trim_duration:.3f}s"
                )
            
            # 當沒有活躍請求且已結束時，清理 session
            # 但添加一個短暫延遲，讓其他併發請求有機會完成
            if session.active_requests == 0 and session.is_ended:
                # 保留 session 資訊供查詢，但標記為可清理
                # cleanup thread 會在下一輪清理它
                session.cleanup_ready = True
                trim_logger.info(f"[SESSION] Marked for cleanup: uid={audio_uid}")
            else:
                trim_logger.debug(
                    f"[RELEASE] uid={audio_uid}, active={session.active_requests}, "
                    f"is_ended={session.is_ended}"
                )
    
    def get_trim_info(self, audio_uid: str) -> Tuple[float, str]:
        """
        獲取當前的 trim 資訊
        
        Returns:
            (trim_duration, trim_text)
        """
        with self.session_lock:
            if audio_uid not in self.sessions:
                return 0.0, ""
            
            session = self.sessions[audio_uid]
            return session.trim_duration, session.trim_text
    
    def add_result_and_check(
        self, 
        audio_uid: str, 
        segments: List[dict],
        trim_duration: float
    ) -> TrimResult:
        """
        添加轉譯結果到 window 並檢查穩定性
        
        Args:
            audio_uid: 音訊 UID
            segments: 本次轉譯的 segments（時間戳相對於 trim 後音訊）
            trim_duration: 發送這個請求時的 trim_duration
        
        Returns:
            TrimResult 包含是否需要更新 trim
        """
        result = TrimResult()
        
        # 過濾空的 segments（不加入 window）
        if not segments:
            trim_logger.debug(f"[SKIP] Empty segments: uid={audio_uid}")
            return result
        
        with self.session_lock:
            if audio_uid not in self.sessions:
                return result
            
            session = self.sessions[audio_uid]
            
            # 將 segments 轉換為絕對時間戳並加入 window
            absolute_segments = []
            for seg in segments:
                absolute_segments.append({
                    'index': seg.get('index', 0),
                    'start': trim_duration + seg.get('start', 0.0),
                    'end': trim_duration + seg.get('end', 0.0),
                    'text': seg.get('text', '')
                })
            
            # 加入 window buffer
            session.window_buffer.append({
                'segments': absolute_segments,
                'timestamp': time.time()
            })
            
            trim_logger.debug(
                f"[WINDOW+] uid={audio_uid}, window_size={len(session.window_buffer)}/{TRIM_WINDOW_SIZE}, "
                f"segments={len(segments)}"
            )
            
            # logger.info(f" | [WINDOW] uid={audio_uid} | window_buffer: {session.window_buffer} | ")
            
            # 檢查穩定性
            if len(session.window_buffer) >= TRIM_WINDOW_SIZE:
                result = self._check_stability_and_update(session)
        
        return result
    
    def get_session_info(self, audio_uid: str) -> dict:
        """獲取 session 的詳細資訊（用於輸出）"""
        with self.session_lock:
            if audio_uid not in self.sessions:
                return {
                    "trim_duration": 0.0,
                    "trim_text": "",
                    "window_count": 0,
                    "trim_updated": False
                }
            
            session = self.sessions[audio_uid]
            return {
                "trim_duration": session.trim_duration,
                "trim_text": session.trim_text,
                "window_count": len(session.window_buffer),
                "trim_updated": False  # 這個欄位在 add_result_and_check 中設定
            }
    
    def compose_output_text(
        self, 
        transcription_text: str,
        send_trim_text: str = "",
        trim_updated: bool = False,
        new_trim_text: str = ""
    ) -> Tuple[str, str, str]:
        """
        組合輸出文本
        
        Args:
            transcription_text: 本次轉譯結果（不含 trim 部分）
            send_trim_text: 發送請求時的 trim_text（只補上這部分，避免重複）
            trim_updated: 這次請求是否觸發了 trim 更新
            new_trim_text: 更新後的完整 stable text（只在 trim_updated=True 時使用）
        
        Returns:
            (full_text, stable_text, unstable_text)
        """
        if trim_updated and new_trim_text:
            # Trim 更新了，使用新的 stable_text
            stable_text = new_trim_text
            
            # 計算新增的穩定部分（這次新確認的文字）
            if send_trim_text:
                # 舊的 stable 有值，新增部分 = new - old
                new_stable_part = new_trim_text.replace(send_trim_text, '', 1).strip()
            else:
                # 舊的 stable 是空的，全部都是新增
                new_stable_part = new_trim_text
            
            # 從 transcription_text 移除新增的穩定部分，得到 unstable
            ori = transcription_text.strip()
            new_part = new_stable_part.strip()
            if ori.startswith(new_part):
                unstable_text = ori[len(new_part):].strip()
            else:
                # 不完全匹配時保守處理
                unstable_text = transcription_text
            
            # 組合 full_text
            full_text = (stable_text + " " + unstable_text).strip() if unstable_text else stable_text
        else:
            # 沒有更新，使用發送時的 trim_text
            stable_text = send_trim_text
            unstable_text = transcription_text
            full_text = (stable_text + unstable_text).strip() if stable_text else unstable_text
        
        return full_text, stable_text, unstable_text
    
    # ========== 私有方法 ==========
    
    def _check_stability_and_update(self, session: TrimSession) -> TrimResult:
        """
        檢查 window 中的 segment 穩定性並更新 trim
        
        邏輯：
        1. W0 是最早的（最短音訊），取 W0 的每個 Segment
        2. W0 的 Segment 文本必須是 W1~W9 各自「合併文本」的前綴
        3. 時間戳只取「單一 Segment 文本完全相同」的 window，取最小值
        4. 跨 segment 的不考慮時間戳
        """
        result = TrimResult()
        
        if len(session.window_buffer) < TRIM_WINDOW_SIZE:
            return result
        
        windows = list(session.window_buffer)
        w0 = windows[0]
        w0_segments = w0.get('segments', [])
        
        if not w0_segments:
            return result
        
        trim_logger.debug(
            f"[CHECK] Starting stability check for uid={session.audio_uid}, "
            f"window_size={len(windows)}, W0_segments={len(w0_segments)}"
        )
        
        # 預先計算 W1~W9 的合併文本（用於前綴檢查）
        merged_texts = []  # W1~W9 的合併文本
        for w in windows[1:]:
            segments = w.get('segments', [])
            merged = "".join(seg.get('text', '') for seg in segments)
            merged_texts.append(normalize_text(merged))
        
        # 從 W0 的 segment 0 開始逐個檢查
        for seg_index, w0_seg in enumerate(w0_segments):
            w0_text = w0_seg.get('text', '')
            w0_normalized = normalize_text(w0_text)
            
            if not w0_normalized:
                trim_logger.debug(f"[CHECK] W0 S{seg_index} is empty, stopping")
                break
            
            # Step 1: 檢查 W0 S{seg_index} 是否為 W1~W9 合併文本的前綴
            is_stable = True
            for i, merged in enumerate(merged_texts):
                if not merged.startswith(w0_normalized):
                    trim_logger.debug(
                        f"[CHECK] W0 S{seg_index} NOT prefix of W{i+1} merged: "
                        f"'{w0_normalized[:30]}' vs '{merged[:30]}'"
                    )
                    is_stable = False
                    break
            
            if not is_stable:
                trim_logger.debug(f"[CHECK] S{seg_index} is NOT stable, stopping")
                break
            
            trim_logger.debug(f"[CHECK] S{seg_index} is STABLE: '{w0_text[:30]}...'")
            
            # Step 2: 找時間戳 - 只取「單一 Segment 完全相同」的 window
            matching_ends = []
            
            # W0 自己的 segment
            matching_ends.append(w0_seg.get('end', 0.0))
            
            # 檢查 W1~W9
            for w_idx, w in enumerate(windows[1:], start=1):
                w_segments = w.get('segments', [])
                
                # 找這個 window 中是否有「完全相同」的單一 segment
                for seg in w_segments:
                    seg_text = seg.get('text', '')
                    if normalize_text(seg_text) == w0_normalized:
                        # 完全相同！記錄 end 時間
                        matching_ends.append(seg.get('end', 0.0))
                        trim_logger.debug(
                            f"[CHECK] W{w_idx} has exact match: end={seg.get('end', 0.0):.3f}s"
                        )
                        break  # 每個 window 只取一個
                # 如果沒找到（跨 segment），就不加入 matching_ends
            
            # 取最小的 end 時間
            if matching_ends:
                min_end = min(matching_ends)
                
                result.should_update = True
                result.new_trim_duration = min_end
                result.new_trim_text = (session.trim_text + w0_text).strip()
                result.stable_segments.append({
                    'index': seg_index,
                    'text': w0_text,
                    'end': min_end,
                    'matching_windows': len(matching_ends)
                })
                
                trim_logger.debug(
                    f"[CHECK] S{seg_index} min_end={min_end:.3f}s from {len(matching_ends)} matches"
                )
        
        # 如果有穩定的 segment，更新 session
        if result.should_update:
            old_trim = session.trim_duration
            session.trim_duration = result.new_trim_duration
            session.trim_text = result.new_trim_text
            session.trim_updates += 1
            session.window_buffer.clear()  # 清空 window
            
            trim_logger.info(
                f"[TRIM] Updated: uid={session.audio_uid}, "
                f"duration={old_trim:.3f}s -> {result.new_trim_duration:.3f}s, "
                f"stable_segments={len(result.stable_segments)}, "
                f"text='{result.new_trim_text[:50]}...'"
            )
        
        return result


# ============================================================================
# 全局實例
# ============================================================================

# 單例模式，整個應用共用一個 manager
trim_session_manager = TrimSessionManager()


# ============================================================================
# 便利函數（供外部調用）
# ============================================================================

def get_trim_manager() -> TrimSessionManager:
    """獲取 TrimSessionManager 實例"""
    return trim_session_manager
