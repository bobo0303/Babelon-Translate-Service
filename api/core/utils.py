import os
import re
from typing import Dict
from datetime import datetime

# Global variables to track current state
current_meeting_id = None
audio_uid_line_map = {}  # {audio_uid: line_number}
audio_uid_times_map = {}  # {audio_uid: latest_times}
current_line_number = 0

def write_txt(zh_text: str, en_text: str, de_text: str, ja_text: str, ko_text: str, meeting_id: str, audio_uid: str, times: str = None):
    """
    Write translation results to txt files
    
    Args:
        zh_text: Chinese translation
        en_text: English translation  
        de_text: German translation
        meeting_id: Meeting ID
        audio_uid: Audio segment ID
        times: Timestamp string, used to determine if this is the latest version
    """
    global current_meeting_id, audio_uid_line_map, audio_uid_times_map, current_line_number
    
    # Check if we need to start a new file (meeting_id changed)
    if current_meeting_id != meeting_id:
        current_meeting_id = meeting_id
        audio_uid_line_map.clear()
        audio_uid_times_map.clear()
        current_line_number = 0
        
        # Clear old files or create new files
        for lang, text in [("zh", zh_text), ("en", en_text), ("de", de_text), ("ja", ja_text), ("ko", ko_text)]:
            filename = f"transcription_{lang}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write("")  # Clear file
    
    # Update times record (removed time comparison check as it was causing data loss)
    if times is not None:
        audio_uid_times_map[audio_uid] = times
    
    # Check if audio_uid already exists
    if audio_uid in audio_uid_line_map:
        # Same audio_uid, overwrite corresponding line
        line_number = audio_uid_line_map[audio_uid]
        _update_line_in_files(line_number, zh_text, en_text, de_text, ja_text, ko_text)
    else:
        # New audio_uid, add a new line
        current_line_number += 1
        audio_uid_line_map[audio_uid] = current_line_number
        _append_line_to_files(zh_text, en_text, de_text, ja_text, ko_text)

def _update_line_in_files(line_number: int, zh_text: str, en_text: str, de_text: str, ja_text: str, ko_text: str):
    """Update the content of a specific line in files"""
    for lang, text in [("zh", zh_text), ("en", en_text), ("de", de_text), ("ja", ja_text), ("ko", ko_text)]:
        filename = f"transcription_{lang}.txt"
        
        # Read all lines
        lines = []
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                lines = f.readlines()
        
        # Ensure we have enough lines
        while len(lines) < line_number:
            lines.append("\n")
        
        # Update the specified line (line_number starts from 1, so subtract 1)
        lines[line_number - 1] = text + "\n"
        
        # Write back to file
        with open(filename, "w", encoding="utf-8") as f:
            f.writelines(lines)

def _append_line_to_files(zh_text: str, en_text: str, de_text: str, ja_text: str, ko_text: str):
    """Append a new line to the end of files"""
    for lang, text in [("zh", zh_text), ("en", en_text), ("de", de_text), ("ja", ja_text), ("ko", ko_text)]:
        filename = f"transcription_{lang}.txt"
        with open(filename, "a", encoding="utf-8") as f:
            f.write(text + "\n")
            
def remove_punctuation(translation_dict: dict) -> dict:
    """
    Remove punctuation from translated text in all languages.
    Preserves: % symbol and decimal points in numbers (e.g., 3.14, 0.5)
    
    Args:
        translation_dict: Dictionary with language codes as keys and translated text as values.
    Returns:
        Dictionary with punctuation removed from each language's text.
    """
    cleaned_dict = {}
    
    for lang, text in translation_dict.items():
        # Protect decimal points in numbers (e.g., 3.14, 0.5)
        text_protected = re.sub(r'(\d)\.(\d)', r'\1<DOT>\2', text)
        
        # Remove punctuation except % (keep word chars, spaces, %, and temporary markers)
        text_no_punct = re.sub(r'[^\w\s%<>]', ' ', text_protected)
        
        # Restore decimal points
        text_restored = text_no_punct.replace('<DOT>', '.')
        
        # Standardize whitespace
        text_standardized = re.sub(r'\s+', ' ', text_restored).strip()
        cleaned_dict[lang] = text_standardized
        
    return cleaned_dict


def format_text_spacing(translation_dict: dict) -> dict:
    """
    Format spacing between CJK (Chinese/Japanese/Korean) and non-CJK characters.
    
    Rules:
    - Remove spaces between CJK characters
    - Ensure spaces between CJK and non-CJK (English/numbers)
    
    Args:
        translation_dict: Dictionary with language codes as keys and text as values.
    Returns:
        Dictionary with properly formatted spacing.
    """
    # CJK character ranges: Chinese, Japanese (Hiragana, Katakana), Korean (Hangul)
    cjk_pattern = r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]'
    
    formatted_dict = {}
    
    for lang, text in translation_dict.items():
        # Step 1: Remove spaces between CJK characters
        # Match: CJK + spaces + CJK → Remove spaces
        text = re.sub(f'({cjk_pattern})\\s+({cjk_pattern})', r'\1\2', text)
        
        # Step 2: Ensure space between CJK and non-CJK (alphanumeric)
        # Match: CJK + (no space) + alphanumeric → Add space
        text = re.sub(f'({cjk_pattern})([a-zA-Z0-9%])', r'\1 \2', text)
        # Match: alphanumeric + (no space) + CJK → Add space
        text = re.sub(f'([a-zA-Z0-9%])({cjk_pattern})', r'\1 \2', text)
        
        # Step 3: Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        formatted_dict[lang] = text
        
    return formatted_dict
    
def format_cleaning(translation_dict: dict) -> dict:
    """
    Apply formatting and cleaning to translation dictionary.
    
    Args:
        translation_dict: Dictionary with language codes as keys and text as values.        
    Returns:
        Dictionary with formatted and cleaned text for each language.
    """
    try:    
        cleaned_dict = remove_punctuation(translation_dict)
        formatted_dict = format_text_spacing(cleaned_dict)
        return formatted_dict
    except Exception as e:
        print(f" | Error in format_cleaning: {e} | ")
        return translation_dict

# Response tracking for race condition prevention
import uuid
import threading

class ResponseTracker:
    """
    Tracks pending translation requests to prevent race conditions where older 
    responses return after newer ones.
    
    Each request is tracked with:
    - audio_uid: identifies the audio being processed
    - task_id: unique identifier for each request
    - times: timestamp for request ordering
    - cancelled: flag to mark if request should be ignored
    """
    
    def __init__(self):
        # Structure: {audio_uid: {task_id: {"times": datetime, "cancelled": bool}}}
        self.pending_requests = {}
        self.lock = threading.Lock()
    
    def register_request(self, audio_uid: str, times: str) -> str:
        """
        Register a new request and return its unique task_id.
        
        Args:
            audio_uid: The audio file identifier
            times: Timestamp string for the request
            
        Returns:
            task_id: Unique identifier for this request
        """
        task_id = str(uuid.uuid4())
        
        with self.lock:
            if audio_uid not in self.pending_requests:
                self.pending_requests[audio_uid] = {}
            
            self.pending_requests[audio_uid][task_id] = {
                "times": times,
                "cancelled": False
            }
        
        return task_id
    
    def check_cancelled(self, audio_uid: str, task_id: str) -> bool:
        """
        Check if a request has been cancelled.
        
        Args:
            audio_uid: The audio file identifier
            task_id: The request's unique identifier
            
        Returns:
            True if cancelled, False otherwise
        """
        with self.lock:
            if audio_uid in self.pending_requests:
                if task_id in self.pending_requests[audio_uid]:
                    return self.pending_requests[audio_uid][task_id]["cancelled"]
        return False
    
    def complete_and_cancel_older(self, audio_uid: str, task_id: str, times: str):
        """
        Mark older pending requests for the same audio_uid as cancelled.
        This is called when a request completes successfully.
        
        Args:
            audio_uid: The audio file identifier
            task_id: The completing request's identifier
            times: The completing request's timestamp
        """
        with self.lock:
            if audio_uid in self.pending_requests:
                for other_task_id, info in self.pending_requests[audio_uid].items():
                    if other_task_id != task_id:
                        # Compare timestamps - cancel if older
                        if info["times"] < times:
                            if not info["cancelled"]:
                                info["cancelled"] = True
                                from lib.core.logging_config import setup_application_logger
                                logger = setup_application_logger(__name__)
                                logger.info(f" | Task {other_task_id} (audio_uid: {audio_uid}, times: {info['times']}) cancelled due to newer request (times: {times}). [QUEUED] | ")
    
    def cleanup(self, audio_uid: str, task_id: str):
        """
        Remove a completed request from tracking.
        
        Args:
            audio_uid: The audio file identifier
            task_id: The request's unique identifier
        """
        with self.lock:
            if audio_uid in self.pending_requests:
                if task_id in self.pending_requests[audio_uid]:
                    del self.pending_requests[audio_uid][task_id]
                
                # Clean up empty audio_uid entries
                if not self.pending_requests[audio_uid]:
                    del self.pending_requests[audio_uid] 