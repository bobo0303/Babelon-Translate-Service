import os
from typing import Dict
from datetime import datetime

# Global variables to track current state
current_meeting_id = None
audio_uid_line_map = {}  # {audio_uid: line_number}
audio_uid_times_map = {}  # {audio_uid: latest_times}
current_line_number = 0

def write_txt(zh_text: str, en_text: str, de_text: str, meeting_id: str, audio_uid: str, times: str = None):
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
        for lang, text in [("zh", zh_text), ("en", en_text), ("de", de_text)]:
            filename = f"transcription_{lang}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write("")  # Clear file
    
    # If times is provided, check if this is a more recent version
    if times is not None and audio_uid in audio_uid_times_map:
        if times <= audio_uid_times_map[audio_uid]:
            # Current times is older than or equal to existing, don't write
            print(f"Skipping write for audio_uid {audio_uid}: current times {times} <= existing times {audio_uid_times_map[audio_uid]}")
            return
    
    # Update times record
    if times is not None:
        audio_uid_times_map[audio_uid] = times
    
    # Check if audio_uid already exists
    if audio_uid in audio_uid_line_map:
        # Same audio_uid, overwrite corresponding line
        line_number = audio_uid_line_map[audio_uid]
        _update_line_in_files(line_number, zh_text, en_text, de_text)
    else:
        # New audio_uid, add a new line
        current_line_number += 1
        audio_uid_line_map[audio_uid] = current_line_number
        _append_line_to_files(zh_text, en_text, de_text)

def _update_line_in_files(line_number: int, zh_text: str, en_text: str, de_text: str):
    """Update the content of a specific line in files"""
    for lang, text in [("zh", zh_text), ("en", en_text), ("de", de_text)]:
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

def _append_line_to_files(zh_text: str, en_text: str, de_text: str):
    """Append a new line to the end of files"""
    for lang, text in [("zh", zh_text), ("en", en_text), ("de", de_text)]:
        filename = f"transcription_{lang}.txt"
        with open(filename, "a", encoding="utf-8") as f:
            f.write(text + "\n") 