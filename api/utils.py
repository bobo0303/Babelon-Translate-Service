import os
from typing import Dict
from datetime import datetime

# 全局變量追蹤當前狀態
current_meeting_id = None
audio_uid_line_map = {}  # {audio_uid: line_number}
audio_uid_times_map = {}  # {audio_uid: latest_times}
current_line_number = 0

def write_txt(zh_text: str, en_text: str, de_text: str, meeting_id: str, audio_uid: str, times: str = None):
    """
    寫入翻譯結果到txt文件
    
    Args:
        zh_text: 中文翻譯
        en_text: 英文翻譯  
        de_text: 德文翻譯
        meeting_id: 會議ID
        audio_uid: 音頻段落ID
        times: 時間戳字符串，用於判斷是否為最新版本
    """
    global current_meeting_id, audio_uid_line_map, audio_uid_times_map, current_line_number
    
    # 檢查是否需要開新文件（meeting_id 改變）
    if current_meeting_id != meeting_id:
        current_meeting_id = meeting_id
        audio_uid_line_map.clear()
        audio_uid_times_map.clear()
        current_line_number = 0
        
        # 清空舊文件或創建新文件
        for lang, text in [("zh", zh_text), ("en", en_text), ("de", de_text)]:
            filename = f"transcription_{lang}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write("")  # 清空文件
    
    # 如果提供了times，檢查是否為更新的版本
    if times is not None and audio_uid in audio_uid_times_map:
        if times <= audio_uid_times_map[audio_uid]:
            # 當前times比已有的舊或相同，不寫入
            print(f"Skipping write for audio_uid {audio_uid}: current times {times} <= existing times {audio_uid_times_map[audio_uid]}")
            return
    
    # 更新times記錄
    if times is not None:
        audio_uid_times_map[audio_uid] = times
    
    # 檢查audio_uid是否已存在
    if audio_uid in audio_uid_line_map:
        # 同一個audio_uid，覆寫對應行
        line_number = audio_uid_line_map[audio_uid]
        _update_line_in_files(meeting_id, line_number, zh_text, en_text, de_text)
    else:
        # 新的audio_uid，新增一行
        current_line_number += 1
        audio_uid_line_map[audio_uid] = current_line_number
        _append_line_to_files(meeting_id, zh_text, en_text, de_text)

def _update_line_in_files(meeting_id: str, line_number: int, zh_text: str, en_text: str, de_text: str):
    """更新文件中指定行的內容"""
    for lang, text in [("zh", zh_text), ("en", en_text), ("de", de_text)]:
        filename = f"transcription_{lang}.txt"
        
        # 讀取所有行
        lines = []
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                lines = f.readlines()
        
        # 確保有足夠的行數
        while len(lines) < line_number:
            lines.append("\n")
        
        # 更新指定行（line_number從1開始，所以要-1）
        lines[line_number - 1] = text + "\n"
        
        # 寫回文件
        with open(filename, "w", encoding="utf-8") as f:
            f.writelines(lines)

def _append_line_to_files(meeting_id: str, zh_text: str, en_text: str, de_text: str):
    """在文件末尾新增一行"""
    for lang, text in [("zh", zh_text), ("en", en_text), ("de", de_text)]:
        filename = f"transcription_{lang}.txt"
        with open(filename, "a", encoding="utf-8") as f:
            f.write(text + "\n")