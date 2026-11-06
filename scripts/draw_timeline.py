"""
從 tts_monitor.csv 產生 Gantt-style timeline 圖，每 30 秒一張。
X 軸: 0~30 秒 (window 內相對時間)
Y 軸: 每個 segment 一行
顏色: recording=綠, transcribe=藍, translate=橘, tts_batch=紅
"""
import os
import csv
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

CSV_PATH = "./static/tts_monitor.csv"
OUTPUT_DIR = "./static/charts"
WINDOW_SEC = 30

COLORS = {
    "recording": "#4CAF50",    # 綠
    "transcribe": "#2196F3",   # 藍
    "translate": "#FF9800",    # 橘
    "tts_batch": "#F44336",    # 紅
    "playback": "#9C27B0",     # 紫
}

# 個別語言 TTS 顏色 (淺色系)
TTS_LANG_COLORS = {
    "zh": "#FFCDD2",
    "en": "#C8E6C9",
    "ja": "#BBDEFB",
    "ko": "#FFE0B2",
    "de": "#E1BEE7",
}


def load_csv(path):
    """讀取 CSV，回傳 list of dicts"""
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['segment_idx'] = int(row['segment_idx'])
            row['start'] = float(row['start'])
            row['end'] = float(row['end'])
            row['duration'] = float(row['duration'])
            # rec_real_time: 收音真實開始時間 (僅 recording 行有值)
            rrt = row.get('rec_real_time', '')
            row['rec_real_time'] = float(rrt) if rrt else None
            rows.append(row)
    return rows


def draw_window(rows_in_window, window_start, window_end, window_idx, output_dir):
    """畫一個 30 秒窗口的 Gantt 圖"""
    
    # 找出此窗口涉及的 segments (main stages only)
    main_stages = {"recording", "transcribe", "translate", "tts_batch", "playback"}
    
    # 收集每個 segment 的 main stage bars
    seg_data = defaultdict(list)  # {segment_idx: [(stage, rel_start, rel_end), ...]}
    seg_tts_lang = defaultdict(list)  # {segment_idx: [(lang, rel_start, rel_end, audio_dur), ...]}
    seg_rec_real_time = {}  # {segment_idx: rec_real_time (absolute seconds from session start)}
    
    for row in rows_in_window:
        seg_idx = row['segment_idx']
        stage = row['stage']
        
        # clip to window
        bar_start = max(row['start'], window_start) - window_start
        bar_end = min(row['end'], window_end) - window_start
        
        if bar_end <= bar_start:
            continue
        
        if stage in main_stages:
            seg_data[seg_idx].append((stage, bar_start, bar_end))
            if stage == "recording":
                # rec_real_time 欄位優先，沒有則用 recording 的 absolute start
                rrt = row.get('rec_real_time')
                if rrt is not None:
                    seg_rec_real_time[seg_idx] = rrt
                else:
                    seg_rec_real_time[seg_idx] = row['start']
        elif stage.startswith("tts_") and stage != "tts_batch":
            lang = row.get('language', '')
            audio_dur_str = row.get('tts_output_audio_duration', '')
            audio_dur = float(audio_dur_str) if audio_dur_str else 0
            seg_tts_lang[seg_idx].append((lang, bar_start, bar_end, audio_dur))
    
    if not seg_data:
        return
    
    # Sort segments
    all_segs = sorted(seg_data.keys())
    n_segs = len(all_segs)
    
    # Figure height based on segment count
    fig_height = max(4, n_segs * 0.8 + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    
    bar_height = 0.6
    tts_lang_height = 0.08
    
    for i, seg_idx in enumerate(all_segs):
        y_pos = n_segs - 1 - i  # top to bottom
        
        # Draw main stage bars
        for stage, s, e in seg_data[seg_idx]:
            color = COLORS.get(stage, '#999999')
            ax.barh(y_pos, e - s, left=s, height=bar_height, 
                    color=color, edgecolor='white', linewidth=0.5, alpha=0.85)
            
            # 在 bar 上標示 duration
            dur = e - s
            if dur > 0.8:  # 夠長才標
                label = f"{dur:.1f}s"
                ax.text(s + dur/2, y_pos, label, ha='center', va='center', 
                        fontsize=7, color='white', fontweight='bold')
        
        # Draw per-language TTS bars (thin bars below main bar)
        if seg_idx in seg_tts_lang:
            langs = seg_tts_lang[seg_idx]
            for j, (lang, s, e, audio_dur) in enumerate(langs):
                y_offset = y_pos - bar_height/2 - 0.02 - (j + 1) * (tts_lang_height + 0.01)
                color = TTS_LANG_COLORS.get(lang, '#CCCCCC')
                ax.barh(y_offset, e - s, left=s, height=tts_lang_height,
                        color=color, edgecolor='none', alpha=0.9)
        
        # 在每段最前面標示收音真實時間點
        if seg_idx in seg_rec_real_time:
            rec_time = seg_rec_real_time[seg_idx]
            # 找到這段最早的 bar 起點 (window-relative)
            all_starts = [s for _, s, _ in seg_data[seg_idx]]
            min_start = min(all_starts) if all_starts else 0
            ax.text(min_start, y_pos + bar_height/2 + 0.05, f"@{rec_time:.1f}s",
                    ha='left', va='bottom', fontsize=7, color='#333333',
                    fontstyle='italic')
    
    # Formatting
    ax.set_xlim(0, WINDOW_SEC)
    ax.set_ylim(-1, n_segs)
    ax.set_xlabel('Time (seconds)', fontsize=10)
    ax.set_yticks(range(n_segs))
    ax.set_yticklabels([f'Seg {all_segs[n_segs - 1 - i]}' for i in range(n_segs)], fontsize=9)
    ax.set_title(f'Pipeline Timeline — Window {window_idx+1} ({window_start:.0f}s ~ {window_end:.0f}s)', fontsize=12)
    
    # Grid
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, alpha=0.3, linestyle='--')
    ax.axvline(x=0, color='gray', linewidth=0.5)
    
    # Legend
    legend_patches = [
        mpatches.Patch(color=COLORS["recording"], label='Recording'),
        mpatches.Patch(color=COLORS["transcribe"], label='Transcribe'),
        mpatches.Patch(color=COLORS["translate"], label='Translate'),
        mpatches.Patch(color=COLORS["tts_batch"], label='TTS'),
        mpatches.Patch(color=COLORS["playback"], label='Playback'),
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=8, framealpha=0.9)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"window_{window_idx+1:02d}_{window_start:.0f}s-{window_end:.0f}s.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    rows = load_csv(CSV_PATH)
    if not rows:
        print("No data in CSV")
        return
    
    # Find total time range
    max_time = max(r['end'] for r in rows)
    n_windows = math.ceil(max_time / WINDOW_SEC)
    
    print(f"Total time: {max_time:.1f}s, generating {n_windows} charts (each {WINDOW_SEC}s)...")
    
    for w in range(n_windows):
        w_start = w * WINDOW_SEC
        w_end = (w + 1) * WINDOW_SEC
        
        # Filter rows that overlap with this window
        window_rows = [r for r in rows if r['end'] > w_start and r['start'] < w_end]
        
        if window_rows:
            draw_window(window_rows, w_start, w_end, w, OUTPUT_DIR)
    
    print(f"\nDone! {n_windows} charts saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
