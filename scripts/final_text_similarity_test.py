#!/usr/bin/env python3
"""
Final Text Similarity Test

只使用每個 UID 的最後一筆音檔來測試 without_trim，
然後比較 with_trim vs without_trim 的最終文本相似度。

使用方式:
    1. 確保 main.py 正在運行且 ENABLE_TRIM = False
    2. 執行: python3 scripts/final_text_similarity_test.py
"""

import os
import sys
import json
import asyncio
import aiohttp
import argparse
from datetime import datetime
from collections import defaultdict
from difflib import SequenceMatcher
from typing import List, Dict, Tuple

# 配置
API_URL = "http://localhost:80/translate_pipeline"
AUDIO_DIR = "/mnt/audio/5857d384804b3e0e8086399423b99169"
WITH_TRIM_REPORT = "/mnt/benchmark_results/formal_with_trim_20260121_175519.json"
MEETING_ID = "5857d384804b3e0e8086399423b99169"


def parse_timestamp_from_filename(filename: str) -> datetime:
    """從檔名解析時間戳"""
    parts = filename.replace('.wav', '').split('_')
    if len(parts) >= 3:
        date_part = parts[1]
        time_part = parts[2].replace(';', ':')
        dt_str = f"{date_part} {time_part}"
        return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")
    return datetime.now()


def load_last_audio_per_uid() -> Dict[str, str]:
    """載入每個 UID 的最後一筆音檔"""
    uid_to_files = defaultdict(list)
    for f in os.listdir(AUDIO_DIR):
        if f.endswith('.wav') and '2026-01-21' in f:
            uid = f.split('_')[0]
            ts = parse_timestamp_from_filename(f)
            uid_to_files[uid].append((f, ts))
    
    # 取每個 UID 的最後一個檔案
    last_files = {}
    for uid, files in uid_to_files.items():
        sorted_files = sorted(files, key=lambda x: x[1])
        last_files[uid] = sorted_files[-1][0]  # 最後一個
    
    return last_files


def load_with_trim_final_texts() -> Dict[str, str]:
    """從 with_trim 報告中載入每個 UID 的 final_text"""
    with open(WITH_TRIM_REPORT) as f:
        report = json.load(f)
    
    return {
        uid: stats['final_text']
        for uid, stats in report['uid_statistics'].items()
    }


async def send_request(session: aiohttp.ClientSession, 
                       audio_path: str, 
                       uid: str) -> Tuple[bool, str]:
    """發送請求並返回結果文本"""
    # 從檔名解析時間戳
    filename = os.path.basename(audio_path)
    ts = parse_timestamp_from_filename(filename)
    times_str = ts.strftime("%Y-%m-%d %H:%M:%S.%f")
    
    data = aiohttp.FormData()
    data.add_field('meeting_id', MEETING_ID)
    data.add_field('device_id', '123')
    data.add_field('audio_uid', uid)
    data.add_field('times', times_str)
    data.add_field('o_lang', 'zh')
    data.add_field('t_lang', 'zh,en,ja,ko,de')
    data.add_field('prev_text', '')
    data.add_field('multi_strategy_transcription', '4')
    data.add_field('transcription_post_processing', 'true')
    data.add_field('multi_translate', 'true')
    
    with open(audio_path, 'rb') as f:
        audio_data = f.read()
    data.add_field('file', audio_data, filename=filename, content_type='audio/wav')
    
    try:
        async with session.post(API_URL, data=data, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status == 200:
                result = await resp.json()
                # 取得轉錄文本 - 從 data 欄位中取得
                data_obj = result.get('data', {})
                text = data_obj.get('transcription_text', '')
                # 確保是字串
                if not isinstance(text, str):
                    text = str(text) if text else ''
                return True, text
            else:
                return False, f"HTTP {resp.status}"
    except Exception as e:
        return False, str(e)


def calculate_similarity(text1: str, text2: str) -> float:
    """計算兩個文本的相似度 (0-100%)"""
    return SequenceMatcher(None, text1, text2).ratio() * 100


async def main(limit: int = None, concurrent: int = 10):
    """主程式"""
    print("="*70)
    print("🔍 Final Text Similarity Test")
    print("   比較 With Trim vs Without Trim 的最終文本相似度")
    print("="*70)
    
    # 載入 with_trim 的 final_text
    print("\n📋 載入 with_trim 報告...")
    with_trim_texts = load_with_trim_final_texts()
    print(f"   總 UID 數: {len(with_trim_texts)}")
    
    # 載入每個 UID 的最後一筆音檔
    print("\n📂 載入音檔（每個 UID 最後一筆）...")
    last_audio = load_last_audio_per_uid()
    
    # 找出有對應音檔的 UID
    valid_uids = [uid for uid in with_trim_texts.keys() if uid in last_audio]
    print(f"   有對應音檔的 UID: {len(valid_uids)}")
    
    if limit:
        valid_uids = valid_uids[:limit]
        print(f"   限制測試: 前 {limit} 個 UID")
    
    print(f"\n   將發送 {len(valid_uids)} 個請求（每個 UID 一個）")
    
    # 確認
    print("\n" + "-"*70)
    input("按 Enter 開始測試...")
    
    # 開始測試 - 一個一個順序執行
    async with aiohttp.ClientSession() as session:
        print(f"\n🚀 開始發送請求（順序執行）...")
        
        results = {}
        
        for i, uid in enumerate(valid_uids):
            audio_file = last_audio[uid]
            audio_path = os.path.join(AUDIO_DIR, audio_file)
            success, text = await send_request(session, audio_path, uid)
            print(f"\r[{i+1}/{len(valid_uids)}] {uid[:16]}... {'✓' if success else '✗'}", end='', flush=True)
            if success:
                results[uid] = text
    
    print(f"\n\n✅ 完成！成功 {len(results)}/{len(valid_uids)} 個")
    
    # 計算相似度
    print("\n" + "="*70)
    print("📊 文本相似度比較")
    print("="*70)
    
    similarities = []
    comparison_results = []
    
    for uid in valid_uids:
        if uid not in results:
            continue
        
        with_trim_text = with_trim_texts[uid]
        without_trim_text = results[uid]
        sim = calculate_similarity(with_trim_text, without_trim_text)
        similarities.append(sim)
        
        comparison_results.append({
            'uid': uid,
            'similarity': sim,
            'with_trim': with_trim_text,
            'without_trim': without_trim_text
        })
    
    # 排序顯示（從低到高）
    comparison_results.sort(key=lambda x: x['similarity'])
    
    print(f"\n{'序號':<6} {'相似度':>10} {'With Trim':<30} {'Without Trim':<30}")
    print("-"*80)
    
    for i, r in enumerate(comparison_results[:20]):  # 顯示前 20 個最不相似的
        wt = r['with_trim'][:28] + '..' if len(r['with_trim']) > 28 else r['with_trim']
        wot = r['without_trim'][:28] + '..' if len(r['without_trim']) > 28 else r['without_trim']
        sim_val = r['similarity']
        print(f"{i+1:<6} {sim_val:>9.1f}% {wt:<30} {wot:<30}")
    
    # 統計
    print("\n" + "="*70)
    print("📈 統計摘要")
    print("="*70)
    
    if similarities:
        avg_sim = sum(similarities) / len(similarities)
        min_sim = min(similarities)
        max_sim = max(similarities)
        
        # 分布
        perfect = sum(1 for s in similarities if s == 100)
        high = sum(1 for s in similarities if 90 <= s < 100)
        medium = sum(1 for s in similarities if 70 <= s < 90)
        low = sum(1 for s in similarities if s < 70)
        
        print(f"\n總比較 UID 數: {len(similarities)}")
        print(f"\n相似度統計:")
        print(f"  平均: {avg_sim:.1f}%")
        print(f"  最低: {min_sim:.1f}%")
        print(f"  最高: {max_sim:.1f}%")
        print(f"\n分布:")
        print(f"  100%（完全相同）: {perfect} ({perfect/len(similarities)*100:.1f}%)")
        print(f"  90-99%（高度相似）: {high} ({high/len(similarities)*100:.1f}%)")
        print(f"  70-89%（中度相似）: {medium} ({medium/len(similarities)*100:.1f}%)")
        print(f"  <70%（低相似度）: {low} ({low/len(similarities)*100:.1f}%)")
    
    # 儲存結果
    output_path = f"/mnt/benchmark_results/similarity_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_compared': len(similarities),
                'avg_similarity': avg_sim if similarities else 0,
                'min_similarity': min_sim if similarities else 0,
                'max_similarity': max_sim if similarities else 0,
            },
            'distribution': {
                'perfect_100': perfect if similarities else 0,
                'high_90_99': high if similarities else 0,
                'medium_70_89': medium if similarities else 0,
                'low_below_70': low if similarities else 0,
            },
            'details': comparison_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 結果已儲存: {output_path}")
    print("\n" + "="*70)
    print("✅ 測試完成!")
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Final Text Similarity Test')
    parser.add_argument('--limit', type=int, help='限制測試 UID 數量')
    parser.add_argument('--concurrent', type=int, default=10, help='並行數（預設 10）')
    args = parser.parse_args()
    
    asyncio.run(main(limit=args.limit, concurrent=args.concurrent))
