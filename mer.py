import re
from jiwer import wer, cer
import os
import sys

def read_file_content(file_path):
    """
    讀取文件內容，支持 txt 和 docx 格式
    """
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return ""
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_ext == '.docx':
            try:
                from docx import Document
                doc = Document(file_path)
                content = []
                for paragraph in doc.paragraphs:
                    content.append(paragraph.text)
                return '\n'.join(content)
            except ImportError:
                print("需要安裝 python-docx 庫來讀取 docx 文件: pip install python-docx")
                return ""
        else:
            print(f"不支持的文件格式: {file_ext}")
            return ""
    except Exception as e:
        print(f"讀取文件時發生錯誤: {e}")
        return ""

def normalize_text(text):
    """
    標準化文本：去除符號、統一格式
    - 去除所有標點符號
    - 保留英文單詞間的空格
    - 統一成一行文本
    """
    # 去除換行符和多餘空白
    text = re.sub(r'\s+', ' ', text.replace('\n', ' ').strip())
    
    # 去除標點符號，但保留英文單詞間的空格
    # 先標記英文單詞位置
    words = []
    i = 0
    while i < len(text):
        if text[i].isalpha():
            # 英文字母開始
            word_start = i
            while i < len(text) and (text[i].isalpha() or text[i].isdigit()):
                i += 1
            words.append((word_start, i, text[word_start:i]))
        else:
            i += 1
    
    # 重建文本：中文字符連在一起，英文單詞用空格分隔
    result = ""
    last_pos = 0
    
    for start, end, word in words:
        # 添加前面的中文字符（去除符號）
        chinese_part = text[last_pos:start]
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', chinese_part)
        result += "".join(chinese_chars)
        
        # 添加英文單詞
        if result and result[-1] != ' ' and chinese_chars:
            # 如果前面有中文字符，不需要空格
            result += word
        else:
            # 如果前面是英文或開頭，需要空格分隔
            if result and not result.endswith(' '):
                result += ' '
            result += word
        
        last_pos = end
    
    # 處理最後剩餘的中文字符
    chinese_part = text[last_pos:]
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', chinese_part)
    result += "".join(chinese_chars)
    
    return result.strip()

def detect_language(text):
    """
    分離中英文文本
    """
    # 先標準化文本
    text = normalize_text(text)
    
    # 提取中文字符（連續）
    zh_chars = re.findall(r'[\u4e00-\u9fff]', text)
    zh_text = "".join(zh_chars)
    
    # 提取英文單詞（保持空格分隔）
    en_words = re.findall(r'[a-zA-Z]+', text)
    en_text = " ".join(en_words)
    
    return zh_text, en_text

def mixed_error_rate(reference, hypothesis):
    """
    計算混合錯誤率（MER）
    返回百分比格式
    """
    ref_zh, ref_en = detect_language(reference)
    hyp_zh, hyp_en = detect_language(hypothesis)
    
    # 計算 CER 和 WER
    cer_score = cer(ref_zh, hyp_zh) if ref_zh else 0
    wer_score = wer(ref_en, hyp_en) if ref_en else 0
    
    # 合併成 MER: 加權平均，權重為字數/單詞數
    zh_len = len(ref_zh)
    en_len = len(ref_en.split()) if ref_en.strip() else 0
    total_len = zh_len + en_len
    
    if total_len == 0:
        return 0.0
    print(f"CER: {cer_score * 100:.2f}%")
    print(f"WER: {wer_score * 100:.2f}%")
    mer = (cer_score * zh_len + wer_score * en_len) / total_len
    return mer * 100  # 轉換為百分比

def generate_comparison_file(reference, hypothesis, output_file="comparison.txt"):
    """
    生成對照文件，標示出 hypothesis 與 reference 的不同之處
    - reference: 正確答案
    - hypothesis: 識別結果
    - output_file: 輸出文件名
    """
    from difflib import SequenceMatcher
    import re
    
    # 標準化兩段文本
    ref_normalized = normalize_text(reference)
    hyp_normalized = normalize_text(hypothesis)
    
    # 使用 difflib 進行序列比對
    matcher = SequenceMatcher(None, ref_normalized, hyp_normalized)
    
    result_text = ""
    ref_pos = 0
    hyp_pos = 0
    
    for tag, ref_start, ref_end, hyp_start, hyp_end in matcher.get_opcodes():
        if tag == 'equal':
            # 相同的部分直接添加
            result_text += hyp_normalized[hyp_start:hyp_end]
        elif tag == 'replace':
            # 替換的部分：用 "" 標記錯誤，() 標記正確答案
            wrong_text = hyp_normalized[hyp_start:hyp_end]
            correct_text = ref_normalized[ref_start:ref_end]
            result_text += f'"{wrong_text}"({correct_text})'
        elif tag == 'delete':
            # reference 有但 hypothesis 沒有的部分（遺漏）
            missing_text = ref_normalized[ref_start:ref_end]
            result_text += f'""({missing_text})'
        elif tag == 'insert':
            # hypothesis 有但 reference 沒有的部分（多餘）
            extra_text = hyp_normalized[hyp_start:hyp_end]
            result_text += f'"{extra_text}"()'
    
    # 寫入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== 文本對照報告 ===\n")
        f.write("格式說明：\n")
        f.write('- "錯誤文本"(正確文本) - 表示識別錯誤\n')
        f.write('- "多餘文本"() - 表示多識別的內容\n')
        f.write('- ""(遺漏文本) - 表示漏識別的內容\n')
        f.write("\n" + "="*50 + "\n\n")
        f.write(result_text)
        f.write("\n\n=== 結束 ===")
    
    print(f"對照文件已生成: {output_file}")
    return result_text

# 範例 - 修改為讀取文件
# 支持命令行參數或預設文件名
if len(sys.argv) >= 3:
    reference_file = sys.argv[1]
    hypothesis_file = sys.argv[2]
else:
    # 預設文件名，請根據需要修改
    reference_file = "James_Q1_中文原文.docx"
    hypothesis_file = "Jame_Q1_zh.txt"
    print(f"使用預設文件名: {reference_file}, {hypothesis_file}")
    print("用法: python mer.py <參考文件> <假設文件>")

print(f"讀取參考文件: {reference_file}")
print(f"讀取假設文件: {hypothesis_file}")

# 從文件讀取內容
reference = read_file_content(reference_file)
hypothesis = read_file_content(hypothesis_file)

# 檢查是否成功讀取文件
if not reference:
    print(f"警告: 無法讀取參考文件 {reference_file}")
if not hypothesis:
    print(f"警告: 無法讀取假設文件 {hypothesis_file}")

if reference and hypothesis:
    mer_score = mixed_error_rate(reference, hypothesis)
    print(f"MER: {mer_score:.2f}%")

    # 生成對照文件
    print("\n=== 生成文本對照文件 ===")
    generate_comparison_file(reference, hypothesis, "text_comparison.txt")

    # 測試標準化功能
    print("\n=== 測試標準化功能 ===")
    test_ref = normalize_text(reference)
    test_hyp = normalize_text(hypothesis)
    print(f"標準化後參考文本長度: {len(test_ref)}")
    print(f"標準化後假設文本長度: {len(test_hyp)}")

    # 顯示分離後的中英文
    ref_zh, ref_en = detect_language(reference)
    print(f"\n中文字符數: {len(ref_zh)}")
    print(f"英文單詞數: {len(ref_en.split()) if ref_en.strip() else 0}")
    print(f"英文部分前100字符: {ref_en[:100]}...")
    print(f"中文部分前50字符: {ref_zh[:50]}...")
else:
    print("無法繼續執行，請檢查文件路徑和內容。")
