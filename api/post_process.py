import re
import logging  
from lib.constant import CONTAINS_UNUSUAL, ONLY_UNUSUAL

logger = logging.getLogger(__name__)  


def post_process(text, audio_duration=None):
    """Post-process the transcribed text based on the specified strategy.

    :param text: str
        The transcribed text to be post-processed.
    :param audio_duration: float, optional
        Duration of the audio file in seconds.
    :return: tuple
        A tuple containing a boolean indicating if reprocessing is needed and the processed text.
    """
    if not isinstance(text, str):
        logger.error(f" | Input text is not a string: {type(text)} | ")
        return True, ""
    
    cleaned_text = text
    retry_flag = False
    
    # 1. First merge hyphenated words, then split English words from adjacent Chinese characters
    # A - B -> A-B
    cleaned_text = re.sub(r'([a-zA-Z0-9]+)\s*-\s*([a-zA-Z0-9]+)', r'\1-\2', cleaned_text)
    # XXABXX -> XX AB XX (but exclude cases where English/alphanumeric is directly followed by symbols)
    cleaned_text = re.sub(r'([^\sa-zA-Z0-9.-])([a-zA-Z0-9])', r'\1 \2', cleaned_text) 
    # For characters following English/alphanumeric, handle differently:
    # - If followed by Chinese characters, add space
    # - If followed by symbols, don't add space
    cleaned_text = re.sub(r'([a-zA-Z0-9])([\u4e00-\u9fff])', r'\1 \2', cleaned_text)  # English followed by Chinese
    cleaned_text = re.sub(r'([\u4e00-\u9fff])([a-zA-Z0-9])', r'\1 \2', cleaned_text)  # Chinese followed by English
    
    # 2. Remove leading/trailing whitespace
    cleaned_text = cleaned_text.strip() 
    
    # 3. Clean 【】 format and inside text
    if '【' in cleaned_text or '】' in cleaned_text:
        original_text = cleaned_text
        cleaned_text = re.sub(r'【[^】]*】', '', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        logger.warning(f" | Clean 【】format: '{original_text}' → '{cleaned_text}' | ")

    # Helper function to count mixed language units
    def count_mixed_language_units(text):
        """Count units in mixed language text: Chinese by characters, English by words"""
        # Text has already been preprocessed in step 1, so we can directly count
        
        # Count Chinese characters (using original text)
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        # Count English words (including letter+number combinations, hyphens, decimals)
        english_words = len(re.findall(r'\b[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*(?:\.[a-zA-Z0-9]+)*\b', text))
        # Count standalone numbers 
        numbers = len(re.findall(r'\b\d+\b', text))
        
        total_units = chinese_chars + english_words + numbers
        return total_units

    # 4. Check audio duration vs text length ratio using mixed language counting
    if audio_duration is not None and audio_duration > 0:
        # Always use mixed language counting: Chinese chars + English words + numbers
        total_units = count_mixed_language_units(cleaned_text)
        ratio = total_units / audio_duration
        min_ratio, max_ratio = 2.0, 6.0  # Adjusted range for mixed counting
        unit = "units"
        
        if ratio < min_ratio:
            logger.warning(f" | Text too short for audio duration: {ratio:.2f}{unit}/sec < {min_ratio}{unit}/sec (duration: {audio_duration}s | ")
            retry_flag = True
        elif ratio > max_ratio:
            logger.warning(f" | Text too long for audio duration: {ratio:.2f}{unit}/sec > {max_ratio}{unit}/sec (duration: {audio_duration}s | ")
            retry_flag = True

    # 5. Check for obvious format anomalies
    # if (cleaned_text.startswith('[') and cleaned_text.endswith(']')) or \
    #    (cleaned_text.startswith('{') and cleaned_text.endswith('}')) or \
    #    (cleaned_text.startswith('(') and cleaned_text.endswith(')')) or \
    #    '#' in cleaned_text or '**' in cleaned_text:
    #     logger.warning(f"Detected format markers: '{cleaned_text[:50]}...'")
    #     retry_flag = True
    #     return retry_flag, cleaned_text

    # 6. Check for word repetition hallucinations
    # Split by both spaces and Chinese punctuation
    words = re.split(r'[\s、，,]+', cleaned_text)
    # Filter out empty strings
    words = [word.strip() for word in words if word.strip()]
    
    if len(words) >= 4:
        # Check for obvious repeated words (3 consecutive identical words)
        for i in range(len(words) - 2):
            word = words[i]
            if len(word) > 2 and words[i] == words[i+1] == words[i+2]:
                logger.warning(f" | Repeated word hallucination: '{word}' appears 3+ times consecutively | ")
                retry_flag = True
                break
        
        # Check for repeated phrases (2-word combinations)
        if not retry_flag and len(words) >= 6:
            for i in range(len(words) - 3):
                phrase1 = ' '.join(words[i:i+2])
                phrase2 = ' '.join(words[i+2:i+4])
                if phrase1 == phrase2 and len(phrase1.strip()) > 4:
                    logger.warning(f" | Repeated phrase hallucination: '{phrase1}' appears consecutively | ")
                    retry_flag = True
                    break

    # 7. Check for unusual character patterns (excluding JSON-safe characters)
    if re.search(r'[^\w\s\u4e00-\u9fff.,!?;:""''()（），。！？；：、—–+=%$€£¥@&*/<>|\\-]', cleaned_text):
        unusual_chars = re.findall(r'[^\w\s\u4e00-\u9fff.,!?;:""''()（），。！？；：、—–+=%$€£¥@&*/<>|\\-]', cleaned_text)
        logger.warning(f" | Unusual characters detected: {set(unusual_chars)} in '{cleaned_text[:50]}...' | ")

    # 8. Common Hallucination Check
    text_for_check = re.sub(r'[^\w\s\u4e00-\u9fff]', '', cleaned_text)
    text_for_check = re.sub(r'\s+', ' ', text_for_check).strip()
    
    if any(phrase in text_for_check for phrase in CONTAINS_UNUSUAL):
        logger.warning(f" | Found common hallucination (contains) | ")
        for phrase in CONTAINS_UNUSUAL:
            if phrase in text_for_check:
                cleaned_text = cleaned_text.replace(phrase, '')
                logger.info(f" | Removed hallucination phrase: '{phrase}' | ")
        
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        retry_flag = True
        
    if any(phrase == text_for_check for phrase in ONLY_UNUSUAL):
        logger.warning(f" | Found common hallucination (exact match) | ")
        retry_flag = True
        cleaned_text = ""
    
    return retry_flag, cleaned_text

