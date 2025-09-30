import re
import logging  

  
logger = logging.getLogger(__name__)  


def post_process(text):
    """Post-process the transcribed text based on the specified strategy.

    :param text: str
        The transcribed text to be post-processed.
    :return: tuple
        A tuple containing a boolean indicating if reprocessing is needed and the processed text.
    """
    if not isinstance(text, str):
        logger.error(f"Input text is not a string: {type(text)}")
        return True, ""
    
    cleaned_text = text
    retry_flag = False
    
    # 1. Remove leading/trailing whitespace
    cleaned_text = cleaned_text.strip() 
    
    # 2. Clean 【】 format and inside text
    if '【' in cleaned_text or '】' in cleaned_text:
        original_text = cleaned_text
        cleaned_text = re.sub(r'【[^】]*】', '', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        logger.warning(f"Clean 【】format: '{original_text}' → '{cleaned_text}'")

    # 3. Check if text is empty or too short after cleaning
    if not cleaned_text or len(cleaned_text.replace('.', '').replace('。', '').replace(',', '').replace('，', '').strip()) <= 1:
        logger.warning(f"Text too short or empty after cleaning: '{cleaned_text}'")
        retry_flag = True
        return retry_flag, cleaned_text

    # 4. Check for obvious format anomalies
    # if (cleaned_text.startswith('[') and cleaned_text.endswith(']')) or \
    #    (cleaned_text.startswith('{') and cleaned_text.endswith('}')) or \
    #    (cleaned_text.startswith('(') and cleaned_text.endswith(')')) or \
    #    '#' in cleaned_text or '**' in cleaned_text:
    #     logger.warning(f"Detected format markers: '{cleaned_text[:50]}...'")
    #     retry_flag = True
    #     return retry_flag, cleaned_text

    # 4. Check for word repetition hallucinations
    words = cleaned_text.split()
    if len(words) >= 4:
        # Check for obvious repeated words (3 consecutive identical words)
        for i in range(len(words) - 2):
            word = words[i]
            if len(word) > 2 and words[i] == words[i+1] == words[i+2]:
                logger.warning(f"Repeated word hallucination: '{word}' appears 3+ times consecutively")
                retry_flag = True
                break
        
        # Check for repeated phrases (2-word combinations)
        if not retry_flag and len(words) >= 6:
            for i in range(len(words) - 3):
                phrase1 = ' '.join(words[i:i+2])
                phrase2 = ' '.join(words[i+2:i+4])
                if phrase1 == phrase2 and len(phrase1.strip()) > 4:
                    logger.warning(f"Repeated phrase hallucination: '{phrase1}' appears consecutively")
                    retry_flag = True
                    break

    # 5. Check for unusual character patterns (excluding JSON-safe characters)
    if re.search(r'[^\w\s\u4e00-\u9fff.,!?;:""''()（），。！？；：、\-—–+=%$€£¥@&*/<>|\\]', cleaned_text):
        unusual_chars = re.findall(r'[^\w\s\u4e00-\u9fff.,!?;:""''()（），。！？；：、\-—–+=%$€£¥@&*/<>|\\]', cleaned_text)
        logger.warning(f"Unusual characters detected: {set(unusual_chars)} in '{cleaned_text[:50]}...'")

    return retry_flag, cleaned_text