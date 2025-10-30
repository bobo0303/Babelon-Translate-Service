import re
import logging
import logging.handlers
from lib.constant import CONTAINS_UNUSUAL, ONLY_UNUSUAL, Q1, Q3, IQR_RATIO, TOLERANCE_RATE

logger = logging.getLogger(__name__)  
  
# Configure logger settings (if not already configured)  
if not logger.handlers:  
    log_format = "%(asctime)s - %(message)s"  
    log_file = "logs/app.log"  
    logging.basicConfig(level=logging.INFO, format=log_format)  
  
    # Create file handler  
    file_handler = logging.handlers.RotatingFileHandler(  
        log_file, maxBytes=10*1024*1024, backupCount=5  
    )  
    file_handler.setFormatter(logging.Formatter(log_format))  
  
    # Create console handler  
    console_handler = logging.StreamHandler()  
    console_handler.setFormatter(logging.Formatter(log_format))  
  
    logger.addHandler(file_handler)  
    logger.addHandler(console_handler)  
  
logger.setLevel(logging.INFO)  
logger.propagate = False  

def post_process(text, audio_duration=None, prompt_name=None):
    """Post-process the transcribed text based on the specified strategy.

    :param text: str
        The transcribed text to be post-processed.
    :param audio_duration: float, optional
        Duration of the audio file in seconds.
    :return: tuple
        A tuple containing a boolean indicating if reprocessing is needed and the processed text.
    """
    try:
        if not isinstance(text, str):
            logger.error(f" | Input text is not a string: {type(text)} | ")
            return True, ""
        
        cleaned_text = text
        retry_flag = False
        
        # 1. First merge hyphenated words, then split English words from adjacent Chinese characters
        try:
            # A - B -> A-B
            cleaned_text = re.sub(r'([a-zA-Z0-9]+)\s*-\s*([a-zA-Z0-9]+)', r'\1-\2', cleaned_text)
            # XXABXX -> XX AB XX (but exclude cases where English/alphanumeric is directly followed by symbols)
            cleaned_text = re.sub(r'([^\sa-zA-Z0-9.-])([a-zA-Z0-9])', r'\1 \2', cleaned_text) 
            # For characters following English/alphanumeric, handle differently:
            # - If followed by Chinese characters, add space
            # - If followed by symbols, don't add space
            cleaned_text = re.sub(r'([a-zA-Z0-9])([\u4e00-\u9fff])', r'\1 \2', cleaned_text)  # English followed by Chinese
            cleaned_text = re.sub(r'([\u4e00-\u9fff])([a-zA-Z0-9])', r'\1 \2', cleaned_text)  # Chinese followed by English
        except Exception as e:
            logger.error(f" | Step 1 (text normalization) error: {e} | ")
        
        # 2. Remove leading/trailing whitespace
        try:
            cleaned_text = cleaned_text.strip() 
        except Exception as e:
            logger.error(f" | Step 2 (whitespace removal) error: {e} | ")
        
        # 3. Clean 【】 format and inside text
        try:
            if '【' in cleaned_text or '】' in cleaned_text:
                original_text = cleaned_text
                cleaned_text = re.sub(r'【[^】]*】', '', cleaned_text)
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
                logger.warning(f" | Clean 【】format: '{original_text}' → '{cleaned_text}' | ")
        except Exception as e:
            logger.error(f" | Step 3 (bracket cleaning) error: {e} | ")

        # Helper function to count mixed language units
        def count_mixed_language_units(text):
            """Count units in mixed language text: Chinese by characters, English by words"""
            try:
                # Text has already been preprocessed in step 1, so we can directly count
                
                # Count Chinese characters (using original text)
                chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
                # Count English words (letters only, including hyphens and dots)
                english_words = len(re.findall(r'\b[a-zA-Z]+(?:-[a-zA-Z]+)*(?:\.[a-zA-Z]+)*\b', text))
                # Count standalone numbers (including decimals)
                numbers = len(re.findall(r'\b\d+(?:\.\d+)?\b', text))
                
                total_units = chinese_chars + english_words + numbers
                return total_units
            except Exception as e:
                logger.error(f" | count_mixed_language_units error: {e} | ")
                return 0

        # 4. Check audio duration vs text length ratio using IQR-based dynamic range
        try:
            if audio_duration is not None and audio_duration > 0:
                # Always use mixed language counting: Chinese chars + English words + numbers
                total_units = count_mixed_language_units(cleaned_text)
                
                # Get IQR-based range for the specific duration
                duration_seconds = round(audio_duration)
                if 1 <= duration_seconds <= 20:
                    # Array index is duration - 1 (since arrays are 0-indexed)
                    q1_value = Q1[duration_seconds - 1]
                    q3_value = Q3[duration_seconds - 1]
                    
                    # Calculate IQR bounds for expected absolute unit counts
                    iqr = q3_value - q1_value
                    min_units = max(0, q1_value - IQR_RATIO * iqr)  # Ensure non-negative
                    max_units = q3_value + IQR_RATIO * iqr
                    
                    # Apply tolerance rate for flexibility
                    min_units *= (1 - TOLERANCE_RATE)
                    max_units *= (1 + TOLERANCE_RATE)
                    
                    unit = "units"
                    
                    if total_units < min_units:
                        logger.warning(f" | Text too short for audio duration: {total_units}{unit} < {min_units:.1f}{unit} (IQR-based, duration: {audio_duration}s) | ")
                        retry_flag = True
                    elif total_units > max_units:
                        logger.warning(f" | Text too long for audio duration: {total_units}{unit} > {max_units:.1f}{unit} (IQR-based, duration: {audio_duration}s) | ")
                        retry_flag = True
                elif duration_seconds == 0:
                    pass  # Skip duration 0 case
                else:
                    # Fallback to ratio-based method for durations outside 1-20 seconds
                    ratio = total_units / audio_duration
                    min_ratio, max_ratio = 0.68, 9.14
                    unit = "units/sec"
                    
                    if ratio < min_ratio:
                        logger.warning(f" | Text too short for audio duration: {ratio:.2f}{unit} < {min_ratio}{unit} (fallback method, duration: {audio_duration}s) | ")
                        retry_flag = True
                    elif ratio > max_ratio:
                        logger.warning(f" | Text too long for audio duration: {ratio:.2f}{unit} > {max_ratio}{unit} (fallback method, duration: {audio_duration}s) | ")
                        retry_flag = True
        except Exception as e:
            logger.error(f" | Step 4 (audio duration check) error: {e} | ")

        # 5. Check for prompt leakage - detect if transcription contains prompt words
        try:
            if prompt_name is not None and prompt_name.strip() != "":
                # Extract individual words from prompt_name, normalize for comparison
                prompt_words = []
                # Remove punctuation and split into words
                prompt_clean = re.sub(r'[.,!?;:\u3002\uff0c\uff01\uff1f\uff1b\uff1a\u201c\u201d\u2018\u2019\uff08\uff09\u3001]', ' ', prompt_name.strip())
                prompt_words = [word.strip().lower() for word in prompt_clean.split() if word.strip() and len(word.strip()) > 1]
                
                if prompt_words:
                    # Normalize text for comparison (remove punctuation, convert to lowercase)
                    text_for_comparison = re.sub(r'[.,!?;:\u3002\uff0c\uff01\uff1f\uff1b\uff1a\u201c\u201d\u2018\u2019\uff08\uff09\u3001]', ' ', cleaned_text.lower())
                    text_words = [word.strip() for word in text_for_comparison.split() if word.strip()]
                    
                    # Check for consecutive prompt words in transcription (must be in correct order)
                    # Find ALL sequences of consecutive prompt words that meet criteria
                    all_prompt_segments = []
                    
                    # Find all possible consecutive prompt word sequences
                    for start_idx in range(len(text_words)):
                        # Try starting from each position in the prompt sequence
                        for prompt_start in range(len(prompt_words)):
                            consecutive_count = 0
                            prompt_idx = prompt_start
                            
                            # Try to match prompt words in order starting from this position
                            for text_idx in range(start_idx, len(text_words)):
                                if prompt_idx < len(prompt_words) and text_words[text_idx] == prompt_words[prompt_idx]:
                                    consecutive_count += 1
                                    prompt_idx += 1
                                else:
                                    break  # Stop if we can't continue the sequence
                            
                            # If this segment meets criteria, add it to the list
                            if consecutive_count >= 3:
                                all_prompt_segments.append({
                                    'start': start_idx,
                                    'end': start_idx + consecutive_count - 1,
                                    'count': consecutive_count,
                                    'prompt_start': prompt_start
                                })
                                break  # Found a valid sequence from this text position, no need to try other prompt starts
                    
                    # Remove overlapping segments (keep the longer ones)
                    def remove_overlapping_segments(segments):
                        if not segments:
                            return []
                        
                        # Sort by start position
                        segments.sort(key=lambda x: x['start'])
                        result = []
                        
                        for segment in segments:
                            # Check if this segment overlaps with any existing segment
                            overlaps = False
                            for existing in result:
                                if not (segment['end'] < existing['start'] or segment['start'] > existing['end']):
                                    # There's overlap, keep the longer one
                                    if segment['count'] > existing['count']:
                                        result.remove(existing)
                                        result.append(segment)
                                    overlaps = True
                                    break
                            
                            if not overlaps:
                                result.append(segment)
                        
                        return result
                    
                    # Remove overlapping segments
                    final_segments = remove_overlapping_segments(all_prompt_segments)
                    
                    # Function to remove multiple prompt segments from original text
                    def remove_multiple_prompt_segments_from_text(original_text, segments_to_remove):
                        # Split original text into words while preserving spacing
                        original_words = original_text.split()
                        
                        # Build accurate mapping from original index to normalized index
                        original_to_normalized = {}
                        normalized_words = []
                        
                        for i, word in enumerate(original_words):
                            normalized = re.sub(r'[.,!?;:\u3002\uff0c\uff01\uff1f\uff1b\uff1a\u201c\u201d\u2018\u2019\uff08\uff09\u3001]', '', word.lower())
                            if normalized:  # Only add non-empty normalized words
                                original_to_normalized[i] = len(normalized_words)
                                normalized_words.append(normalized)
                            # If empty, this original index has no corresponding normalized index
                        
                        # Find which original words correspond to all the segments to remove
                        remove_indices = set()
                        for orig_idx in original_to_normalized:
                            norm_idx = original_to_normalized[orig_idx]
                            # Check if this normalized index falls within any segment to remove
                            for segment in segments_to_remove:
                                if norm_idx >= segment['start'] and norm_idx <= segment['end']:
                                    remove_indices.add(orig_idx)
                                    break
                        
                        # Remove the identified words
                        result_words = [word for i, word in enumerate(original_words) if i not in remove_indices]
                        return ' '.join(result_words)
                    
                    # Rule 1: 3+ consecutive prompt words → remove all segments that meet criteria
                    if final_segments:
                        total_removed_count = sum(seg['count'] for seg in final_segments)
                        
                        # Extract the actual prompt words that will be removed
                        removed_prompt_sequences = []
                        for segment in final_segments:
                            start_idx = segment['start']
                            end_idx = segment['end']
                            removed_words = text_words[start_idx:end_idx+1]
                            removed_prompt_sequences.append(' '.join(removed_words))
                        
                        cleaned_text = remove_multiple_prompt_segments_from_text(cleaned_text, final_segments)
                        
                        # Log with specific removed prompt sequences
                        sequences_str = "', '".join(removed_prompt_sequences)
                        logger.warning(f" | Prompt leakage detected: {len(final_segments)} segments removed: '{sequences_str}' | Total {total_removed_count} prompt words | ")
                        # Continue processing (don't return early)
                    
                    # Rule 2: 2 consecutive prompt words + repetition pattern → check for any 2+ consecutive segments
                    else:
                        # Check for 2+ consecutive segments with repetition
                        two_plus_segments = []
                        for start_idx in range(len(text_words)):
                            # Try starting from each position in the prompt sequence
                            for prompt_start in range(len(prompt_words)):
                                consecutive_count = 0
                                prompt_idx = prompt_start
                                
                                for text_idx in range(start_idx, len(text_words)):
                                    if prompt_idx < len(prompt_words) and text_words[text_idx] == prompt_words[prompt_idx]:
                                        consecutive_count += 1
                                        prompt_idx += 1
                                    else:
                                        break
                                
                                if consecutive_count >= 2:
                                    two_plus_segments.append({
                                        'start': start_idx,
                                        'end': start_idx + consecutive_count - 1,
                                        'count': consecutive_count,
                                        'prompt_start': prompt_start
                                    })
                                    break  # Found a valid sequence from this text position
                        
                        if two_plus_segments:
                            # Check if there's also repetition in the text
                            word_count = {}
                            for word in text_words:
                                if len(word) > 1:  # Skip single characters
                                    word_count[word] = word_count.get(word, 0) + 1
                            
                            # If any word appears 3+ times, combined with 2 consecutive prompt words → remove prompt segments
                            has_repetition = any(count >= 3 for count in word_count.values())
                            if has_repetition:
                                # Remove overlapping segments
                                final_two_plus_segments = remove_overlapping_segments(two_plus_segments)
                                total_removed_count = sum(seg['count'] for seg in final_two_plus_segments)
                                
                                # Extract the actual prompt words that will be removed
                                removed_prompt_sequences = []
                                for segment in final_two_plus_segments:
                                    start_idx = segment['start']
                                    end_idx = segment['end']
                                    removed_words = text_words[start_idx:end_idx+1]
                                    removed_prompt_sequences.append(' '.join(removed_words))
                                
                                cleaned_text = remove_multiple_prompt_segments_from_text(cleaned_text, final_two_plus_segments)
                                
                                # Log with specific removed prompt sequences
                                sequences_str = "', '".join(removed_prompt_sequences)
                                logger.warning(f" | Prompt leakage with repetition detected: {len(final_two_plus_segments)} segments removed: '{sequences_str}' | Total {total_removed_count} prompt words | Remaining text: '{cleaned_text}' | ")
                                # Continue processing (don't return early)
        except Exception as e:
            logger.error(f" | Step 5 (prompt leakage check) error: {e} | ")

        # 6. Check for word repetition hallucinations
        repetition_cleaned = False  # Flag specifically for repetition cleaning in step 6
        
        try:
            # First check for single character repetitions (like 嗯嗯嗯嗯...)
            # Detect patterns where the same character repeats 5 or more times consecutively
            char_repeat_pattern = r'(.)\1{4,}'  # Same character 5+ times
            if re.search(char_repeat_pattern, cleaned_text):
                matches = list(re.finditer(char_repeat_pattern, cleaned_text))  # Convert to list to avoid iterator modification
                # Process matches from end to start to avoid position shifts
                for match in reversed(matches):
                    start, end = match.span()
                    repeated_char = match.group(1)
                    repeat_count = end - start
                    logger.warning(f" | Repeated character hallucination: '{repeated_char}' appears {repeat_count} times consecutively, removing all | ")
                    # Remove only this specific position match
                    cleaned_text = cleaned_text[:start] + cleaned_text[end:]
                    retry_flag = True
                    repetition_cleaned = True
        except Exception as e:
            logger.error(f" | Step 6a (character repetition check) error: {e} | ")
        
        try:
            # Split by both spaces and Chinese punctuation
            words = re.split(r'[\s、，,]+', cleaned_text)
            # Filter out empty strings
            words = [word.strip() for word in words if word.strip()]
            
            if len(words) >= 4:
                # Check for obvious repeated words (3 consecutive identical words)
                cleaned_words = []
                i = 0
                while i < len(words):
                    word = words[i]
                    
                    # Check for consecutive repetitions (removed length restriction)
                    repeat_count = 1
                    j = i + 1
                    while j < len(words) and words[j] == word:
                        repeat_count += 1
                        j += 1
                    
                    if repeat_count >= 3:
                        logger.warning(f" | Repeated word hallucination: '{word}' appears {repeat_count} times consecutively, removing all | ")
                        retry_flag = True
                        repetition_cleaned = True
                        i = j  # Skip all repetitions without adding to cleaned_words
                    else:
                        cleaned_words.append(word)
                        i += 1
                
                # Check for repeated phrases (2-word combinations) and clean them
                if len(cleaned_words) >= 6:
                    final_words = []
                    removed_phrases = []
                    i = 0
                    while i < len(cleaned_words) - 1:
                        if i < len(cleaned_words) - 3:
                            phrase1 = ' '.join(cleaned_words[i:i+2])
                            phrase2 = ' '.join(cleaned_words[i+2:i+4])
                            
                            if phrase1 == phrase2 and len(phrase1.strip()) > 4:
                                final_words.extend(cleaned_words[i:i+2])  # Keep only first occurrence
                                removed_phrases.append(phrase1)
                                retry_flag = True
                                repetition_cleaned = True
                                i += 4  # Skip the repeated phrase
                            else:
                                final_words.append(cleaned_words[i])
                                i += 1
                        else:
                            final_words.append(cleaned_words[i])
                            i += 1
                    
                    # Add remaining words
                    if i < len(cleaned_words):
                        final_words.extend(cleaned_words[i:])
                    
                    # Log once with total count and specific phrases
                    if removed_phrases:
                        phrases_str = "', '".join(removed_phrases)
                        logger.warning(f" | Repeated phrase hallucination: removed {len(removed_phrases)} duplicate phrase(s): '{phrases_str}' | ")
                    
                    cleaned_words = final_words
                
                # Reconstruct the text from cleaned words only if repetition was actually cleaned
                if repetition_cleaned:
                    # Try to maintain original spacing by using the most common separator
                    if '，' in cleaned_text:
                        separator = '，'
                    elif ', ' in cleaned_text:
                        separator = ', '
                    else:
                        separator = ' '
                    
                    cleaned_text = separator.join(cleaned_words)
                    logger.info(f" | Cleaned repetition hallucinations: result length {len(cleaned_words)} words | ")
        except Exception as e:
            logger.error(f" | Step 6b (word repetition check) error: {e} | ")

        # 7. Common Hallucination Check (moved before unusual character cleaning)
        try:
            # For CONTAINS_UNUSUAL, we still need to remove punctuation for matching
            text_for_check = re.sub(r'[^a-zA-Z0-9\s\u4e00-\u9fff]', '', cleaned_text)
            text_for_check = re.sub(r'\s+', ' ', text_for_check).strip()
            
            if any(phrase in text_for_check for phrase in CONTAINS_UNUSUAL):
                logger.warning(f" | Found common hallucination (contains) | ")
                for phrase in CONTAINS_UNUSUAL:
                    if phrase in text_for_check:
                        # Create flexible pattern that allows punctuation between characters
                        if len(phrase) > 1:
                            # Build pattern allowing punctuation between characters
                            pattern_chars = list(phrase)
                            flexible_pattern = pattern_chars[0]
                            for char in pattern_chars[1:]:
                                flexible_pattern += r'[^a-zA-Z0-9\u4e00-\u9fff]*' + re.escape(char)
                            
                            # Use flexible pattern to remove from original text
                            if re.search(flexible_pattern, cleaned_text, re.IGNORECASE):
                                cleaned_text = re.sub(flexible_pattern, '', cleaned_text, flags=re.IGNORECASE)
                                logger.info(f" | Removed hallucination phrase: '{phrase}' | ")
                        else:
                            # For single character, direct replacement
                            cleaned_text = cleaned_text.replace(phrase, '')
                            logger.info(f" | Removed hallucination phrase: '{phrase}' | ")
                
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
                retry_flag = True
                
            # For ONLY_UNUSUAL, use original text to preserve punctuation like "Amara.org"
            cleaned_text_normalized = re.sub(r'\s+', ' ', cleaned_text).strip()
            if any(phrase == cleaned_text_normalized for phrase in ONLY_UNUSUAL):
                logger.warning(f" | Found common hallucination (exact match): '{cleaned_text_normalized}' | ")
                retry_flag = True
                cleaned_text = ""
        except Exception as e:
            logger.error(f" | Step 7 (common hallucination check) error: {e} | ")

        # 8. Check for unusual character patterns and clean them
        try:
            # Define allowed characters: Latin (a-zA-Z), digits (0-9), Chinese, common punctuation, spaces
            # Explicitly exclude Arabic, Cyrillic, and other non-target languages
            if re.search(r'[^a-zA-Z0-9\s\u4e00-\u9fff.,!?;:""''()（），。！？；：、—–+=%$€£¥@&*/<>|\\-]', cleaned_text):
                unusual_chars = re.findall(r'[^a-zA-Z0-9\s\u4e00-\u9fff.,!?;:""''()（），。！？；：、—–+=%$€£¥@&*/<>|\\-]', cleaned_text)
                logger.warning(f" | Unusual characters detected: {set(unusual_chars)} in '{cleaned_text[:50]}...' | ")
                
                # Remove unusual characters (like Arabic, symbols, corrupted Unicode, etc.)
                original_text = cleaned_text
                cleaned_text = re.sub(r'[^a-zA-Z0-9\s\u4e00-\u9fff.,!?;:""''()（），。！？；：、—–+=%$€£¥@&*/<>|\\-]', '', cleaned_text)
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # Clean up extra spaces
                
                if cleaned_text != original_text:
                    logger.info(f" | Removed unusual characters: '{original_text[:50]}...' → '{cleaned_text[:50]}...' | ")
                    retry_flag = True
        except Exception as e:
            logger.error(f" | Step 8 (unusual character check) error: {e} | ")

        return retry_flag, cleaned_text
        
    except Exception as e:
        # Global error handling - if anything goes wrong, log and return original text
        logger.error(f" | Global post_process error: {e} | ")
        return True, text  # Return original text with retry flag

