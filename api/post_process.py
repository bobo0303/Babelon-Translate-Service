import re
import logging
import logging.handlers
import opencc
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

# Initialize OpenCC converter for simplified to traditional Chinese
try:
    s2t_converter = opencc.OpenCC('s2tw')  # simplified to Taiwan traditional (modern forms)
except Exception as e:
    logger.error(f"Failed to initialize OpenCC converter: {e}")
    s2t_converter = None

def convert_simplified_to_traditional(text):
    """Convert simplified Chinese to traditional Chinese using OpenCC"""
    try:
        if s2t_converter is None:
            logger.warning("OpenCC converter not available, returning original text")
            return text
        
        result = s2t_converter.convert(text)
        return result
    except Exception as e:
        logger.error(f" | OpenCC conversion error: {e} | ")
        return text  # Return original text if conversion fails  
    
def normalize_for_comparison(text):
    """Apply same normalization as constant.py preprocessing"""
    if not text or not text.strip():
        return ""
    
    # Step 1: Remove punctuation 
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Step 2: Remove ALL spaces between Chinese characters 
    prev_text = ""
    while prev_text != text:
        prev_text = text
        text = re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', text)
    
    # Final cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    return text

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
        
        # 3. Clean 【】 and 《》 format and inside text
        try:
            # Check for 【】 brackets
            if '【' in cleaned_text or '】' in cleaned_text:
                original_text = cleaned_text
                cleaned_text = re.sub(r'【[^】]*】', '', cleaned_text)
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
                logger.warning(f" | Clean 【】format: '{original_text}' → '{cleaned_text}' | ")
            
            # Check for 《》 brackets
            if '《' in cleaned_text or '》' in cleaned_text:
                original_text = cleaned_text
                cleaned_text = re.sub(r'《[^》]*》', '', cleaned_text)
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
                logger.warning(f" | Clean 《》format: '{original_text}' → '{cleaned_text}' | ")
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
        
        # 4. Convert simplified Chinese to traditional Chinese (before hallucination checks)
        try:
            original_text = cleaned_text
            cleaned_text = convert_simplified_to_traditional(cleaned_text)
            if original_text != cleaned_text:
                logger.info(f" | Simplified to Traditional conversion applied | ")
        except Exception as e:
            logger.error(f" | Step 4 (simplified to traditional conversion) error: {e} | ")

        # 5. Check for unusual character patterns and clean them (moved up for better processing order)
        try:
            # Define allowed characters: Latin (a-zA-Z), digits (0-9), Chinese, common punctuation, spaces
            # Explicitly exclude Arabic, Cyrillic, and other non-target languages
            if re.search(r'[^a-zA-Z0-9\s\u4e00-\u9fff.,!?;:""''\'()（），。！？；：、—–+=%$€£¥@&*/<>|\\-]', cleaned_text):
                unusual_chars = re.findall(r'[^a-zA-Z0-9\s\u4e00-\u9fff.,!?;:""''\'()（），。！？；：、—–+=%$€£¥@&*/<>|\\-]', cleaned_text)
                logger.warning(f" | Unusual characters detected: {set(unusual_chars)} in '{cleaned_text[:50]}...' | ")
                
                # Remove unusual characters (like Arabic, symbols, corrupted Unicode, etc.)
                original_text = cleaned_text
                cleaned_text = re.sub(r'[^a-zA-Z0-9\s\u4e00-\u9fff.,!?;:""''\'()（），。！？；：、—–+=%$€£¥@&*/<>|\\-]', '', cleaned_text)
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # Clean up extra spaces
                
                if cleaned_text != original_text:
                    logger.info(f" | Removed unusual characters: '{original_text[:50]}...' → '{cleaned_text[:50]}...' | ")
        except Exception as e:
            logger.error(f" | Step 5 (unusual character check) error: {e} | ")

        # 6. Check audio duration vs text length ratio using IQR-based dynamic range
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
            logger.error(f" | Step 6 (audio duration check) error: {e} | ")

        # 7. Check for prompt leakage - detect if transcription contains prompt terms
        try:
            if prompt_name is not None and prompt_name.strip() != "":
                # Extract terms from comma-separated prompt (vocabulary list)
                prompt_terms = []
                # Split by comma and clean each term
                raw_terms = [term.strip() for term in prompt_name.split(',') if term.strip()]
                prompt_terms = [term.lower() for term in raw_terms if term.strip()]
                
                if prompt_terms:
                    # Normalize text for comparison (remove punctuation, convert to lowercase)
                    text_for_comparison = re.sub(r'[.,!?;:\u3002\uff0c\uff01\uff1f\uff1b\uff1a\u201c\u201d\u2018\u2019\uff08\uff09\u3001]', ' ', cleaned_text.lower())
                    text_words = [word.strip() for word in text_for_comparison.split() if word.strip()]
                    
                    # Check for consecutive prompt terms in transcription (must be in correct order)
                    # Find ALL sequences of consecutive prompt terms that meet criteria
                    all_prompt_segments = []
                    
                    # Find all possible consecutive prompt term sequences
                    for start_idx in range(len(text_words)):
                        # Try starting from each position in the prompt sequence
                        for prompt_start in range(len(prompt_terms)):
                            consecutive_count = 0
                            prompt_idx = prompt_start
                            
                            # Try to match prompt terms in order starting from this position
                            for text_idx in range(start_idx, len(text_words)):
                                if prompt_idx < len(prompt_terms) and text_words[text_idx] == prompt_terms[prompt_idx]:
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
                    
                    # Rule 1: 3+ consecutive prompt terms → remove all segments that meet criteria
                    if final_segments:
                        total_removed_count = sum(seg['count'] for seg in final_segments)
                        
                        # Extract the actual prompt terms that will be removed
                        removed_prompt_sequences = []
                        for segment in final_segments:
                            start_idx = segment['start']
                            end_idx = segment['end']
                            removed_words = text_words[start_idx:end_idx+1]
                            removed_prompt_sequences.append(' '.join(removed_words))
                        
                        cleaned_text = remove_multiple_prompt_segments_from_text(cleaned_text, final_segments)
                        
                        # Log with specific removed prompt sequences
                        sequences_str = "', '".join(removed_prompt_sequences)
                        logger.warning(f" | Prompt leakage detected: {len(final_segments)} segments removed: '{sequences_str}' | Total {total_removed_count} prompt terms | ")
                        retry_flag = True
                        # Continue processing (don't return early)
                    
                    # Rule 2: 2 consecutive prompt terms + repetition pattern → check for any 2+ consecutive segments
                    else:
                        # Check for 2+ consecutive segments with repetition
                        two_plus_segments = []
                        for start_idx in range(len(text_words)):
                            # Try starting from each position in the prompt sequence
                            for prompt_start in range(len(prompt_terms)):
                                consecutive_count = 0
                                prompt_idx = prompt_start
                                
                                for text_idx in range(start_idx, len(text_words)):
                                    if prompt_idx < len(prompt_terms) and text_words[text_idx] == prompt_terms[prompt_idx]:
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
                            
                            # If any word appears 3+ times, combined with 2 consecutive prompt terms → remove prompt segments
                            has_repetition = any(count >= 3 for count in word_count.values())
                            if has_repetition:
                                # Remove overlapping segments
                                final_two_plus_segments = remove_overlapping_segments(two_plus_segments)
                                total_removed_count = sum(seg['count'] for seg in final_two_plus_segments)
                                
                                # Extract the actual prompt terms that will be removed
                                removed_prompt_sequences = []
                                for segment in final_two_plus_segments:
                                    start_idx = segment['start']
                                    end_idx = segment['end']
                                    removed_words = text_words[start_idx:end_idx+1]
                                    removed_prompt_sequences.append(' '.join(removed_words))
                                
                                cleaned_text = remove_multiple_prompt_segments_from_text(cleaned_text, final_two_plus_segments)
                                
                                # Log with specific removed prompt sequences
                                sequences_str = "', '".join(removed_prompt_sequences)
                                logger.warning(f" | Prompt leakage with repetition detected: {len(final_two_plus_segments)} segments removed: '{sequences_str}' | Total {total_removed_count} prompt terms | Remaining text: '{cleaned_text}' | ")
                                retry_flag = True
                                # Continue processing (don't return early)
                    
                    # Rule 3: Single prompt term only → if text contains only one or more prompt terms, remove entire text
                    if not final_segments and not (two_plus_segments and has_repetition):
                        # Check if the entire text consists of only prompt terms
                        # Remove punctuation and extra spaces from the text
                        text_normalized = re.sub(r'[.,!?;:\u3002\uff0c\uff01\uff1f\uff1b\uff1a\u201c\u201d\u2018\u2019\uff08\uff09\u3001\s]+', ' ', cleaned_text).strip()
                        
                        # Split into words and remove empty strings
                        text_only_words = [word.strip() for word in text_normalized.split() if word.strip()]
                        
                        # Check if text contains only one or more words that are ALL prompt terms
                        if text_only_words:  # Make sure there are words to check
                            all_words_are_prompt_terms = True
                            matched_prompt_terms = []
                            
                            for word in text_only_words:
                                word_lower = word.lower()
                                # Check if this word matches any prompt term (case insensitive)
                                word_is_prompt_term = False
                                for prompt_term in prompt_terms:
                                    if word_lower == prompt_term.lower():
                                        matched_prompt_terms.append(word)
                                        word_is_prompt_term = True
                                        break
                                
                                # If any word is NOT a prompt term, break
                                if not word_is_prompt_term:
                                    all_words_are_prompt_terms = False
                                    break
                            
                            # If all words in the text are prompt terms, remove the entire text
                            if all_words_are_prompt_terms and matched_prompt_terms:
                                logger.warning(f" | Single prompt term(s) only detected: '{', '.join(matched_prompt_terms)}' → removing entire text | ")
                                cleaned_text = ""
                                retry_flag = True
                                return retry_flag, cleaned_text

        except Exception as e:
            logger.error(f" | Step 7 (prompt leakage check) error: {e} | ")

        # 8. Check for word repetition hallucinations
        repetition_cleaned = False  # Flag specifically for repetition cleaning in step 8
        
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
            logger.error(f" | Step 8a (character repetition check) error: {e} | ")
        
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
                
        except Exception as e:
            logger.error(f" | Step 8b (word repetition check) error: {e} | ")

        # Step 8c: Check for repeated phrases (2-word combinations) and clean them
        try:
            # Split by both spaces and Chinese punctuation if not already done in 8b
            if 'words' not in locals() or not words:
                words = re.split(r'[\s、，,]+', cleaned_text)
                words = [word.strip() for word in words if word.strip()]
            
            # Use cleaned_words from 8b if available, otherwise use words
            if 'cleaned_words' in locals() and cleaned_words:
                phrase_check_words = cleaned_words
            else:
                phrase_check_words = words
            
            if len(phrase_check_words) >= 6:
                final_words = []
                removed_phrases = []
                i = 0
                while i < len(phrase_check_words) - 1:
                    if i < len(phrase_check_words) - 3:
                        phrase1 = ' '.join(phrase_check_words[i:i+2])
                        phrase2 = ' '.join(phrase_check_words[i+2:i+4])
                        
                        if phrase1 == phrase2 and len(phrase1.strip()) > 4:
                            # Remove all occurrences of repeated phrase (both first and second)
                            removed_phrases.append(phrase1)
                            retry_flag = True
                            repetition_cleaned = True
                            i += 4  # Skip both repeated phrases without adding to final_words
                        else:
                            final_words.append(phrase_check_words[i])
                            i += 1
                    else:
                        final_words.append(phrase_check_words[i])
                        i += 1
                
                # Add remaining words
                if i < len(phrase_check_words):
                    final_words.extend(phrase_check_words[i:])
                
                # Log once with total count and specific phrases
                if removed_phrases:
                    phrases_str = "', '".join(removed_phrases)
                    logger.warning(f" | Repeated phrase hallucination: removed {len(removed_phrases)} duplicate phrase(s): '{phrases_str}' | ")
                
                # Update the words list for text reconstruction
                phrase_check_words = final_words
            
            # Reconstruct the text from cleaned words only if repetition was actually cleaned
            if repetition_cleaned:
                # Try to maintain original spacing by using the most common separator
                if '，' in cleaned_text:
                    separator = '，'
                elif ', ' in cleaned_text:
                    separator = ', '
                else:
                    separator = ' '
                
                cleaned_text = separator.join(phrase_check_words)
                logger.info(f" | Cleaned repetition hallucinations: result length {len(phrase_check_words)} words | ")
        except Exception as e:
            logger.error(f" | Step 8c (phrase repetition check) error: {e} | ")

        # 9. Common Hallucination Check (Enhanced with Normalization)
        try:
            # Normalize input text for comparison
            normalized_text = normalize_for_comparison(cleaned_text)
            
            # Step 9a: Check ONLY_UNUSUAL first (exact match with normalized text)
            if normalized_text in ONLY_UNUSUAL:
                logger.warning(f" | Found common hallucination (exact match): '{normalized_text}' | ")
                retry_flag = True
                cleaned_text = ""
            else:
                # Step 9b: Check CONTAINS_UNUSUAL only if not exact match (if any phrase is found in normalized text)
                found_contains = False
                for phrase in CONTAINS_UNUSUAL:
                    if phrase in normalized_text:
                        logger.warning(f" | Found common hallucination (contains): '{phrase}' | ")
                        found_contains = True
                        break
                
                if found_contains:
                    # Remove the matched phrases while preserving original punctuation structure
                    result_text = cleaned_text
                    for phrase in CONTAINS_UNUSUAL:
                        if phrase in normalized_text:
                            # Create a flexible pattern that matches the phrase with optional punctuation
                            # Split phrase into characters and allow punctuation between them
                            if len(phrase) > 1:
                                pattern_chars = list(phrase)
                                flexible_pattern = re.escape(pattern_chars[0])
                                for char in pattern_chars[1:]:
                                    # Allow optional punctuation and spaces between characters
                                    flexible_pattern += r'[^\w\u4e00-\u9fff]*' + re.escape(char)
                                
                                # Remove matched pattern from original text
                                if re.search(flexible_pattern, result_text, re.IGNORECASE):
                                    result_text = re.sub(flexible_pattern, '', result_text, flags=re.IGNORECASE)
                                    logger.info(f" | Removed hallucination phrase: '{phrase}' | ")
                            else:
                                # For single character phrases
                                result_text = result_text.replace(phrase, '')
                                logger.info(f" | Removed hallucination phrase: '{phrase}' | ")
                    
                    cleaned_text = re.sub(r'\s+', ' ', result_text).strip()
                    retry_flag = True
                
        except Exception as e:
            logger.error(f" | Step 9 (common hallucination check) error: {e} | ")

        # 10. Final text is empty call retry
        try:
            if cleaned_text.strip() == "":
                logger.warning(f" | Final cleaned text is empty after post-processing | ")
                retry_flag = True
                cleaned_text = ""   
        except Exception as e:
            logger.error(f" | Step 10 (final empty check) error: {e} | ")

        return retry_flag, cleaned_text
        
    except Exception as e:
        # Global error handling - if anything goes wrong, log and return original text
        logger.error(f" | Global post_process error: {e} | ")
        return True, text  # Return original text with retry flag

