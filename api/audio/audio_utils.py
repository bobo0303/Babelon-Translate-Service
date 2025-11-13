import soundfile as sf
import logging

logger = logging.getLogger(__name__)

def get_audio_duration(audio_file_path):
    """
    Get audio duration using soundfile (faster than librosa for metadata only).
    
    Args:
        audio_file_path: Path to audio file
        
    Returns:
        float: Audio duration in seconds, or None if failed
    """
    import time
    start_time = time.time()
    
    try:
        # Use soundfile to get audio info without loading the entire file
        info = sf.info(audio_file_path)
        duration = info.frames / info.samplerate
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.debug(f" | get_audio_duration execution time: {execution_time:.8f}s | Audio duration: {duration:.2f}s | File: {audio_file_path} | ")
        return duration
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        logger.warning(f" | get_audio_duration failed in {execution_time:.8f}s | Error: {e} | File: {audio_file_path} | ")
        return None

def calculate_rtf(audio_file_path, transcription_time, translation_time=0):
    """
    Calculate Real Time Factor (RTF) for audio processing.
    
    Args:
        audio_file_path: Path to audio file
        transcription_time: Time spent on transcription (seconds)
        translation_time: Time spent on translation (seconds), default 0
        
    Returns:
        float: RTF value, or 0 if calculation failed
    """
    try:
        audio_duration = get_audio_duration(audio_file_path)
        if audio_duration is not None and audio_duration > 0:
            total_processing_time = transcription_time + translation_time
            rtf = total_processing_time / audio_duration
            logger.debug(f" | RTF: {rtf:.4f} | Audio: {audio_duration:.2f}s | Processing: {total_processing_time:.2f}s | ")
            return rtf
        else:
            logger.debug(f" | No valid audio duration for RTF calculation | ")
            return 0.0
    except Exception as e:
        logger.error(f" | RTF calculation error: {e} | ")
        return 0.0