import time
import librosa
import soundfile as sf
import numpy as np
from io import BytesIO

from lib.config.constant import SILENCE_PADDING
from lib.core.logging_config import get_logger

# Get logger instance
logger = get_logger(__name__)

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

def calculate_rtf(audio_duration, transcription_time, translation_time=0):
    """
    Calculate Real Time Factor (RTF) for audio processing.
    
    Args:
        audio_duration: Duration of audio in seconds (from transcription result)
        transcription_time: Time spent on transcription (seconds)
        translation_time: Time spent on translation (seconds), default 0
        
    Returns:
        float: RTF value, or 0 if calculation failed
    """
    try:
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
    
    
def add_silence_padding(audio, sr, padding_duration=0.05):  # Reduce to 0.05 seconds (0.3 original)
        """
        Add silence padding to the beginning and end of audio file.
        
        Args:
            audio: numpy.ndarray of audio data
            padding_duration: Duration of silence to add in seconds
            
        Returns:
            numpy.ndarray: Audio with silence padding added
        """
        start_time = time.time()
        # Add silence at beginning and end
        padding_samples = int(padding_duration * sr)
        silence = np.zeros(padding_samples, dtype=audio.dtype)
        
        # Add silence before and after the audio
        padded_audio = np.concatenate([silence, audio, silence])
        
        end_time = time.time()
        execution_time = end_time - start_time
        original_duration = len(audio) / sr
        padded_duration = len(padded_audio) / sr
        
        logger.debug(f" | _add_silence_padding execution time: {execution_time:.8f}s | Original: {original_duration:.2f}s | Padded: {padded_duration:.2f}s | ")
        
        return padded_audio

        
def audio_preprocess(audio_path, padding_duration=0.05, max_duration=28.0, reject_duration=60.0):
    # Check duration before loading (fast metadata read)
    try:
        info = sf.info(audio_path)
        raw_duration = info.frames / info.samplerate
        if reject_duration and raw_duration > reject_duration:
            logger.warning(f" | audio_preprocess rejected: duration {raw_duration:.2f}s > {reject_duration:.2f}s | File: {audio_path} | ")
            return None, 0.0
    except Exception as e:
        logger.warning(f" | audio_preprocess info check failed: {e} | File: {audio_path} | ")
    
    # fast load
    try:
        audio, sr = sf.read(audio_path)
        audio_length = len(audio) / sr
        
        # Resample to 16kHz if needed
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000 
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Truncate to max_duration if exceeded
        if max_duration and audio_length > max_duration:
            max_samples = int(max_duration * sr)
            audio = audio[:max_samples]
            logger.debug(f" | audio_preprocess truncated from {audio_length:.2f}s to {max_duration:.2f}s | File: {audio_path} | ")
            audio_length = max_duration
            
        # silence padding
        if SILENCE_PADDING:
            try:
                audio = add_silence_padding(audio, sr, padding_duration)
            except Exception as e:
                logger.warning(f" | audio_preprocess silence padding error: {e} | File: {audio_path} | use original audio | ")    
        
        # Convert to float32 for whisper.cpp compatibility
        audio = audio.astype(np.float32)
    except Exception as e:  
        logger.error(f" | audio_preprocess error: {e} | File: {audio_path} | ")
        audio = None
        audio_length = 0.0
    
    return audio, audio_length


def audio_preprocess_from_bytes(
    audio_bytes: bytes,
    padding_duration=0.05,
    max_duration=28.0,
    reject_duration=60.0
):
    """
    Preprocess audio directly from bytes.
    
    Args:
        audio_bytes: Raw audio file bytes (e.g., WAV format)
        padding_duration: Silence padding duration in seconds (default: 0.05s)
        max_duration: Maximum audio duration to process (default: 28.0s)
        reject_duration: Reject audio longer than this (default: 60.0s)
        
    Returns:
        tuple: (audio_array, audio_length)
            - audio_array: numpy.ndarray (float32, 16kHz, mono) or None if failed
            - audio_length: float, duration in seconds
    """
    try:
        # Step 1: Read audio from bytes (in-memory, no disk I/O)
        audio_io = BytesIO(audio_bytes)
        audio, sr = sf.read(audio_io)
        audio_length = len(audio) / sr
        
        # Step 2: Duration validation
        if reject_duration and audio_length > reject_duration:
            logger.warning(f" | audio_preprocess_from_bytes rejected: duration {audio_length:.2f}s > {reject_duration:.2f}s | ")
            return None, 0.0
        
        # Step 3: Resample to 16kHz if needed
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Step 4: Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Step 5: Truncate to max_duration if exceeded
        if max_duration and audio_length > max_duration:
            max_samples = int(max_duration * sr)
            audio = audio[:max_samples]
            logger.debug(f" | audio_preprocess_from_bytes truncated from {audio_length:.2f}s to {max_duration:.2f}s | ")
            audio_length = max_duration
        
        # Step 6: Add silence padding (optional)
        if SILENCE_PADDING:
            try:
                audio = add_silence_padding(audio, sr, padding_duration)
            except Exception as e:
                logger.warning(f" | audio_preprocess_from_bytes silence padding error: {e} | Using original audio | ")
        
        # Step 7: Convert to float32 for whisper.cpp compatibility
        audio = audio.astype(np.float32)
        
        return audio, audio_length
        
    except Exception as e:
        logger.error(f" | audio_preprocess_from_bytes error: {e} | ")
        return None, 0.0