import logging  
import threading  
import ctypes
import uuid
import time

from api.audio.audio_utils import calculate_rtf
from lib.config.constant import DEFAULT_RESULT  

  
logger = logging.getLogger(__name__)  

def _cleanup_transcription_task(transcribe_manager, task_id):
    """
    Clean up a transcription task that was interrupted or cancelled.
    
    :param transcribe_manager: TranscribeManager instance
    :param task_id: Task ID to clean up
    """
    try:
        with transcribe_manager.task_lock:
            if task_id in transcribe_manager.task_results:
                # Mark task as cancelled
                transcribe_manager.task_results[task_id]['cancelled'] = True
                transcribe_manager.task_results[task_id]['event'].set()
                logger.info(f" | Task {task_id} marked as cancelled for cleanup. | ")
    except Exception as e:
        logger.warning(f" | Error cleaning up task {task_id}: {e} | ")  

def audio_pipeline_coordinator(transcribe_manager, translate_manager, audio_file, o_lang, t_lang, 
                               multi_strategy_transcription, transcription_post_processing, 
                               prev_text, multi_translate, audio_uid, times):
    """
    Coordinate transcription and translation pipeline with proper decoupling.
    
    This function handles the complete pipeline:
    1. Submit transcription task to TranscribeManager
    2. Wait for transcription completion
    3. Submit translation task to TranslateManager if needed
    4. Return combined results
    
    :param transcribe_manager: TranscribeManager instance
    :param translate_manager: TranslateManager instance  
    :param audio_file: Path to audio file
    :param o_lang: Original language
    :param t_lang: Target language(s)
    :param multi_strategy_transcription: Transcription strategy
    :param transcription_post_processing: Post-processing flag
    :param prev_text: Previous context text
    :param multi_translate: Multi-translation flag
    :param audio_uid: Audio unique identifier
    :param times: Timestamp
    :return: Tuple of (ori_pred, translated_pred, rtf, transcription_time, translate_time, translate_method, timing_dict)
    """
    # Initialize default output structure
    result = {
        'ori_pred': "",
        'translated_pred': DEFAULT_RESULT.copy(),
        'rtf': 0,
        'transcription_time': 0,
        'translate_time': 0,
        'translate_method': "none",
        'timing_dict': {}
    }
    
    start_time = time.time()
    task_id = str(uuid.uuid4())
    transcription_event = None
    
    audio_tags = ""
    if multi_strategy_transcription == 1:
        audio_tags = "audio_start"
    elif multi_strategy_transcription == 4:
        audio_tags = "audio_end"
    
    try:
        # Step 1: Submit transcription task
        transcription_event = transcribe_manager.add_task(
            task_id, audio_file, o_lang, multi_strategy_transcription,
            transcription_post_processing, prev_text, audio_uid, times
        )
        
        # Step 2: Wait for transcription completion
        transcription_event.wait()
        
        # Step 3: Get transcription result
        with transcribe_manager.task_lock:
            # Handle case where task was cancelled before queuing (optimized out)
            if task_id not in transcribe_manager.task_results:
                logger.debug(f" | Pipeline task {task_id} cancelled before queuing (not in task_results). | ")
                result['ori_pred'] = None
                result['translate_method'] = "cancelled_before_queue"
                return tuple(result.values()), None
            
            if transcribe_manager.task_results[task_id].get('cancelled', False):
                logger.debug(f" | Pipeline task {task_id} cancelled during transcription. | ")
                # Clean up cancelled task before returning
                del transcribe_manager.task_results[task_id]
                result['ori_pred'] = None
                result['translate_method'] = "cancelled"
                return tuple(result.values()), None
            
            transcription_result = transcribe_manager.task_results[task_id]['result']
            # Clean up task result to free memory
            del transcribe_manager.task_results[task_id]
    
    except Exception as e:
        # Handle any interruption or cancellation
        logger.warning(f" | Pipeline coordinator interrupted: {e} | ")
        _cleanup_transcription_task(transcribe_manager, task_id)
        result['translate_method'] = "interrupted"
        return tuple(result.values()), None
    
    # Continue with the rest of the function...
    if transcription_result is None:
        logger.error(f" | Transcription failed for task {task_id}. | ")
        result['translate_method'] = "transcription_failed"
        return tuple(result.values()), None
    
    result['ori_pred'], result['transcription_time'], audio_length = transcription_result
    
    
    # Step 4: Handle translation if needed
    if t_lang:
        try:
            result['translated_pred'], result['translate_time'], result['translate_method'], result['timing_dict'] = \
                translate_manager.translate(result['ori_pred'], o_lang, t_lang, prev_text, multi_translate)
        except Exception as e:
            logger.error(f" | Translation failed for task {task_id}: {e} | ")
            # Clean up any translation threads if needed
            try:
                translate_manager.cleanup_translation_threads()
            except:
                pass
            result['translate_method'] = "translation_failed"
    
    # Step 5: Calculate RTF and return results
    result['rtf'] = calculate_rtf(audio_file, result['transcription_time'], result['translate_time'])
    
    other_info = {
            "audio_length": audio_length,
            "task_id": task_id,
            "audio_uid": audio_uid,
            "audio_file_name": None,
            "use_translate": True if t_lang else False,
            "use_prev_text": True if prev_text else False,
            "prev_text": prev_text,
            "post_processing": transcription_post_processing,
            "audio_tags": audio_tags,
            "strategy": multi_strategy_transcription,
            "rtf": result['rtf'],
            "process_method": "pipeline_coordinator"
        }
     
    total_time = time.time() - start_time
    logger.debug(f" | Pipeline task {task_id} completed in {total_time:.2f}s (transcription: {result['transcription_time']:.2f}s, translation: {result['translate_time']:.2f}s). | ")
    
    return tuple(result.values()), other_info

def audio_translate(transcribe_manager, translate_manager, audio_file_path, result_queue, o_lang, t_lang, 
                    stop_event, strategy, post_processing, prev_text, multi_translate):  
    """  
    Transcribe and translate an audio file, then store the results in a queue.  
  
    :param transcribe_manager: The TranscribeManager used for transcription.  
    :param translate_manager: The TranslateManager used for translation.
    :param audio_file_path: str  
        The path to the audio file to be processed.  
    :param result_queue: Queue  
        The queue to store the results.  
    :param ori: str  
        The original language of the audio.  
    :param tar: str or list  
        Target language(s) for translation  
    :param stop_event: threading.Event  
        The event used to signal stopping.  
    """  
    try:
        ori_pred, inference_time, audio_length = transcribe_manager.transcribe(audio_file_path, o_lang, strategy, post_processing, prev_text)
        if t_lang:
            # t_lang is already a list from main.py
            translated_pred, translate_time, translate_method, timing_dict = translate_manager.translate(ori_pred, o_lang, t_lang, prev_text, multi_translate)  
        else:
            translated_pred = DEFAULT_RESULT.copy()
            translate_time = 0
            translate_method = "none"
            timing_dict = {}
        
        # Calculate RTF using audio_utils
        rtf = calculate_rtf(audio_file_path, inference_time, translate_time)
        
        audio_tags = ""
        if strategy == 1:
            audio_tags = "audio_start"
        elif strategy == 4:
            audio_tags = "audio_end"
        
        other_info = {
            "audio_length": audio_length,
            "task_id": "no_need",
            "audio_uid": None,
            "audio_file_name": None,
            "use_translate": True if t_lang else False,
            "use_prev_text": True if prev_text else False,
            "prev_text": prev_text,
            "post_processing": post_processing,
            "audio_tags": audio_tags,
            "strategy": strategy,
            "rtf": rtf,
            "process_method": "threading_sequential"
        }

        result_queue.put((ori_pred, translated_pred, rtf, inference_time, translate_time, translate_method, timing_dict, other_info))  
        stop_event.set()  # Signal to stop the waiting thread
    finally:
        # Clean up any active translation threads when this thread is stopped
        try:
            translate_manager.cleanup_translation_threads()
        except Exception as e:
            logger.error(f" | Error during cleanup: {e} | ")

def texts_translate(translate_manager, text, result_queue, ori, tar, stop_event, multi_translate=True):  
    """  
    Translate a given text using TranslateManager.  
  
    :param translate_manager: The TranslateManager used for translation.
    :param text: str  
        The text to be translated.  
    :param result_queue: Queue  
        The queue to store the results.  
    :param ori: str  
        The original language of the text.
    :param tar: str or list
        Target language(s) for translation
    :param stop_event: threading.Event  
        The event used to signal stopping.
    :param multi_translate: bool
        If True, distribute tasks across multiple LLMs; If False, use single LLM
    """  
    try:
        # tar is already a list from main.py
        translated_pred, translate_time, translate_method, timing_dict = translate_manager.translate(text, ori, tar, prev_text="", multi_translate=multi_translate)  

        result_queue.put((translated_pred, translate_time, translate_method, timing_dict))  
        stop_event.set()  # Signal to stop the waiting thread
    finally:
        # Clean up any active translation threads when this thread is stopped
        try:
            translate_manager.cleanup_translation_threads()
        except Exception as e:
            logger.error(f" | Error during cleanup: {e} | ")  
  
def audio_translate_sse(transcribe_manager, translate_manager, audio_file_path, ori, other_information, stop_event):  
    """  
    Transcribe and translate an audio file for SSE, then store the results in the transcribe_manager's result queue.  
  
    :param transcribe_manager: The TranscribeManager used for transcription.  
    :param translate_manager: The TranslateManager used for translation.
    :param audio_file_path: str  
        The path to the audio file to be processed.  
    :param ori: str  
        The original language of the audio.
    :param other_information: dict
        Contains target_langs, multi_translate and other settings
    :param stop_event: threading.Event  
        The event used to signal stopping.  
    """  
    try:
        transcribe_manager.processing = True
        
        ori_pred, inference_time = transcribe_manager.transcribe(audio_file_path, ori, other_information["multi_strategy_transcription"], other_information["transcription_post_processing"], other_information["prev_text"])
        if other_information["use_translate"]:
            # Get target_langs from other_information or default to all languages
            target_langs = other_information.get("target_langs", [lang for lang in ['zh', 'en', 'de', 'ja', 'ko'] if lang != ori])
            multi_translate = other_information.get("multi_translate", True)
            translated_pred, translate_time, translate_method, timing_dict = translate_manager.translate(ori_pred, ori, target_langs, other_information["prev_text"], multi_translate)  
        else:
            translated_pred = DEFAULT_RESULT.copy()
            translate_time = 0
            translate_method = "none"
            timing_dict = {}
        
        # Calculate RTF using audio_utils
        rtf = calculate_rtf(audio_file_path, inference_time, translate_time) 
        
        transcribe_manager.result_queue.put((ori_pred, translated_pred, rtf, inference_time, translate_time, translate_method, timing_dict))
        transcribe_manager.processing = False
        stop_event.set()  # Signal to stop the waiting thread
    finally:
        # Clean up any active translation threads when this thread is stopped
        try:
            translate_manager.cleanup_translation_threads()
        except Exception as e:
            logger.error(f" | Error during cleanup: {e} | ")

def get_thread_id(thread):  
    """  
    Get the thread ID of a given thread.  
  
    :param thread: threading.Thread  
        The thread whose ID is to be retrieved.  
    :return: int  
        The ID of the thread.  
    """  
    if not thread.is_alive():  
        return None  
    for tid, tobj in threading._active.items():  
        if tobj is thread:  
            return tid  
    logger.debug(" | Could not determine the thread ID | ")  
    raise AssertionError(" | Could not determine the thread ID | ")  
  
def stop_thread(thread):  
    """  
    Stop a given thread.  
  
    :param thread: threading.Thread  
        The thread to be stopped.  
    """  
    thread_id = get_thread_id(thread)  
    if thread_id is not None:  
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), ctypes.py_object(SystemExit))  
        if res == 0:  
            logger.debug(" | Invalid thread ID | ")  
            raise ValueError(" | Invalid thread ID | ")  
        elif res != 1:  
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)  
            logger.debug(" | PyThreadState_SetAsyncExc failed | ")  
            raise SystemError(" | PyThreadState_SetAsyncExc failed | ")  
  
def waiting_times(stop_event, transcribe_manager, times):  
    """  
    Wait for a specified amount of time or until an event is set.  
  
    :param stop_event: threading.Event  
        The event used to signal stopping.  
    :param transcribe_manager: The TranscribeManager whose processing state is to be updated.  
    :param times: float  
        The amount of time to wait.  
    """  
    stop_event.wait(times)  # Wait for the event or timeout  
    transcribe_manager.processing = False  
    
    