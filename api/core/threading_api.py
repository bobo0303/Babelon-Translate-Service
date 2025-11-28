import logging  
import threading  
import ctypes

from api.audio.audio_utils import calculate_rtf
from lib.config.constant import DEFAULT_RESULT  

  
logger = logging.getLogger(__name__)  

def audio_translate(transcribe_manager, translate_manager, audio_file_path, result_queue, ori, tar, stop_event, strategy, post_processing, prev_text, multi_translate):  
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
        ori_pred, inference_time = transcribe_manager.transcribe(audio_file_path, ori, strategy, post_processing, prev_text)
        if tar:
            # tar is already a list from main.py
            translated_pred, translate_time, translate_method, timing_dict = translate_manager.translate(ori_pred, ori, tar, prev_text, multi_translate)  
        else:
            translated_pred = DEFAULT_RESULT.copy()
            translate_time = 0
            translate_method = "none"
            timing_dict = {}
        
        # Calculate RTF using audio_utils
        rtf = calculate_rtf(audio_file_path, inference_time, translate_time)

        result_queue.put((ori_pred, translated_pred, rtf, inference_time, translate_time, translate_method, timing_dict))  
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
    
    