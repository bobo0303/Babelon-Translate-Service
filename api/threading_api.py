import logging  
import threading  
import ctypes

from api.audio_utils import calculate_rtf
from lib.constant import DEFAULT_RESULT  
  
logger = logging.getLogger(__name__)  
  
def audio_translate(model, audio_file_path, result_queue, ori, stop_event, multi_strategy_transcription=1, transcription_post_processing=True, prev_text="", use_translate=True):  
    """  
    Transcribe and translate an audio file, then store the results in a queue.  
  
    :param model: The model used for transcription and translation.  
    :param audio_file_path: str  
        The path to the audio file to be processed.  
    :param result_queue: Queue  
        The queue to store the results.  
    :param ori: str  
        The original language of the audio.  
    :param stop_event: threading.Event  
        The event used to signal stopping.
    """  
    ori_pred, inference_time = model.transcribe(audio_file_path, ori, multi_strategy_transcription, transcription_post_processing, prev_text)
    if use_translate:
        translated_pred, translate_time, translate_method = model.translate(ori_pred, ori)  
    else:
        translated_pred = DEFAULT_RESULT.copy()
        translate_time = 0
        translate_method = "none"
    
    # Calculate RTF using audio_utils
    rtf = calculate_rtf(audio_file_path, inference_time, translate_time)

    result_queue.put((ori_pred, translated_pred, rtf, inference_time, translate_time, translate_method))  
    stop_event.set()  # Signal to stop the waiting thread  
    
def texts_translate(model, text, result_queue, ori, stop_event):  
    """  
    Translate a given text using the specified model.  
  
    :param model: The model used for translation.  
    :param text: str  
        The text to be translated.  
    :param result_queue: Queue  
        The queue to store the results.  
    :param ori: str  
        The original language of the text.  
    :param stop_event: threading.Event  
        The event used to signal stopping.  
    """  
    translated_pred, translate_time, translate_method = model.translate(text, ori)  

    result_queue.put((translated_pred, translate_time, translate_method))  
    stop_event.set()  # Signal to stop the waiting thread  
  
def audio_translate_sse(model, audio_file_path, ori, other_information, stop_event):  
    """  
    Transcribe and translate an audio file for SSE, then store the results in the model's result queue.  
  
    :param model: The model used for transcription and translation.  
    :param audio_file_path: str  
        The path to the audio file to be processed.  
    :param ori: str  
        The original language of the audio.  
    :param stop_event: threading.Event  
        The event used to signal stopping.  
    """  
    model.processing = True
    
    ori_pred, inference_time = model.transcribe(audio_file_path, ori, other_information["multi_strategy_transcription"], other_information["transcription_post_processing"], other_information["prev_text"])
    if other_information["use_translate"]:
        translated_pred, translate_time, translate_method = model.translate(ori_pred, ori)  
    else:
        translated_pred = DEFAULT_RESULT.copy()
        translate_time = 0
        translate_method = "none"
    
    # Calculate RTF using audio_utils
    rtf = calculate_rtf(audio_file_path, inference_time, translate_time) 
    
    model.result_queue.put((ori_pred, translated_pred, rtf, inference_time, translate_time, translate_method))
    model.processing = False
    stop_event.set()  # Signal to stop the waiting thread

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
  
def waiting_times(stop_event, model, times):  
    """  
    Wait for a specified amount of time or until an event is set.  
  
    :param stop_event: threading.Event  
        The event used to signal stopping.  
    :param model: The model whose processing state is to be updated.  
    :param times: float  
        The amount of time to wait.  
    """  
    stop_event.wait(times)  # Wait for the event or timeout  
    model.processing = False  
    
    