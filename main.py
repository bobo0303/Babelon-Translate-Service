from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os  
import time  
import pytz  
import asyncio  
import logging  
import uvicorn  
import datetime  
import threading
from queue import Queue  
from threading import Thread, Event  
from api.core.transcribe_manager import TranscribeManager
from api.core.translate_manager import TranslateManager
from api.core.threading_api import audio_translate, texts_translate, waiting_times, stop_thread, audio_translate_sse, audio_pipeline_coordinator
from lib.core.response_manager import storage_upload
from wjy3 import BaseResponse, Status
from lib.config.constant import AudioTranslationResponse, TextTranslationResponse, WAITING_TIME, LANGUAGE_LIST, TRANSCRIPTION_METHODS, TRANSLATE_METHODS, DEFAULT_PROMPTS, DEFAULT_RESULT, MAX_NUM_STRATEGIES, set_global_model
from api.utils import write_txt, format_text_spacing, format_cleaning
from api.core.utils import ResponseTracker
from api import websocket_router
from lib.core.logging_config import setup_application_logger

# Create necessary directories if they don't exist
if not os.path.exists("./audio"):  
    os.mkdir("./audio")  
if not os.path.exists("./logs"):  
    os.mkdir("./logs")
if not os.path.exists("./static"):
    os.mkdir("./static")
    
# Configure logging using centralized configuration
logger = setup_application_logger(
    logger_name=__name__,
    log_level=logging.INFO,
    log_file="logs/app.log",
    max_bytes=10*1024*1024,
    backup_count=5,
    console_output=True
)  
  
# Configure UTC+8 time  
utc_now = datetime.datetime.now(pytz.utc)  
tz = pytz.timezone('Asia/Taipei')  
local_now = utc_now.astimezone(tz)  
use_pretext = True
  
# Initialize global objects and variables
transcribe_manager = TranscribeManager()
translate_manager = TranslateManager()
response_tracker = ResponseTracker()  # Tracker to prevent race conditions
waiting_list = []  # Queue for waiting translation requests
sse_stop_event = Event()  # Global event to control SSE connection
service_stop_event = Event()  # Event to control service shutdown  

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler for startup and shutdown events.
    """
    try:
        # Startup
        logger.info(f" | ##################################################### | ")  
        logger.info(f" | Start to loading default model. | ")  
        # load model  
        default_model = "ggml_breeze_asr_25"  
        transcribe_manager.load_model(default_model)  # Directly load the default model  
        logger.info(f" | Default model {default_model} has been loaded successfully. Model ID: {id(transcribe_manager)} | ")  
        # preheat  
        logger.info(f" | Start to preheat model. | ")  
        default_audio = "audio/test.wav"  
        start = time.time()  
        for _ in range(5):  
            transcribe_manager.transcribe(default_audio, "en", post_processing=False)  
        end = time.time()  
        logger.info(f" | Preheat model has been completed in {end - start:.2f} seconds. | ")  
        # set default prompt
        transcribe_manager.set_prompt(DEFAULT_PROMPTS["DEFAULT"])
        logger.info(f" | Default prompt has been set. | ")  
        
        # 設置 websocket 的 model
        # set_global_model(transcribe_manager)
        # logger.info(f" | WebSocket model has been set. Model ID: {id(transcribe_manager)} | ")
        
        # Start pipeline worker thread
        transcribe_manager.start_worker()
        logger.info(f" | Pipeline worker thread has been started. | ")
        logger.info(f" | ##################################################### | ")  
        # delete_old_audio_files()
        
        # Start daily task scheduling  
        task_thread = Thread(target=schedule_daily_task, args=(service_stop_event,))  
        task_thread.start()
        
        yield  # Application starts receiving requests
        
    except asyncio.CancelledError:
        # Handle graceful shutdown when Ctrl+C is pressed
        logger.info(" | Application shutdown initiated (Ctrl+C pressed) | ")
    except Exception as e:
        logger.error(f" | Error during application lifespan: {e} | ")
    finally:
        # Shutdown - always executed regardless of how we got here
        try:
            logger.info(" | Starting shutdown... | ")
            service_stop_event.set()  
            task_thread.join()
            
            # Stop pipeline worker
            transcribe_manager.stop_worker_thread()
            
            # Close translation manager
            translate_manager.close()
            logger.info(" | shutdown completed successfully | ")
        except Exception as shutdown_error:
            logger.error(f" | Error during shutdown: {shutdown_error} | ")

app = FastAPI(lifespan=lifespan)
app.include_router(websocket_router)

# Mount static files for admin page
if os.path.exists("./static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

##############################################################################  
# API Endpoints
##############################################################################  

@app.get("/")  
def HelloWorld(name:str=None):  
    """Health check endpoint."""
    return {"Hello": f"World {name}"}  

# @app.get("/admin", response_class=HTMLResponse)
# async def admin_page():
#     """
#     WebSocket Admin Console page.
    
#     Provides a web-based interface for:
#     - Monitoring WebSocket connections
#     - Testing WebSocket messages
#     - Streaming audio from microphone
#     - Viewing real-time message logs
#     """
#     admin_html_path = os.path.join("static", "admin.html")
#     if os.path.exists(admin_html_path):
#         return FileResponse(admin_html_path)
#     else:
#         return HTMLResponse(
#             content="<h1>Admin page not found</h1><p>Please ensure static/admin.html exists.</p>",
#             status_code=404
#         )

##############################################################################  

@app.get("/get_current_model")  
async def get_items():  
    """
    Get information about currently loaded models.
    
    Returns:
        BaseResponse: Current transcription and translation models
    """
    logger.info(f" | ############### Transcription model ########################### | ")  
    logger.info(f" | current transcription model is {transcribe_manager.transcription_method} | ")  
    logger.info(f" | ################# Translate methods ########################### | ")  
    logger.info(f" | current translation model is {translate_manager.translation_method} | ")  
    logger.info(f" | ############################################################### | ")  
    return BaseResponse(message=f" | current transcription model is {transcribe_manager.transcription_method} | current translation model is {translate_manager.translation_method} | ", data=[transcribe_manager.transcription_method, translate_manager.translation_method])  

@app.get("/list_optional_items")  
async def get_items():  
    """  
    List the optional items for inference models and translate methods.  
  
    This endpoint provides information about the available inference models  
    and translation methods that can be selected.  
  
    :rtype: str: A string listing the available inference models and translation methods.  
    """  
    logger.info(f" | ################# Transcription model #################change_############ | ")  
    logger.info(f" | You can choose {TRANSCRIPTION_METHODS} | ")  
    logger.info(f" | ################### Translate methods ############################# | ")  
    logger.info(f" | You can choose {TRANSLATE_METHODS} | ")  
    logger.info(f" | ################################################################### | ")  
    return BaseResponse(message=f" | Transcription method: You can choose '{TRANSCRIPTION_METHODS}' | Translate method: You can choose '{TRANSLATE_METHODS}' | ", data=None)  
   
@app.post("/change_transcription_model")  
async def change_transcription_model(model_name: str = Form(...)):  
    """  
    Load a specified model.  
      
    This endpoint allows the user to load a specified model for inference.  
      
    :param request: LoadModelRequest  
        The request object containing the model's name to be loaded.  
    :rtype: BaseResponse  
        A response indicating the success or failure of the model loading process.  
    """
    # Convert the model's name to lowercase  
    model_name = model_name.lower()  
    
    # Normalize model name: replace dashes with underscores for attribute lookup
    model_name = model_name.replace('-', '_')
      
    # Check if the model's name exists in the model's path  
    if not hasattr(transcribe_manager.models_path, model_name):  
        # Raise an HTTPException if the model is not found  
        logger.info(f" | Model '{model_name}' not found. | ")
        return BaseResponse(status=Status.FAILED, message=f" | Model '{model_name}' not found. | ", data=None)  
      
    # Load the specified model
    message = transcribe_manager.load_model(model_name)  
      
    # Return a response indicating the success of the model loading process  
    if message is None:  
        return BaseResponse(status=Status.OK, message=f" | Model {model_name} has been loaded successfully. | ", data=None)  
    else:  
        return BaseResponse(status=Status.FAILED, message=f" | Model {model_name} loading failed: {message} | ", data=None)  
    
@app.post("/set_prompt")
async def set_prompt(prompts = Form(None)):
    """
    Set a custom prompt for the transcription model.
    
    Args:
        prompts: Custom prompt text or None to clear
        
    Returns:
        BaseResponse: Status of prompt setting operation
    """
    
    if prompts is None or prompts == "" or (isinstance(prompts, str) and prompts.strip() == ""):
        prompt_message = transcribe_manager.set_prompt(None)
    else:
        prompt_message = transcribe_manager.set_prompt(prompts)
    
    if prompt_message:
        return BaseResponse(status=Status.FAILED, message=prompt_message, data=None)
    else:
        return BaseResponse(status=Status.OK, message=" | Prompt has been set successfully. | ", data=None)
    
@app.post("/enable_pretext")
async def enable_pretext():
    """
    Enable the use of previous text context for transcription.
    
    Returns:
        BaseResponse: Status of the operation
    """
    global use_pretext
    use_pretext = True
    logger.info(f" | Previous text context has been enabled. | ")
    return BaseResponse(message=" | Previous text context has been enabled. | ", data={"use_pretext": use_pretext})

@app.post("/disable_pretext")
async def disable_pretext():
    """
    Disable the use of previous text context for transcription.
    
    Returns:
        BaseResponse: Status of the operation
    """
    global use_pretext
    use_pretext = False
    logger.info(f" | Previous text context has been disabled. | ")
    return BaseResponse(message=" | Previous text context has been disabled. | ", data={"use_pretext": use_pretext})

@app.get("/get_pretext_status")
async def get_pretext_status():
    """
    Get the current status of previous text context usage.
    
    Returns:
        BaseResponse: Current status of use_pretext
    """
    logger.info(f" | Current pretext status: {use_pretext} | ")
    return BaseResponse(message=f" | Current pretext status: {'enabled' if use_pretext else 'disabled'} | ", data={"use_pretext": use_pretext})


@app.post("/translate")
async def translate(
    file: UploadFile = File(...),  
    meeting_id: str = Form(123),  
    device_id: str = Form(123),  
    audio_uid: str = Form(123),  
    times: datetime.datetime = Form(...),  
    o_lang: str = Form("zh"),  
    t_lang: str = Form("zh,en,ja,ko,de"),
    prev_text: str = Form(""),
    multi_strategy_transcription: int = Form(4), # 1~MAX_NUM_STRATEGIES others 1
    transcription_post_processing: bool = Form(True), # True/False
    multi_translate: bool = Form(True)
):  
    """  
    Transcribe and translate an audio file.  
      
    This endpoint receives an audio file and its associated metadata, and  
    performs transcription and translation on the audio file.  
      
    :param file: UploadFile  
        The audio file to be transcribed.  
    :param meeting_id: str  
        The ID of the meeting.  
    :param device_id: str  
        The ID of the device.  
    :param audio_uid: str  
        The unique ID of the audio.  
    :param times: datetime.datetime  
        The start time of the audio.  
    :param o_lang: str  
        The original language of the audio.  
    :param t_lang: Optional[str]
        Target languages for translation. Can be single language 'en' or comma-separated 'en,ja,ko'. If None or empty, no translation.
    :param prev_text: str
        The previous text for context (will be overridden by global previous translation)
    :rtype: BaseResponse  
        A response containing the transcription results.  
    """  
    
    # Handle t_lang parameter
    if t_lang:
        # Convert comma-separated string to list
        if ',' in t_lang:
            t_lang = [lang.strip().lower() for lang in t_lang.split(',')]
        else:
            t_lang = [t_lang.strip().lower()]
    else:
        # Empty or None means no translation
        t_lang = []
    
    # 20251118 we found if we use prev_text in 0.5 sec audio it will cause worse results
    if multi_strategy_transcription == 1:
        prev_text = ""  # Clear previous text if only one strategy is used
    
    # 20251121 use_pretext global control
    if not use_pretext:
        logger.info(f" | Previous text context usage is disabled. Overriding prev_text to empty. | ")
        prev_text = ""
    
    # Convert times to string format  
    times = str(times)  
    # Convert original language to lowercase  
    o_lang = o_lang.lower()
    multi_strategy_transcription = multi_strategy_transcription if 0 < multi_strategy_transcription <= MAX_NUM_STRATEGIES else 1
    
    # Create response data structure  
    response_data = AudioTranslationResponse(  
        meeting_id=meeting_id,  
        device_id=device_id,  
        ori_lang=o_lang,  
        transcription_text="",
        n_segments=0,
        segments=[],  
        text=DEFAULT_RESULT.copy(),  
        times=str(times),  
        audio_uid=audio_uid,  
        transcribe_time=0.0,  
        translate_time=0.0,  
    )  
  
    # Save the uploaded audio file  
    filename = (
                f"{audio_uid}_{times.replace(':', ';').replace(' ', '_')}.wav"
            )
    os.makedirs(f"audio/{meeting_id}", exist_ok=True)
    audio_buffer = f"audio/{meeting_id}/{filename}"  
    
    # Read file content once
    file_content = file.file.read()
    
    with open(audio_buffer, 'wb') as f:  
        f.write(file_content)
  
    # Check if the audio file exists  
    if not os.path.exists(audio_buffer):  
        return BaseResponse(status=Status.FAILED, message=" | The audio file does not exist, please check the audio path. | ", data=response_data)  
  
    # Check if the model has been loaded  
    if transcribe_manager.transcription_method is None:  
        return BaseResponse(status=Status.FAILED, message=" | model haven't been load successfully. may out of memory please check again | ", data=response_data)  
  
    # Check if the languages are in the supported language list  
    if o_lang not in LANGUAGE_LIST:  
        logger.info(f" | The original language is not in LANGUAGE_LIST: {LANGUAGE_LIST}. | ")  
        return BaseResponse(status=Status.FAILED, message=f" | The original language is not in LANGUAGE_LIST: {LANGUAGE_LIST}. | ", data=response_data)  
  
    # Check if all target languages are in the supported language list
    for lang in t_lang:
        if lang not in LANGUAGE_LIST:  
            logger.info(f" | The target language '{lang}' is not in LANGUAGE_LIST: {LANGUAGE_LIST}. | ")  
            return BaseResponse(status=Status.FAILED, message=f" | The target language '{lang}' is not in LANGUAGE_LIST: {LANGUAGE_LIST}. | ", data=response_data)  
  
    try:  
        # Create a queue to hold the return value  
        result_queue = Queue()  
        # Create an event to signal stopping  
        stop_event = threading.Event()  
  
        # Create timing thread and inference thread  
        time_thread = threading.Thread(target=waiting_times, args=(stop_event, transcribe_manager, WAITING_TIME))  
        inference_thread = threading.Thread(target=audio_translate, args=(transcribe_manager, translate_manager, audio_buffer, result_queue, o_lang, t_lang, stop_event, multi_strategy_transcription, transcription_post_processing, prev_text, multi_translate))
  
        # Start the threads  
        time_thread.start()  
        inference_thread.start()  
  
        # Wait for timing thread to complete and check if the inference thread is active to close  
        time_thread.join()  
        stop_thread(inference_thread)  
  
        # Remove the audio buffer file  
        # if os.path.exists(audio_buffer):
        #     os.remove(audio_buffer)  
  
        # Get the result from the queue  
        if not result_queue.empty():  
            ori_pred, n_segments, segments, result, rtf, transcription_time, translate_time, translate_method, timing_dict, other_info = result_queue.get()  
            response_data.transcription_text = ori_pred
            response_data.text = result  
            response_data.n_segments = n_segments
            response_data.segments = segments
            response_data.transcribe_time = transcription_time  
            response_data.translate_time = translate_time  
            zh_result = response_data.text.get("zh", "")
            en_result = response_data.text.get("en", "")
            de_result = response_data.text.get("de", "")
            ja_result = response_data.text.get("ja", "")
            ko_result = response_data.text.get("ko", "")
            
            # Format timing_dict: each task as separate entry
            timing_parts = []
            if timing_dict:
                for translator, time_lang_pairs in timing_dict.items():
                    if time_lang_pairs and isinstance(time_lang_pairs[0], tuple):
                        for t, lang in time_lang_pairs:
                            timing_parts.append(f"{translator}: {t:.2f}s ({lang})")
                    else:
                        for t in time_lang_pairs:
                            timing_parts.append(f"{translator}: {t:.2f}s")
            timing_str = " | ".join(timing_parts) if timing_parts else "N/A"
            
            logger.debug(f" | {response_data.model_dump_json()} | ")  
            logger.info(f" | meeting_id: {response_data.meeting_id} | audio_uid: {response_data.audio_uid} | source language: {o_lang} | translate_method: {translate_method} | time: {times} | ")  
            logger.info(f" | Transcription: {ori_pred} | ")    
            logger.info(f" | {n_segments} | segments: {segments} | ")            
            if timing_str != "N/A": 
                logger.info(f" | {timing_str} | ")
            if t_lang:
                logger.info(f" | {'#' * 75} | ")
                logger.info(f" | ZH: {zh_result} | ")  
                logger.info(f" | EN: {en_result} | ")  
                logger.info(f" | DE: {de_result} | ")  
                logger.info(f" | JA: {ja_result} | ")  
                logger.info(f" | KO: {ko_result} | ")  
                logger.info(f" | {'#' * 75} | ")
            logger.info(f" | RTF: {rtf} | total time: {transcription_time + translate_time:.2f} seconds. | transcribe {transcription_time:.2f} seconds. | translate {translate_time:.2f} seconds. | strategy: {multi_strategy_transcription} | ")  
            state = Status.OK
        else:  
            logger.info(f" | Translation has exceeded the upper limit time and has been stopped |")  
            ori_pred = zh_result = en_result = de_result = ja_result = ko_result = ""
            state = Status.FAILED
            
        # write_txt(zh_result, en_result, de_result, ja_result, ko_result, meeting_id, audio_uid, times)
        if other_info:
            other_info['audio_uid'] = audio_uid
            other_info['audio_file_name'] = f"{audio_uid}_{times.replace(':', ';').replace(' ', '_')}.wav"
            storage_upload(logger, response_data, other_info) 

        return BaseResponse(status=state, message=f" | Transcription: {ori_pred} | ZH: {zh_result} | EN: {en_result} | DE: {de_result} | JA: {ja_result} | KO: {ko_result} | ", data=response_data)  
    except Exception as e:  
        logger.error(f" | Translation() error: {e} | ")  
        return BaseResponse(status=Status.FAILED, message=f" | Translation() error: {e} | ", data=response_data)  


@app.post("/translate_pipeline")
async def translate_pipeline(
    file: UploadFile = File(...),  
    meeting_id: str = Form(123),  
    device_id: str = Form(123),  
    audio_uid: str = Form(123),  
    times: datetime.datetime = Form(...),  
    o_lang: str = Form("zh"),  
    t_lang: str = Form("zh,en,ja,ko,de"),
    prev_text: str = Form(""),
    multi_strategy_transcription: int = Form(4), # 1~MAX_NUM_STRATEGIES others 1
    transcription_post_processing: bool = Form(True), # True/False
    multi_translate: bool = Form(True)
):  
    """
    Pipeline endpoint for transcription and translation.
    
    This endpoint uses a queue-based worker system to enable parallel resource utilization.
    Transcription tasks are queued and processed sequentially, but translation happens
    in parallel, allowing the next transcription to start while the previous translation
    is still running.
    """
    # start = time.time()
    
    logger.debug(f" | Received pipeline translation request: audio_uid={audio_uid}, times={times}) | ")
    
    # Handle t_lang parameter
    if t_lang:
        if ',' in t_lang:
            t_lang = [lang.strip().lower() for lang in t_lang.split(',')]
        else:
            t_lang = [t_lang.strip().lower()]
    else:
        t_lang = []
    
    # Handle prev_text
    if multi_strategy_transcription == 1:
        prev_text = ""
    
    if not use_pretext:
        logger.info(f" | Previous text context usage is disabled. Overriding prev_text to empty. | ")
        prev_text = ""
    
    times = str(times)  
    o_lang = o_lang.lower()
    multi_strategy_transcription = multi_strategy_transcription if 0 < multi_strategy_transcription <= MAX_NUM_STRATEGIES else 1
    
    # Register this request in the tracker
    task_id = response_tracker.register_request(audio_uid, times)
    
    # Create response data structure  
    response_data = AudioTranslationResponse(  
        meeting_id=meeting_id,  
        device_id=device_id,  
        ori_lang=o_lang,  
        transcription_text="",
        n_segments=0,
        segments=[],
        text=DEFAULT_RESULT.copy(),  
        times=str(times),  
        audio_uid=audio_uid,  
        transcribe_time=0.0,  
        translate_time=0.0,  
    )  
  
    # Save the uploaded audio file  
    filename = f"{audio_uid}_{times.replace(':', ';').replace(' ', '_')}.wav"
    os.makedirs(f"audio/{meeting_id}", exist_ok=True)
    audio_buffer = f"audio/{meeting_id}/{filename}"  
    
    file_content = file.file.read()
    with open(audio_buffer, 'wb') as f:  
        f.write(file_content)
  
    # Validation checks
    if not os.path.exists(audio_buffer):  
        return BaseResponse(status=Status.FAILED, message=" | The audio file does not exist, please check the audio path. | ", data=response_data)  
  
    if transcribe_manager.transcription_method is None:  
        return BaseResponse(status=Status.FAILED, message=" | model haven't been load successfully. may out of memory please check again | ", data=response_data)  
  
    if o_lang not in LANGUAGE_LIST:  
        logger.info(f" | The original language is not in LANGUAGE_LIST: {LANGUAGE_LIST}. | ")  
        return BaseResponse(status=Status.FAILED, message=f" | The original language is not in LANGUAGE_LIST: {LANGUAGE_LIST}. | ", data=response_data)  
  
    for lang in t_lang:
        if lang not in LANGUAGE_LIST:  
            logger.info(f" | The target language '{lang}' is not in LANGUAGE_LIST: {LANGUAGE_LIST}. | ")  
            return BaseResponse(status=Status.FAILED, message=f" | The target language '{lang}' is not in LANGUAGE_LIST: {LANGUAGE_LIST}. | ", data=response_data)  
  
    # Use the new pipeline coordinator for better decoupling with timeout
    try:
        result, other_info = await asyncio.wait_for(
            asyncio.to_thread(
                audio_pipeline_coordinator,
                transcribe_manager=transcribe_manager,
                translate_manager=translate_manager,
                audio_file=audio_buffer,
                o_lang=o_lang,
                t_lang=t_lang,
                multi_strategy_transcription=multi_strategy_transcription,
                transcription_post_processing=transcription_post_processing,
                prev_text=prev_text,
                multi_translate=multi_translate,
                audio_uid=audio_uid,
                times=times
            ),
            timeout=WAITING_TIME  # Use the same timeout as the original design
        )
    except asyncio.TimeoutError:
        logger.warning(f" | Translation timeout for audio_uid: {audio_uid}, force terminating task. | ")
        
        # Force terminate the currently executing task
        try:
            terminated = transcribe_manager.force_terminate_current_task(audio_uid)
            if terminated:
                logger.debug(f" | Successfully force terminated task for audio_uid: {audio_uid}. | ")
            else:
                logger.info(f" | No active task found to terminate for audio_uid: {audio_uid}. | ")
        except Exception as term_error:
            logger.error(f" | Error during force termination: {term_error} | ")
        
        # Try to clean up translation threads if they exist
        try:
            translate_manager.cleanup_translation_threads()
        except Exception as cleanup_error:
            logger.warning(f" | Error during timeout cleanup: {cleanup_error} | ")
        
        result = None  # Set result to None when timeout occurs
        other_info = None  # Initialize other_info to prevent UnboundLocalError  

    # Check if this request was cancelled by a newer one
    if response_tracker.check_cancelled(audio_uid, task_id):
        logger.info(f" | Task {task_id} (audio_uid: {audio_uid}, times: {times}) cancelled due to newer request | ")
        response_tracker.cleanup(audio_uid, task_id)
        return BaseResponse(status=Status.OK, message=" | Translation task was cancelled due to newer request | ", data=response_data)
    
    if result and result[0] is not None:  # Check if cancelled or failed
        # Mark older pending requests as cancelled
        response_tracker.complete_and_cancel_older(audio_uid, task_id, times)
        
        ori_pred, n_segments, segments, translated_result, rtf, transcription_time, translate_time, translate_method, timing_dict = result
        
        # Check if this result indicates cancellation from other_info
        if other_info and 'cancelled_by_times' in other_info:
            cancelled_by = other_info['cancelled_by_times']
            logger.info(f" | Task {task_id} (audio_uid: {audio_uid}, times: {times}) cancelled by newer request (times: {cancelled_by}) | ")
            response_tracker.cleanup(audio_uid, task_id)
            return BaseResponse(status=Status.OK, message=" | Translation task was cancelled due to newer request | ", data=response_data)
        response_data.transcription_text = ori_pred
        response_data.n_segments = n_segments
        response_data.segments = segments
        response_data.text = format_text_spacing(translated_result) if multi_strategy_transcription == 4 else format_cleaning(translated_result)
        response_data.transcribe_time = transcription_time  
        response_data.translate_time = translate_time  
        zh_result = response_data.text.get("zh", "")
        en_result = response_data.text.get("en", "")
        de_result = response_data.text.get("de", "")
        ja_result = response_data.text.get("ja", "")
        ko_result = response_data.text.get("ko", "")
        
        # Format timing_dict
        timing_parts = []
        if timing_dict:
            for translator, time_lang_pairs in timing_dict.items():
                if time_lang_pairs and isinstance(time_lang_pairs[0], tuple):
                    for t, lang in time_lang_pairs:
                        timing_parts.append(f"{translator}: {t:.2f}s ({lang})")
                else:
                    for t in time_lang_pairs:
                        timing_parts.append(f"{translator}: {t:.2f}s")
        timing_str = " | ".join(timing_parts) if timing_parts else "N/A"
        
        logger.debug(f" | {response_data.model_dump_json()} | ")  
        logger.info(f" | meeting_id: {response_data.meeting_id} | audio_uid: {response_data.audio_uid} | source language: {o_lang} | translate_method: {translate_method} | time: {times} | ")  
        logger.info(f" | Transcription: {ori_pred} | ")       
        logger.info(f" | {n_segments} | segments: {segments} | ")         
        if timing_str != "N/A": 
            logger.info(f" | {timing_str} | ")
        if t_lang:
            logger.info(f" | {'#' * 75} | ")
            logger.info(f" | ZH: {zh_result} | ")  
            logger.info(f" | EN: {en_result} | ")  
            logger.info(f" | DE: {de_result} | ")  
            logger.info(f" | JA: {ja_result} | ")  
            logger.info(f" | KO: {ko_result} | ")  
            logger.info(f" | {'#' * 75} | ")
        logger.info(f" | RTF: {rtf} | total time: {transcription_time + translate_time:.2f} seconds. | transcribe {transcription_time:.2f} seconds. | translate {translate_time:.2f} seconds. | strategy: {multi_strategy_transcription} | ")  
        state = Status.OK
        message = f" | Transcription: {ori_pred} | ZH: {zh_result} | EN: {en_result} | DE: {de_result} | JA: {ja_result} | KO: {ko_result} | "
    else:  
        state = Status.FAILED
        # Determine failure reason and set appropriate message
        if result is None:
            # Timeout occurred
            message = " | Translation has exceeded the upper limit time and has been stopped | "
        elif result and result[0] is None:
            # Task was cancelled (e.g., duplicate request)
            message = " | Translation task was cancelled due to newer request | "
        else:
            # Other failure reasons (interrupted, transcription failed, etc.)
            message = " | Pipeline processing failed | "
        logger.warning(message)
    
    if other_info:
        other_info['audio_file_name'] = f"{audio_uid}_{times.replace(':', ';').replace(' ', '_')}.wav"
        storage_upload(logger, response_data, other_info)

    # Clean up this request from tracker
    response_tracker.cleanup(audio_uid, task_id)

    # end = time.time()
    # logger.info(f" | Total pipeline processing time: {end - start:.2f} seconds. | ")
    return BaseResponse(status=state, message=message, data=response_data)
    

@app.post("/text_translate")  
async def text_translate(  
    text: str = Form(...),
    source_language: str = Form("zh"),
    target_language: str = Form("zh,en,ja,ko,de"),
    multi_translate: bool = Form(True)
):  
    """  
    Translate a text.  
  
    This endpoint receives text and its associated metadata, and performs translation on the text.  
  
    :param text: str
        The text to be translated.
    :param source_language: str
        The source language code.
    :param target_language: str
        Target languages for translation. Can be single language 'en' or comma-separated 'en,ja,ko'. If None or empty, no translation.
    :param multi_translate: bool
        If True, distribute tasks across multiple LLMs; If False, use single LLM to translate all languages at once
    :rtype: BaseResponse  
        A response containing the translation results.  
    """  
    # Handle target_language parameter
    if target_language:
        # Convert comma-separated string to list
        if ',' in target_language:
            target_language = [lang.strip().lower() for lang in target_language.split(',')]
        else:
            target_language = [target_language.strip().lower()]
    else:
        # Empty or None means no translation
        target_language = []
    
    source_language = source_language.lower()
  
    # Create response data structure first (for error handling)
    response_data = TextTranslationResponse(  
        ori_lang=source_language,
        text=DEFAULT_RESULT.copy(),
        translate_time=0.0
    )
    
    # Check if the languages are in the supported language list  
    if source_language not in LANGUAGE_LIST:  
        logger.info(f" | The original language is not in LANGUAGE_LIST: {LANGUAGE_LIST}. | ")  
        return BaseResponse(status=Status.FAILED, message=f" | The original language is not in LANGUAGE_LIST: {LANGUAGE_LIST}. | ", data=response_data)  
  
    # Check if all target languages are in the supported language list
    for lang in target_language:
        if lang not in LANGUAGE_LIST:  
            logger.info(f" | The target language '{lang}' is not in LANGUAGE_LIST: {LANGUAGE_LIST}. | ")  
            return BaseResponse(status=Status.FAILED, message=f" | The target language '{lang}' is not in LANGUAGE_LIST: {LANGUAGE_LIST}. | ", data=response_data)  
  
    try:  
        # Create a queue to hold the return value  
        result_queue = Queue()  
        # Create an event to signal stopping  
        stop_event = threading.Event()  
  
        # Create timing thread and inference thread  
        time_thread = threading.Thread(target=waiting_times, args=(stop_event, transcribe_manager, WAITING_TIME))  
        inference_thread = threading.Thread(target=texts_translate, args=(translate_manager, text, result_queue, source_language, target_language, stop_event, multi_translate))  
  
        # Start the threads  
        time_thread.start()  
        inference_thread.start()  
  
        # Wait for timing thread to complete and check if the inference thread is active to close  
        time_thread.join()  
        stop_thread(inference_thread)  
  
        # Get the result from the queue  
        if not result_queue.empty():  
            result, translate_time, translate_method, timing_dict = result_queue.get()  
            response_data.text = result  
            response_data.translate_time = translate_time
            zh_result = response_data.text.get("zh", "")
            en_result = response_data.text.get("en", "")
            de_result = response_data.text.get("de", "")
            ja_result = response_data.text.get("ja", "")
            ko_result = response_data.text.get("ko", "")
            
            # Format timing_dict: each task as separate entry
            timing_parts = []
            if timing_dict:
                for translator, time_lang_pairs in timing_dict.items():
                    if time_lang_pairs and isinstance(time_lang_pairs[0], tuple):
                        for t, lang in time_lang_pairs:
                            timing_parts.append(f"{translator}: {t:.2f}s ({lang})")
                    else:
                        for t in time_lang_pairs:
                            timing_parts.append(f"{translator}: {t:.2f}s")
            timing_str = " | ".join(timing_parts) if timing_parts else "N/A"
  
            logger.debug(f" | {response_data.model_dump_json()} | ")  
            logger.info(f" | source language: {source_language} -> target language: {target_language} | translate_method: {translate_method} |")
            if timing_str != "N/A":
                logger.info(f" | {timing_str} | ")
            if target_language:
                logger.info(f" | {'#' * 75} | ")
                logger.info(f" | ZH: {zh_result} | ")  
                logger.info(f" | EN: {en_result} | ")  
                logger.info(f" | DE: {de_result} | ")  
                logger.info(f" | JA: {ja_result} | ")  
                logger.info(f" | KO: {ko_result} | ")
                logger.info(f" | {'#' * 75} | ")
            logger.info(f" | translate has been completed in {translate_time:.2f} seconds. |")  
            state = Status.OK
        else:
            logger.info(f" | translation has exceeded the upper limit time and has been stopped |")
            zh_result = en_result = de_result = ja_result = ko_result = ""
            state = Status.FAILED

        return BaseResponse(status=state, message=f" | ZH: {zh_result} | EN: {en_result} | DE: {de_result} | JA: {ja_result} | KO: {ko_result} | ", data=response_data)
    except Exception as e:
        logger.error(f" | translation() error: {e} | ")
        return BaseResponse(status=Status.FAILED, message=f" | translation() error: {e} | ", data=response_data)

@app.post("/sse_audio_translate")
async def sse_audio_translate(
    file: UploadFile = File(...),
    meeting_id: str = Form(...),  
    device_id: str = Form(...),  
    audio_uid: str = Form(...),  
    times: datetime.datetime = Form(...),  
    o_lang: str = Form(...),  
    prev_text: str = Form(""),
    multi_strategy_transcription: int = Form(1), # 1~MAX_NUM_STRATEGIES others 1
    transcription_post_processing: bool = Form(True), # True/False
    use_translate: bool = Form(True), # True/False
    multi_translate: bool = Form(True) # True/False
):  
    """  
    Transcribe and translate an audio file.  
      
    This endpoint receives an audio file and its associated metadata, and  
    performs transcription and translation on the audio file.  
      
    :param file: UploadFile  
        The audio file to be transcribed.  
    :param meeting_id: str  
        The ID of the meeting.  
    :param device_id: str  
        The ID of the device.  
    :param audio_uid: str  
        The unique ID of the audio.  
    :param times: datetime.datetime  
        The start time of the audio.  
    :param o_lang: str  
        The original language of the audio.  
    :rtype: BaseResponse  
        A response containing the transcription results.  
    """  
    # Convert times to string format  
    times = str(times)  
    # Convert original language and target language to lowercase  
    o_lang = o_lang.lower()  
    
    # Create response data structure first  
    response_data = AudioTranslationResponse(  
        meeting_id=meeting_id,  
        device_id=device_id,  
        ori_lang=o_lang,  
        transcription_text="",
        text=DEFAULT_RESULT.copy(),  
        times=str(times),  
        audio_uid=audio_uid,  
        transcribe_time=0.0,  
        translate_time=0.0,  
    )  
    
    # Check if the model has been loaded  
    if transcribe_manager.transcription_method is None:  
        return BaseResponse(status=Status.FAILED, message=" | model haven't been load successful. may out of memory please check again | ", data=response_data)  
        
    # Check if the languages are in the supported language list  
    if o_lang not in LANGUAGE_LIST:  
        logger.info(f" | The original language is not in LANGUAGE_LIST: {LANGUAGE_LIST}. | ")  
        return BaseResponse(status=Status.FAILED, message=f" | The original language is not in LANGUAGE_LIST: {LANGUAGE_LIST}. | ", data=response_data)  
  
    other_information = {
        "prev_text": prev_text,
        "multi_strategy_transcription": multi_strategy_transcription,
        "transcription_post_processing": transcription_post_processing,
        "use_translate": use_translate,
        "multi_translate": multi_translate
    }

    try:  
        previous_waiting_list = waiting_list.copy()  
        audio_uid_exist = False  
          
        # Check if the audio UID already exists in the waiting list  
        for item in previous_waiting_list:  
            item_response_data = item[0]
            if item_response_data.audio_uid == response_data.audio_uid:  
                audio_uid_exist = True  
                if item_response_data.times < response_data.times:  
                    waiting_list.remove(item)  
                    waiting_list.append([response_data, other_information])  
                    audio = f"audio/{item_response_data.times}.wav"  
                    if os.path.exists(audio):  
                        os.remove(audio)  
                break  
          
        if not audio_uid_exist:
            waiting_list.append([response_data, other_information])  
          
        if previous_waiting_list != waiting_list:  
            filename = (
                f"{audio_uid}_{times.replace(':', ';').replace(' ', '_')}.wav"
            )
            os.makedirs(f"audio/{response_data.meeting_id}", exist_ok=True)
            audio_buffer = f"audio/{response_data.meeting_id}/{filename}"  
            
            # Read file content once and save
            file_content = file.file.read()
            with open(audio_buffer, 'wb') as f:  
                f.write(file_content)  
          
        return BaseResponse(status=Status.OK, message=" | Request added to the waiting list. | ", data=None)  
    except Exception as e:  
        logger.error(f' | save info error: {e} | ')  
        return BaseResponse(status=Status.FAILED, message=f" | save info error: {e} | ", data=response_data)    
    
    
@app.get("/sse_audio_translate")  
async def sse_audio_translate():  
    """  
    Server-Sent Events endpoint to handle real-time translation.  
  
    This endpoint checks the waiting list and processes the translation if the model is not busy.  
    """  
    sse_stop_event.clear()  

    async def event_stream():  
        try:
            while not sse_stop_event.is_set():  
                if waiting_list and not transcribe_manager.processing:  
                    response_data, other_information = waiting_list.pop(0)  
                    audio_buffer = f"audio/{response_data.meeting_id}/{response_data.times}.wav" 
                    o_lang = response_data.ori_lang  
      
                    try:  
                        # Create an event to signal stopping  
                        stop_event = threading.Event()  
                        # Create timing thread and inference thread  
                        time_thread = threading.Thread(target=waiting_times, args=(stop_event, transcribe_manager, WAITING_TIME))  
                        inference_thread = threading.Thread(target=audio_translate_sse, args=(transcribe_manager, translate_manager, audio_buffer, o_lang, other_information, stop_event))  
      
                        # Start the threads  
                        time_thread.start()  
                        inference_thread.start()  
      
                        # Wait for timing thread to complete and stop the inference thread if still running  
                        time_thread.join()  
                        stop_thread(inference_thread)  
                        
                        # if os.path.exists(audio_buffer):
                        #     os.remove(audio_buffer)  
      
                        # Process all available results from the result queue
                        while not transcribe_manager.result_queue.empty():
                            ori_pred, result, rtf, transcription_time, translate_time, translate_method, timing_dict = transcribe_manager.result_queue.get()
                            response_data.transcription_text = ori_pred
                            response_data.text = result  
                            response_data.transcribe_time = transcription_time  
                            response_data.translate_time = translate_time  
                            zh_result = response_data.text.get("zh", "")
                            en_result = response_data.text.get("en", "")
                            de_result = response_data.text.get("de", "")
                            ja_result = response_data.text.get("ja", "")
                            ko_result = response_data.text.get("ko", "")
                            
                            # Format timing_dict: each task as separate entry
                            timing_parts = []
                            if timing_dict:
                                for translator, time_lang_pairs in timing_dict.items():
                                    if time_lang_pairs and isinstance(time_lang_pairs[0], tuple):
                                        for t, lang in time_lang_pairs:
                                            timing_parts.append(f"{translator}: {t:.2f}s ({lang})")
                                    else:
                                        for t in time_lang_pairs:
                                            timing_parts.append(f"{translator}: {t:.2f}s")
                            timing_str = " | ".join(timing_parts) if timing_parts else "N/A"
                            
                            logger.debug(f" | {response_data.model_dump_json()} | ")  
                            logger.info(f" | meeting_id: {response_data.meeting_id} | audio_uid: {response_data.audio_uid} | source language: {o_lang} | translate_method: {translate_method} |")  
                            logger.info(f" | Transcription: {ori_pred} | ")
                            if timing_str != "N/A":
                                logger.info(f" | {timing_str} | ")
                            if other_information["use_translate"]:
                                logger.info(f" | {'#' * 75} | ")
                                logger.info(f" | ZH: {zh_result} | ")  
                                logger.info(f" | EN: {en_result} | ")  
                                logger.info(f" | DE: {de_result} | ")  
                                logger.info(f" | JA: {ja_result} | ")  
                                logger.info(f" | KO: {ko_result} | ")  
                                logger.info(f" | {'#' * 75} | ")
                            else:
                                zh_result = en_result = de_result = ja_result = ko_result = ""
                                
                            logger.info(f" | RTF: {rtf} | total time: {transcription_time + translate_time:.2f} seconds. | transcribe {transcription_time:.2f} seconds. | translate {translate_time:.2f} seconds. | strategy: {other_information['multi_strategy_transcription']} | ")  

                            # Write translation results to txt files
                            # write_txt(zh_result, en_result, de_result, ja_result, ko_result, response_data.meeting_id, response_data.audio_uid, response_data.times)

                            base_response = BaseResponse(  
                                status=Status.OK,  
                                message=f" | Transcription: {ori_pred} | ZH: {zh_result} | EN: {en_result} | DE: {de_result } | JA: {ja_result} | KO: {ko_result} | ",  
                                data=response_data  
                            )  
                            yield f"data: {base_response.model_dump_json()}\n\n"
                        
                    except Exception as e:  
                        logger.error(f' | inference() error: {e} | ')  
                        base_response = BaseResponse(  
                            status=Status.FAILED,  
                            message=f" | inference() error: {e} | ",  
                            data=response_data  
                        )  
                        yield f"data: {base_response.model_dump_json()}\n\n"  
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            # 客戶端斷線時會觸發這個異常
            logger.info(" | Client disconnected from SSE | ")
            raise
        except Exception as e:
            logger.error(f" | SSE stream error: {e} | ")
        finally:
            # 確保清理資源
            logger.info(" | SSE stream ended | ")  
  
    return StreamingResponse(event_stream(), media_type="text/event-stream") 

@app.post("/stop_sse")  
async def stop_sse():  
    """Endpoint to stop the SSE connection."""  
    sse_stop_event.set() 
    return BaseResponse(status=Status.OK, message=" | SSE connection has been stopped | ", data=None)  


##############################################################################  
# Utility Functions
##############################################################################

# Clean up audio files  
def delete_old_audio_files():  
    """  
    The process of deleting old audio files recursively and removing empty directories  
    :param  
    ----------  
    None: The function does not take any parameters  
    :rtype  
    ----------  
    None: The function does not return any value  
    :logs  
    ----------  
    Deleted old files and empty directories  
    """  
    current_time = time.time()  
    audio_dir = "./audio"  
    
    # Recursively walk through all subdirectories
    for root, dirs, files in os.walk(audio_dir, topdown=False):
        # Process files in current directory
        for filename in files:
            if filename == "test.wav":  # Skip specific file
                continue
            file_path = os.path.join(root, filename)
            file_creation_time = os.path.getctime(file_path)
            # Delete files older than a day
            if current_time - file_creation_time > 24 * 60 * 60:
                os.remove(file_path)
                logger.info(f" | Deleted old file: {file_path} | ")
        
        # Remove empty directories (skip the main audio directory)
        if root != audio_dir:
            try:
                if not os.listdir(root):  # Check if directory is empty
                    os.rmdir(root)
                    logger.info(f" | Deleted empty directory: {root} | ")
            except OSError:
                # Directory not empty or other error, continue
                pass  
  
# Daily task scheduling  
def schedule_daily_task(stop_event):  
    """
    Schedule daily cleanup tasks.
    
    Args:
        stop_event: Event to signal stopping the scheduler
    """
    while not stop_event.is_set():  
        if local_now.hour == 0 and local_now.minute == 0:  
            delete_old_audio_files()  
            time.sleep(60)  # Prevent triggering multiple times within the same minute  
        time.sleep(1)  
  
if __name__ == "__main__":  
    port = int(os.environ.get("PORT", 80))  
    uvicorn.config.LOGGING_CONFIG["formatters"]["default"]["fmt"] = "%(asctime)s [%(name)s] %(levelprefix)s %(message)s"  
    uvicorn.config.LOGGING_CONFIG["formatters"]["access"]["fmt"] = '%(asctime)s [%(name)s] %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'  
    uvicorn.run(app, log_level='info', host='0.0.0.0', port=port)   
    
    
 