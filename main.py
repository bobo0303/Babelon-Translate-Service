from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse  
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
from api.model import Model  
from api.threading_api import audio_translate, texts_translate, waiting_times, stop_thread, audio_translate_sse
from lib.base_object import BaseResponse, Status
from lib.constant import AudioTranslationResponse, TextTranslationResponse, WAITING_TIME, LANGUAGE_LIST, TRANSCRIPTION_METHODS, TRANSLATE_METHODS, DEFAULT_PROMPTS, DEFAULT_RESULT, MAX_NUM_STRATEGIES
from api.utils import write_txt
from api import websocket_router

# Create necessary directories if they don't exist
if not os.path.exists("./audio"):  
    os.mkdir("./audio")  
if not os.path.exists("./logs"):  
    os.mkdir("./logs")  
    
# Configure logging  
log_format = "%(asctime)s - %(message)s"  # Output timestamp and message content  
log_file = "logs/app.log"  
logging.basicConfig(level=logging.INFO, format=log_format)  
  
# Create a file handler  
file_handler = logging.handlers.RotatingFileHandler(  
    log_file, maxBytes=10*1024*1024, backupCount=5  
)  
file_handler.setFormatter(logging.Formatter(log_format))  
  
# Create a console handler  
console_handler = logging.StreamHandler()  
console_handler.setFormatter(logging.Formatter(log_format))  
  
logger = logging.getLogger(__name__)  
logger.setLevel(logging.INFO)  # Ensure logger level is set to INFO or lower  
  
# Clear existing handlers to prevent duplicate logs  
if not logger.handlers:  
    logger.addHandler(file_handler)  
    logger.addHandler(console_handler)  # Add console handler 

logger.propagate = False  
  
# Configure UTC+8 time  
utc_now = datetime.datetime.now(pytz.utc)  
tz = pytz.timezone('Asia/Taipei')  
local_now = utc_now.astimezone(tz)  
  
# Initialize global objects and variables
model = Model()  
waiting_list = []  # Queue for waiting translation requests
sse_stop_event = Event()  # Global event to control SSE connection
service_stop_event = Event()  # Event to control service shutdown  

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler for startup and shutdown events.
    """
    # Startup
    logger.info(f" | ##################################################### | ")  
    logger.info(f" | Start to loading default model. | ")  
    # load model  
    default_model = "large_v2"  
    model.load_model(default_model)  # Directly load the default model  
    logger.info(f" | Default model {default_model} has been loaded successfully. | ")  
    # preheat  
    logger.info(f" | Start to preheat model. | ")  
    default_audio = "audio/test.wav"  
    start = time.time()  
    for _ in range(5):  
        model.transcribe(default_audio, "en", post_processing=False)  
    end = time.time()  
    logger.info(f" | Preheat model has been completed in {end - start:.2f} seconds. | ")  
    # set default prompt
    model.set_prompt(DEFAULT_PROMPTS["DEFAULT"])
    logger.info(f" | Default prompt has been set. | ")  
    logger.info(f" | ##################################################### | ")  
    # delete_old_audio_files()
    
    # Start daily task scheduling  
    # task_thread = Thread(target=schedule_daily_task, args=(service_stop_event,))  
    # task_thread.start()
    
    yield  # Application starts receiving requests
    
    # Shutdown
    service_stop_event.set()  
    # task_thread.join()  
    model.close()
    logger.info(" | Scheduled task has been stopped. | ")

app = FastAPI(lifespan=lifespan)
app.include_router(websocket_router)

##############################################################################  
# API Endpoints
##############################################################################  

@app.get("/")  
def HelloWorld(name:str=None):  
    """Health check endpoint."""
    return {"Hello": f"World {name}"}  

##############################################################################  

@app.get("/get_current_model")  
async def get_items():  
    """
    Get information about currently loaded models.
    
    Returns:
        BaseResponse: Current transcription and translation models
    """
    logger.info(f" | ############### Transcription model ########################### | ")  
    logger.info(f" | current transcription model is {model.model_version} | ")  
    logger.info(f" | ################# Translate methods ########################### | ")  
    logger.info(f" | current translation model is {model.translate_method} | ")  
    logger.info(f" | ############################################################### | ")  
    return BaseResponse(message=f" | current transcription model is {model.model_version} | current translation model is {model.translate_method} | ", data=[model.model_version, model.translate_method])  

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
      
    # Check if the model's name exists in the model's path  
    if not hasattr(model.models_path, model_name):  
        # Raise an HTTPException if the model is not found  
        return BaseResponse(status=Status.FAILED, message=f" | Model '{model_name}' not found. | ", data=None)  
      
    # Load the specified model  
    message = model.load_model(model_name)  
      
    # Return a response indicating the success of the model loading process  
    if message is None:  
        return BaseResponse(status=Status.OK, message=f" | Model {model_name} has been loaded successfully. | ", data=None)  
    else:  
        return BaseResponse(status=Status.FAILED, message=f" | Model {model_name} loading failed: {message} | ", data=None)  
    
@app.post("/change_translation_method")  
async def change_translation_method(method_name: str = Form(...)):  
    """  
    Change the translation method.  
      
    This endpoint allows the user to change the translation method used  
    by the model.  
      
    :param request: LoadMethodRequest  
        The request object containing the new translation method's name.  
    :rtype: BaseResponse  
        A response indicating the success or failure of changing the translation method.  
    """  
    # Convert the method name to lowercase  
    method_name = method_name.lower()  
      
    # Check if the method name is in the list of supported translation methods  
    if method_name not in TRANSLATE_METHODS:  
        logger.info(f" | Translate method '{method_name}' is not supported. Supported methods: {TRANSLATE_METHODS}. | ")  
        return BaseResponse(status=Status.FAILED, message=f" | Translate method '{method_name}' is not supported. Supported methods: {TRANSLATE_METHODS}. | ", data=None)
    
    # Change the translation method  
    model.change_translate_method(method_name)  
    active_method = getattr(model, "translate_method", None)

    if active_method == method_name:
        logger.info(f" | Translate method '{method_name}' has been changed successfully. | ")  
        return BaseResponse(message=f" | Translate method '{method_name}' has been changed successfully. | ", data=active_method)
    elif active_method is None:
        logger.info(f" | Translate method change failed. and all fallback methods are failed. Can't translate now. | ")
        return BaseResponse(status=Status.FAILED, message=f" | Translate method change failed. and all fallback methods are failed. Can't translate now. | ", data=active_method)
    else:
        logger.info(f" | Translate method change failed. Fallback to '{active_method}'. | ")
        return BaseResponse(message=f" | Translate method change failed. Fallback to '{active_method}'. | ", data=active_method)
        
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
        model.set_prompt(None)
        logger.info(f" | Prompt has been cleared. | ")
        return BaseResponse(message=" | Prompt has been cleared. | ", data=None)
    else:
        error = model.set_prompt(prompts.strip()) 
        if error is None:
            return BaseResponse(message=f" | Prompt has been set | ", data=None)
        else:
            return BaseResponse(message=f" | Prompt setting failed. Error: {error} | ", data=None)


@app.post("/translate")
async def translate(
    file: UploadFile = File(...),  
    meeting_id: str = Form(123),  
    device_id: str = Form(123),  
    audio_uid: str = Form(123),  
    times: datetime.datetime = Form(...),  
    o_lang: str = Form("zh"),  
    prev_text: str = Form(""),
    multi_strategy_transcription: int = Form(1), # 1~MAX_NUM_STRATEGIES others 1
    transcription_post_processing: bool = Form(True), # True/False
    use_translate: bool = Form(True) # True/False
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
    :param prev_text: str
        The previous text for context (will be overridden by global previous translation)
    :rtype: BaseResponse  
        A response containing the transcription results.  
    """  
    
    # Convert times to string format  
    times = str(times)  
    # Convert original language and target language to lowercase  
    o_lang = o_lang.lower()  
    multi_strategy_transcription = multi_strategy_transcription if 0 < multi_strategy_transcription <= MAX_NUM_STRATEGIES else 1
    
    # Create response data structure  
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
  
    # Save the uploaded audio file  
    file_name = times + ".wav"  
    audio_buffer = f"audio/{file_name}"  
    
    # Read file content once
    file_content = file.file.read()
    
    with open(audio_buffer, 'wb') as f:  
        f.write(file_content)
  
    # Check if the audio file exists  
    if not os.path.exists(audio_buffer):  
        return BaseResponse(status=Status.FAILED, message=" | The audio file does not exist, please check the audio path. | ", data=response_data)  
  
    # Check if the model has been loaded  
    if model.model_version is None:  
        return BaseResponse(status=Status.FAILED, message=" | model haven't been load successfully. may out of memory please check again | ", data=response_data)  
  
    # Check if the languages are in the supported language list  
    if o_lang not in LANGUAGE_LIST:  
        logger.info(f" | The original language is not in LANGUAGE_LIST: {LANGUAGE_LIST}. | ")  
        return BaseResponse(status=Status.FAILED, message=f" | The original language is not in LANGUAGE_LIST: {LANGUAGE_LIST}. | ", data=response_data)  
  
    try:  
        # Create a queue to hold the return value  
        result_queue = Queue()  
        # Create an event to signal stopping  
        stop_event = threading.Event()  
  
        # Create timing thread and inference thread  
        time_thread = threading.Thread(target=waiting_times, args=(stop_event, model, WAITING_TIME))  
        inference_thread = threading.Thread(target=audio_translate, args=(model, audio_buffer, result_queue, o_lang, stop_event, multi_strategy_transcription, transcription_post_processing, prev_text, use_translate))  
  
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
            ori_pred, result, rtf, transcription_time, translate_time, translate_method = result_queue.get()  
            response_data.transcription_text = ori_pred
            response_data.text = result  
            response_data.transcribe_time = transcription_time  
            response_data.translate_time = translate_time  
            zh_result = response_data.text.get("zh", "")
            en_result = response_data.text.get("en", "")
            de_result = response_data.text.get("de", "")
            
            logger.debug(response_data.model_dump_json())  
            logger.info(f" | device_id: {response_data.device_id} | audio_uid: {response_data.audio_uid} | source language: {o_lang} | translate_method: {translate_method} | time: {times} | ")  
            logger.info(f" | Transcription: {ori_pred} | ")
            if use_translate:
                logger.info(f" | {'#' * 75} | ")
                logger.info(f" | ZH: {zh_result} | ")  
                logger.info(f" | EN: {en_result} | ")  
                logger.info(f" | DE: {de_result} | ")  
                logger.info(f" | {'#' * 75} | ")
            logger.info(f" | RTF: {rtf} | total time: {transcription_time + translate_time:.2f} seconds. | transcribe {transcription_time:.2f} seconds. | translate {translate_time:.2f} seconds. | strategy: {multi_strategy_transcription} | ")  
            state = Status.OK
        else:  
            logger.info(f" | Translation has exceeded the upper limit time and has been stopped |")  
            ori_pred = zh_result = en_result = de_result = ""
            state = Status.FAILED

        return BaseResponse(status=state, message=f" | Transcription: {ori_pred} | ZH: {zh_result} | EN: {en_result} | DE: {de_result} | ", data=response_data)  
    except Exception as e:  
        logger.error(f" | Translation() error: {e} | ")  
        return BaseResponse(status=Status.FAILED, message=f" | Translation() error: {e} | ", data=response_data)  

@app.post("/text_translate")  
async def text_translate(  
    text: str = Form(...),
    language: str = Form(...)
):  
    """  
    Translate a text.  
  
    This endpoint receives text and its associated metadata, and performs translation on the text.  
  
    :param translate_request: TextData  
        The request containing the text to be translated.  
    :rtype: BaseResponse  
        A response containing the translation results.  
    """  
    language = language.lower()  
  
    # Create response data structure  
    response_data = TextTranslationResponse(  
        ori_lang="",
        text=DEFAULT_RESULT.copy(),
        translate_time=0.0
    )  
  
    try:  
        # Create a queue to hold the return value  
        result_queue = Queue()  
        # Create an event to signal stopping  
        stop_event = threading.Event()  
  
        # Create timing thread and inference thread  
        time_thread = threading.Thread(target=waiting_times, args=(stop_event, model, WAITING_TIME))  
        inference_thread = threading.Thread(target=texts_translate, args=(model, text, result_queue, language, stop_event))  
  
        # Start the threads  
        time_thread.start()  
        inference_thread.start()  
  
        # Wait for timing thread to complete and check if the inference thread is active to close  
        time_thread.join()  
        stop_thread(inference_thread)  
  
        # Get the result from the queue  
        if not result_queue.empty():  
            result, translate_time, translate_method = result_queue.get()  
            response_data.text = result  
            response_data.translate_time = translate_time
            zh_result = response_data.text.get("zh", "")
            en_result = response_data.text.get("en", "")
            de_result = response_data.text.get("de", "")
  
            logger.debug(response_data.model_dump_json())  
            logger.info(f" | original language: {language} | translate_method: {translate_method} |")  
            logger.info(f" | ZH: {zh_result} | ")  
            logger.info(f" | EN: {en_result} | ")  
            logger.info(f" | DE: {de_result} | ")  
            logger.info(f" | translate has been completed in {translate_time:.2f} seconds. |")  
            state = Status.OK
        else:
            logger.info(f" | translation has exceeded the upper limit time and has been stopped |")
            state = Status.FAILED

        return BaseResponse(status=state, message=f" | ZH: {zh_result} | EN: {en_result} | DE: {de_result} | ", data=response_data)
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
    use_translate: bool = Form(True) # True/False
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
    if model.model_version is None:  
        return BaseResponse(status=Status.FAILED, message=" | model haven't been load successful. may out of memory please check again | ", data=response_data)  
        
    # Check if the languages are in the supported language list  
    if o_lang not in LANGUAGE_LIST:  
        logger.info(f" | The original language is not in LANGUAGE_LIST: {LANGUAGE_LIST}. | ")  
        return BaseResponse(status=Status.FAILED, message=f" | The original language is not in LANGUAGE_LIST: {LANGUAGE_LIST}. | ", data=response_data)  
  
    other_information = {
        "prev_text": prev_text,
        "multi_strategy_transcription": multi_strategy_transcription,
        "transcription_post_processing": transcription_post_processing,
        "use_translate": use_translate
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
            file_name = f"{response_data.times}.wav"  
            audio_buffer = f"audio/{file_name}"  
            
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
                if waiting_list and not model.processing:  
                    response_data, other_information = waiting_list.pop(0)  
                    audio_buffer = f"audio/{response_data.times}.wav"  
                    o_lang = response_data.ori_lang  
      
                    try:  
                        # Create an event to signal stopping  
                        stop_event = threading.Event()  
                        # Create timing thread and inference thread  
                        time_thread = threading.Thread(target=waiting_times, args=(stop_event, model, WAITING_TIME))  
                        inference_thread = threading.Thread(target=audio_translate_sse, args=(model, audio_buffer, o_lang, other_information, stop_event))  
      
                        # Start the threads  
                        time_thread.start()  
                        inference_thread.start()  
      
                        # Wait for timing thread to complete and stop the inference thread if still running  
                        time_thread.join()  
                        stop_thread(inference_thread)  
                        
                        if os.path.exists(audio_buffer):
                            os.remove(audio_buffer)  
      
                        # Process all available results from the result queue
                        while not model.result_queue.empty():
                            ori_pred, result, rtf, transcription_time, translate_time, translate_method = model.result_queue.get()
                            response_data.transcription_text = ori_pred
                            response_data.text = result  
                            response_data.transcribe_time = transcription_time  
                            response_data.translate_time = translate_time  
                            zh_result = response_data.text.get("zh", "")
                            en_result = response_data.text.get("en", "")
                            de_result = response_data.text.get("de", "")
                            
                            logger.debug(response_data.model_dump_json())  
                            logger.info(f" | device_id: {response_data.device_id} | audio_uid: {response_data.audio_uid} | source language: {o_lang} | translate_method: {translate_method} |")  
                            logger.info(f" | Transcription: {ori_pred} | ")
                            if other_information["use_translate"]:
                                logger.info(f" | {'#' * 75} | ")
                                logger.info(f" | ZH: {zh_result} | ")  
                                logger.info(f" | EN: {en_result} | ")  
                                logger.info(f" | DE: {de_result} | ")  
                                logger.info(f" | {'#' * 75} | ")
                            else:
                                zh_result = en_result = de_result = ""
                                
                            logger.info(f" | RTF: {rtf} | total time: {transcription_time + translate_time:.2f} seconds. | transcribe {transcription_time:.2f} seconds. | translate {translate_time:.2f} seconds. | strategy: {other_information['multi_strategy_transcription']} | ")  

                            base_response = BaseResponse(  
                                status=Status.OK,  
                                message=f" | Transcription: {ori_pred} | ZH: {zh_result} | EN: {en_result} | DE: {de_result } | ",  
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
    The process of deleting old audio files  
    :param  
    ----------  
    None: The function does not take any parameters  
    :rtype  
    ----------  
    None: The function does not return any value  
    :logs  
    ----------  
    Deleted old files  
    """  
    current_time = time.time()  
    audio_dir = "./audio"  
    for filename in os.listdir(audio_dir):  
        if filename == "test.wav":  # Skip specific file  
            continue  
        file_path = os.path.join(audio_dir, filename)  
        if os.path.isfile(file_path):  
            file_creation_time = os.path.getctime(file_path)  
            # Delete files older than a day  
            if current_time - file_creation_time > 24 * 60 * 60:  
                os.remove(file_path)  
                logger.info(f" | Deleted old file: {file_path} | ")  
  
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
    
    
 