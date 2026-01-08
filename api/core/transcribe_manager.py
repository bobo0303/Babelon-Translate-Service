import logging  
import logging.handlers
import threading

from queue import Queue  

from api.transcraption.whisper_transformer import WhisperTransformer
from api.transcraption.whisper_cpp import WhisperCpp 
from api.audio.audio_utils import audio_preprocess

from lib.config.constant import ModelPath
  
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

class TranscribeManager:  
    def __init__(self):  
        """Initialize the TranscribeManager class with default attributes."""
        self.models_path = ModelPath()
        self.result_queue = Queue()  
        self.transcribers = {
            "whisper_transformer": WhisperTransformer(self.result_queue),
            "whisper_cpp": WhisperCpp(self.result_queue)
        }
        self.transcription_method = None
        self.transcriber = None
        self.prompt = None
        
        # Pipeline queue system (event-driven, no polling)
        self.task_queue = Queue()  # Thread-safe queue for tasks
        self.task_results = {}  # {task_id: {'result': ..., 'event': Event()}}
        self.task_lock = threading.Lock()  # Lock for task_results dict
        self.stop_worker = False
        self.worker_thread = None
        self.processing = False
        self.current_task_id = None  # Track currently executing task
        self.task_complete_event = threading.Event()  # Event for task completion notification
  
  
    def load_model(self, model_name):  
        """Load the specified model based on the model's name."""  
        try:  
            # Release old model resources
            if self.transcriber:
                self.transcriber.release_model()
                
            # Load new model based on the model name
            if model_name.startswith("ggml"):
                self.transcriber = self.transcribers["whisper_cpp"]
            else:
                self.transcriber = self.transcribers["whisper_transformer"]

            model_path = getattr(self.models_path, model_name)  
            msg = self.transcriber.load_model(model_name, model_path)
            self.transcription_method = model_name
            logger.debug(f" | TranscribeManager: {msg} | ")
        except Exception as e:  
            # On error, release any partially loaded model
            if self.transcriber:
                self.transcriber.release_model()
            self.transcription_method = None
            self.transcriber = None
            logger.error(f" | TranscribeManager: load_model() model_name: '{model_name}' error: {e} | ")
            logger.error(f" | model has been released. Please use correct model name to reload the model before using | ")
            return e

        if self.prompt is not None:
            self.set_prompt(self.prompt) 
            
        return None

    def set_prompt(self, prompt):  
        """  
        Set the prompt for the transcription model.  
  
        :param prompt: str  
            The name of the prompt to be used.  
        :rtype: None  
        """  
        self.prompt = " ".join(prompt.strip().split())
        
        # Build prompt text based on language
        if self.prompt :
            if not self.prompt .endswith((',', '.', '。', '!', '！', '?', '？')):
                self.prompt  += '.'
            self.prompt  = f"These are our prompts {self.prompt } Let's continue."
        
        try:
            self.transcriber.set_prompt(prompt=self.prompt )
        except Exception as e:
            msg = f" | TranscribeManager: set_prompt() error: {e} | "
            logger.error(msg)
            return msg 
        
    def start_worker(self):
        """Start the background worker thread for pipeline processing."""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.stop_worker = False
            self.worker_thread = threading.Thread(target=self._queue_worker, daemon=True)
            self.worker_thread.start()
            logger.debug(" | Pipeline worker thread started. | ")
    
    def stop_worker_thread(self):
        """Stop the background worker thread."""
        self.stop_worker = True
        self.task_queue.put(None)  # Send stop signal
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
            logger.info(" | Pipeline worker thread stopped. | ")
    
    def force_terminate_current_task(self, audio_uid):
        """Force terminate the currently executing task if it matches the audio_uid.
        
        Strategy: Try graceful abort first, then force kill if needed.
        """
        with self.task_lock:
            if self.current_task_id is None:
                logger.debug(f" | No task currently executing to terminate for audio_uid: {audio_uid}. | ")
                return False
            
            task_info = self.task_results.get(self.current_task_id)
            if task_info and task_info.get('audio_uid') == audio_uid:
                task_id = self.current_task_id
                logger.warning(f" | Force terminating task {task_id} (audio_uid: {audio_uid}). | ")
                
                # Mark task as cancelled
                task_info['cancelled'] = True
                task_info['result'] = None
                task_info['event'].set()
                
                # Step 1: Try graceful abort via C++ callback (fast)
                if self.transcriber:
                    logger.debug(f" | Requesting graceful abort via C++ callback... | ")
                    self.transcriber.request_abort()
                
                # Clear the completion event before waiting
                self.task_complete_event.clear()
        
        # Step 2: Wait for task completion event (no polling, instant notification)
        completed = self.task_complete_event.wait(timeout=0.5)  # Wait max 500ms
        
        if completed:
            logger.info(f" | Task gracefully aborted (C++ responded to abort). | ")
            return True
        
        # Step 3: Graceful abort timeout, force kill thread (slow backup)
        with self.task_lock:
            # Re-check if task already completed by another force_terminate call
            if self.current_task_id is None:
                logger.debug(f" | Task already terminated by another force_terminate call. | ")
                return True
            
            logger.warning(f" | Graceful abort timeout after 500ms, force killing worker thread... | ")
            
            if self.worker_thread and self.worker_thread.is_alive():
                self.stop_worker = True
                # Force thread termination (dangerous but necessary)
                try:
                    import ctypes
                    thread_id = self.worker_thread.ident
                    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                        ctypes.c_long(thread_id), 
                        ctypes.py_object(SystemExit)
                    )
                    if res == 0:
                        logger.error(f" | Failed to terminate thread {thread_id}. | ")
                    elif res > 1:
                        # If it returns more than 1, revert the exception
                        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, None)
                        logger.error(f" | Exception raised in multiple threads. | ")
                    else:
                        logger.info(f" | Worker thread {thread_id} terminated. | ")
                except Exception as e:
                    logger.error(f" | Error terminating thread: {e} | ")
                
                # Wait a bit for thread to die
                self.worker_thread.join(timeout=0.3)
            
            # Clear state
            self.processing = False
            self.current_task_id = None
            
            # Restart worker
            self.start_worker()
            logger.info(f" | Worker thread restarted after force termination. | ")
            return True
        
        return False
    
    def add_task(self, task_id, audio_file, o_lang, multi_strategy_transcription, 
                 transcription_post_processing, prev_text, audio_uid, times):
        """Add a task to the queue and return an Event for blocking wait."""
        task_event = threading.Event()
        should_add = True
        
        # Check and cancel duplicate audio_uid in pending tasks
        with self.task_lock:
            should_add = self._check_and_cancel_duplicate(audio_uid, task_id, times)
            
            if should_add:
                self.task_results[task_id] = {
                    'result': None,
                    'event': task_event,
                    'audio_uid': audio_uid,
                    'times': times,
                    'cancelled': False,
                    'processing': False,
                    'task_id': task_id,
                }
        
        # Queue optimization: Only add to queue if not cancelled
        if should_add:
            task = (task_id, audio_file, o_lang, multi_strategy_transcription,
                    transcription_post_processing, prev_text)
            self.task_queue.put(task)
            logger.debug(f" | Task {task_id} (audio_uid: {audio_uid}, times: {times}) added to queue. | ")
        else:
            # Task cancelled before queuing, immediately wake up API thread
            task_event.set()
            logger.debug(f" | Task {task_id} event set without queuing (cancelled before queue). | ")
        
        return task_event
    
    def _check_and_cancel_duplicate(self, audio_uid, new_task_id, new_times):
        """Check for duplicate audio_uid in task_results and cancel older ones, keep only the latest.
        
        Returns:
            bool: True if new task should be added, False if new task should be cancelled
        """
        for task_id, task_info in self.task_results.items():
            if (task_id != new_task_id and 
                task_info.get('audio_uid') == audio_uid and 
                task_info['result'] is None and
                not task_info.get('processing', False)):  # Only cancel queued tasks, not executing ones
                
                existing_times = task_info.get('times', '')
                
                # Case 1: Existing task is older → cancel it, allow new task
                if existing_times < new_times:
                    task_info['cancelled'] = True
                    task_info['cancelled_by_times'] = new_times  # Record who cancelled it
                    task_info['event'].set()  # Immediately wake up waiting endpoint
                    logger.debug(f" | Task {task_id} (audio_uid: {audio_uid}, times: {existing_times}) cancelled due to newer request (times: {new_times}). [QUEUED] | ")
                
                # Case 2: Existing task is newer or equal → cancel new task
                elif existing_times >= new_times:
                    logger.info(f" | New task {new_task_id} (audio_uid: {audio_uid}, times: {new_times}) cancelled (older/equal than existing task {task_id}: {existing_times}). [NOT_QUEUED] | ")
                    return False
        
        return True
    
    def _queue_worker(self):
        """Background worker thread - pure event-driven, zero polling."""
        logger.debug(" | Queue worker started and waiting for tasks... | ")
        
        while True:
            try:
                # Block indefinitely until task arrives (pure event-driven)
                task = self.task_queue.get(block=True)
                
                if task is None:  # Stop signal
                    break
                
                # Unpack task
                task_id, audio_file, o_lang, multi_strategy_transcription, \
                    transcription_post_processing, prev_text = task
                
                # Check if task was cancelled or already cleaned up
                with self.task_lock:
                    if task_id not in self.task_results:
                        logger.debug(f" | Task {task_id} skipped (already cleaned up). | ")
                        continue
                    if self.task_results[task_id].get('cancelled', False):
                        logger.info(f" | Task {task_id} skipped (cancelled). | ")
                        self.task_results[task_id]['event'].set()
                        continue
                
                logger.debug(f" | Worker processing task {task_id}... | ")
                
                # Mark task as processing to prevent cancellation
                with self.task_lock:
                    if task_id not in self.task_results:
                        logger.info(f" | Task {task_id} disappeared before processing. | ")
                        continue
                    self.task_results[task_id]['processing'] = True
                    self.current_task_id = task_id  # Track current executing task
                
                # Execute transcription (set processing = True)
                self.processing = True
                ori_pred, transcription_time, audio_length = self.transcribe(
                    audio_file, o_lang, multi_strategy_transcription, 
                    transcription_post_processing, prev_text
                )
                
                # Critical: Release transcribe_manager immediately after transcription
                self.processing = False
                logger.debug(f" | Task {task_id}: Transcription complete, transcribe_manager released. | ")
                
                # Store transcription result and notify
                with self.task_lock:
                    self.current_task_id = None  # Clear current task
                    if task_id in self.task_results:
                        self.task_results[task_id]['result'] = (ori_pred, transcription_time, audio_length)
                        self.task_results[task_id]['event'].set()  # Wake up waiting endpoint
                        logger.debug(f" | Task {task_id} completed. | ")
                    else:
                        logger.info(f" | Task {task_id} was cleaned up before completion could be stored. | ")
                    
                    # Notify force_terminate that task is complete
                    self.task_complete_event.set()
                
            except Exception as e:
                logger.error(f" | Worker error: {e} | ")
                self.processing = False  # Ensure processing flag is reset on error
                
                # Notify API thread and clean up to prevent permanent blocking
                with self.task_lock:
                    self.current_task_id = None  # Clear current task on error
                    if task_id in self.task_results:
                        self.task_results[task_id]['result'] = None  # Error result
                        self.task_results[task_id]['event'].set()  # Wake up waiting API thread
                        logger.info(f" | Task {task_id} event set after error. | ")
                    
                    # Notify force_terminate that task is complete (even if error)
                    self.task_complete_event.set()
        
        logger.info(" | Queue worker stopped. | ")

    def transcribe(self, audio_path, ori, multi_strategy_transcription=1, post_processing=True, prev_text=""):  
        """
        Docstring for transcribe
        
        :param self: Description
        :param audio_file_path: Description
        :param ori: Description
        :param multi_strategy_transcription: Description
        :param post_processing: Description
        :param prev_text: Description
        """
        
        audio, audio_length = audio_preprocess(audio_path, padding_duration=0.05)
                
        # Process previous text context
        if prev_text.strip() != "" and len(prev_text.replace('.', '').replace('。', '').replace(',', '').replace('，', '').strip()) >= 1:
            if not prev_text.endswith(('.', '。', '!', '！', '?', '？')):
                prev_text += '。' 
        else:
            prev_text = ""
                
        return self.transcriber.transcribe(audio_path,
                                           audio, 
                                           audio_length,
                                           ori, 
                                           multi_strategy_transcription, 
                                           post_processing, 
                                           prev_text)

    
