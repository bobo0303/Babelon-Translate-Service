import logging  
import logging.handlers
import threading

from queue import Queue  

from api.transcraption.whisper_transformer import WhisperTransformer
from api.transcraption.whisper_cpp import WhisperCpp  

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
        
        # Pipeline queue system (event-driven, no polling)
        self.task_queue = Queue()  # Thread-safe queue for tasks
        self.task_results = {}  # {task_id: {'result': ..., 'event': Event()}}
        self.task_lock = threading.Lock()  # Lock for task_results dict
        self.stop_worker = False
        self.worker_thread = None
        self.processing = False
  
  
    def load_model(self, models_name):  
        """Load the specified model based on the model's name."""  
        try:  
            # Release old model resources
            if self.transcriber:
                self.transcriber.release_model()
            # Load new model based on the model name
            if models_name.startswith("whisper_transformer"):
                self.transcriber = self.transcribers["whisper_transformer"]
            elif models_name.startswith("whisper_cpp"):
                self.transcriber = self.transcribers["whisper_cpp"]
            else:
                raise ValueError(f" | Unsupported transcription method: {models_name} | ")
            model_path = getattr(self.models_path, models_name)  
            msg = self.transcriber.load_model(models_name, model_path)
            self.transcription_method = models_name
            logger.debug(f" | TranscribeManager: {msg} | ")
        except Exception as e:  
            # On error, release any partially loaded model
            if self.transcriber:
                self.transcriber.release_model()
            self.transcription_method = None
            self.transcriber = None
            logger.error(f" | TranscribeManager: load_model() models_name: '{models_name}' error: {e} | ")
            logger.error(f" | model has been released. Please use correct model name to reload the model before using | ")
            return e
        return None

    def set_prompt(self, prompt, language="zh"):  
        """  
        Set the prompt for the transcription model.  
  
        :param prompt: str  
            The name of the prompt to be used.  
        :rtype: None  
        """  
        language = language.lower()
        
        # Build prompt text based on language
        if prompt:
            if language == "zh":
                if not prompt.endswith((',', '.', '。', '!', '！', '?', '？')):
                    prompt += '。'
                prompt = f"這是我們的提示詞 {prompt} 讓我們繼續吧。"
            else:
                if not prompt.endswith((',', '.', '。', '!', '！', '?', '？')):
                    prompt += '.'
                prompt = f"These are our prompts {prompt} Let's continue."
        
        try:
            self.transcriber.set_prompt(prompt=prompt)
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
            logger.info(" | Pipeline worker thread started. | ")
    
    def stop_worker_thread(self):
        """Stop the background worker thread."""
        self.stop_worker = True
        self.task_queue.put(None)  # Send stop signal
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
            logger.info(" | Pipeline worker thread stopped. | ")
    
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
                    'processing': False
                }
        
        # 🚀 Queue optimization: Only add to queue if not cancelled
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
                    task_info['event'].set()  # Immediately wake up waiting endpoint
                    logger.info(f" | Task {task_id} (audio_uid: {audio_uid}, times: {existing_times}) cancelled due to newer request (times: {new_times}). [QUEUED] | ")
                
                # Case 2: Existing task is newer or equal → cancel new task
                elif existing_times >= new_times:
                    logger.info(f" | New task {new_task_id} (audio_uid: {audio_uid}, times: {new_times}) cancelled (older/equal than existing task {task_id}: {existing_times}). [NOT_QUEUED] | ")
                    return False
        
        return True
    
    def _queue_worker(self):
        """Background worker thread - pure event-driven, zero polling."""
        logger.info(" | Queue worker started and waiting for tasks... | ")
        
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
                        logger.info(f" | Task {task_id} skipped (already cleaned up). | ")
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
                
                # Execute transcription (set processing = True)
                self.processing = True
                ori_pred, transcription_time = self.transcribe(
                    audio_file, o_lang, multi_strategy_transcription, 
                    transcription_post_processing, prev_text
                )
                
                # Critical: Release transcribe_manager immediately after transcription
                self.processing = False
                logger.debug(f" | Task {task_id}: Transcription complete, transcribe_manager released. | ")
                
                # Store transcription result and notify
                with self.task_lock:
                    if task_id in self.task_results:
                        self.task_results[task_id]['result'] = (ori_pred, transcription_time)
                        self.task_results[task_id]['event'].set()  # Wake up waiting endpoint
                        logger.debug(f" | Task {task_id} completed. | ")
                    else:
                        logger.info(f" | Task {task_id} was cleaned up before completion could be stored. | ")
                
            except Exception as e:
                logger.error(f" | Worker error: {e} | ")
                self.processing = False  # Ensure processing flag is reset on error
                
                # Notify API thread and clean up to prevent permanent blocking
                with self.task_lock:
                    if task_id in self.task_results:
                        self.task_results[task_id]['result'] = None  # Error result
                        self.task_results[task_id]['event'].set()  # Wake up waiting API thread
                        logger.info(f" | Task {task_id} event set after error. | ")
        
        logger.info(" | Queue worker stopped. | ")

    def transcribe(self, audio_file_path, ori, multi_strategy_transcription=1, post_processing=True, prev_text=""):  
        """
        Docstring for transcribe
        
        :param self: Description
        :param audio_file_path: Description
        :param ori: Description
        :param multi_strategy_transcription: Description
        :param post_processing: Description
        :param prev_text: Description
        """
        return self.transcriber.transcribe(audio_file_path, 
                                           ori, 
                                           multi_strategy_transcription, 
                                           post_processing, 
                                           prev_text)

    
