import time  
import torch
import logging  
import logging.handlers
import threading
import ctypes
import uuid

from queue import Queue  

# from api.translation.gemma_translate import Gemma4BTranslate  
from api.translation.ollama_translate import OllamaChat
from api.translation.gpt_translate import GptTranslate  

from lib.config.constant import OLLAMA_MODEL, DEFAULT_RESULT, TaskContext, SharedResources
  
  
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

class TranslateManager:  
    def __init__(self):  

        try:
            self.ollama_gemma_translator = OllamaChat(OLLAMA_MODEL['ollama-gemma'])  # Use correct key
        except Exception as e:
            logger.warning(f" | Failed to initialize Ollama translator: {e} | ")
            self.ollama_gemma_translator = None
        
        self.translation_method = "gpt-4o"
        self.fallback_translate = 'ollama-gemma'
        # Initialize 20 GPT-4o translators
        self.gpt_4o_translators = []
        for i in range(20):
            try:
                translator = GptTranslate(model_version=self.translation_method)
                if translator.test_gpt_model():
                    self.gpt_4o_translators.append(translator)
                    logger.debug(f" | {self.translation_method} translator #{i+1} initialized successfully | ")
                else:
                    logger.warning(f" | {self.translation_method} translator #{i+1} test failed | ")
            except Exception as e:
                logger.warning(f" | Failed to initialize {self.translation_method} translator #{i+1}: {e} | ")

        # Create translation_method dict with 10 translator instances
        translation_method_name = self.translation_method
        self.translation_method = {}
        for i, translator in enumerate(self.gpt_4o_translators):
            self.translation_method[f"{translation_method_name} #{i+1}"] = {"translator": translator, "busy": False}
        
        # Thread management
        self.lock = threading.Lock()  # Protect busy status and task_groups
        self.translator_available = threading.Condition(self.lock)  # Condition variable for translator availability
        self.task_groups = {}  # {task_group_id: {"threads": [], "stop_events": [], "target_langs": []}}
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  
        self.prompt = None
        self.prompt_name = None  # Store original prompt name for post-processing
        self.processor = None
        self.pipe = None  
        self.model_version = None  
        self.result_queue = Queue()  
        self.processing = False
    
    def _create_default_result(self, ori_pred, ori):
        """Helper function to create default translation result"""
        result = DEFAULT_RESULT.copy()
        result[ori] = ori_pred
        return result
    
    def _get_available_translator(self):
        """
        Find any available (not busy) translator. All translator instances have equal capability.
        
        IMPORTANT: Caller must already hold self.translator_available lock.
        
        Returns:
            tuple: (translator_name, translator_instance) or (None, None) if all busy
        """
        for method_name, method_info in self.translation_method.items():
            if method_info and method_info["translator"] is not None and not method_info["busy"]:
                method_info["busy"] = True
                return method_name, method_info["translator"]
        
        # All translators are busy
        logger.warning(f" | ##### All translators are busy ##### | ")
        return None, None
    
    def _release_translator(self, translator_name):
        """
        Release translator (set busy=False) and notify one waiting dispatcher.
        
        Args:
            translator_name: Name of the translator to release
        """
        with self.translator_available:
            if translator_name in self.translation_method:
                self.translation_method[translator_name]["busy"] = False
                # Notify one waiting dispatcher that a translator is available
                self.translator_available.notify()
    
    def _translate_single_task(self, task_ctx: TaskContext, shared: SharedResources):
        """
        Execute single translation task in a thread.
        
        Args:
            task_ctx: TaskContext with text, languages, translator info
            shared: SharedResources with thread-safe result storage and synchronization
        """
        logger.debug(f" | _translate_single_task started: {task_ctx.target_lang} by {task_ctx.translator_name} | ")
        start_time = time.time()
        try:
            if shared.stop_event.is_set():
                logger.warning(f" | Task {task_ctx.target_lang} stopped early (stop_event set) | ")
                return
            
            # Call translator based on type
            if task_ctx.translator_name.startswith("gpt"):
                logger.debug(f" | Calling GPT translator for {task_ctx.target_lang} | ")
                translated_result = task_ctx.translator.translate(
                    source_text=task_ctx.text, 
                    source_lang=task_ctx.source_lang, 
                    target_lang=task_ctx.target_lang, 
                    prev_text=task_ctx.prev_text
                )
                logger.debug(f" | GPT translation result for {task_ctx.target_lang}: {type(translated_result)} | ")
                
                # Check for 403 Forbidden
                if translated_result == "403_Forbidden":
                    with shared.result_lock:
                        # Only log and set fallback if not already triggered
                        if "_fallback_triggered" not in shared.result_dict:
                            shared.result_dict["_fallback_triggered"] = task_ctx.translator_name
                            shared.result_dict["_fallback_reason"] = "403_Forbidden"
                            logger.warning(f" | First 403 detected by {task_ctx.translator_name}, triggering fallback | ")
                        else:
                            logger.debug(f" | {task_ctx.translator_name} also got 403 (fallback already triggered) | ")
                    if hasattr(shared, 'fallback_event') and shared.fallback_event:
                        shared.fallback_event.set()  # Signal fallback immediately
                    return
                
                # Store result (thread-safe)
                with shared.result_lock:
                    if translated_result and task_ctx.target_lang in translated_result:
                        shared.result_dict[task_ctx.target_lang] = translated_result[task_ctx.target_lang]
                        logger.debug(f" | Stored result for {task_ctx.target_lang} | ")
                    else:
                        shared.result_dict[task_ctx.target_lang] = ""
                        logger.warning(f" | Empty result for {task_ctx.target_lang} | ")
                    
            elif task_ctx.translator_name.startswith("ollama"):
                translated_result = task_ctx.translator.translate(
                    source_text=task_ctx.text, 
                    source_lang=task_ctx.source_lang, 
                    target_lang=task_ctx.target_lang, 
                    prev_text=task_ctx.prev_text
                )
                
                # Store result (thread-safe)
                with shared.result_lock:
                    if translated_result and task_ctx.target_lang in translated_result:
                        shared.result_dict[task_ctx.target_lang] = translated_result[task_ctx.target_lang]
                    else:
                        shared.result_dict[task_ctx.target_lang] = ""
            
        except Exception as e:
            logger.error(f" | _translate_single_task error for {task_ctx.target_lang}: {e} | ")
            with shared.result_lock:
                shared.result_dict["_fallback_triggered"] = task_ctx.translator_name
                shared.result_dict["_fallback_reason"] = str(e)
            if hasattr(shared, 'fallback_event') and shared.fallback_event:
                shared.fallback_event.set()  # Signal fallback immediately
        finally:
            elapsed_time = time.time() - start_time
            # Record timing with language info if timing_dict is provided
            if shared.timing_dict is not None:
                with shared.result_lock:
                    if task_ctx.translator_name not in shared.timing_dict:
                        shared.timing_dict[task_ctx.translator_name] = []
                    # Store tuple of (time, language) for detailed tracking
                    shared.timing_dict[task_ctx.translator_name].append((elapsed_time, task_ctx.target_lang))
            
            # After completing task, actively check for more tasks in queue (event-driven)
            # Keep translator busy and reuse it if more tasks available
            if shared.task_queue is not None and shared.task_group_id is not None:
                picked_next = self._try_pick_next_task(task_ctx, shared)
                # Only release if no next task was picked up
                if not picked_next:
                    self._release_translator(task_ctx.translator_name)
            else:
                # No task queue context, release immediately
                self._release_translator(task_ctx.translator_name)
    
    def _try_pick_next_task(self, task_ctx: TaskContext, shared: SharedResources) -> bool:
        """
        Event-driven: After completing a task, actively check queue for next task.
        This replaces the passive polling approach.
        Translator remains busy during this check and will be released by caller if no task found.
        
        Args:
            task_ctx: Current task context (will be updated with next target_lang)
            shared: Shared resources including task queue
            
        Returns:
            bool: True if picked up next task, False if queue empty or fallback triggered
        """
        # Check if fallback was triggered
        with shared.result_lock:
            if "_fallback_triggered" in shared.result_dict:
                return False
        
        # Try to get next task from queue (non-blocking)
        try:
            next_target_lang = shared.task_queue.get_nowait()
        except:
            # Queue is empty, no more tasks
            return False
        
        # Translator is still marked as busy from previous task, continue using it
        logger.debug(f" | {task_ctx.translator_name} immediately picked up next task: {next_target_lang} | ")
        
        # Update task context with new target language
        next_task_ctx = TaskContext(
            text=task_ctx.text,
            source_lang=task_ctx.source_lang,
            target_lang=next_target_lang,
            prev_text=task_ctx.prev_text,
            translator_name=task_ctx.translator_name,
            translator=task_ctx.translator
        )
        
        # Execute the new task (recursive call, will continue chain until queue empty)
        # Note: translator remains busy, will be released when chain ends
        self._translate_single_task(next_task_ctx, shared)
        
        return True  # Successfully picked up next task
    
    def _stop_all_threads_in_group(self, task_group_id):
        """
        Stop all threads in a task group.
        
        Args:
            task_group_id: ID of the task group to stop
        """
        if task_group_id not in self.task_groups:
            return
        
        task_group = self.task_groups[task_group_id]
        
        # Set all stop events
        for stop_event in task_group["stop_events"]:
            stop_event.set()
        
        # Force terminate threads
        for thread in task_group["threads"]:
            if thread.is_alive():
                try:
                    thread_id = self._get_thread_id(thread)
                    if thread_id:
                        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), ctypes.py_object(SystemExit))
                        logger.info(f" | Force stopped thread {thread_id} | ")
                except Exception as e:
                    logger.warning(f" | Failed to force stop thread: {e} | ")
    
    def _get_thread_id(self, thread):
        """Get thread ID for force termination."""
        if not thread.is_alive():
            return None
        for tid, tobj in threading._active.items():
            if tobj is thread:
                return tid
        return None
    
    def _single_llm_translate_all(self, text, source_lang, target_langs, prev_text):
        """
        Use a single available LLM to translate all target languages at once.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_langs: List of target language codes
            prev_text: Previous context text
            
        Returns:
            tuple: (result_dict, translate_time, methods_used, timing_dict)
        """
        logger.debug(f" | _single_llm_translate_all called: source={source_lang}, targets={target_langs} | ")
        start_time = time.time()
        
        # Try to get an available translator
        translator_name, translator = self._get_available_translator()
        
        if translator_name is None or translator is None:
            logger.warning(f" | No translator available, using fallback | ")
            result_dict = self._fallback_translate_all(text, source_lang, target_langs, prev_text)
            elapsed_time = time.time() - start_time
            methods_used = f"{self.fallback_translate} (fallback)"
            timing_dict = {self.fallback_translate: [(elapsed_time, ", ".join(target_langs))]}
            
            return result_dict, elapsed_time, methods_used, timing_dict
        
        try:
            logger.debug(f" | Using {translator_name} for batch translation of {len(target_langs)} languages | ")
            
            # Call translator to translate all languages at once
            result_dict = translator.translate(
                source_text=text,
                source_lang=source_lang,
                target_lang=target_langs,
                prev_text=prev_text
            )
            
            logger.debug(f" | Translation result from {translator_name}: keys={list(result_dict.keys()) if isinstance(result_dict, dict) else result_dict} | ")
            
            # Add source language to result
            if source_lang not in result_dict:
                result_dict[source_lang] = text
            
            elapsed_time = time.time() - start_time
            methods_used = self.translation_method if isinstance(self.translation_method, str) else "GPT-4o"
            # Store timing with all target languages as a single batch
            timing_dict = {translator_name: [(elapsed_time, ", ".join(target_langs))]}
            
            # Check for 403 Forbidden or missing translations
            missing_langs = [lang for lang in target_langs if lang not in result_dict]
            if result_dict == "403_Forbidden" or missing_langs:
                logger.warning(f" | {translator_name} failed (missing: {missing_langs}), using fallback | ")
                self._release_translator(translator_name)
                result_dict = self._fallback_translate_all(text, source_lang, target_langs, prev_text)
                elapsed_time = time.time() - start_time
                methods_used = f"{self.fallback_translate} (fallback)"
                timing_dict = {self.fallback_translate: [(elapsed_time, ", ".join(target_langs))]}
            else:
                self._release_translator(translator_name)
            
            return result_dict, elapsed_time, methods_used, timing_dict
                
        except Exception as e:
            logger.error(f" | {translator_name} translation failed: {e}, using fallback | ")
            self._release_translator(translator_name)
            result_dict = self._fallback_translate_all(text, source_lang, target_langs, prev_text)
            elapsed_time = time.time() - start_time
            methods_used = f"{self.fallback_translate} (fallback)"
            timing_dict = {self.fallback_translate: [(elapsed_time, ", ".join(target_langs))]}
            
            return result_dict, elapsed_time, methods_used, timing_dict
    
    def _fallback_translate_all(self, text, source_lang, target_langs, prev_text):
        """
        Fallback: Use fallback translator to translate all target languages at once.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_langs: List of target language codes
            prev_text: Previous context text
            
        Returns:
            dict: Translation results {lang: translated_text}
        """
        logger.warning(f" | Fallback to {self.fallback_translate} for batch translation | ")
        
        if self.ollama_gemma_translator is None:
            logger.error(f" | {self.fallback_translate} translator not available for fallback | ")
            # Return default result with source text
            default_result = self._create_default_result(text, source_lang)
            return default_result
        
        try:
            # Use fallback translator to translate all target languages at once
            result = self.ollama_gemma_translator.translate(
                source_text=text, 
                source_lang=source_lang,
                target_lang=target_langs,  # Pass list of target languages
                prev_text=prev_text
            )
            
            # Verify result contains all required languages
            if result and all(lang in result for lang in target_langs):
                return result
            else:
                logger.error(f" | Fallback result missing some languages | ")
                # Fill in missing languages with empty strings
                default_result = self._create_default_result(text, source_lang)
                for lang in target_langs:
                    if lang in result:
                        default_result[lang] = result[lang]
                return default_result
            
        except Exception as e:
            logger.error(f" | Fallback translation failed: {e} | ")
            # Return default result with source text
            default_result = self._create_default_result(text, source_lang)
            return default_result
    
    def translate(self, text, source_lang, target_langs, prev_text="", multi_translate=True):
        """
        Public method to translate text to multiple target languages.
        
        Args:
            text: Text to translate
            source_lang: Source language code (e.g., 'zh', 'en')
            target_langs: Target language code(s), can be:
                - str: single language (e.g., 'en')
                - list: multiple languages (e.g., ['en', 'de', 'ja', 'ko'])
            prev_text: Previous text context (optional)
            multi_translate: If True, distribute tasks across multiple LLMs; 
                           If False, use single LLM to translate all languages at once
            
        Returns:
            tuple: (result_dict, translate_time, methods_used, timing_dict)
                - result_dict: {lang: translated_text} including source language
                - translate_time: Time taken for translation
                - methods_used: Translator name(s) used
                - timing_dict: {translator_name: [(time, lang), ...]} - detailed timing info
        """
        # Convert target_langs to list if it's a single string
        if isinstance(target_langs, str):
            target_langs = [target_langs]
        
        try:
            return self.assign_task(
                text=text,
                source_lang=source_lang,
                target_langs=target_langs,
                prev_text=prev_text,
                multi_translate=multi_translate
            )
        except Exception as e:
            logger.error(f" | translate() error: {e} | ")
            # Return default result with source text
            default_result = DEFAULT_RESULT.copy()
            default_result[source_lang] = text
            return default_result, 0.0, "FAILED", {}
    
    def assign_task(self, text, source_lang, target_langs, prev_text="", multi_translate=True):
        """
        Assign translation tasks for multiple target languages using task queue.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_langs: List of target language codes
            prev_text: Previous context text
            multi_translate: If True, distribute tasks across multiple LLMs; 
                           If False, use single LLM to translate all languages at once
            
        Returns:
            tuple: (result_dict, translate_time, methods_used, timing_dict)
                - result_dict: {lang: translated_text}
                - translate_time: Total time taken
                - methods_used: Translator name used (self.translation_method or f"{self.fallback_translate} (fallback)")
                - timing_dict: {translator_name: [time1, time2, ...]} - detailed timing info for each translator
        """
        logger.debug(f" | assign_task: source={source_lang}, targets={target_langs}, multi={multi_translate} | ")
        start_time = time.time()
        
        # If multi_translate=False, use single LLM for all languages at once
        if not multi_translate:
            return self._single_llm_translate_all(text, source_lang, target_langs, prev_text)
        task_group_id = str(uuid.uuid4())
        
        # Thread-safe shared data structures
        result_dict = {source_lang: text}  # Start with source text
        result_lock = threading.Lock()  # Protect result_dict access
        timing_dict = {}  # Store detailed timing info for each translator
        fallback_event = threading.Event()  # Event to signal fallback trigger (no polling needed)
        
        # Create task queue and put all target languages
        task_queue = Queue()
        for target_lang in target_langs:
            task_queue.put(target_lang)
        
        logger.debug(f" | Task queue initialized with {len(target_langs)} languages | ")
        
        # Register task group (initially empty, will be updated as threads start)
        with self.lock:
            self.task_groups[task_group_id] = {
                "threads": [],
                "stop_events": [],
                "target_langs": target_langs,
                "task_queue": task_queue
            }
        
        # Task dispatcher: wait for translators using Condition Variable, then event-driven
        def task_dispatcher():
            """Dispatcher waits for available translators using Condition Variable. Subsequent tasks are picked up event-driven."""
            initial_assigned = 0
            logger.debug(f" | Dispatcher started: queue_size={task_queue.qsize()} | ")
            while not task_queue.empty():
                # Check if fallback triggered at loop start
                with result_lock:
                    if "_fallback_triggered" in result_dict:
                        logger.info(f" | Dispatcher detected fallback at loop start | ")
                        return
                
                # Wait for available translator using Condition Variable
                with self.translator_available:
                    translator_name, translator = self._get_available_translator()
                    
                    # If no translator available, wait for notification
                    while not translator_name:
                        # Check fallback before waiting
                        with result_lock:
                            if "_fallback_triggered" in result_dict:
                                logger.info(f" | Dispatcher detected fallback before wait | ")
                                return
                        
                        # Wait for translator to be released (blocks until notified)
                        self.translator_available.wait()
                        
                        # After wake-up, immediately check fallback
                        with result_lock:
                            if "_fallback_triggered" in result_dict:
                                logger.info(f" | Dispatcher detected fallback after wake-up | ")
                                return
                        
                        # Try to get translator again
                        translator_name, translator = self._get_available_translator()
                # Lock released here
                
                if not translator_name:
                    # Should not happen, but safety check
                    continue
                
                # Final fallback check before dispatching
                with result_lock:
                    if "_fallback_triggered" in result_dict:
                        logger.info(f" | Dispatcher detected fallback before dispatch | ")
                        self._release_translator(translator_name)
                        return
                
                # Get a task from queue
                try:
                    target_lang = task_queue.get_nowait()
                except:
                    # Queue empty, all tasks assigned
                    self._release_translator(translator_name)
                    break
                
                # Start translation thread
                stop_event = threading.Event()
                
                # Create task context and shared resources
                task_ctx = TaskContext(
                    text=text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    prev_text=prev_text,
                    translator_name=translator_name,
                    translator=translator
                )
                
                shared = SharedResources(
                    result_dict=result_dict,
                    result_lock=result_lock,
                    stop_event=stop_event,
                    timing_dict=timing_dict,
                    task_queue=task_queue,
                    task_group_id=task_group_id,
                    fallback_event=fallback_event
                )
                
                thread = threading.Thread(
                    target=self._translate_single_task,
                    args=(task_ctx, shared)
                )
                thread.start()
                initial_assigned += 1
                logger.debug(f" | Dispatched {target_lang} to {translator_name} (thread started) | ")
                
                # Update task group (thread-safe)
                with self.lock:
                    if task_group_id in self.task_groups:
                        self.task_groups[task_group_id]["threads"].append(thread)
                        self.task_groups[task_group_id]["stop_events"].append(stop_event)
            
            logger.debug(f" | Dispatcher completed: {initial_assigned} tasks initially assigned | ")
        
        # Start dispatcher thread
        dispatcher_thread = threading.Thread(target=task_dispatcher, daemon=True)
        dispatcher_thread.start()
        
        # Wait for either fallback trigger or dispatcher completion (event-driven, no polling)
        while dispatcher_thread.is_alive():
            # Wait for fallback event with timeout (to periodically check dispatcher status)
            if fallback_event.wait(timeout=0.1):  # Event-driven: wake up immediately when fallback triggered
                logger.warning(f" | Fallback triggered immediately, stopping all tasks | ")
                
                # Clear task queue
                while not task_queue.empty():
                    try:
                        task_queue.get_nowait()
                    except:
                        break
                
                # Stop all running threads
                self._stop_all_threads_in_group(task_group_id)
                break
        
        # Wait for dispatcher to fully complete 
        dispatcher_thread.join()
        
        # Ensure all threads have finished
        with self.lock:
            active_threads = self.task_groups.get(task_group_id, {}).get("threads", [])
        
        logger.debug(f" | Waiting for {len(active_threads)} translation threads to complete | ")
        for thread in active_threads:
            if thread.is_alive():
                thread.join(timeout=5.0)  # Give enough time for translation to complete
        
        # Check if fallback was triggered (thread-safe)
        with result_lock:
            fallback_triggered = "_fallback_triggered" in result_dict
        
        if fallback_triggered:
            with result_lock:
                logger.warning(f" | Fallback triggered by {result_dict['_fallback_triggered']}: {result_dict.get('_fallback_reason', 'unknown')} | ")
                
                # Clear partial results (keep source language)
                for lang in target_langs:
                    if lang in result_dict:
                        del result_dict[lang]
            
            # Use fallback to translate all at once
            fallback_results = self._fallback_translate_all(text, source_lang, target_langs, prev_text)
            
            with result_lock:
                result_dict.update(fallback_results)
            
            methods_used = f"{self.fallback_translate} (fallback)"
        else:
            methods_used = self.translation_method if isinstance(self.translation_method, str) else "GPT-4o"
        
        # Clean up task group
        with self.lock:
            if task_group_id in self.task_groups:
                del self.task_groups[task_group_id]
        
        end_time = time.time()
        translate_time = end_time - start_time
        
        logger.debug(f" | assign_task complete: result_keys={list(result_dict.keys())}, time={translate_time:.2f}s, methods={methods_used}, timing_keys={list(timing_dict.keys())} | ")
        
        return result_dict, translate_time, methods_used, timing_dict
    
    def cleanup_translation_threads(self):
        """Clean up all active translation threads."""
        try:
            # Stop all active task groups
            with self.lock:
                task_group_ids = list(self.task_groups.keys())
            
            for task_group_id in task_group_ids:
                logger.info(f" | Cleaning up translation task group: {task_group_id} | ")
                self._stop_all_threads_in_group(task_group_id)
            
            logger.debug(" | All translation threads cleaned up | ")
        except Exception as e:
            logger.error(f" | Error cleaning up translation threads: {e} | ")

    def close(self):
        """
        Close any resources held by the translators.
        """
        if self.ollama_gemma_translator is not None:
            try:    
                self.ollama_gemma_translator.close()
                logger.info(f" | Closed Ollama Gemma translator successfully. | ")
            except Exception as e:
                logger.warning(f" | Failed to close Ollama Gemma translator: {e} | ")
        # if self.ollama_qwen_translator is not None:
        #     try:
        #         self.ollama_qwen_translator.close()
        #         logger.info(f" | Closed Ollama Qwen translator successfully. | ")
        #     except Exception as e:
        #         logger.warning(f" | Failed to close Ollama Qwen translator: {e} | ")

