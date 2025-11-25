import gc  
import time  
import torch
import librosa
import logging  
import logging.handlers
import numpy as np
import threading
import ctypes
import uuid

from transformers import pipeline, AutoProcessor
from queue import Queue  

# from api.translation.gemma_translate import Gemma4BTranslate  
from api.translation.ollama_translate import OllamaChat
from api.translation.gpt_translate import GptTranslate  

from lib.config.constant import ModelPath, LANGUAGE_LIST, OLLAMA_MODEL, SILENCE_PADDING, DEFAULT_RESULT, MAX_NUM_STRATEGIES, FALLBACK_METHOD, get_system_prompt_dynamic_language
  
  
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
        

        # Initialize 10 GPT-4o translators
        self.gpt_4o_translators = []
        for i in range(10):
            try:
                translator = GptTranslate(model_version='gpt-4o')
                if translator.test_gpt_model():
                    self.gpt_4o_translators.append(translator)
                    logger.debug(f" | GPT-4o translator #{i+1} initialized successfully | ")
                else:
                    logger.warning(f" | GPT-4o translator #{i+1} test failed | ")
            except Exception as e:
                logger.warning(f" | Failed to initialize GPT-4o translator #{i+1}: {e} | ")

        # Create translation_method with 10 GPT-4o instances
        self.translation_method = {}
        for i, translator in enumerate(self.gpt_4o_translators):
            self.translation_method[f"gpt-4o-{i+1}"] = {"translator": translator, "busy": False}
        
        # self.translation_priority = ["gpt-4o", "gpt-4.1", "gpt-4.1-mini", "ollama-gemma", "ollama-qwen"]  # ,"gemma4b"
        self.translation_priority = [f"gpt-4o-{i+1}" for i in range(len(self.gpt_4o_translators))]
        
        # Thread management
        self.lock = threading.Lock()  # Protect busy status and task_groups
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
        Find the first available (not busy) translator based on priority.
        
        Returns:
            tuple: (translator_name, translator_instance) or (None, None) if all busy
        """
        with self.lock:
            for method_name in self.translation_priority:
                method_info = self.translation_method.get(method_name)
                if method_info and method_info["translator"] is not None and not method_info["busy"]:
                    method_info["busy"] = True
                    return method_name, method_info["translator"]
        return None, None
    
    def _release_translator(self, translator_name):
        """
        Release translator (set busy=False).
        
        Args:
            translator_name: Name of the translator to release
        """
        with self.lock:
            if translator_name in self.translation_method:
                self.translation_method[translator_name]["busy"] = False
    
    def _translate_single_task(self, text, source_lang, target_lang, prev_text, translator_name, translator, result_dict, result_lock, stop_event, timing_dict=None):
        """
        Execute single translation task in a thread.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            prev_text: Previous context text
            translator_name: Name of translator being used
            translator: Translator instance
            result_dict: Shared dict to store results {target_lang: translated_text}
            result_lock: Lock to protect result_dict access
            stop_event: Event to signal stop
            timing_dict: Optional dict to store timing info {translator_name: time}
        """
        start_time = time.time()
        try:
            if stop_event.is_set():
                return
            
            # Call translator based on type
            if translator_name.startswith("gpt"):
                translated_result = translator.translate(source_text=text, source_lang=source_lang, target_lang=target_lang, prev_text=prev_text)
                
                # Check for 403 Forbidden
                if translated_result == "403_Forbidden":
                    with result_lock:
                        result_dict["_fallback_triggered"] = translator_name
                        result_dict["_fallback_reason"] = "403_Forbidden"
                    return
                
                # Store result (thread-safe)
                with result_lock:
                    if translated_result and target_lang in translated_result:
                        result_dict[target_lang] = translated_result[target_lang]
                    else:
                        result_dict[target_lang] = ""
                    
            elif translator_name.startswith("ollama"):
                translated_result = translator.translate(source_text=text, source_lang=source_lang, target_lang=target_lang, prev_text=prev_text)
                
                # Store result (thread-safe)
                with result_lock:
                    if translated_result and target_lang in translated_result:
                        result_dict[target_lang] = translated_result[target_lang]
                    else:
                        result_dict[target_lang] = ""
            
        except Exception as e:
            logger.error(f" | _translate_single_task error for {target_lang}: {e} | ")
            with result_lock:
                result_dict["_fallback_triggered"] = translator_name
                result_dict["_fallback_reason"] = str(e)
        finally:
            elapsed_time = time.time() - start_time
            # Record timing if timing_dict is provided
            if timing_dict is not None:
                with result_lock:
                    if translator_name not in timing_dict:
                        timing_dict[translator_name] = []
                    timing_dict[translator_name].append(elapsed_time)
            self._release_translator(translator_name)
    
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
    
    def _single_llm_translate_all(self, text, source_lang, target_langs, prev_text, return_timing=False):
        """
        Use a single available LLM to translate all target languages at once.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_langs: List of target language codes
            prev_text: Previous context text
            return_timing: Whether to return detailed timing info
            
        Returns:
            tuple: (result_dict, translate_time, methods_used) or with timing_dict if return_timing=True
        """
        start_time = time.time()
        
        # Try to get an available translator
        translator_name, translator = self._get_available_translator()
        
        if translator_name is None or translator is None:
            logger.warning(f" | No translator available, using fallback | ")
            result_dict = self._fallback_translate_all(text, source_lang, target_langs, prev_text)
            elapsed_time = time.time() - start_time
            methods_used = ["ollama-gemma (fallback)"]
            timing_dict = {"ollama-gemma": [elapsed_time]} if return_timing else None
            
            if return_timing:
                return result_dict, elapsed_time, methods_used, timing_dict
            else:
                return result_dict, elapsed_time, methods_used
        
        try:
            logger.info(f" | Using {translator_name} for batch translation of {len(target_langs)} languages | ")
            
            # Call translator to translate all languages at once
            result_dict = translator.translate(
                source_text=text,
                source_lang=source_lang,
                target_lang=target_langs,
                prev_text=prev_text
            )
            
            # Add source language to result
            if source_lang not in result_dict:
                result_dict[source_lang] = text
            
            elapsed_time = time.time() - start_time
            methods_used = [translator_name]
            timing_dict = {translator_name: [elapsed_time]} if return_timing else None
            
            # Check for 403 Forbidden or missing translations
            if result_dict == "403_Forbidden" or not all(lang in result_dict for lang in target_langs):
                logger.warning(f" | {translator_name} failed, using fallback | ")
                self._release_translator(translator_name)
                result_dict = self._fallback_translate_all(text, source_lang, target_langs, prev_text)
                elapsed_time = time.time() - start_time
                methods_used = ["ollama-gemma (fallback)"]
                timing_dict = {"ollama-gemma": [elapsed_time]} if return_timing else None
            else:
                self._release_translator(translator_name)
            
            if return_timing:
                return result_dict, elapsed_time, methods_used, timing_dict
            else:
                return result_dict, elapsed_time, methods_used
                
        except Exception as e:
            logger.error(f" | {translator_name} translation failed: {e}, using fallback | ")
            self._release_translator(translator_name)
            result_dict = self._fallback_translate_all(text, source_lang, target_langs, prev_text)
            elapsed_time = time.time() - start_time
            methods_used = ["ollama-gemma (fallback)"]
            timing_dict = {"ollama-gemma": [elapsed_time]} if return_timing else None
            
            if return_timing:
                return result_dict, elapsed_time, methods_used, timing_dict
            else:
                return result_dict, elapsed_time, methods_used
    
    def _fallback_translate_all(self, text, source_lang, target_langs, prev_text):
        """
        Fallback: Use ollama-gemma to translate all target languages at once.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_langs: List of target language codes
            prev_text: Previous context text
            
        Returns:
            dict: Translation results {lang: translated_text}
        """
        logger.warning(f" | Fallback to ollama-gemma for batch translation | ")
        
        if self.ollama_gemma_translator is None:
            logger.error(f" | ollama-gemma translator not available for fallback | ")
            # Return default result with source text
            default_result = self._create_default_result(text, source_lang)
            return default_result
        
        try:
            # Use ollama-gemma to translate all target languages at once
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
    
    def assign_task(self, text, source_lang, target_langs, prev_text="", return_timing=False, multi_translate=True):
        """
        Assign translation tasks for multiple target languages using task queue.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_langs: List of target language codes
            prev_text: Previous context text
            return_timing: Whether to return detailed timing info for each translator
            multi_translate: If True, distribute tasks across multiple LLMs; 
                           If False, use single LLM to translate all languages at once
            
        Returns:
            tuple: (result_dict, translate_time, methods_used) or 
                   (result_dict, translate_time, methods_used, timing_dict) if return_timing=True
                - result_dict: {lang: translated_text}
                - translate_time: Total time taken
                - methods_used: List of translator names used
                - timing_dict: {translator_name: [time1, time2, ...]} (if return_timing=True)
        """
        start_time = time.time()
        
        # If multi_translate=False, use single LLM for all languages at once
        if not multi_translate:
            return self._single_llm_translate_all(text, source_lang, target_langs, prev_text, return_timing)
        task_group_id = str(uuid.uuid4())
        
        # Thread-safe shared data structures
        result_dict = {source_lang: text}  # Start with source text
        result_lock = threading.Lock()  # Protect result_dict access
        methods_used_list = []
        methods_lock = threading.Lock()  # Protect methods_used_list access
        timing_dict = {} if return_timing else None  # Store timing info if requested
        
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
        
        # Task dispatcher: continuously checks queue and assigns tasks
        def task_dispatcher():
            """Background thread that assigns tasks from queue to available translators"""
            while True:
                # Check if fallback triggered (stop dispatching) - thread-safe read
                with result_lock:
                    if "_fallback_triggered" in result_dict:
                        logger.info(f" | Dispatcher detected fallback, stopping | ")
                        break
                
                # Check if queue is empty and all threads finished
                if task_queue.empty():
                    with self.lock:
                        active_threads = self.task_groups.get(task_group_id, {}).get("threads", [])
                    if not active_threads or not any(t.is_alive() for t in active_threads):
                        logger.info(f" | All tasks completed, dispatcher exiting | ")
                        break
                
                # Try to get a task from queue (blocking with timeout)
                try:
                    target_lang = task_queue.get(timeout=0.1)  # 100ms timeout, let Queue handle waiting
                except:
                    # Queue empty or timeout, continue to check conditions
                    continue
                
                # Try to get available translator
                translator_name, translator = self._get_available_translator()
                
                if translator_name and translator:
                    # Start translation thread
                    stop_event = threading.Event()
                    
                    thread = threading.Thread(
                        target=self._translate_single_task,
                        args=(text, source_lang, target_lang, prev_text, translator_name, translator, result_dict, result_lock, stop_event, timing_dict)
                    )
                    thread.start()
                    logger.debug(f" | Dispatched {target_lang} to {translator_name} | ")
                    
                    # Update task group and methods_used_list (thread-safe)
                    with self.lock:
                        if task_group_id in self.task_groups:
                            self.task_groups[task_group_id]["threads"].append(thread)
                            self.task_groups[task_group_id]["stop_events"].append(stop_event)
                    
                    with methods_lock:
                        methods_used_list.append(translator_name)
                else:
                    # No translator available, put task back to queue
                    task_queue.put(target_lang)
                    time.sleep(0.02)  # 20ms - balanced retry interval
        
        # Start dispatcher thread
        dispatcher_thread = threading.Thread(target=task_dispatcher, daemon=True)
        dispatcher_thread.start()
        
        # Monitor for fallback trigger
        while dispatcher_thread.is_alive():
            # Check if fallback was triggered (immediate response) - thread-safe read
            with result_lock:
                fallback_triggered = "_fallback_triggered" in result_dict
            
            if fallback_triggered:
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
            
            time.sleep(0.025)  # 25ms - balanced between responsiveness and CPU usage
        
        # Wait for dispatcher to finish (shorter timeout for faster completion)
        dispatcher_thread.join(timeout=0.5)
        
        # Ensure all threads have finished
        with self.lock:
            active_threads = self.task_groups.get(task_group_id, {}).get("threads", [])
        
        for thread in active_threads:
            if thread.is_alive():
                thread.join(timeout=0.2)  # Reduced timeout for faster completion
        
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
            
            methods_used = ["ollama-gemma (fallback)"]
        else:
            with methods_lock:
                methods_used = list(set(methods_used_list))  # Remove duplicates
        
        # Clean up task group
        with self.lock:
            if task_group_id in self.task_groups:
                del self.task_groups[task_group_id]
        
        end_time = time.time()
        translate_time = end_time - start_time
        
        if return_timing:
            return result_dict, translate_time, methods_used, timing_dict
        else:
            return result_dict, translate_time, methods_used
    

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


if __name__ == "__main__":
    """測試 TranslateManager 的多語言翻譯功能"""
    print("\n" + "="*80)
    print("開始測試 TranslateManager 多語言並行翻譯功能")
    print("="*80 + "\n")
    
    # 初始化 TranslateManager
    manager = TranslateManager()
    
    
    # 測試案例 4: 三句文本同時翻譯 - 批次 vs 並行派發任務
    print("\n" + "="*80)
    print("測試案例 4: 三句文本同時翻譯 - 批次 vs 並行派發")
    print("="*80 + "\n")
    
    source_lang = "zh"
    test_texts = [
        "這個是我們共同帶領團隊要去做的事。再來,這個, 拍劇昨天那個 Ben 跟我說的,做業務呢,每年的一月",
        "下面叫做基本,我們建立所有的員工全球員工的永續素養,右邊要建立數位化的 ESG,為什麼要數位化 ESG,剛剛大家看到我們所有的平板",
        "OS 的 pool 往上去層, 希望這樣來優化, 那我們有 define 幾個,就是說我們希望不要太多太,所以像 display 的,所以像 display 的,所以像 display 的,所以像 display plus 或 M"
    ]
    target_langs = ["en", "de", "ja", "ko"]
    num_tests = 10
    
    print(f"📝 測試文本數量: {len(test_texts)} 句")
    for idx, text in enumerate(test_texts, 1):
        print(f"  句子 {idx}: {text[:50]}{'...' if len(text) > 50 else ''}")
    print(f"🎯 目標語言: {target_langs}")
    print(f"🔄 測試次數: {num_tests} 次")
    print("\n" + "-"*80)
    
    # 儲存測試結果
    batch_all_times = []  # 批次翻譯所有文本的時間
    parallel_dispatch_times = []  # 並行派發所有任務的時間
    all_dispatch_timing_data = []  # 儲存每次測試的各 LLM 耗時
    
    # 執行 10 次測試
    for i in range(num_tests):
        print(f"\n第 {i+1}/{num_tests} 次測試:")
        
        # 方法 1: 批次翻譯 - 對每句文本依序調用 GPT 批次翻譯所有語言
        if manager.gpt_4o_translator is not None:
            try:
                start_batch = time.time()
                for text in test_texts:
                    gpt_result = manager.gpt_4o_translator.translate(
                        source_text=text,
                        source_lang=source_lang,
                        target_lang=target_langs,
                        prev_text=""
                    )
                batch_all_time = time.time() - start_batch
                batch_all_times.append(batch_all_time)
                print(f"  方法1 批次翻譯 (依序處理3句): {batch_all_time:.3f} 秒")
            except Exception as e:
                print(f"  ❌ 批次翻譯失敗: {e}")
                batch_all_times.append(None)
        else:
            batch_all_times.append(None)
        
        # 方法 2: 並行派發 - 對每句文本同時派發翻譯任務 (3句 x 4語言 = 12個任務)
        try:
            start_dispatch = time.time()
            dispatch_threads = []
            all_results = []
            all_timings = []
            
            def translate_one_text(text_content, text_idx):
                """在線程中翻譯一句文本"""
                result, trans_time, methods, timing = manager.assign_task(
                    text=text_content,
                    source_lang=source_lang,
                    target_langs=target_langs,
                    prev_text="",
                    return_timing=True
                )
                all_results.append((text_idx, result, trans_time, methods, timing))
            
            # 為每句文本創建一個線程
            for idx, text in enumerate(test_texts):
                thread = threading.Thread(target=translate_one_text, args=(text, idx))
                thread.start()
                dispatch_threads.append(thread)
            
            # 等待所有線程完成
            for thread in dispatch_threads:
                thread.join()
            
            parallel_dispatch_time = time.time() - start_dispatch
            parallel_dispatch_times.append(parallel_dispatch_time)
            
            # 合併所有 timing 數據
            combined_timing = {}
            for _, _, _, _, timing in all_results:
                for llm, times in timing.items():
                    if llm not in combined_timing:
                        combined_timing[llm] = []
                    combined_timing[llm].extend(times)
            all_dispatch_timing_data.append(combined_timing)
            
            # 顯示各 LLM 耗時
            llm_times_str = ", ".join([f"{llm}:{sum(times):.3f}s" for llm, times in combined_timing.items()])
            print(f"  方法2 並行派發 (同時處理3句): {parallel_dispatch_time:.3f} 秒 ({llm_times_str})")
        except Exception as e:
            print(f"  ❌ 並行派發失敗: {e}")
            parallel_dispatch_times.append(None)
            all_dispatch_timing_data.append({})
    
    # 計算統計數據
    print("\n" + "="*80)
    print("📊 測試統計結果")
    print("="*80 + "\n")
    
    # 方法1統計
    valid_batch_times = [t for t in batch_all_times if t is not None]
    if valid_batch_times:
        batch_avg = sum(valid_batch_times) / len(valid_batch_times)
        batch_min = min(valid_batch_times)
        batch_max = max(valid_batch_times)
        print(f"方法 1️⃣ 批次翻譯 (GPT-4o 依序處理 3句x4語言):")
        print(f"  平均耗時: {batch_avg:.3f} 秒")
        print(f"  最快: {batch_min:.3f} 秒")
        print(f"  最慢: {batch_max:.3f} 秒")
        print(f"  成功次數: {len(valid_batch_times)}/{num_tests}")
    else:
        print("方法 1️⃣ 批次翻譯: 全部失敗")
        batch_avg = None
    
    print()
    
    # 方法2統計
    valid_dispatch_times = [t for t in parallel_dispatch_times if t is not None]
    if valid_dispatch_times:
        dispatch_avg = sum(valid_dispatch_times) / len(valid_dispatch_times)
        dispatch_min = min(valid_dispatch_times)
        dispatch_max = max(valid_dispatch_times)
        print(f"方法 2️⃣ 並行派發 (TranslateManager 同時處理 3句x4語言=12任務):")
        print(f"  平均耗時: {dispatch_avg:.3f} 秒")
        print(f"  最快: {dispatch_min:.3f} 秒")
        print(f"  最慢: {dispatch_max:.3f} 秒")
        print(f"  成功次數: {len(valid_dispatch_times)}/{num_tests}")
    else:
        print("方法 2️⃣ 並行派發: 全部失敗")
        dispatch_avg = None
    
    # 計算調度開銷統計
    dispatch_overheads = []
    for i, timing_dict in enumerate(all_dispatch_timing_data):
        if parallel_dispatch_times[i] is not None and timing_dict:
            max_llm_time = max(sum(times) for times in timing_dict.values())
            overhead = parallel_dispatch_times[i] - max_llm_time
            dispatch_overheads.append(overhead)
    
    if dispatch_overheads:
        overhead_avg = sum(dispatch_overheads) / len(dispatch_overheads)
        overhead_min = min(dispatch_overheads)
        overhead_max = max(dispatch_overheads)
        print(f"\n方法 2️⃣ 調度開銷分析:")
        print(f"  平均開銷: {overhead_avg:.3f} 秒 ({overhead_avg/dispatch_avg*100:.1f}% of 並行時間)")
        print(f"  最小開銷: {overhead_min:.3f} 秒")
        print(f"  最大開銷: {overhead_max:.3f} 秒")
    
    # 對比分析
    if batch_avg is not None and dispatch_avg is not None:
        print("\n" + "-"*80)
        print("🏆 效能對比")
        print("-"*80)
        
        if dispatch_avg < batch_avg:
            speedup = batch_avg / dispatch_avg
            time_saved = batch_avg - dispatch_avg
            winner = "並行派發"
            print(f"✅ {winner} 較快，提升 {speedup:.2f}x 倍")
            print(f"✅ 平均每次節省 {time_saved:.3f} 秒")
        else:
            speedup = dispatch_avg / batch_avg
            time_saved = dispatch_avg - batch_avg
            winner = "批次翻譯"
            print(f"✅ {winner} 較快，提升 {speedup:.2f}x 倍")
            print(f"⚠️ 並行派發反而慢了 {time_saved:.3f} 秒")
    
    # 計算各 LLM 的統計數據
    llm_stats = {}
    for timing_dict in all_dispatch_timing_data:
        for llm, times in timing_dict.items():
            if llm not in llm_stats:
                llm_stats[llm] = []
            llm_stats[llm].extend(times)
    
    if llm_stats:
        print("\n" + "="*80)
        print("📋 各 LLM 耗時統計 (方法2並行派發)")
        print("="*80 + "\n")
        
        for llm in sorted(llm_stats.keys()):
            times = llm_stats[llm]
            if times:
                avg = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                print(f"{llm:15s} | 平均: {avg:.3f}s | 最快: {min_time:.3f}s | 最慢: {max_time:.3f}s | 使用次數: {len(times)}")
    
    print("\n" + "="*80)
    print("📋 表格數據（可直接複製到 Excel）")
    print("="*80 + "\n")
    
    # 取得所有使用過的 LLM 列表
    all_llms = sorted(set(llm for timing_dict in all_dispatch_timing_data for llm in timing_dict.keys()))
    
    # 生成表格標題
    header = "次數\t批次翻譯(秒)\t並行派發(秒)\tmax(LLM時間)\t調度開銷(秒)"
    for llm in all_llms:
        header += f"\t{llm}(秒)"
    print(header)
    print("-" * 120)
    
    # 生成每次測試的數據
    for i in range(num_tests):
        batch_time = f"{batch_all_times[i]:.3f}" if batch_all_times[i] is not None else "失敗"
        dispatch_time = f"{parallel_dispatch_times[i]:.3f}" if parallel_dispatch_times[i] is not None else "失敗"
        
        # 計算此次測試的最大 LLM 時間和調度開銷
        timing_dict = all_dispatch_timing_data[i]
        max_llm_time = max(sum(times) for times in timing_dict.values()) if timing_dict else 0
        overhead = parallel_dispatch_times[i] - max_llm_time if parallel_dispatch_times[i] is not None else 0
        
        row = f"{i+1}\t{batch_time}\t{dispatch_time}\t{max_llm_time:.3f}\t{overhead:.3f}"
        
        # 加入各 LLM 的時間
        for llm in all_llms:
            if llm in timing_dict and timing_dict[llm]:
                llm_time = sum(timing_dict[llm])
                row += f"\t{llm_time:.3f}"
            else:
                row += "\t-"
        print(row)
    
    print("-" * 100)
    
    # 生成平均、最快、最慢
    if batch_avg is not None and dispatch_avg is not None:
        # 計算 max(LLM時間) 和調度開銷的統計
        avg_max_llm = sum(max(sum(times) for times in td.values()) for td in all_dispatch_timing_data if td) / len([td for td in all_dispatch_timing_data if td]) if all_dispatch_timing_data else 0
        min_max_llm = min(max(sum(times) for times in td.values()) for td in all_dispatch_timing_data if td) if all_dispatch_timing_data else 0
        max_max_llm = max(max(sum(times) for times in td.values()) for td in all_dispatch_timing_data if td) if all_dispatch_timing_data else 0
        
        avg_row = f"平均\t{batch_avg:.3f}\t{dispatch_avg:.3f}\t{avg_max_llm:.3f}\t{overhead_avg:.3f}"
        min_row = f"最快\t{batch_min:.3f}\t{dispatch_min:.3f}\t{min_max_llm:.3f}\t{overhead_min:.3f}"
        max_row = f"最慢\t{batch_max:.3f}\t{dispatch_max:.3f}\t{max_max_llm:.3f}\t{overhead_max:.3f}"
        
        for llm in all_llms:
            if llm in llm_stats and llm_stats[llm]:
                times = llm_stats[llm]
                avg_row += f"\t{sum(times)/len(times):.3f}"
                min_row += f"\t{min(times):.3f}"
                max_row += f"\t{max(times):.3f}"
            else:
                avg_row += "\t-"
                min_row += "\t-"
                max_row += "\t-"
        
        print(avg_row)
        print(min_row)
        print(max_row)
    
    print("\n" + "="*80 + "\n")
    
    
    # 測試案例 3: GPT 一次性多語言翻譯 vs 並行翻譯時間對比 (10次測試)
    print("\n" + "="*80)
    print("測試案例 3: GPT 一次性多語言翻譯 vs 並行翻譯 (10次重複測試)")
    print("="*80 + "\n")
    source_lang = "zh"
    test_text_3 = "這個是我們共同帶領團隊要去做的事。再來,這個, 拍劇昨天那個 Ben 跟我說的,做業務呢,每年的一月"
    target_langs_3 = ["en", "de", "ja", "ko"]  # 使用 LANGUAGE_LIST 允許的語言
    num_tests = 10
    
    print(f"📝 測試文本: {test_text_3}")
    print(f"🎯 目標語言: {target_langs_3}")
    print(f"🔄 測試次數: {num_tests} 次")
    print("\n" + "-"*80)
    
    # 儲存測試結果
    gpt_batch_times = []
    parallel_times = []
    all_timing_data = []  # 儲存每次測試的各 LLM 耗時
    
    # 執行 10 次測試
    for i in range(num_tests):
        print(f"\n第 {i+1}/{num_tests} 次測試:")
        
        # 方法 1: GPT 一次性翻譯所有語言
        if manager.gpt_4o_translator is not None:
            try:
                start_gpt_batch = time.time()
                gpt_result = manager.gpt_4o_translator.translate(
                    source_text=test_text_3,
                    source_lang=source_lang,
                    target_lang=target_langs_3,
                    prev_text=""
                )
                gpt_batch_time = time.time() - start_gpt_batch
                gpt_batch_times.append(gpt_batch_time)
                print(f"  GPT 批次翻譯: {gpt_batch_time:.3f} 秒")
            except Exception as e:
                print(f"  ❌ GPT 批次翻譯失敗: {e}")
                gpt_batch_times.append(None)
        
        # 方法 2: TranslateManager 並行翻譯 (取得詳細時間)
        try:
            result_dict_3, parallel_time, methods_used_3, timing_dict = manager.assign_task(
                text=test_text_3,
                source_lang=source_lang,
                target_langs=target_langs_3,
                prev_text="",
                return_timing=True
            )
            parallel_times.append(parallel_time)
            all_timing_data.append(timing_dict)
            
            # 顯示各 LLM 耗時
            llm_times_str = ", ".join([f"{llm}:{sum(times):.3f}s" for llm, times in timing_dict.items()])
            print(f"  並行翻譯:     {parallel_time:.3f} 秒 ({llm_times_str})")
        except Exception as e:
            print(f"  ❌ 並行翻譯失敗: {e}")
            parallel_times.append(None)
            all_timing_data.append({})
    
    # 計算統計數據
    print("\n" + "="*80)
    print("📊 10次測試統計結果")
    print("="*80 + "\n")
    
    # GPT 批次翻譯統計
    valid_gpt_times = [t for t in gpt_batch_times if t is not None]
    if valid_gpt_times:
        gpt_avg = sum(valid_gpt_times) / len(valid_gpt_times)
        gpt_min = min(valid_gpt_times)
        gpt_max = max(valid_gpt_times)
        print(f"方法 1️⃣ GPT-4o 批次翻譯 (單執行緒, 1次API):")
        print(f"  平均耗時: {gpt_avg:.2f} 秒")
        print(f"  最快: {gpt_min:.2f} 秒")
        print(f"  最慢: {gpt_max:.2f} 秒")
        print(f"  成功次數: {len(valid_gpt_times)}/{num_tests}")
    else:
        print("方法 1️⃣ GPT-4o 批次翻譯: 全部失敗")
        gpt_avg = None
    
    print()
    
    # 並行翻譯統計
    valid_parallel_times = [t for t in parallel_times if t is not None]
    if valid_parallel_times:
        parallel_avg = sum(valid_parallel_times) / len(valid_parallel_times)
        parallel_min = min(valid_parallel_times)
        parallel_max = max(valid_parallel_times)
        print(f"方法 2️⃣ TranslateManager 並行翻譯 (多執行緒):")
        print(f"  平均耗時: {parallel_avg:.2f} 秒")
        print(f"  最快: {parallel_min:.2f} 秒")
        print(f"  最慢: {parallel_max:.2f} 秒")
        print(f"  成功次數: {len(valid_parallel_times)}/{num_tests}")
    else:
        print("方法 2️⃣ TranslateManager 並行翻譯: 全部失敗")
        parallel_avg = None
    
    # 計算調度開銷統計
    overheads = []
    for i, timing_dict in enumerate(all_timing_data):
        if parallel_times[i] is not None and timing_dict:
            max_llm_time = max(sum(times) for times in timing_dict.values())
            overhead = parallel_times[i] - max_llm_time
            overheads.append(overhead)
    
    if overheads:
        overhead_avg = sum(overheads) / len(overheads)
        overhead_min = min(overheads)
        overhead_max = max(overheads)
        print(f"\n方法 2️⃣ 調度開銷分析:")
        print(f"  平均開銷: {overhead_avg:.3f} 秒 ({overhead_avg/parallel_avg*100:.1f}% of 並行時間)")
        print(f"  最小開銷: {overhead_min:.3f} 秒")
        print(f"  最大開銷: {overhead_max:.3f} 秒")
        print(f"  說明: 調度開銷 = 並行總時間 - max(各LLM時間)")
    
    # 對比分析
    if gpt_avg is not None and parallel_avg is not None:
        print("\n" + "-"*80)
        print("🏆 效能對比")
        print("-"*80)
        
        if parallel_avg < gpt_avg:
            speedup = gpt_avg / parallel_avg
            time_saved = gpt_avg - parallel_avg
            winner = "並行翻譯"
        else:
            speedup = parallel_avg / gpt_avg
            time_saved = parallel_avg - gpt_avg
            winner = "GPT 批次翻譯"
        
        print(f"✅ {winner} 較快，提升 {speedup:.2f}x 倍")
        print(f"✅ 平均每次節省 {abs(time_saved):.3f} 秒")
    
    # 計算各 LLM 的統計數據
    llm_stats = {}
    for timing_dict in all_timing_data:
        for llm, times in timing_dict.items():
            if llm not in llm_stats:
                llm_stats[llm] = []
            llm_stats[llm].extend(times)
    
    print("\n" + "="*80)
    print("📋 各 LLM 耗時統計")
    print("="*80 + "\n")
    
    for llm in sorted(llm_stats.keys()):
        times = llm_stats[llm]
        if times:
            avg = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            print(f"{llm:15s} | 平均: {avg:.3f}s | 最快: {min_time:.3f}s | 最慢: {max_time:.3f}s | 使用次數: {len(times)}")
    
    print("\n" + "="*80)
    print("📋 表格數據（可直接複製到 Excel）")
    print("="*80 + "\n")
    
    # 取得所有使用過的 LLM 列表
    all_llms = sorted(set(llm for timing_dict in all_timing_data for llm in timing_dict.keys()))
    
    # 生成表格標題
    header = "次數\tGPT批次(秒)\t並行總時間(秒)\tmax(LLM時間)\t調度開銷(秒)"
    for llm in all_llms:
        header += f"\t{llm}(秒)"
    print(header)
    print("-" * 120)
    
    # 生成每次測試的數據
    for i in range(num_tests):
        gpt_time = f"{gpt_batch_times[i]:.3f}" if gpt_batch_times[i] is not None else "失敗"
        para_time = f"{parallel_times[i]:.3f}" if parallel_times[i] is not None else "失敗"
        
        # 計算此次測試的最大 LLM 時間和調度開銷
        timing_dict = all_timing_data[i]
        max_llm_time = max(sum(times) for times in timing_dict.values()) if timing_dict else 0
        overhead = parallel_times[i] - max_llm_time if parallel_times[i] is not None else 0
        
        row = f"{i+1}\t{gpt_time}\t{para_time}\t{max_llm_time:.3f}\t{overhead:.3f}"
        
        # 加入各 LLM 的時間
        timing_dict = all_timing_data[i]
        for llm in all_llms:
            if llm in timing_dict and timing_dict[llm]:
                llm_time = sum(timing_dict[llm])
                row += f"\t{llm_time:.3f}"
            else:
                row += "\t-"
        print(row)
    
    print("-" * 100)
    
    # 生成平均、最快、最慢
    if gpt_avg is not None and parallel_avg is not None:
        # 計算 max(LLM時間) 和調度開銷的統計
        avg_max_llm = sum(max(sum(times) for times in td.values()) for td in all_timing_data) / len(all_timing_data) if all_timing_data else 0
        min_max_llm = min(max(sum(times) for times in td.values()) for td in all_timing_data) if all_timing_data else 0
        max_max_llm = max(max(sum(times) for times in td.values()) for td in all_timing_data) if all_timing_data else 0
        
        avg_row = f"平均\t{gpt_avg:.3f}\t{parallel_avg:.3f}\t{avg_max_llm:.3f}\t{overhead_avg:.3f}"
        min_row = f"最快\t{gpt_min:.3f}\t{parallel_min:.3f}\t{min_max_llm:.3f}\t{overhead_min:.3f}"
        max_row = f"最慢\t{gpt_max:.3f}\t{parallel_max:.3f}\t{max_max_llm:.3f}\t{overhead_max:.3f}"
        
        for llm in all_llms:
            if llm in llm_stats and llm_stats[llm]:
                times = llm_stats[llm]
                avg_row += f"\t{sum(times)/len(times):.3f}"
                min_row += f"\t{min(times):.3f}"
                max_row += f"\t{max(times):.3f}"
            else:
                avg_row += "\t-"
                min_row += "\t-"
                max_row += "\t-"
        
        print(avg_row)
        print(min_row)
        print(max_row)
    
    print("\n" + "="*80 + "\n")
    
    # 清理資源
    manager.close()
    
    print("\n" + "="*80)
    print("✅ 測試完成！")
    print("="*80 + "\n")


