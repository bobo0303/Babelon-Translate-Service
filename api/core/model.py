import gc  
import time  
import torch
import librosa
import logging  
import logging.handlers
import numpy as np

from transformers import pipeline, AutoProcessor
from queue import Queue  

from api.core.post_process import post_process
from api.audio.audio_utils import get_audio_duration
from api.core.translate_manager import TranslateManager

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

class Model:  
    def __init__(self):  
        """Initialize the Model class with default attributes."""  
        self.models_path = ModelPath()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  
        self.prompt = None
        self.prompt_name = None  # Store original prompt name for post-processing
        self.processor = None
        self.pipe = None  
        self.model_version = None  
        self.result_queue = Queue()  
        self.processing = False
        
        # Initialize TranslateManager for parallel translation
        try:
            self.translate_manager = TranslateManager()
            logger.info(" | TranslateManager initialized successfully | ")
        except Exception as e:
            logger.error(f" | Failed to initialize TranslateManager: {e} | ")
            self.translate_manager = None
  
    def load_model(self, models_name):  
        """Load the specified model based on the model's name."""  
        start = time.time()  
        try:  
            # Release old model resources  
            self._release_model()  
            self.model_version = models_name  
  
            self.model_path = getattr(self.models_path, models_name)  # Use getattr to access ModelPath attributes
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.pipe = pipeline(
                task="automatic-speech-recognition",
                model=self.model_path,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                device=self.device
            )            
            end = time.time()  
            logger.info(f" | Loading '{models_name}' in {end - start:.2f} seconds and successful | ")
        except Exception as e:  
            self.model_version = None
            logger.error(f" | load_model() models_name: '{models_name}' error: {e} | ")
            logger.error(f" | model has been released. Please use correct model name to reload the model before using | ")
            return e
        return None

    def _release_model(self):  
        """Release the resources occupied by the current model."""  
        if self.pipe is not None:  
            del self.pipe  
            gc.collect()  
            self.pipe = None  
            torch.cuda.empty_cache()  
            logger.info(" | Previous model resources have been released. | ")  
  
        
    def set_prompt(self, prompt_name):  
        """  
        Set the prompt for the transcription model.  
  
        :param prompt_name: str  
            The name of the prompt to be used.  
        :rtype: None  
        """  
        
        if prompt_name is None:
            self.prompt = None
            self.prompt_name = None
            return
        
        # Store original prompt name for post-processing
        self.prompt_name = prompt_name
        
        start = time.time()
        if not prompt_name.endswith(('.', '。', '!', '！', '?', '？')):
            prompt_name += '.'
        # prompt_text = f"Our prompts are {prompt_name}"
        prompt_text = f"These are our prompts {prompt_name} Let's continue."
        
        try:
            # 如果 processor 未初始化，先初始化它
            if self.processor is None:
                logger.warning(f" | AutoProcessor not initialized, initializing now... | ")
                self.processor = AutoProcessor.from_pretrained(self.model_path)
            
            # 生成 prompt_ids 並移到正確的設備上
            self.prompt = self.processor.get_prompt_ids(prompt_text, return_tensors="pt")
            # 確保 prompt 在正確的設備上
            if self.device == "cuda" and torch.cuda.is_available():
                self.prompt = self.prompt.to(self.device)
                logger.debug(f" | Prompt moved to device: {self.device} | ")
            end = time.time()
            logger.info(f" | spend {end - start:.2f} seconds | Prompt has been set to: {prompt_name} | ")
        except Exception as e:
            logger.error(f" | set_prompt() error: {e} | ")
            self.prompt = None
            logger.error(f" | Prompt setting failed. | ")
            return e    
        
    def _add_silence_padding(self, audio_file, padding_duration=0.3):  # Reduce to 0.3 seconds
        """
        Add silence padding to the beginning and end of audio file.
        
        Args:
            audio_file: Path to audio file
            padding_duration: Duration of silence to add in seconds
            
        Returns:
            numpy.ndarray: Audio with silence padding added
        """
        start_time = time.time()
        
        try:
            audio, sr = librosa.load(audio_file, sr=16000)
            
            # Add silence at beginning and end
            padding_samples = int(padding_duration * sr)
            silence = np.zeros(padding_samples, dtype=audio.dtype)
            
            # Add silence before and after the audio
            padded_audio = np.concatenate([silence, audio, silence])
            
            end_time = time.time()
            execution_time = end_time - start_time
            original_duration = len(audio) / sr
            padded_duration = len(padded_audio) / sr
            
            logger.debug(f" | _add_silence_padding execution time: {execution_time:.8f}s | Original: {original_duration:.2f}s | Padded: {padded_duration:.2f}s | File: {audio_file} | ")
            
            return padded_audio
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            logger.error(f" | _add_silence_padding failed in {execution_time:.8f}s | Error: {e} | File: {audio_file} | ")
            # Return original file path if padding fails
            return audio_file

    def transcribe(self, audio_file_path, ori, multi_strategy_transcription=1, post_processing=True, prev_text=""):  
        """  
        Perform transcription on the given audio file.  
    
        :param audio_file_path: str  
            The path to the audio file to be transcribed.  
        :param ori: str  
            The original language of the audio.
        :rtype: tuple  
            A tuple containing the original transcription and inference time.  
        :logs: Inference status and time.  
        """  
        start = time.time()  # Start timing the transcription process  

        try:
            # Store original file path for duration calculation
            original_audio_file_path = audio_file_path
            
            if SILENCE_PADDING:
                audio_file_path = self._add_silence_padding(audio_file_path)
                
            # Process previous text context
            if prev_text.strip() != "" and len(prev_text.replace('.', '').replace('。', '').replace(',', '').replace('，', '').strip()) >= 1:
                if not prev_text.endswith(('.', '。', '!', '！', '?', '？')):
                    prev_text += '。' 
            else:
                prev_text = ""
                
            # Multi-strategy transcription:
            # Strategy 1: temp=0.0, do_sample=False, prompt=self.prompt+prev_text
            # Strategy 2: temp=0.0, do_sample=False, prompt=self.prompt 
            # Strategy 3: temp=0.0, do_sample=False, prompt=None
            # Strategy 4: temp=[0.2,0.4,0.6,0.8,1.0], do_sample=True, prompt=None
            for strategy in range(multi_strategy_transcription):
                retry_flag = False
                # Build generate_kwargs conditionally to avoid conflicts
                generate_kwargs = {
                    "language": ori,  
                    "task": "transcribe",  
                    "temperature": 0.0 if strategy < MAX_NUM_STRATEGIES - 1 else [0.2, 0.4, 0.6, 0.8, 1.0],
                    "do_sample": False if strategy < MAX_NUM_STRATEGIES - 1 else True,
                }
                
                if strategy < MAX_NUM_STRATEGIES - 2:
                    # if available strategy > 3 and not prompt and not prev_text -> skip to strategy 3 
                    if multi_strategy_transcription >= MAX_NUM_STRATEGIES - 1 and self.prompt is None:
                        if strategy == 0 and prev_text != "":
                            pass
                        else:
                            continue
                    # if no prev_text -> strategy 0 already handled
                    if prev_text == "" and strategy == 1:
                        continue
                    # strategy 0 with prev_text
                    if strategy == 0 and prev_text != "":
                        prev_prompt = self.processor.get_prompt_ids(prev_text, return_tensors="pt")
                        prev_prompt = prev_prompt.to(self.device) if self.device == "cuda" else prev_prompt
                        prompt = torch.cat([self.prompt, prev_prompt], dim=-1) if self.prompt is not None else prev_prompt
                        prompt_size = list(prompt.size())[0]  # Get the size as an integer
                        # logger.debug(self.prompt_name)
                        # logger.debug(prev_text)
                        # logger.debug(f" | len of prompt with prev_text: {prompt_size} tokens. | ")
                        if prompt_size >= 400:    
                            logger.warning(f" | len of prompt: {prompt_size} voer the limit 448 tokens. Use no prev_text prompt. | ")
                            generate_kwargs["prompt_ids"] = self.prompt.to(self.device) if self.device == "cuda" else self.prompt
                        else:
                            generate_kwargs["prompt_ids"] = prompt
                    # strategy 0 without prev_text (handled strategy 1)
                    else:
                        if self.prompt is not None:
                            generate_kwargs["prompt_ids"] = self.prompt.to(self.device) if self.device == "cuda" else self.prompt
                
                transcription_result = self.pipe(
                    audio_file_path, 
                    generate_kwargs=generate_kwargs,
                    # return_timestamps=True  
                )
                
                ori_pred = transcription_result["text"]
                logger.debug(f" | Raw Transcription: {ori_pred} | ")
                
                if post_processing:
                    audio_duration = get_audio_duration(original_audio_file_path)
                    retry_flag, ori_pred = post_process(ori_pred, audio_duration, self.prompt_name)
                
                if retry_flag:
                    end = time.time() 
                    if strategy < multi_strategy_transcription - 1:
                        logger.info(f" | Strategy {strategy+1} | Transcription: {ori_pred} | ")
                        logger.info(f" | Strategy {strategy+1} FAILED: retry strategy {strategy+2} | now process time '{end - start:.2f}' seconds | ")
                    else:
                        logger.info(f" | Strategy {strategy+1} | Transcription: {ori_pred} | ")
                        logger.info(f" | Strategy {strategy+1} FAILED: no more retry strategies | now process time '{end - start:.2f}' seconds | ")
                else:
                    break  
            
            end = time.time() 
            inference_time = end - start  
        except Exception as e:
            ori_pred = ""
            inference_time = 0
            logger.error(f" | transcribe() error: {e} | ") 

        return ori_pred, inference_time  
    
    def translate(self, text, source_lang, target_langs, prev_text="", multi_translate=True):
        """
        Translate text to multiple target languages using TranslateManager.
        
        Args:
            text: Text to translate
            source_lang: Source language code (e.g., 'zh', 'en')
            target_langs: Target language code(s), can be:
                - str: single language (e.g., 'en')
                - list: multiple languages (e.g., ['en', 'de', 'ja', 'ko'])
            prev_text: Previous text context (optional)
            
        Returns:
            tuple: (result_dict, translate_time, translate_method)
                - result_dict: {lang: translated_text} including source language
                - translate_time: Time taken for translation
                - translate_method: List of translators used
        """
        if self.translate_manager is None:
            logger.error(" | TranslateManager not available, returning empty results | ")
            return DEFAULT_RESULT.copy(), 0.0, "FAILED"
        
        # Convert target_langs to list if it's a single string
        if isinstance(target_langs, str):
            target_langs = [target_langs]
        
        try:
            # Store current task group ID for cleanup if needed
            self.current_translation_task_group = None
            
            # Use TranslateManager's assign_task for parallel translation
            result_dict, translate_time, methods_used, timing_dict = self.translate_manager.assign_task(
                text=text,
                source_lang=source_lang,
                target_langs=target_langs,
                prev_text=prev_text,
                multi_translate=multi_translate
            )
            
            # methods_used is already a string from translate_manager
            translate_method = methods_used if methods_used else "UNKNOWN"
            
            return result_dict, translate_time, translate_method, timing_dict
            
        except Exception as e:
            logger.error(f" | translate() error: {e} | ")
            # Return default result with source text
            default_result = DEFAULT_RESULT.copy()
            default_result[source_lang] = text
            return default_result, 0.0, "FAILED", {}
    
    def cleanup_translation_threads(self):
        """Clean up all active translation threads in TranslateManager."""
        if self.translate_manager is None:
            return
        
        try:
            # Stop all active task groups
            with self.translate_manager.lock:
                task_group_ids = list(self.translate_manager.task_groups.keys())
            
            for task_group_id in task_group_ids:
                logger.info(f" | Cleaning up translation task group: {task_group_id} | ")
                self.translate_manager._stop_all_threads_in_group(task_group_id)
            
            logger.debug(" | All translation threads cleaned up | ")
        except Exception as e:
            logger.error(f" | Error cleaning up translation threads: {e} | ")
    