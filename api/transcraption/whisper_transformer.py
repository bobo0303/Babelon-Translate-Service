import gc  
import time  
import torch
import logging  
import logging.handlers

from transformers import pipeline, AutoProcessor

from api.core.post_process import post_process
from api.audio.audio_utils import get_audio_duration, add_silence_padding

from lib.config.constant import SILENCE_PADDING, MAX_NUM_STRATEGIES
  
  
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

class WhisperTransformer:  
    def __init__(self, result_queue):  
        """Initialize the TranscribeManager class with default attributes."""
        self.result_queue = result_queue
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  
        self.prompt_token = None
        self.prompt = None  # Store original prompt name for post-processing
        self.processor = None
        self.pipe = None  
        
    def load_model(self, model_name, model_path):  
        """Load the specified model based on the model's name."""  
        start = time.time()  
        try:  
            self.model_path = model_path  # Use getattr to access ModelPath attributes
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.pipe = pipeline(
                task="automatic-speech-recognition",
                model=self.model_path,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                device=self.device
            )            
            end = time.time()  
            logger.info(f" | Loading '{model_name}' in {end - start:.2f} seconds and successful | ")
        except Exception as e:  
            logger.error(f" | load_model() models_name: '{model_name}' error: {e} | ")
            logger.error(f" | model has been released. Please use correct model name to reload the model before using | ")
            return e
        return None

    def release_model(self):  
        """Release the resources occupied by the current model."""  
        if self.pipe is not None:  
            del self.pipe  
            gc.collect()  
            self.pipe = None  
            torch.cuda.empty_cache()  
            logger.info(" | Previous model resources have been released. | ")  
  
        
    def set_prompt(self, prompt: str):  
        """  
        Set the prompt for the transcription model.  
  
        :param prompt: str  
            The name of the prompt to be used.  
        :rtype: None  
        """  
        
        if prompt is None or prompt == "":
            self.prompt_token = None
            self.prompt = None
            logger.info(f" | Prompt has been cleared. | ")
            return
        
        # Store original prompt name for post-processing
        self.prompt = prompt
        
        start = time.time()

        try:
            # 如果 processor 未初始化，先初始化它
            if self.processor is None:
                logger.warning(f" | AutoProcessor not initialized, initializing now... | ")
                self.processor = AutoProcessor.from_pretrained(self.model_path)
            
            # 生成 prompt_ids 並移到正確的設備上
            self.prompt_token = self.processor.get_prompt_ids(prompt, return_tensors="pt")
            # 確保 prompt 在正確的設備上
            if self.device == "cuda" and torch.cuda.is_available():
                self.prompt_token = self.prompt_token.to(self.device)
                logger.debug(f" | Prompt moved to device: {self.device} | ")
            end = time.time()
            logger.info(f" | spend {end - start:.2f} seconds | Prompt has been set to: {prompt} | ")
        except Exception as e:
            logger.error(f" | set_prompt() error: {e} | ")
            self.prompt = None
            self.prompt_token = None
            logger.error(f" | Prompt setting failed. | ")
            return e    
    
    
    def transcribe(self, audio_path, audio, audio_length, ori, multi_strategy_transcription=1, post_processing=True, prev_text=""):  
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
            # Strategy 1: temp=0.0, do_sample=False, prompt=self.prompt_token+prev_text
            # Strategy 2: temp=0.0, do_sample=False, prompt=self.prompt_token 
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
                    "forced_decoder_ids": None,
                }
                
                if strategy < MAX_NUM_STRATEGIES - 2:
                    # if available strategy > 3 and not prompt and not prev_text -> skip to strategy 3 
                    if multi_strategy_transcription >= MAX_NUM_STRATEGIES - 1 and self.prompt_token is None:
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
                        prompt = torch.cat([self.prompt_token, prev_prompt], dim=-1) if self.prompt_token is not None else prev_prompt
                        prompt_size = list(prompt.size())[0]  # Get the size as an integer
                        # logger.debug(self.prompt)
                        # logger.debug(prev_text)
                        # logger.debug(f" | len of prompt with prev_text: {prompt_size} tokens. | ")
                        if prompt_size >= 400:    
                            logger.warning(f" | len of prompt: {prompt_size} voer the limit 448 tokens. Use no prev_text prompt. | ")
                            generate_kwargs["prompt_ids"] = self.prompt_token.to(self.device) if self.device == "cuda" else self.prompt_token
                        else:
                            generate_kwargs["prompt_ids"] = prompt
                    # strategy 0 without prev_text (handled strategy 1)
                    else:
                        if self.prompt_token is not None:
                            generate_kwargs["prompt_ids"] = self.prompt_token.to(self.device) if self.device == "cuda" else self.prompt_token
                
                # prepare input audio if unread give audio path 
                audio_input = audio if audio is not None else audio_path
                transcription_result = self.pipe(
                    audio_input, 
                    generate_kwargs=generate_kwargs,
                    # return_timestamps=True  
                )
                
                ori_pred = transcription_result["text"]
                logger.debug(f" | Raw Transcription: {ori_pred} | ")
                
                if post_processing:
                    audio_duration = get_audio_duration(audio_path) if audio_length is None else audio_length
                    retry_flag, ori_pred = post_process(ori_pred, audio_duration, self.prompt)
                
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
            audio_length = 0.0
            logger.error(f" | transcribe() error: {e} | ") 

        return ori_pred, inference_time, audio_length

    
