#!/usr/bin/env python3
import gc
import os
import sys
import time
import torch
import ctypes
import librosa
import logging  
import logging.handlers
import numpy as np
import soundfile as sf

from pathlib import Path
from statistics import mean, stdev
from huggingface_hub import hf_hub_download

from lib.config.constant import MAX_NUM_STRATEGIES

from api.audio.audio_utils import get_audio_duration
from api.core.post_process import post_process

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

##############################################################################  

# whisper.cpp 相關
WHISPER_LIB = "/mnt/quantification/whisper.cpp/build/src/libwhisper.so"
lib = ctypes.CDLL(WHISPER_LIB)

##############################################################################  

class WhisperContext(ctypes.Structure):
    pass


class WhisperContextParams(ctypes.Structure):
    _fields_ = [
        ("use_gpu", ctypes.c_bool),
        ("flash_attn", ctypes.c_bool),
        ("gpu_device", ctypes.c_int),
        ("dtw_token_timestamps", ctypes.c_bool),
        ("dtw_aheads_preset", ctypes.c_int),
        ("dtw_n_top", ctypes.c_int),
        ("dtw_aheads", ctypes.c_void_p),
        ("dtw_mem_size", ctypes.c_size_t),
    ]


class WhisperFullParams(ctypes.Structure):
    _fields_ = [
        ("strategy", ctypes.c_int),
        ("n_threads", ctypes.c_int),
        ("n_max_text_ctx", ctypes.c_int),
        ("offset_ms", ctypes.c_int),
        ("duration_ms", ctypes.c_int),
        ("translate", ctypes.c_bool),
        ("no_context", ctypes.c_bool),
        ("no_timestamps", ctypes.c_bool),
        ("single_segment", ctypes.c_bool),
        ("print_special", ctypes.c_bool),
        ("print_progress", ctypes.c_bool),
        ("print_realtime", ctypes.c_bool),
        ("print_timestamps", ctypes.c_bool),
        ("token_timestamps", ctypes.c_bool),
        ("thold_pt", ctypes.c_float),
        ("thold_ptsum", ctypes.c_float),
        ("max_len", ctypes.c_int),
        ("split_on_word", ctypes.c_bool),
        ("max_tokens", ctypes.c_int),
        ("debug_mode", ctypes.c_bool),
        ("audio_ctx", ctypes.c_int),
        ("tdrz_enable", ctypes.c_bool),
        ("suppress_regex", ctypes.c_char_p),
        ("initial_prompt", ctypes.c_char_p),
        ("carry_initial_prompt", ctypes.c_bool),
        ("prompt_tokens", ctypes.POINTER(ctypes.c_int)),
        ("prompt_n_tokens", ctypes.c_int),
        ("language", ctypes.c_char_p),
        ("detect_language", ctypes.c_bool),
        ("suppress_blank", ctypes.c_bool),
        ("suppress_nst", ctypes.c_bool),
        ("temperature", ctypes.c_float),
        ("max_initial_ts", ctypes.c_float),
        ("length_penalty", ctypes.c_float),
        ("temperature_inc", ctypes.c_float),
        ("entropy_thold", ctypes.c_float),
        ("logprob_thold", ctypes.c_float),
        ("no_speech_thold", ctypes.c_float),
        ("greedy_best_of", ctypes.c_int),
        ("beam_search_beam_size", ctypes.c_int),
        ("beam_search_patience", ctypes.c_float),
        ("new_segment_callback", ctypes.c_void_p),
        ("new_segment_callback_user_data", ctypes.c_void_p),
        ("progress_callback", ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("encoder_begin_callback", ctypes.c_void_p),
        ("encoder_begin_callback_user_data", ctypes.c_void_p),
        ("abort_callback", ctypes.c_void_p),
        ("abort_callback_user_data", ctypes.c_void_p),
        ("logits_filter_callback", ctypes.c_void_p),
        ("logits_filter_callback_user_data", ctypes.c_void_p),
        ("grammar_rules", ctypes.c_void_p),
        ("n_grammar_rules", ctypes.c_size_t),
        ("i_start_rule", ctypes.c_size_t),
        ("grammar_penalty", ctypes.c_float),
        ("vad", ctypes.c_bool),
        ("vad_model_path", ctypes.c_char_p),
        ("vad_params_threshold", ctypes.c_float),
        ("vad_params_min_speech_duration_ms", ctypes.c_int),
        ("vad_params_min_silence_duration_ms", ctypes.c_int),
        ("vad_params_max_speech_duration_s", ctypes.c_float),
        ("vad_params_speech_pad_ms", ctypes.c_int),
        ("vad_params_samples_overlap", ctypes.c_float),
    ]

##############################################################################  

# load mdoel
lib.whisper_context_default_params.argtypes = []
lib.whisper_context_default_params.restype = WhisperContextParams

lib.whisper_init_from_file_with_params.argtypes = [ctypes.c_char_p, WhisperContextParams]
lib.whisper_init_from_file_with_params.restype = ctypes.POINTER(WhisperContext)

# inference
lib.whisper_full_default_params.argtypes = [ctypes.c_int]
lib.whisper_full_default_params.restype = WhisperFullParams

lib.whisper_full.argtypes = [
    ctypes.POINTER(WhisperContext),
    WhisperFullParams,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int
]
lib.whisper_full.restype = ctypes.c_int

# ask inference results
lib.whisper_full_n_segments.argtypes = [ctypes.POINTER(WhisperContext)]
lib.whisper_full_n_segments.restype = ctypes.c_int

# get inference text
lib.whisper_full_get_segment_text.argtypes = [ctypes.POINTER(WhisperContext), ctypes.c_int]
lib.whisper_full_get_segment_text.restype = ctypes.c_char_p

# release model
lib.whisper_free.argtypes = [ctypes.POINTER(WhisperContext)]
lib.whisper_free.restype = None

# log control - define callback function type
GGML_LOG_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)

# set log callback
lib.whisper_log_set.argtypes = [GGML_LOG_CALLBACK, ctypes.c_void_p]
lib.whisper_log_set.restype = None

# Define a silent log callback to suppress whisper.cpp logs
@GGML_LOG_CALLBACK
def silent_log_callback(level, text, user_data):
    """Suppress all whisper.cpp C-level logs"""
    pass

# Apply silent logging globally for whisper.cpp
lib.whisper_log_set(silent_log_callback, None)

##############################################################################  

class WhisperCpp:
    def __init__(self, result_queue):
        """Initialize the TranscribeManager class with default attributes."""
        self.result_queue = result_queue
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  
        self.prompt = None  # Store original prompt name for post-processing
        self.transcriber = None  
        
    
    def load_model(self, model_name, model_path):  
        """Load the specified model based on the model's name."""  
        
        if not os.path.exists(model_path):
            start = time.time()
            model_dir = os.path.dirname(model_path)
            os.makedirs(model_dir, exist_ok=True)
            try:
                hf_hub_download(
                    repo_id="ggerganov/whisper.cpp",
                    filename=model_name + ".bin",
                    local_dir=model_dir,
                    local_dir_use_symlinks=False
                    )
                end = time.time()
                logger.info(f" | Model downloaded '{model_name}' in {end - start:.2f} seconds and successful | ")
            except Exception as e:
                self.transcriber =  None
                logger.error(f" | load_model() models_name: '{model_name}' error: {e} | ")
                logger.error(f" | model has been released. Please use correct model name to reload the model before using | ")
                return e
            
        try:
            start = time.time()
            cparams = lib.whisper_context_default_params()
            cparams.use_gpu = True if self.device == "cuda" else False
            cparams.flash_attn = True
            cparams.gpu_device = 0
            
            self.transcriber = lib.whisper_init_from_file_with_params(
                model_path.encode('utf-8'),
                cparams
            )
            
            if not self.transcriber:
                raise RuntimeError(" | CPP: Failed to load model | ")
            
            end = time.time()
            logger.info(f" | Loading '{model_name}' in {end - start:.2f} seconds and successful | ")
        except Exception as e:
            self.transcriber =  None
            logger.error(f" | load_model() models_name: '{model_name}' error: {e} | ")
            logger.error(f" | model has been released. Please use correct model name to reload the model before using | ")
            return e


    def release_model(self):  
        """Release the resources occupied by the current model."""  
        if self.transcriber:  
            lib.whisper_free(self.transcriber)
            del self.transcriber  
            gc.collect()  
            self.transcriber = None  
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
            self.prompt = None
            logger.info(f" | Prompt has been cleared. | ")
            return
        
        # Store original prompt string (not bytes) for later use
        self.prompt = prompt
        logger.info(f" | Prompt has been set to: {prompt} | ")
            
    
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
                
                # Set whisper.cpp full params                
                params = lib.whisper_full_default_params(0)
                params.language = ori.encode('utf-8')
                params.temperature = 0.0 
                params.temperature_inc = 0.0 if strategy < MAX_NUM_STRATEGIES - 1 else 0.2
                # params.logprob_thold=-1.0
                # params.entropy_thold=2.4
                params.n_threads = 8
                params.beam_search_beam_size = 1
                params.vad = False 
                
                # Set initial prompt based on strategy (match transformer logic)
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
                        prompt_text = (self.prompt + " " + prev_text) if self.prompt else prev_text
                        # Rough estimation: whisper.cpp limit is 224 tokens (~448 chars for mixed CN/EN)
                        params.initial_prompt = prompt_text.encode('utf-8')
                    # strategy 0 without prev_text (handled as strategy 1)
                    else:
                        if self.prompt:
                            params.initial_prompt = self.prompt.encode('utf-8')
                
                result = lib.whisper_full(
                    self.transcriber,
                    params,
                    audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    len(audio)
                )
                
                if result != 0:
                    ori_pred = ""
                    inference_time = 0
                    logger.error(f" | transcribe() failed with error code: {result} | ") 
                    return ori_pred, inference_time
                
                n_segments = lib.whisper_full_n_segments(self.transcriber)
                text_parts = []
                
                for j in range(n_segments):
                    text = lib.whisper_full_get_segment_text(self.transcriber, j)
                    if text:
                        try:
                            text_parts.append(text.decode('utf-8', errors='ignore'))
                        except:
                            text_parts.append(text.decode('utf-8', errors='replace'))
                            
                ori_pred = ''.join(text_parts)
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


if __name__ == "__main__":
    from queue import Queue  
    from api.audio.audio_utils import audio_preprocess
    
    result_queue = Queue()
    cpp_whisper = WhisperCpp(result_queue)
    
    audio_path = "/mnt/old/2025_Q2_Frank/segment_001.wav"
    audio = audio_preprocess(audio_path, 0.0)
    
    cpp_whisper.load_model("ggml-large-v2", "/mnt/models/ggml-large-v2.bin")
    
    # cpp_whisper.set_prompt("")
    # cpp_whisper.set_prompt("拉貨力道, 出貨力道, 放量, 換機潮, 業說會, pull in, 曝險, BOM, deal, 急單, foreX, NT dollars, Monitor, MS, BS, china car, FindARTs, DSBG, low temp, Tier 2, Tier 3, Notebook, RD, TV, 8B, In-Cell Touch, Vertical, 主管, Firmware, AecoPost, DaaS, OLED, AmLED, Polarizer, Tartan Display, 達擎, ADP team, Legamaster, AVOCOR, RISEvision, JECTOR, SatisCtrl, Karl Storz, Schwarz, NATISIX, Pillar, 凌華, ComQi, paul, AUO, 彭双浪, 柯富仁")
    
    ori_pred, inference_time = cpp_whisper.transcribe(audio_path, audio, "zh", multi_strategy_transcription=1, post_processing=True, prev_text="")
    
    print(f"Transcription: {ori_pred}")
    print(f"Inference time: {inference_time:.2f} seconds")
    
    