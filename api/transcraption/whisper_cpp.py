#!/usr/bin/env python3
import sys
import time
import ctypes
import numpy as np
import soundfile as sf
import librosa
import torch
from pathlib import Path
from statistics import mean, stdev

from api.audio.audio_utils import get_audio_duration, add_silence_padding

# whisper.cpp 相關
WHISPER_LIB = "/mnt/quantification/whisper.cpp/build/src/libwhisper.so"
lib = ctypes.CDLL(WHISPER_LIB)


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


class WhisperCpp:
    def __init__(self, result_queue):
        """Initialize the TranscribeManager class with default attributes."""
        self.result_queue = result_queue
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  
        self.prompt_token = None
        self.prompt = None  # Store original prompt name for post-processing
        self.pipe = None  
        self.processing = False
        
    
    def load_model(self, models_name, models_path):  
        """Load the specified model based on the model's name."""  
        pass
       

    def release_model(self):  
        """Release the resources occupied by the current model."""  
        pass
        
  
        
    def set_prompt(self, prompt: str):  
        """  
        Set the prompt for the transcription model.  
  
        :param prompt: str  
            The name of the prompt to be used.  
        :rtype: None  
        """  
        pass
        
    
    def transcribe(self, audio_file_path, ori, multi_strategy_transcription=1, post_processing=True, prev_text=""):  
       pass
    

    