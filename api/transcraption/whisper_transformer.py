import gc  
import time  
import torch
import librosa

from transformers import pipeline, AutoProcessor

from api.core.post_process import post_process
from api.audio.audio_utils import get_audio_duration, add_silence_padding
from lib.config.constant import SILENCE_PADDING, MAX_NUM_STRATEGIES
from lib.core.logging_config import get_logger

# 獲取日誌器
logger = get_logger(__name__)  

class WhisperTransformer:  
    def __init__(self, result_queue):  
        """Initialize the TranscribeManager class with default attributes."""
        self.result_queue = result_queue
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  
        self.prompt_token = None
        self.prompt = None  # Store original prompt name for post-processing
        self.processor = None
        self.pipe = None
        self.abort_flag = False  # Flag for abort request
        
    def request_abort(self):
        """
        Request abort for current transcription task.
        Note: Transformer models don't support graceful abort like whisper.cpp.
        This only sets a flag but cannot interrupt the pipeline mid-inference.
        Force thread termination will be used instead.
        """
        self.abort_flag = True
        logger.warning(" | Abort requested for Transformer (no graceful abort support, will force terminate) | ")
    
    def clear_abort(self):
        """Clear abort flag."""
        self.abort_flag = False
    
    def supports_graceful_abort(self):
        """Check if this transcriber supports graceful abort."""
        return False
        
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
    
    
    def transcribe(self, audio, audio_length, ori, multi_strategy_transcription=1, post_processing=True, prev_text="", trim_text=""):  
        """  
        Perform transcription on the given audio.  
    
        :param audio: numpy array
            The preprocessed audio data (16kHz float32).
        :param audio_length: float
            Duration of the audio in seconds.
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
                    
                    # strategy 0 without prev_text (handled strategy 1)
                    if self.prompt_token is not None:
                        generate_kwargs["prompt_ids"] = self.prompt_token.to(self.device) if self.device == "cuda" else self.prompt_token
                 
                    # strategy 0 with prev_text
                    if strategy == 0 and prev_text != "":
                        prev_prompt = self.processor.get_prompt_ids(prev_text, return_tensors="pt")
                        prev_prompt = prev_prompt.to(self.device) if self.device == "cuda" else prev_prompt
                        prompt = torch.cat([self.prompt_token, prev_prompt], dim=-1) if self.prompt_token is not None else prev_prompt
                        prompt_size = list(prompt.size())[0]  # Get the size as an integer
                        if prompt_size >= 400:    
                            logger.warning(f" | len of prompt: {prompt_size} over the limit 448 tokens. Use no prev_text prompt. | ")
                        else:
                            generate_kwargs["prompt_ids"] = prompt
                        # logger.debug(self.prompt)
                        # logger.debug(prev_text)
                        # logger.debug(f" | len of prompt with prev_text: {prompt_size} tokens. | ")
                            
                if trim_text != "":
                    trim_prompt = self.processor.get_prompt_ids(trim_text, return_tensors="pt")
                    trim_prompt = trim_prompt.to(self.device) if self.device == "cuda" else trim_prompt
                    if "prompt_ids" in generate_kwargs and generate_kwargs["prompt_ids"] is not None:
                        generate_kwargs["prompt_ids"] = torch.cat([generate_kwargs["prompt_ids"], trim_prompt], dim=-1)
                        if list(generate_kwargs["prompt_ids"].size())[0] > 400:
                            generate_kwargs["prompt_ids"] = torch.cat([self.prompt_token, trim_prompt], dim=-1) if self.prompt_token is not None else trim_prompt
                            logger.warning(f" | add trim text | len of prompt with trim_text over the limit 448 tokens. Use original prompt with trim prompt. (ignored prev text) | ")                              
                    else:
                        generate_kwargs["prompt_ids"] = trim_prompt
                    
                # prepare input audio if unread give audio path 
                # audio_input = audio if audio is not None else audio_path
                
                # whole pipelne not support return detecct language yet (20260316 bug issue). We use another transcribe way below 
                # transcription_result = self.pipe(
                #     audio_input, 
                #     generate_kwargs=generate_kwargs,
                #     return_timestamps=True  
                # )
                
                # ori_pred = transcription_result["text"]
                # logger.debug(f" | Raw Transcription: {ori_pred} | ")
                
                # chunks = transcription_result.get("chunks", [])
                # segments = []
                # for idx, chunk in enumerate(chunks):
                #     ts = chunk.get("timestamp", (0.0, 0.0))
                #     segments.append({
                #         'index': idx,
                #         'start': ts[0] if ts[0] is not None else 0.0,
                #         'end': ts[1] if ts[1] is not None else 0.0,
                #         'text': chunk.get("text", "").strip()
                #     })
                # n_segments = len(segments)
                
                # audio is always provided (preprocessed in transcribe_manager)
                audio_input = audio
                    
                # audio to mel spectrogram
                audio_input = self.pipe.feature_extractor(
                    audio_input, 
                    sampling_rate=16000, 
                    return_tensors="pt"
                ).input_features.to(device=self.device, dtype=self.pipe.model.dtype)

                # transcribe with model.generate and get language tag from token and do token to text by ourself
                transcription_result = self.pipe.model.generate(
                    audio_input,
                    return_timestamps=True,
                    return_dict_in_generate=True,
                    language=generate_kwargs["language"] if generate_kwargs["language"] != "auto" else None, 
                    task=generate_kwargs["task"],
                    temperature=generate_kwargs["temperature"],
                    do_sample=generate_kwargs["do_sample"],
                    forced_decoder_ids=generate_kwargs["forced_decoder_ids"],
                    prompt_ids=generate_kwargs.get("prompt_ids", None),
                )
                
                # 初始化變數
                detected_language = ori if ori != "auto" else "unknown"
                transcription_parts = []
                segments = []
                
                for batch_segments in transcription_result['segments']:
                    for seg in batch_segments:
                        # time stamps
                        start_time = seg['start'].item() if hasattr(seg['start'], 'item') else float(seg['start'])
                        end_time = seg['end'].item() if hasattr(seg['end'], 'item') else float(seg['end'])
                        
                        # 從 result['sequences'] 提取語言 (僅在 auto 模式且尚未檢測時)
                        if detected_language == "unknown":
                            result_seq = seg['result']['sequences']
                            # 找 <|startoftranscript|> (50258) 後面的語言 token
                            start_token_id = 50258
                            for i, t in enumerate(result_seq):
                                t_val = t.item() if hasattr(t, 'item') else t
                                if t_val == start_token_id and i + 1 < len(result_seq):
                                    lang_token_id = result_seq[i + 1].item() if hasattr(result_seq[i + 1], 'item') else result_seq[i + 1]
                                    lang_token_str = self.pipe.tokenizer.decode([lang_token_id])
                                    detected_language = lang_token_str.strip("<|>")
                                    break
                        
                        # Decode segment text
                        seg_text = self.pipe.tokenizer.decode(seg['tokens'], skip_special_tokens=True)
                        transcription_parts.append(seg_text)
                        
                        # 建立 segment (原本格式)
                        segments.append({
                            'index': len(segments),
                            'start': start_time if start_time is not None else 0.0,
                            'end': end_time if end_time is not None else 0.0,
                            'text': seg_text.strip()
                        })
                
                # 合併所有 segments 的文字
                ori_pred = ''.join(transcription_parts)
                n_segments = len(segments)
                
                # 更新 ori 為檢測到的語言 (供後續使用)
                if ori == "auto":
                    detected_lang = detected_language
                else:
                    detected_lang = ""
                    
                logger.debug(f" | Raw Transcription: {ori_pred} | ")
                
                if post_processing:
                    audio_duration = audio_length
                    retry_flag, ori_pred = post_process(ori_pred, audio_duration, self.prompt)
                # retry_flag = True
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
            detected_lang = ""
            ori_pred = ""
            inference_time = 0
            audio_length = 0.0
            n_segments = 0
            segments = []
            logger.error(f" | transcribe() error: {e} | ") 

        return detected_lang, ori_pred, n_segments, segments, inference_time, audio_length

    def detect_language(self, audio, count_confidence=True):
        """Detect language of the given audio using the current transcriber."""

        try:
            # prepare input feature 
            input_features = self.pipe.feature_extractor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.to(self.pipe.device, dtype=self.pipe.model.dtype)
            
            # detect language token and confidence (if count_confidence=True)
            with torch.no_grad():
                # detect_language ID (50259 ~ 50358)
                lang_token_ids = self.pipe.model.detect_language(input_features)
                
                # ID to language string
                detected_token_id = lang_token_ids[0].item()
                detected_language = self.pipe.tokenizer.decode(detected_token_id).strip("<|>")
                
                if count_confidence:
                    # encoder + once decoder step to get confidence
                    encoder_outputs = self.pipe.model.get_encoder()(input_features)
                    decoder_input_ids = torch.tensor([[self.pipe.model.config.decoder_start_token_id]]).to(self.pipe.device)
                    outputs = self.pipe.model(encoder_outputs=encoder_outputs, decoder_input_ids=decoder_input_ids)
                    logits = outputs.logits[0, 0]
                    probs = torch.softmax(logits, dim=-1)
                    confidence = probs[detected_token_id].item()
                else:
                    confidence = 0.0
                    
        except Exception as e:
            detected_language = "unknown"
            confidence = 0.0
            logger.error(f" | detect_language() error: {e} | ")
        
        return detected_language, confidence
