# This process is for websocket STT manager
import os
import threading
import numpy as np
from queue import Queue  
from datetime import datetime
from api.core.threading_api import audio_translate, waiting_times, stop_thread

from lib.config.constant import SAMPLERATE, DEFAULT_RESULT, LANGUAGE_LIST, WAITING_TIME, AudioTranslationResponse, get_global_model
from lib.core.response_manager import process_stt_response

is_call_stt = False
call_web_api_threads = {}

audio_info_list = []
audio_info_dict = {}

class WebSocketSttManager:
    def __init__(self, logger, payload_data: dict, connection, connection_id):
        self.recording_id = payload_data.get("recording_id", "default_recording_id")
        self.meeting_id = payload_data.get("meeting_id", "default_meeting_id")
        self.speaker_id = payload_data.get("speaker_id", "default_speaker_id")
        self.speaker_name = payload_data.get("speaker_name", "default_speaker_name")
        self.device_id = payload_data.get("device_id", "default_device_id")
        self.connection = connection
        self.connection_id = connection_id
        self.use_translate = payload_data.get("use_translate", True)
        self.use_prev_text = payload_data.get("use_prev_text", False)
        self.prev_text = ""
        self.prev_text_timestamp = datetime.now()  # 直接存 datetime 對象
        self.output_directory = f"audio/{payload_data.get('meeting_id', 'default_meeting_id')}/"
        self.logger = logger
        self.stt_lock = threading.Lock()

        self.model = get_global_model()
        if self.model:
            self.logger.info(f" | AudioProcessor 已獲取 model 實例: {id(self.model)} | ")
        else:
            self.logger.warning(" | Model 尚未設置到註冊表中 | ")
            
        self.post_processing = True
        self.language = "zh"
        
    def send_to_stt(self, audio_data: np.ndarray, audio_uid: str, frame_timestamp: str, audio_tags=""):
        
        global is_call_stt
        global call_web_api_thread
        global audio_info_list
        global audio_info_dict
        
        try:
            with self.stt_lock:
                chunk_len = len(audio_data)
                
                input_info = {
                            "chunk_len": chunk_len,
                            "audio_uid": audio_uid,
                            "frame_timestamp": frame_timestamp,
                            "audio_tags": audio_tags,
                            "meeting_id": self.meeting_id,
                            "device_id": self.device_id,
                            "ori_lang": self.language,
                            "prev_text": self.prev_text,
                            "use_prev_text": self.use_prev_text,
                            "post_processing": self.post_processing,
                            "use_translate": self.use_translate,
                        }
                
                if is_call_stt:
                    if audio_uid not in audio_info_list:
                        audio_info_list.append(audio_uid)
                    audio_info_dict[audio_uid] = input_info

                    self.logger.debug(" | wait stt response. | ")
                    return

                # 設定正在呼叫 STT Flag
                is_call_stt = True
                
            # 將 Call STT 的步驟用 New Thread 處理，避免處理阻塞導致收音報錯
            self._create_stt_thread(input_info)
            
        except Exception as e:
            self.logger.error(f" | Failed to send STT API: {e} | ")
            with self.stt_lock:
                is_call_stt = False
    
    def _create_stt_thread(self, input_info):
        
        # call STT inference 
        call_web_api_thread = threading.Thread(
            target=self._process_stt, 
            kwargs=input_info, 
            daemon=True
        )
        call_web_api_thread.start()
        call_web_api_threads["main"] = call_web_api_thread
        self.logger.debug(" | Started call_web_api_thread for process_stt | ")

    def _process_stt(
        self, 
        chunk_len: int, 
        audio_uid: str, 
        frame_timestamp: str, 
        audio_tags: str,
        meeting_id: str,
        device_id: str,
        ori_lang: str,
        prev_text: str,
        use_prev_text: bool,
        post_processing: bool,
        use_translate: bool,
        ):
        """實際的 STT 處理邏輯"""
        global is_call_stt
        audio_length = round(chunk_len / SAMPLERATE, 2)
        
        # Convert original language and target language to lowercase  
        o_lang = ori_lang.lower()  
        save_prev_text = False
        
        try:
            multi_strategy_transcription = 3

            if "audio_start" in audio_tags:
                multi_strategy_transcription = 1
            elif "audio_end" in audio_tags:
                multi_strategy_transcription = 4
                save_prev_text = use_prev_text
            
            # Create response data structure  
            response_data = AudioTranslationResponse(  
                meeting_id=meeting_id,  
                device_id=device_id,  
                ori_lang=o_lang,  
                transcription_text="",
                text=DEFAULT_RESULT.copy(),  
                times=str(frame_timestamp),  
                audio_uid=audio_uid,  
                transcribe_time=0.0,  
                translate_time=0.0,  
            )  

            # Save the uploaded audio file
            filename = (
                f"{audio_uid}_{frame_timestamp.replace(':', ';').replace(' ', '_')}.wav"
            )
            audio_buffer = os.path.join(self.output_directory, filename)
            
            # Check if the audio file exists  
            if not os.path.exists(audio_buffer):  
                self.logger.error(f" | meeting ID: {self.meeting_id} | audio UID: {audio_uid} | {filename} does not exist | ")
                return
            
            # Check if the model has been loaded  
            if self.model is None or self.model.model_version is None:  
                self.logger.error(f" | model haven't been load successfully. may out of memory please check again | ")
                return
            
            # Check if the languages are in the supported language list  
            if o_lang not in LANGUAGE_LIST:  
                self.logger.info(f" | The original language is not in LANGUAGE_LIST: {LANGUAGE_LIST}. | ")  
                return

            # Create a queue to hold the return value  
            result_queue = Queue()  
            # Create an event to signal stopping  
            stop_event = threading.Event()  
    
            # Create timing thread and inference thread  
            time_thread = threading.Thread(target=waiting_times, args=(stop_event, self.model, WAITING_TIME))  
            inference_thread = threading.Thread(target=audio_translate, args=(self.model, audio_buffer, result_queue, o_lang, stop_event, multi_strategy_transcription, post_processing, prev_text, use_translate))  
    
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
                
                self.logger.debug(response_data.model_dump_json())  
                self.logger.info(f" | device_id: {response_data.device_id} | audio_uid: {response_data.audio_uid} | source language: {o_lang} | translate_method: {translate_method} | frame timestamp: {frame_timestamp} | ")  
                self.logger.info(f" | Transcription: {ori_pred} | ")
                if use_translate:
                    self.logger.info(f" | {'#' * 75} | ")
                    self.logger.info(f" | ZH: {zh_result} | ")  
                    self.logger.info(f" | EN: {en_result} | ")  
                    self.logger.info(f" | DE: {de_result} | ")  
                    self.logger.info(f" | {'#' * 75} | ")
                self.logger.info(f" | RTF: {rtf} | audio length: {audio_length} seconds. | total time: {transcription_time + translate_time:.2f} seconds. | transcribe {transcription_time:.2f} seconds. | translate {translate_time:.2f} seconds. | strategy: {multi_strategy_transcription} | ")  
            else:  
                self.logger.info(f" | Translation has exceeded the upper limit time and has been stopped |")  
                rtf = 0.0
                ori_pred = zh_result = en_result = de_result = ""
            
            if save_prev_text:
                frame_dt = datetime.strptime(frame_timestamp, "%Y-%m-%d %H:%M:%S.%f")
                if frame_dt > self.prev_text_timestamp:
                    self.prev_text_timestamp = frame_dt
                    self.prev_text = ori_pred

            # process the STT response
            other_info = {
                "audio_length": audio_length,
                "recording_id": self.recording_id,
                "speaker_id": self.speaker_id,
                "speaker_name": self.speaker_name,
                "audio_uid": audio_uid,
                "audio_file_name": filename,
                "use_translate": use_translate,
                "use_prev_text": use_prev_text,
                "prev_text": prev_text,
                "prev_text_timestamp": frame_timestamp if use_prev_text else None,
                "post_processing": post_processing,
                "audio_tags": audio_tags,
                "strategy": multi_strategy_transcription,
                "connection": self.connection,
                "connection_id": self.connection_id,
                "rtf": rtf
            }

            process_stt_response(self.logger, response_data, other_info)

        except Exception as e:
            self.logger.error(f" | Error in STT thread for audio_uid:{audio_uid}, frame_timestamp:{frame_timestamp}, error:{e} | ")

        finally:
            with self.stt_lock:
                if len(audio_info_list) == 0:
                    is_call_stt = False
                    self.logger.debug(" | No more audio to process, set is_call_stt to False | ")
                    return

                next_audio_uid = audio_info_list.pop(0)
                next_audio_info = audio_info_dict.pop(next_audio_uid, None)

                # 將 Call STT 的步驟用 New Thread 處理，避免處理阻塞導致收音報錯
                call_web_api_thread = threading.Thread(
                    target=self._process_stt, 
                    kwargs=next_audio_info, 
                    daemon=True
                )
                call_web_api_thread.start()
                call_web_api_threads["main"] = call_web_api_thread
                self.logger.debug(" | Started call_web_api_thread for process_stt | ")
                
                is_call_stt = True
                
