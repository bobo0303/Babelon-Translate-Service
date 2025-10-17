
# GEMMA 4B (https://huggingface.co/google/gemma-3-4b-it)

import os
import sys
import logging  
import torch
import json
import re
from transformers import AutoProcessor, Gemma3ForConditionalGeneration  
from huggingface_hub import login

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.constant import SYSTEM_PROMPT, SYSTEM_PROMPT_V2, GEMMA_4B_IT, LANGUAGE_LIST, DEFAULT_RESULT

logger = logging.getLogger(__name__)

class Gemma4BTranslate:
    def __init__(self):
        # Setup HuggingFace authentication
        self._setup_huggingface_auth()
        
        self.model = Gemma3ForConditionalGeneration.from_pretrained(GEMMA_4B_IT, device_map="auto").eval()  
        self.processor = AutoProcessor.from_pretrained(GEMMA_4B_IT) 
        
    def _setup_huggingface_auth(self):
        """Setup HuggingFace authentication - check existing login first, then try env variable"""
        try:
            # First, try to check if already logged in
            from huggingface_hub import whoami
            try:
                user_info = whoami()
                logger.info(f"Already authenticated with HuggingFace as: {user_info['name']}")
                return
            except Exception:
                # Not logged in yet, continue to token-based login
                pass
            
            # Try to get token from environment variable
            hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN')
            
            if not hf_token:
                # For development/testing purposes, provide clear guidance
                logger.error("HUGGINGFACE_HUB_TOKEN environment variable not found!")
                logger.error("Please set it with: export HUGGINGFACE_HUB_TOKEN='your_token_here'")
                logger.error("Or run with: docker exec -e HUGGINGFACE_HUB_TOKEN='your_token' ...")
                logger.error("Or use: hf auth login (which you may have already done)")
                raise ValueError("HuggingFace token is required to access gated models")
            
            # Use add_to_git_credential=False to avoid modifying git config
            login(token=hf_token, add_to_git_credential=False)
            logger.info("Successfully authenticated with HuggingFace using token")
            
        except Exception as e:
            logger.error(f"Failed to authenticate with HuggingFace: {e}")
            raise RuntimeError(f"HuggingFace authentication failed: {e}") from e
    
    def _parse_response(self, response_text):
        """解析並驗證響應"""
        try:
            # 清理響應文本
            cleaned_response = response_text.strip()
            
            # 嘗試提取 JSON 塊（處理可能的 markdown 包裝）
            import re
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
            else:
                json_str = cleaned_response
            
            # 嘗試解析 JSON
            result = json.loads(json_str)
            
            # 驗證響應格式
            if not isinstance(result, dict):
                logger.warning(" | Response is not a dictionary, ignoring | ")
                return None
            
            # 檢查所需的語言鍵
            for lang in LANGUAGE_LIST:
                if lang not in result:
                    logger.warning(f" | Missing language key: {lang}, ignoring response | ")
                    return None
            
            # 創建標準格式的響應
            formatted_result = DEFAULT_RESULT.copy()
            
            # 設置所有語言的翻譯結果（讓 GPT 決定源語言）
            for lang in LANGUAGE_LIST:
                translated_text = result.get(lang, "").strip()
                formatted_result[lang] = translated_text
            
            return formatted_result
            
        except json.JSONDecodeError as e:
            logger.warning(f" | Failed to parse JSON response, ignoring: {e} | ")
            logger.debug(f" | Raw response: {response_text[:200]}... | ")
            return None
        except Exception as e:
            logger.error(f" | Error parsing response: {e} | ")
            return "403_Forbidden"
    
    def translate(self, source_text):
        """
        Translate text from source language to supported target languages.
        
        Args:
            source_text (str): Text to translate
            
        Returns:
            dict or str: Translation result
        """
        try:
            messages = [
                { 
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT_V2}] 
                },
                { 
                    "role": "user", 
                    "content": [{"type": "text", "text": source_text}] 
                } 
            ]

            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16)

            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = self.model.generate(**inputs, max_new_tokens=4000, do_sample=False)
                generation = generation[0][input_len:]

            decoded = self.processor.decode(generation, skip_special_tokens=True)
            logger.debug(f"GEMMA 4B Translation result: {decoded}")
            
            # Clean and parse the JSON response
            cleaned_result = self._parse_response(decoded)
            return cleaned_result
            
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            return None
        
        