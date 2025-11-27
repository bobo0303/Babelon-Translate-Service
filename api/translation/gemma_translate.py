
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
from lib.config.constant import SYSTEM_PROMPT_V3, SYSTEM_PROMPT_V4_1, SYSTEM_PROMPT_V4_2, SYSTEM_PROMPT_5LANGUAGES_V3, SYSTEM_PROMPT_5LANGUAGES_V4_1, SYSTEM_PROMPT_5LANGUAGES_V4_2, GEMMA_4B_IT, LANGUAGE_LIST, DEFAULT_RESULT, SYSTEM_PROMPT_EAPC_V3, SYSTEM_PROMPT_EAPC_V4_1, SYSTEM_PROMPT_EAPC_V4_2, get_system_prompt_dynamic_language

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
                logger.info(f" | Already authenticated with HuggingFace as: {user_info['name']} | ")
                return
            except Exception:
                # Not logged in yet, continue to token-based login
                pass
            
            # Try to get token from environment variable
            hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN')
            
            if not hf_token:
                # For development/testing purposes, provide clear guidance
                logger.error(f" | HUGGINGFACE_HUB_TOKEN environment variable not found! | ")
                logger.error(f" | Please set it with: export HUGGINGFACE_HUB_TOKEN='your_token_here' | ")
                logger.error(f" | Or run with: docker exec -e HUGGINGFACE_HUB_TOKEN='your_token' ... | ")
                logger.error(f" | Or use: hf auth login (which you may have already done) | ")
                raise ValueError("HuggingFace token is required to access gated models")
            
            # Use add_to_git_credential=False to avoid modifying git config
            login(token=hf_token, add_to_git_credential=False)
            logger.info(f" | Successfully authenticated with HuggingFace using token | ")
            
        except Exception as e:
            logger.error(f" | Failed to authenticate with HuggingFace: {e} | ")
            raise RuntimeError(f"HuggingFace authentication failed: {e}") from e
    
    def _parse_response(self, response_text, expected_languages):
        """Parse and validate response
        
        Args:
            response_text: The response text from Gemma
            expected_languages: List of expected language codes in the response
        """
        try:
            # Clean response text
            cleaned_response = response_text.strip()
            
            # Try to extract JSON block (handle possible markdown wrapping)
            import re
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
            else:
                json_str = cleaned_response
            
            # Fix common GPT errors: {"json":[ -> {"json":{
            json_str = re.sub(r'(\{"json"\s*:\s*)\[', r'\1{', json_str)
            json_str = re.sub(r'\]\s*\}$', r'}}', json_str)
            
            # Try to parse JSON
            result = json.loads(json_str)
            
            # Validate response format
            if not isinstance(result, dict):
                logger.warning(" | Gemma() | Response is not a dictionary, ignoring | ")
                logger.warning(f" | Gemma() | Response type: {type(result)}, Content: {result} | ")
                logger.warning(f" | Gemma() | Raw response: {response_text} | ")
                return None
            
            # Post-processing: Handle wrapped response in extra 'json' key
            # Example: {"json": {"zh": "...", "en": "..."}} -> {"zh": "...", "en": "..."}
            if len(result) == 1 and "json" in result and isinstance(result["json"], dict):
                logger.warning(" | Gemma() | Detected wrapped response with 'json' key, unwrapping | ")
                result = result["json"]
            
            # Check required language keys (only check expected languages)
            for lang in expected_languages:
                if lang not in result:
                    logger.warning(f" | Gemma() | Missing language key: {lang}, ignoring response | ")
                    logger.warning(f" | Gemma() | Expected keys: {expected_languages}, Got keys: {list(result.keys())} | ")
                    logger.warning(f" | Gemma() | Raw response: {response_text} | ")
                    return None
            
            # Create response with only expected languages
            formatted_result = {}
            for lang in expected_languages:
                translated_text = result.get(lang, "").strip()
                formatted_result[lang] = translated_text
            
            return formatted_result
            
        except json.JSONDecodeError as e:
            logger.warning(f" | Gemma() | Failed to parse JSON response: {e} | ")
            logger.warning(f" | Gemma() | Raw response: {response_text} | ")
            return None
        except Exception as e:
            logger.error(f" | Error parsing response: {e} | ")
            return "403_Forbidden"
    
    def translate(self, source_text, source_lang='zh', target_lang='en', prev_text=""):
        """
        Translate text from source language to supported target languages.
        
        Args:
            source_text: Text to translate
            source_lang: Source language code
            target_lang: Target language code or list of codes (supports both str and list)
            prev_text: Previous context text
            
        Returns:
            dict or str: Translation result
        """
        
        # Handle both single string and list for target_lang
        if isinstance(target_lang, list):
            # Multi-language mode
            target_languages = target_lang
        else:
            # Single language mode
            target_languages = [target_lang]
        
        all_languages = [source_lang] + target_languages
        
        # Generate dynamic prompt for all target languages
        system_prompt = get_system_prompt_dynamic_language(all_languages, prev_text)
            
        try:
            messages = [
                { 
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}] 
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
            logger.debug(f" | GEMMA 4B Translation result: {decoded} | ")
            
            # Clean and parse the JSON response with expected languages
            cleaned_result = self._parse_response(decoded, all_languages)
            if cleaned_result is None:
                # When parsing fails, return fallback result
                logger.warning(" | Gemma() | Failed to parse response, using fallback | ")
                result = {source_lang: source_text}
                for lang in target_languages:
                    result[lang] = ""
                return result
            return cleaned_result
            
        except Exception as e:
            logger.error(f" | Translation failed: {str(e)} | ")
            result = {source_lang: source_text}
            for lang in target_languages:
                result[lang] = ""
            return result
        
        