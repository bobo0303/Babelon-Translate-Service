"""
Safe GPT translation module, specifically designed for text_translate API
Prevents prompt injection attacks and ensures stable translation output format
"""

import os
import sys
import yaml  
import json
from openai import AzureOpenAI
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.config.constant import AZURE_CONFIG, LANGUAGE_LIST, DEFAULT_RESULT, SYSTEM_PROMPT_EAPC_V3, SYSTEM_PROMPT_V3, SYSTEM_PROMPT_V4_1, SYSTEM_PROMPT_V4_2, SYSTEM_PROMPT_5LANGUAGES_V3, SYSTEM_PROMPT_5LANGUAGES_V4_1, SYSTEM_PROMPT_5LANGUAGES_V4_2, SYSTEM_PROMPT_EAPC_V3, SYSTEM_PROMPT_EAPC_V4_1, SYSTEM_PROMPT_EAPC_V4_2, get_system_prompt_dynamic_language
from lib.core.logging_config import get_configured_logger

# 獲取配置好的日誌器
logger = get_configured_logger(__name__)

class GptTranslate:
    def __init__(self, model_version='gpt-4o'):
        with open(AZURE_CONFIG, 'r') as file:  
            self.config = yaml.safe_load(file)  

        self.config = self.config['gpt_models'][model_version]

        self.client = AzureOpenAI(api_key=self.config['API_KEY'],
                            api_version=self.config['API_VERSION'],
                            azure_endpoint=self.config['ENDPOINT'],
                            azure_deployment=self.config['DEPLOYMENT']
                            )
        
        # Language mapping
        self.lang_names = {
            'zh': 'Traditional Chinese (繁體中文)',
            'en': 'English',
            'de': 'German (Deutsch)',
            'ja': 'Japanese (日本語)',
            'ko': 'Korean (한국어)'
        }

    def _parse_response(self, response_text, expected_languages):
        """Parse and validate response
        
        Args:
            response_text: The response text from GPT
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
            
            # Try to parse JSON
            result = json.loads(json_str)
            
            # Validate response format
            if not isinstance(result, dict):
                logger.warning(" | Response is not a dictionary, ignoring | ")
                return None
            
            # Check required language keys (only check expected languages)
            for lang in expected_languages:
                if lang not in result:
                    logger.warning(f" | GPT() | Missing language key: {lang}, ignoring response | ")
                    return None
            
            # Create response with only expected languages
            formatted_result = {}
            for lang in expected_languages:
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
        
    def translate(self, source_text, source_lang='zh', target_lang='en', prev_text=""):
        """
        Translation method - using new security strategy
        
        :param source_text: Text to be translated
        :param source_lang: Source language code
        :param target_lang: Target language code or list of codes (supports both str and list)
        :param prev_text: Previous context text
        :return: Translation result dictionary (DEFAULT_RESULT format) or "403_Forbidden"
        """
        try:
            if not source_text.strip():
                result = DEFAULT_RESULT.copy()
                return result
            
            # Handle both single string and list for target_lang
            if isinstance(target_lang, list):
                # Multi-language mode
                target_languages = target_lang
            else:
                # Single language mode
                target_languages = [target_lang]
            
            all_languages = [source_lang] + target_languages
            
            # Check text length and handle appropriately
            if len(source_text) > 8000:  # Approximately 2000 tokens
                logger.warning(f" | Text too long ({len(source_text)} chars), truncating | ")
                source_text = source_text[:8000] + "..."

            # Generate dynamic prompt for all target languages
            system_prompt = get_system_prompt_dynamic_language(all_languages, prev_text)
            user_prompt = source_text

            logger.debug(f" | Translating from {source_lang} to {target_languages}: {source_text[:100]}... | ")
            
            # Call GPT
            response = self.client.chat.completions.create(
                model=self.config['DEPLOYMENT'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=4000,
                temperature=0.0, 
            )
            
            response_text = response.choices[0].message.content.strip()
            
            logger.debug(f" | Raw GPT response: {response_text} | ")
            
            # Parse response with expected languages
            parsed_result = self._parse_response(response_text, all_languages)
            
            if parsed_result is None:
                logger.warning(" | GPT() | Failed to parse response, using fallback | ")
                # When parsing fails, return result containing original text and empty translations
                result = {source_lang: source_text}
                for lang in target_languages:
                    result[lang] = ""
                return result

            logger.debug(" | Translation successful | ")
            return parsed_result
            
        except Exception as e:
            logger.error(f" | GPT translation system error: {e} | ")
            return "403_Forbidden"

    def test_gpt_model(self):
        try:
            response = self.client.chat.completions.create(
                model=self.config['DEPLOYMENT'],
                messages=[{"role": "user", "content": "Hello!"}],
                max_tokens=10,   
                temperature=0     
            )
            logger.debug(f" | Model response: {response.choices[0].message.content} | ")
            return True
        except Exception as e:
            logger.error(f" | Model test failed: {e} | ")
            return False