import os
import sys
import json
import logging  
import yaml
import re
from ollama import Client

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.config.constant import SYSTEM_PROMPT_V3, SYSTEM_PROMPT_V4_1, SYSTEM_PROMPT_V4_2, SYSTEM_PROMPT_5LANGUAGES_V3, SYSTEM_PROMPT_5LANGUAGES_V4_1, SYSTEM_PROMPT_5LANGUAGES_V4_2, LANGUAGE_LIST, DEFAULT_RESULT, SYSTEM_PROMPT_EAPC_V3, SYSTEM_PROMPT_EAPC_V4_1, SYSTEM_PROMPT_EAPC_V4_2, get_system_prompt_dynamic_language

logger = logging.getLogger(__name__)
 
class OllamaChat:
    def __init__(self, config_path):
        """Initialize Ollama chat client
 
        Args:
            config_path: Configuration file path
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.think = "/no_think\n" if "qwen3" in self.config_path else ""
        
         # Initialize Ollama client
        try:
            self.client = Client(host=self.config["HOST"])
            
            self.client.chat(
                model=self.config["MODEL"],
                messages="",
                format="",
                stream=False,
                keep_alive=-1
            )
        except Exception as e:
            logger.error(f" | initial ollama error: {e} (Maybe ollama serve not started) | ")
            raise e
 
    def _load_config(self):
        """Load configuration file"""
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _clean_think_tags(self, text):
        """Clean <think> </think> tags and their content from response
        
        Args:
            text: Original response text
            
        Returns:
            Cleaned text
        """
        if not text:
            return text
        
        cleaned_text = re.sub(r'<think\s*>.*?</think\s*>', '', text, flags=re.DOTALL | re.IGNORECASE)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        logger.debug(f" | Cleaned think tags from response | ")
        return cleaned_text
    
    def _parse_response(self, response_text, expected_languages):
        """Parse and validate response
        
        Args:
            response_text: The response text from Ollama
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
                    logger.warning(f" | Ollama() | Missing language key: {lang}, ignoring response | ")
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

    def translate(
        self,
        source_text="",
        source_lang='zh', 
        target_lang='en',
        prev_text="",
        temperature=0.0,
        stream=False,
        format=""
    ):
        """Send chat request and get response
 
        Args:
            temperature: Temperature parameter to control randomness
            stream: Whether to use streaming output
            format: Output format
            source_text: Text to translate
            source_lang: Source language code
            target_lang: Target language code or list of codes (supports both str and list)
            prev_text: Previous context text
 
        Returns:
            Dict with translation results: {lang_code: translated_text}
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

        messages = [
            {"role": "system", "content": self.think + system_prompt},
            {"role": "user", "content": source_text},
        ]
        try:
            response = self.client.chat(
                model=self.config["MODEL"],
                messages=messages,
                format=format,
                options={"temperature": temperature},
                stream=stream,
                keep_alive=-1
            )

            decoded = response.message.content
            
            # Clean <think> </think> tags and their content
            if decoded:
                decoded = self._clean_think_tags(decoded)
                
        except Exception as e:
            logger.error(f" | ollama Error: {e} | ")
            decoded = None
        logger.debug(f" | OllamaChat Translation result: {decoded} | ")
        
        # Clean and parse the JSON response with expected languages
        if decoded:
            cleaned_result = self._parse_response(decoded, all_languages)
        else:
            cleaned_result = None
        
        # Handle failure: return default result
        if cleaned_result is None:
            logger.warning(f" | Ollama translation failed, returning default result | ")
            cleaned_result = {source_lang: source_text}
            for lang in target_languages:
                cleaned_result[lang] = ""
        
        return cleaned_result
        
    def close(self):
        self.client.chat(
            model=self.config["MODEL"],
            messages="",
            format="",
            options={"temperature": 0.0},
            stream=False,
            keep_alive=0
        )        
        logger.debug(f" | OllamaChat client closed. | ")
   
   
   