import os
import sys
import json
import logging  
import yaml
import re
from ollama import Client

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.constant import SYSTEM_PROMPT, SYSTEM_PROMPT_V2, LANGUAGE_LIST, DEFAULT_RESULT

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
    
    def _parse_response(self, response_text):
        """Parse and validate response"""
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
            
            # Check required language keys
            for lang in LANGUAGE_LIST:
                if lang not in result:
                    logger.warning(f" | Missing language key: {lang}, ignoring response | ")
                    return None
            
            # Create standard format response
            formatted_result = DEFAULT_RESULT.copy()
            
            # Set translation results for all languages (let GPT decide source language)
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

    def translate(
        self,
        temperature = 0.0,
        stream = False,
        format = "",
        source_text = "", 
    ):
        """Send chat request and get response
 
        Args:
            prompt: User question content
            system_prompt: System prompt
            temperature: Temperature parameter to control randomness
            stream: Whether to use streaming output
            format: Output format
 
        Returns:
            If stream=True, returns streaming response generator
            If stream=False, returns complete response
        """

        messages = [
            {"role": "system", "content": self.think + SYSTEM_PROMPT_V2},
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
        logger.debug(f"OllamaChat Translation result: {decoded}")
        
        # Clean and parse the JSON response
        cleaned_result = self._parse_response(decoded)
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
   
   
   