"""
Safe Claude translation module for Azure Anthropic Foundry, specifically designed for text_translate API
Prevents prompt injection attacks and ensures stable translation output format
"""

import os
import sys
import json
from anthropic import AnthropicFoundry
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.config.constant import CLAUDE_CONFIG, LANGUAGE_LIST, DEFAULT_RESULT, get_system_prompt_dynamic_language
from lib.core.logging_config import get_logger

# 獲取日誌器
logger = get_logger(__name__)

class ClaudeTranslate:
    def __init__(self, model_version='claude-haiku-4-5'):
        """
        Initialize Claude translation client via Azure Anthropic Foundry
        
        Args:
            model_version: Claude model to use ('claude-sonnet-4-5' or 'claude-haiku-4-5')
        """
        # Load config from environment variables via CLAUDE_CONFIG
        self.config = CLAUDE_CONFIG.get(model_version)
        if not self.config:
            raise ValueError(f"Unknown model version: {model_version}")

        # Initialize Azure Anthropic Foundry client
        self.client = AnthropicFoundry(
            api_key=self.config['API_KEY'],
            base_url=self.config['ENDPOINT']
        )
        self.model = self.config['DEPLOYMENT']
        
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
            response_text: The response text from Claude
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
            
            # Fix common Claude errors: {"json":[ -> {"json":{
            json_str = re.sub(r'(\{"json"\s*:\s*)\[', r'\1{', json_str)
            json_str = re.sub(r'\]\s*\}$', r'}}', json_str)
            
            # Try to parse JSON
            result = json.loads(json_str)
            
            # Validate response format
            if not isinstance(result, dict):
                logger.warning(" | Claude() | Response is not a dictionary, ignoring | ")
                logger.warning(f" | Claude() | Response type: {type(result)}, Content: {result} | ")
                logger.warning(f" | Claude() | Raw response: {response_text} | ")
                return None
            
            # Post-processing: Handle Claude wrapping response in extra "json" key
            # Example: {"json": {"zh": "...", "en": "..."}} -> {"zh": "...", "en": "..."}
            if len(result) == 1 and "json" in result and isinstance(result["json"], dict):
                logger.debug(" | Claude() | Detected wrapped response with 'json' key, unwrapping | ")
                result = result["json"]
            
            # Check required language keys (only check expected languages)
            for lang in expected_languages:
                if lang not in result:
                    logger.warning(f" | Claude() | Missing language key: {lang}, ignoring response | ")
                    logger.warning(f" | Claude() | Expected keys: {expected_languages}, Got keys: {list(result.keys())} | ")
                    logger.warning(f" | Claude() | Raw response: {response_text} | ")
                    return None
            
            # Create response with only expected languages
            formatted_result = {}
            for lang in expected_languages:
                translated_text = result.get(lang, "").strip()
                formatted_result[lang] = translated_text
            
            return formatted_result
            
        except json.JSONDecodeError as e:
            logger.warning(f" | Claude() | Failed to parse JSON response: {e} | ")
            logger.warning(f" | Claude() | Raw response: {response_text} | ")
            return None
        except Exception as e:
            logger.error(f" | Claude() | Error parsing response: {e} | ")
            return "403_Forbidden"
        
    def translate(self, source_text, source_lang='zh', target_lang='en', prev_text=""):
        """
        Translation method - using Azure Anthropic Foundry
        
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
            
            # Check text length and handle appropriately
            if len(source_text) > 8000:  # Approximately 2000 tokens
                logger.warning(f" | Claude() | Text too long ({len(source_text)} chars), truncating | ")
                source_text = source_text[:8000] + "..."

            # Always include source language in prompt for context
            all_languages = [source_lang] + target_languages
            expected_languages = target_languages  # Only validate target languages by default

            # Generate dynamic prompt (always includes source language for context)
            system_prompt = get_system_prompt_dynamic_language(all_languages, prev_text)
            user_prompt = source_text

            logger.debug(f" | Claude({self.model}) | Translating from {source_lang} to {target_languages}: {source_text[:100]}... | ")
            
            # Call Claude API via Azure Anthropic Foundry
            # Note: Claude uses 'system' parameter separately, not in messages
            message = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                temperature=0.0,
                system=system_prompt,  # System prompt is separate in Claude API
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Extract response text (different structure from OpenAI)
            response_text = message.content[0].text.strip()
            
            logger.debug(f" | Claude() | Raw response: {response_text[:200]}... | ")
            
            # Parse response with expected languages
            parsed_result = self._parse_response(response_text, expected_languages)
            
            if parsed_result is None:
                logger.warning(" | Claude() | Failed to parse response, using fallback | ")
                # When parsing fails, return result containing original text and empty translations
                result = {source_lang: source_text}
                for lang in target_languages:
                    result[lang] = ""
                return result

            # Always ensure source language is included in result
            if source_lang not in parsed_result:
                parsed_result[source_lang] = source_text

            logger.debug(f" | Claude({self.model}) | Translation successful | ")
            return parsed_result
            
        except Exception as e:
            logger.error(f" | Claude({self.model}) | Translation system error: {e} | ")
            return "403_Forbidden"

    def test_claude_model(self):
        """
        Test Claude model with a simple translation
        
        :return: True if successful, False otherwise
        """
        try:
            logger.info(f" | Testing Claude model: {self.model} | ")
            
            test_text = "Hello, world!"
            result = self.translate(
                source_text=test_text,
                source_lang='en',
                target_lang='zh'
            )
            
            if result and 'zh' in result and result['zh']:
                logger.info(f" | Claude model test successful | ")
                logger.info(f" | Test translation: '{test_text}' -> '{result['zh']}' | ")
                return True
            else:
                logger.error(f" | Claude model test failed: no translation result | ")
                return False
                
        except Exception as e:
            logger.error(f" | Claude model test failed: {e} | ")
            return False


def main():
    """Test function"""
    print("=" * 60)
    print("Testing Claude Translation Module (Azure Anthropic Foundry)")
    print("=" * 60)
    
    # Test both Claude models
    models = ['claude-haiku-4-5', 'claude-sonnet-4-5']
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"Testing Model: {model}")
        print(f"{'='*60}")
        
        try:
            translator = ClaudeTranslate(model_version=model)
            
            # Test 1: Chinese to English
            print("\n[Test 1] Chinese to English")
            print("-" * 60)
            result = translator.translate(
                source_text="今天天氣很好，我們一起去公園散步吧！",
                source_lang='zh',
                target_lang='en'
            )
            print(f"Chinese: 今天天氣很好，我們一起去公園散步吧！")
            print(f"English: {result.get('en', 'FAILED')}")
            
            # Test 2: English to Chinese
            print("\n[Test 2] English to Chinese")
            print("-" * 60)
            result = translator.translate(
                source_text="Artificial intelligence is transforming the way we work and live.",
                source_lang='en',
                target_lang='zh'
            )
            print(f"English: Artificial intelligence is transforming the way we work and live.")
            print(f"Chinese: {result.get('zh', 'FAILED')}")
            
            # Test 3: Multi-language translation
            print("\n[Test 3] Multi-language Translation")
            print("-" * 60)
            result = translator.translate(
                source_text="歡迎來到台灣，這裡有美食和美景。",
                source_lang='zh',
                target_lang=['en', 'de', 'ja']
            )
            print(f"Chinese: 歡迎來到台灣，這裡有美食和美景。")
            print(f"English: {result.get('en', 'FAILED')}")
            print(f"German: {result.get('de', 'FAILED')}")
            print(f"Japanese: {result.get('ja', 'FAILED')}")
            
        except Exception as e:
            print(f"\n✗ Model initialization failed: {str(e)}")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
