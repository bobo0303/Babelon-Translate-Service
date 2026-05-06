"""
Azure Translator module, specifically designed for text_translate API
Uses Azure Cognitive Services Translator for fast, cost-effective translation
"""

import os
import sys
import uuid
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.config.constant import DEFAULT_RESULT, LANGUAGE_LIST
from lib.core.logging_config import get_logger

logger = get_logger(__name__)


class AzureTranslate:
    def __init__(self):
        self.key = os.getenv("AZURE_TRANSLATOR_KEY", "")
        self.endpoint = os.getenv("AZURE_TRANSLATOR_ENDPOINT", "https://api.cognitive.microsofttranslator.com")
        self.location = os.getenv("AZURE_TRANSLATOR_LOCATION", "japaneast")
        self.api_version = "3.0"
        self.translate_url = f"{self.endpoint}/translate"
        
        # Azure Translator uses different language codes for Chinese
        self.lang_map = {
            'zh': 'zh-Hant',
            'en': 'en',
            'ja': 'ja',
            'ko': 'ko',
            'de': 'de'
        }
        # Reverse mapping for response parsing
        self.lang_map_reverse = {v: k for k, v in self.lang_map.items()}

    def _get_headers(self):
        """Get request headers."""
        return {
            'Ocp-Apim-Subscription-Key': self.key,
            'Ocp-Apim-Subscription-Region': self.location,
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }

    def _map_lang(self, lang_code):
        """Map internal language code to Azure Translator language code."""
        return self.lang_map.get(lang_code, lang_code)

    def _map_lang_reverse(self, azure_code):
        """Map Azure Translator language code back to internal code."""
        return self.lang_map_reverse.get(azure_code, azure_code)

    def translate(self, source_text, source_lang='zh', target_lang='en', prev_text=""):
        """
        Translation method - using Azure Translator API (batch multi-language)
        
        :param source_text: Text to be translated
        :param source_lang: Source language code
        :param target_lang: Target language code or list of codes (supports both str and list)
        :param prev_text: Previous context text (not used by Azure Translator, kept for API compatibility)
        :return: Translation result dictionary (DEFAULT_RESULT format) or "403_Forbidden"
        """
        try:
            if not source_text.strip():
                return DEFAULT_RESULT.copy()

            # Handle both single string and list for target_lang
            if isinstance(target_lang, list):
                target_languages = target_lang
            else:
                target_languages = [target_lang]

            # Map language codes to Azure format
            azure_source = self._map_lang(source_lang)
            azure_targets = [self._map_lang(lang) for lang in target_languages]

            # Build request
            params = {
                'api-version': self.api_version,
                'from': azure_source,
                'to': azure_targets
            }
            body = [{'text': source_text}]

            logger.debug(f" | Azure Translate | from {source_lang} to {target_languages}: {source_text[:100]}... | ")

            # Call Azure Translator API
            response = requests.post(
                self.translate_url,
                params=params,
                headers=self._get_headers(),
                json=body,
                timeout=10
            )

            if response.status_code != 200:
                logger.error(f" | Azure Translate | API error {response.status_code}: {response.text} | ")
                return "403_Forbidden"

            result_json = response.json()

            # Parse response
            if not isinstance(result_json, list) or len(result_json) == 0:
                logger.warning(f" | Azure Translate | Unexpected response format: {result_json} | ")
                return "403_Forbidden"

            translations = result_json[0].get('translations', [])

            # Build result dictionary
            parsed_result = {source_lang: source_text}
            for t in translations:
                azure_lang = t.get('to', '')
                translated_text = t.get('text', '')
                internal_lang = self._map_lang_reverse(azure_lang)
                parsed_result[internal_lang] = translated_text

            logger.debug(f" | Azure Translate | Translation successful | ")
            return parsed_result

        except requests.exceptions.Timeout:
            logger.error(" | Azure Translate | Request timeout | ")
            return "403_Forbidden"
        except requests.exceptions.ConnectionError:
            logger.error(" | Azure Translate | Connection error | ")
            return "403_Forbidden"
        except Exception as e:
            logger.error(f" | Azure Translate | System error: {e} | ")
            return "403_Forbidden"

    def test_azure_model(self):
        """Test if Azure Translator API is accessible."""
        try:
            params = {
                'api-version': self.api_version,
                'from': 'zh-Hant',
                'to': 'en'
            }
            body = [{'text': '測試'}]
            response = requests.post(
                self.translate_url,
                params=params,
                headers=self._get_headers(),
                json=body,
                timeout=10
            )
            if response.status_code == 200:
                logger.debug(f" | Azure Translate | Test response: {response.json()} | ")
                return True
            else:
                logger.error(f" | Azure Translate | Test failed with status {response.status_code} | ")
                return False
        except Exception as e:
            logger.error(f" | Azure Translate | Test failed: {e} | ")
            return False
