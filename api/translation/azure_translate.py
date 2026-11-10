"""
Azure Translator module, specifically designed for text_translate API
Uses Azure Cognitive Services Translator for fast, cost-effective translation
"""

import os
import sys
import uuid
import http.cookiejar
import requests
from requests.adapters import HTTPAdapter
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.config.constant import DEFAULT_RESULT, LANGUAGE_LIST
from lib.core.logging_config import get_logger
from lib.core.glossary_azure import plan_attempts
from lib.core.glossary_manager import glossary_manager

logger = get_logger(__name__)

# One connection pool shared by every AzureTranslate instance and thread: reusing HTTPS connections cuts a
# request from ~40 ms to ~14 ms. Cookies are blocked so the shared session carries no state between requests.
_session = requests.Session()
_session.mount('https://', HTTPAdapter(pool_connections=4, pool_maxsize=16))
_session.mount('http://', HTTPAdapter(pool_connections=4, pool_maxsize=16))
_session.cookies.set_policy(http.cookiejar.DefaultCookiePolicy(allowed_domains=[]))


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

    def _request(self, text, source_lang, target_languages, html=False):
        """
        Send one translate request (plain text, or HTML when `html` is set).

        Returns {internal_lang: translated_text}, or None on any failure (error already logged).
        """
        params = {
            'api-version': self.api_version,
            'to': [self._map_lang(lang) for lang in target_languages]
        }
        if html:
            params['textType'] = 'html'
        # If source_lang is explicitly specified (not "auto"), set 'from'; otherwise let Azure auto-detect
        if source_lang and source_lang != 'auto':
            params['from'] = self._map_lang(source_lang)

        logger.debug(f" | Azure Translate | from {source_lang} to {target_languages}: {text[:100]}... | ")
        try:
            for attempt in (1, 2):
                try:
                    response = _session.post(
                        self.translate_url,
                        params=params,
                        headers=self._get_headers(),
                        json=[{'text': text}],
                        timeout=10
                    )
                    break
                except requests.exceptions.ConnectionError as e:
                    # A pooled connection that the server closed while idle fails on first use: retry once on a new one
                    if attempt == 2 or isinstance(e, requests.exceptions.Timeout):
                        raise
            if response.status_code != 200:
                logger.error(f" | Azure Translate | API error {response.status_code}: {response.text} | ")
                return None

            result_json = response.json()
            if not isinstance(result_json, list) or len(result_json) == 0:
                logger.warning(f" | Azure Translate | Unexpected response format: {result_json} | ")
                return None

            return {self._map_lang_reverse(t.get('to', '')): t.get('text', '')
                    for t in result_json[0].get('translations', [])}
        except requests.exceptions.Timeout:
            logger.error(" | Azure Translate | Request timeout | ")
        except requests.exceptions.ConnectionError:
            logger.error(" | Azure Translate | Connection error | ")
        except Exception as e:
            logger.error(f" | Azure Translate | System error: {e} | ")
        return None

    def _translate_with_glossary(self, text, source_lang, target_lang, attempts):
        """Try each glossary attempt for one target language; return the final text or None if all failed."""
        for name, request_text, is_html, finish in attempts:
            translated = self._request(request_text, source_lang, [target_lang], html=is_html)
            if translated is None:
                continue
            final = finish(translated.get(target_lang, ''))
            if final is not None:
                logger.debug(f" | Azure Translate | glossary applied ({name}) {source_lang}->{target_lang} | ")
                return final
            logger.warning(f" | Azure Translate | glossary {name} check failed {source_lang}->{target_lang}, trying next | ")
        logger.warning(f" | Azure Translate | glossary skipped {source_lang}->{target_lang}, using plain translation | ")
        return None

    def translate(self, source_text, source_lang='zh', target_lang='en', prev_text=""):
        """
        Translation method - using Azure Translator API (batch multi-language)
        
        Glossary terms (if uploaded) are applied per target language: targets with a glossary match are
        translated in their own request, all other targets share one plain request as before.
        
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

            # "auto" is resolved locally from the glossary terms found in the text, so the glossary requests can
            # carry an explicit source language (dynamic dictionary needs it); other targets still use Azure detection
            glossary_source = glossary_manager.resolve_source(source_lang, source_text)
            if source_lang == 'auto' and glossary_source:
                logger.debug(f" | Azure Translate | glossary source resolved: auto -> {glossary_source} | ")
            plans = {lang: plan_attempts(glossary_source, lang, source_text) for lang in target_languages}
            glossary_targets = [lang for lang, attempts in plans.items() if attempts]
            plain_targets = [lang for lang in target_languages if lang not in glossary_targets]
            # A source language resolved from the glossary also replaces Azure's own detection for the plain
            # requests, which returns short sentences untranslated when it cannot detect them
            plain_source = glossary_source or source_lang

            translations = {}
            failed_glossary_targets = []  # glossary could not be applied, these use the plain translation
            plain = {}
            if glossary_targets:
                # The plain batch (targets without a match) runs in parallel with the glossary requests
                with ThreadPoolExecutor(max_workers=len(glossary_targets) + 1) as pool:
                    plain_future = pool.submit(self._request, source_text, plain_source, plain_targets) if plain_targets else None
                    futures = {lang: pool.submit(self._translate_with_glossary, source_text, glossary_source, lang, plans[lang])
                               for lang in glossary_targets}
                    for lang, future in futures.items():
                        final = future.result()
                        if final is None:
                            failed_glossary_targets.append(lang)
                        else:
                            translations[lang] = final
                    if plain_future is not None:
                        plain = plain_future.result()
            elif plain_targets:
                plain = self._request(source_text, plain_source, plain_targets)

            if plain_targets and plain is None:
                return "403_Forbidden"
            translations.update(plain or {})

            if failed_glossary_targets:
                fallback = self._request(source_text, plain_source, failed_glossary_targets)
                if fallback is None:
                    return "403_Forbidden"
                translations.update(fallback)

            parsed_result = {source_lang: source_text}
            parsed_result.update(translations)

            logger.debug(f" | Azure Translate | Translation successful | ")
            return parsed_result

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
