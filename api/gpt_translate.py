"""
安全的 GPT 翻譯模組，專門為 text_translate API 設計
防止 prompt injection 攻擊，確保穩定的翻譯輸出格式
"""

import os
import sys
import yaml  
import json
import logging  
from openai import AzureOpenAI
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.constant import AZURE_CONFIG, LANGUAGE_LIST, DEFAULT_RESULT, SYSTEM_PROMPT, SYSTEM_PROMPT_V2

logger = logging.getLogger(__name__)

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
        
        # 語言映射
        self.lang_names = {
            'zh': 'Traditional Chinese (繁體中文)',
            'en': 'English',
            'de': 'German (Deutsch)'
        }

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
        
    def translate(self, source_text, source_lang):
        """
        翻譯方法 - 使用新的安全策略
        
        :param source_text: 要翻譯的文本
        :param source_lang: 源語言 ('zh', 'en', 'de') - 僅用於日誌記錄
        :return: 翻譯結果字典 (DEFAULT_RESULT 格式) 或 "403_Forbidden"
        """
        try:
            if not source_text.strip():
                result = DEFAULT_RESULT.copy()
                return result
            
            # 檢查文本長度並適當處理
            if len(source_text) > 8000:  # 約 2000 tokens
                logger.warning(f" | Text too long ({len(source_text)} chars), truncating | ")
                source_text = source_text[:8000] + "..."
            
            system_prompt = SYSTEM_PROMPT_V2
            user_prompt = source_text
            
            logger.debug(f" | Translating from {source_lang}: {source_text[:100]}... | ")
            
            # 調用 GPT
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
            
            # 解析響應
            parsed_result = self._parse_response(response_text)
            
            if parsed_result is None:
                logger.warning(" | Failed to parse response, using fallback | ")
                # 當解析失敗時，返回包含原文的結果
                result = DEFAULT_RESULT.copy()
                result[source_lang] = source_text
                return result

            logger.debug(" | Translation successful | ")
            return parsed_result
            
        except Exception as e:
            logger.error(f" | GPT translation system error: {e} | ")
            return "403_Forbidden"