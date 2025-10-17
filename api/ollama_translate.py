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
        """初始化 Ollama 聊天客戶端
 
        Args:
            config_path: 配置文件路徑
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.think = "/no_think\n" if "qwen3" in self.config_path else ""
        
         # 初始化 Ollama 客戶端
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
        """載入配置文件"""
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _clean_think_tags(self, text):
        """清理回應中的 <think> </think> 標籤及其內容
        
        Args:
            text: 原始回應文本
            
        Returns:
            清理後的文本
        """
        if not text:
            return text
        
        cleaned_text = re.sub(r'<think\s*>.*?</think\s*>', '', text, flags=re.DOTALL | re.IGNORECASE)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        logger.debug(f" | Cleaned think tags from response | ")
        return cleaned_text
    
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

    def translate(
        self,
        temperature = 0.0,
        stream = False,
        format = "",
        source_text = "", 
    ):
        """發送聊天請求並獲取回應
 
        Args:
            prompt: 用戶提問內容
            system_prompt: 系統提示詞
            temperature: 溫度參數，控制隨機性
            stream: 是否使用流式輸出
            format: 輸出格式
 
        Returns:
            如果 stream=True，返回流式響應生成器
            如果 stream=False，返回完整響應
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
            
            # 清理 <think> </think> 標籤及其內容
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
   
   
   