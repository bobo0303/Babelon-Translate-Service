import os
import sys
import logging  
import yaml
import re
from ollama import Client

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.constant import SYSTEM_PROMPT

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
            {"role": "system", "content": self.think + SYSTEM_PROMPT},
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
            
        return decoded
        
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
   
   
   