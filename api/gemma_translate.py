
# GEMMA 4B (https://huggingface.co/google/gemma-3-4b-it)

import os
import sys
import logging  
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration  
from huggingface_hub import login

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.constant import SYSTEM_PROMPT, GEMMA_4B_IT

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
                logger.info(f"Already authenticated with HuggingFace as: {user_info['name']}")
                return
            except Exception:
                # Not logged in yet, continue to token-based login
                pass
            
            # Try to get token from environment variable
            hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN')
            
            if not hf_token:
                # For development/testing purposes, provide clear guidance
                logger.error("HUGGINGFACE_HUB_TOKEN environment variable not found!")
                logger.error("Please set it with: export HUGGINGFACE_HUB_TOKEN='your_token_here'")
                logger.error("Or run with: docker exec -e HUGGINGFACE_HUB_TOKEN='your_token' ...")
                logger.error("Or use: hf auth login (which you may have already done)")
                raise ValueError("HuggingFace token is required to access gated models")
            
            # Use add_to_git_credential=False to avoid modifying git config
            login(token=hf_token, add_to_git_credential=False)
            logger.info("Successfully authenticated with HuggingFace using token")
            
        except Exception as e:
            logger.error(f"Failed to authenticate with HuggingFace: {e}")
            raise RuntimeError(f"HuggingFace authentication failed: {e}") from e
    
    def translate(self, source_text):
        """
        Translate text from source language to supported target languages.
        
        Args:
            source_text (str): Text to translate
            
        Returns:
            dict or str: Translation result
        """
        try:
            messages = [
                { 
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}] 
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
                generation = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
                generation = generation[0][input_len:]

            decoded = self.processor.decode(generation, skip_special_tokens=True)
            return decoded
            
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            return None
        
        