from abc import ABC, abstractmethod
from typing import List, Dict, Any, AsyncGenerator, Optional
import g4f
import logging
import json
import httpx
import asyncio

logger = logging.getLogger(__name__)

class BaseProvider(ABC):
    def __init__(self, name: str, models: List[str], config: Dict[str, Any]):
        self.name = name
        self.models = models
        self.config = config
        self.is_active = True
        self.failure_count = 0

    @abstractmethod
    async def stream_chat(self, model: str, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        pass

class G4FProvider(BaseProvider):
    async def stream_chat(self, model: str, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        try:
            response = g4f.ChatCompletion.create(
                model=model,
                messages=messages,
                stream=True
            )
            for chunk in response:
                if chunk:
                    yield str(chunk)
                    await asyncio.sleep(0) # Yield control
        except Exception as e:
            logger.error(f"G4F Provider {self.name} error: {e}")
            raise e

class OpenAIProvider(BaseProvider):
    def __init__(self, name: str, models: List[str], config: Dict[str, Any]):
        super().__init__(name, models, config)
        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        self.api_key = config.get("api_key", "")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def stream_chat(self, model: str, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "stream": True
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", url, headers=self.headers, json=payload) as response:
                    if response.status_code != 200:
                        error_text = await response.read()
                        raise Exception(f"Provider {self.name} returned {response.status_code}: {error_text}")

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            logger.error(f"OpenAI Provider {self.name} error: {e}")
            raise e
