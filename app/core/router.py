import random
import logging
import asyncio
from typing import List, Dict, Any, AsyncGenerator, Optional
from app.core.providers import BaseProvider, G4FProvider, OpenAIProvider
from app.core.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class ProviderRouter:
    def __init__(self):
        self.providers: List[BaseProvider] = []
        self._load_providers()
        self.current_index = 0

    def _load_providers(self):
        loader = ConfigLoader()
        configs = loader.load_providers()
        
        for cfg in configs:
            ptype = cfg.get("type", "g4f")
            name = cfg.get("name", "unnamed")
            models = cfg.get("models", [])
            
            if ptype == "g4f":
                self.providers.append(G4FProvider(name, models, cfg))
            elif ptype == "openai":
                self.providers.append(OpenAIProvider(name, models, cfg))
                
        logger.info(f"Router initialized with {len(self.providers)} providers")

    def get_provider(self, model: str) -> Optional[BaseProvider]:
        """
        Get a healthy provider that supports the model.
        Simple Round-Robin for now.
        """
        if not self.providers:
            return None
            
        # Try finding a provider starting from current_index
        for _ in range(len(self.providers)):
            self.current_index = (self.current_index + 1) % len(self.providers)
            provider = self.providers[self.current_index]
            
            # Simple health check (failure count)
            if provider.failure_count < 3: # Reset manually or via time logic later
                # Check if model supported (wildcard '*' or explicit list)
                if "*" in provider.models or model in provider.models:
                    return provider
                    
        return None # All failed or no match

    def report_failure(self, provider: BaseProvider):
        provider.failure_count += 1
        logger.warning(f"Provider {provider.name} reported failure. Count: {provider.failure_count}")

    def report_success(self, provider: BaseProvider):
        provider.failure_count = 0

    async def stream_chat(self, model: str, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """
        Stream chat with automatic failover/rotation.
        """
        retries = 3
        attempt_errors = []

        for attempt in range(retries):
            provider = self.get_provider(model)
            if not provider:
                raise Exception("No healthy providers available for this model.")

            logger.info(f"Routing to provider: {provider.name} (Attempt {attempt+1})")
            
            try:
                # Use "yield from" equivalent
                async for chunk in provider.stream_chat(model, messages):
                    self.report_success(provider)
                    yield chunk
                return # Success
                
            except Exception as e:
                logger.error(f"Error with {provider.name}: {e}")
                self.report_failure(provider)
                attempt_errors.append(f"{provider.name}: {str(e)}")
                continue # Retry loop will pick next provider

        raise Exception(f"All retries failed. Errors: {attempt_errors}")
