import logging
from typing import List, Dict, Any, AsyncGenerator
from app.core.router import ProviderRouter

logger = logging.getLogger(__name__)

# Single global instance of Router
_router = ProviderRouter()

async def get_completion(
    model: str,
    messages: List[Dict[str, str]],
    stream: bool = True
) -> AsyncGenerator[str, None]:
    """
    Get completion using the ProviderRouter.
    """
    # Simply delegate to the router
    if not stream:
         # To support non-streaming, we'd need to collect chunks
         # For now, we only implement streaming as per constraints
         pass  

    async for chunk in _router.stream_chat(model, messages):
        yield chunk
