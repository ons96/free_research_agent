from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any
import json
import time
import uuid
import logging
from app.core import llm

logger = logging.getLogger(__name__)
router = APIRouter()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "gpt-3.5-turbo"
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None

@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    """
    req_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    if not request.stream:
        # TODO: Implement non-streaming collection
        return JSONResponse(status_code=400, content={"error": "Only streaming is supported in this version"})

    async def event_generator():
        try:
            # Convert pydantic models to dicts
            messages = [m.model_dump() for m in request.messages]
            
            async for chunk in llm.get_completion(request.model, messages, stream=True):
                data = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(data)}\n\n"
            
            # Final done message
            final_data = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }
                    ]
                }
            yield f"data: {json.dumps(final_data)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Generate error: {e}")
            # In SSE, we can't easily change status code once started, 
            # but we can try to send an error block if headers haven't flushed?
            # Usually better to log and just stop.
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
