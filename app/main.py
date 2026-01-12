from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import json
import logging
import asyncio
from typing import AsyncGenerator

# Import our core modules
from app.core import llm, search, shopping, deep_research, intent, council
from app.api import openai_api

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("free_agent")

app = FastAPI(title="Free Research Agent", description="Zero-cost AI Assistant")

# Mount API Router
app.include_router(openai_api.router, prefix="/v1")

# Mount static for CSS/JS
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Templates
templates = Jinja2Templates(directory="app/templates")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html.j2", {"request": request})


@app.post("/api/chat")
async def chat_endpoint(request: Request):
    """
    Streaming chat endpoint that handles different modes: chat, search, research.
    """
    try:
        data = await request.json()
        message = data.get("message", "")
        mode = data.get("mode", "chat")  # chat, search, research
        history = data.get("history", [])  # List of {role, content}

        logger.info(f"Chat request: mode={mode}, message='{message[:50]}...'")

        return StreamingResponse(
            agent_stream(message, mode, history), media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return {"error": str(e)}


async def agent_stream(
    user_message: str, mode: str, history: list
) -> AsyncGenerator[str, None]:
    """
    Core agent logic generator.
    """
    # 1. Prepare context
    messages = history + [{"role": "user", "content": user_message}]

    # 1.5 Auto-detect mode
    if mode == "auto":
        yield f"data: {json.dumps({'status': 'Detecting intent...'})}\n\n"
        mode = await intent.classify_intent(user_message)
        yield f"data: {json.dumps({'status': f'Mode switched to: {mode}'})}\n\n"

    # 2. Handle modes
    if mode == "council":
        async for chunk in council.run_council_stream(user_message):
            yield chunk
        yield "data: [DONE]\n\n"
        return

    if mode == "deep_research":
        async for chunk in deep_research.deep_research_stream(user_message):
            yield chunk
        yield "data: [DONE]\n\n"
        return

    if mode == "search" or mode == "research":
        # inject search results
        yield f"data: {json.dumps({'status': 'Searching web...'})}\n\n"
        results = await search.search_web(
            user_message, max_results=5 if mode == "search" else 10
        )

        context_str = "\n\n".join(
            [
                f"Source: {r['title']} ({r['href']})\nContent: {r['body']}"
                for r in results
            ]
        )

        if mode == "research":
            yield f"data: {json.dumps({'status': 'Reading content...'})}\n\n"
            # Deep fetch for top 2 results
            for i, r in enumerate(results[:2]):
                full_text = await search.fetch_and_extract(r["href"])
                if full_text:
                    context_str += f"\n\n--- Full Text of {r['title']} ---\n{full_text[:2000]}..."  # Truncate to fit context

        # Shopping Check
        if (
            "price" in user_message.lower()
            or "deal" in user_message.lower()
            or "cost" in user_message.lower()
        ):
            yield f"data: {json.dumps({'status': 'Analyzing prices...'})}\n\n"
            deals = await shopping.analyze_deals(results)
            if deals:
                context_str += "\n\n--- Best Value Analysis ---\n"
                for d in deals[:3]:
                    context_str += f"- {d.name}: {d.currency}{d.price} ({d.unit_amount}{d.unit_type}) = {d.currency}{d.price_per_unit:.4f}/{d.unit_type[0] if d.unit_type else 'u'} [Link]({d.url})\n"

        system_prompt = (
            "You are a helpful research assistant. "
            "Use the provided search results to answer the user's question. "
            "Cite your sources with [Title](URL). "
            "If you cannot verify information, say so."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"User Query: {user_message}\n\nSearch Results:\n{context_str}",
            },
        ]

    elif mode == "chat":
        # Regular chat
        pass

    # 3. Call LLM
    try:
        response_gen = await llm.get_completion("gpt-4o", messages, stream=True)

        async for chunk in response_gen:
            # g4f chunks might be strings or objects depending on provider
            # we assume string here or simple access
            content = str(chunk)
            if content:
                yield f"data: {json.dumps({'token': content})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

    yield "data: [DONE]\n\n"
