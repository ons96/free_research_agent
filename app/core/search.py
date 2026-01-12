import asyncio
from duckduckgo_search import DDGS
import httpx
import trafilatura
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

async def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search the web using DuckDuckGo (lite/keyless).
    """
    logger.info(f"Searching for: {query}")
    results = []
    try:
        # DDGS is synchronous but fast enough; wrap in executor if needed for high load
        # For simplicity in this 'free' agent, we run it directly or in a thread
        with DDGS() as ddgs:
             # text() returns generator
             # region='wt-wt', safesearch='moderate', timelimit=None
             ddg_results = list(ddgs.text(query, max_results=max_results))
             
        for r in ddg_results:
            results.append({
                "title": r.get("title", ""),
                "href": r.get("href", ""),
                "body": r.get("body", "")
            })
    except Exception as e:
        logger.error(f"Search failed: {e}")
        
    return results

async def fetch_and_extract(url: str) -> Optional[str]:
    """
    Fetch a URL and extract its main text content.
    """
    logger.info(f"Fetching: {url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            
            # Extract
            text = trafilatura.extract(resp.text)
            return text
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None
