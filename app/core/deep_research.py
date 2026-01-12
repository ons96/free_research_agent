import logging
import json
import asyncio
from typing import AsyncGenerator
from app.core import llm, search

logger = logging.getLogger(__name__)


async def deep_research_stream(
    query: str, max_depth: int = 2
) -> AsyncGenerator[str, None]:
    """
    Performs multi-step deep research and yields status/content updates.
    Yields SSE-formatted strings: "data: {...}\n\n" or raw text chunks for the final answer.
    """
    context = []
    current_query = query

    yield f"data: {json.dumps({'status': 'Starting deep research...'})}\n\n"

    for depth in range(max_depth):
        # 1. Search
        yield f"data: {json.dumps({'status': f'Searching (Depth {depth + 1}/{max_depth}): {current_query}'})}\n\n"
        results = await search.search_web(current_query, max_results=4)

        if not results:
            yield f"data: {json.dumps({'status': 'No results found, stopping research loop.'})}\n\n"
            break

        # 2. Extract & Summarize (Parallel fetch)
        yield f"data: {json.dumps({'status': 'Reading sources...'})}\n\n"

        # Limit deep fetch to top 2 to save time/bandwidth
        tasks = [search.fetch_and_extract(r["href"]) for r in results[:2]]
        page_contents = await asyncio.gather(*tasks, return_exceptions=True)

        step_context = ""
        found_useful_info = False
        for i, content in enumerate(page_contents):
            if isinstance(content, str) and content:
                found_useful_info = True
                source_title = results[i]["title"]
                source_url = results[i]["href"]
                step_context += f"Source: {source_title} ({source_url})\nContent: {content[:2000]}\n\n"  # Increased limit

        if not found_useful_info:
            # Fallback to snippets if fetch failed
            for r in results:
                step_context += (
                    f"Source: {r['title']} ({r['href']})\nSnippet: {r['body']}\n\n"
                )

        context.append(step_context)

        # 3. Analyze & Plan Next Step (if not last step)
        if depth < max_depth - 1:
            yield f"data: {json.dumps({'status': 'Analyzing findings & planning next step...'})}\n\n"

            prompt = (
                f"Original Goal: {query}\n"
                f"Current Findings:\n{step_context}\n\n"
                "Based on the findings, what is the single most important follow-up search query needed to answer the original goal? "
                "Respond with ONLY the search query. If enough information is gathered, respond with 'DONE'."
            )

            # Use fast model for planning
            next_query = ""
            # We assume get_completion yields strings
            async for chunk in llm.get_completion(
                model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
            ):
                next_query += chunk

            next_query = next_query.strip().strip('"')
            logger.info(f"Deep Research Planner suggested: {next_query}")

            if "DONE" in next_query.upper() or not next_query:
                yield f"data: {json.dumps({'status': 'Sufficient information gathered.'})}\n\n"
                break

            current_query = next_query

    # 4. Final Synthesis
    yield f"data: {json.dumps({'status': 'Synthesizing final comprehensive answer...'})}\n\n"

    full_context = "\n---\n".join(context)
    final_prompt = (
        f"Goal: {query}\n\n"
        f"Research Context:\n{full_context}\n\n"
        "Provide a comprehensive, detailed answer to the goal based on the research context. "
        "Cite sources inline [Title](URL) where appropriate. Use Markdown."
    )

    # Use smart model for synthesis
    async for chunk in llm.get_completion(
        model="gpt-4o", messages=[{"role": "user", "content": final_prompt}]
    ):
        yield chunk
