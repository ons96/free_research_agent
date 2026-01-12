import json
import logging
import re
from dataclasses import dataclass
from typing import List, Optional
from app.core import llm

logger = logging.getLogger(__name__)


@dataclass
class ProductDeal:
    name: str
    price: float
    currency: str
    unit_amount: float
    unit_type: str
    price_per_unit: float
    url: str


async def analyze_deals(search_results: List[dict]) -> List[ProductDeal]:
    """
    Extract product deals from search results using LLM.
    Replaces regex-based heuristic with smart extraction.
    """
    # Prepare context for LLM
    items_text = ""
    for i, r in enumerate(search_results[:8]):  # Limit to top 8 to save context
        items_text += f"Item {i + 1}:\nTitle: {r.get('title')}\nSnippet: {r.get('body')}\nURL: {r.get('href')}\n\n"

    prompt = (
        "You are a smart shopping assistant. Extract product pricing information from the search results below.\n"
        "For each distinct product found, extract:\n"
        "- Name (concise)\n"
        "- Price (numeric only)\n"
        "- Currency (symbol)\n"
        "- Unit Amount (e.g. 500, 1, 2.5). Default to 1 if not specified.\n"
        "- Unit Type (e.g. g, ml, count, lbs, kg). Default to 'count'.\n"
        "- URL (from the source)\n\n"
        "Ignore items without a clear price.\n"
        "Calculate 'price_per_unit' = price / unit_amount.\n"
        "Return ONLY a valid JSON array of objects. No markdown formatting. Example:\n"
        '[{"name": "Apple iPhone 15", "price": 799.0, "currency": "$", "unit_amount": 1, "unit_type": "count", "price_per_unit": 799.0, "url": "..."}]'
        f"\n\nSearch Results:\n{items_text}"
    )

    deals = []
    try:
        # We need a non-streaming completion for JSON
        full_response = ""
        # Use gpt-4o or fast capable model
        async for chunk in llm.get_completion(
            model="gpt-4o", messages=[{"role": "user", "content": prompt}]
        ):
            full_response += chunk

        # Clean response (remove markdown code blocks if present)
        full_response = full_response.strip()
        if full_response.startswith("```"):
            full_response = re.sub(r"^```(?:json)?", "", full_response)
            full_response = re.sub(r"```$", "", full_response)
        full_response = full_response.strip()

        data = json.loads(full_response)
        if isinstance(data, list):
            for item in data:
                try:
                    deals.append(
                        ProductDeal(
                            name=item.get("name", "Unknown"),
                            price=float(item.get("price", 0)),
                            currency=item.get("currency", "$"),
                            unit_amount=float(item.get("unit_amount", 1)),
                            unit_type=item.get("unit_type", "count"),
                            price_per_unit=float(item.get("price_per_unit", 0)),
                            url=item.get("url", ""),
                        )
                    )
                except (ValueError, TypeError):
                    continue

    except Exception as e:
        logger.error(f"LLM shopping extraction failed: {e}")

    # Sort by price per unit (cheapest first)
    deals.sort(key=lambda x: x.price_per_unit if x.price_per_unit > 0 else float("inf"))
    return deals
