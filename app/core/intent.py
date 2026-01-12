import logging
from app.core import llm

logger = logging.getLogger(__name__)


async def classify_intent(message: str) -> str:
    """
    Classify user message into: 'chat', 'search', 'deep_research'.
    """
    prompt = (
        "Classify the following user message into one of these categories:\n"
        "- chat: Casual conversation, coding help, creative writing.\n"
        "- search: Fact retrieval, weather, quick lookups, shopping.\n"
        "- deep_research: Complex topics, extensive reports, multi-step investigation, 'find everything about'.\n\n"
        f"Message: {message}\n\n"
        "Return ONLY the category name (chat, search, deep_research)."
    )

    intent = "chat"  # Default
    try:
        response = ""
        # Use fast model
        async for chunk in llm.get_completion(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
        ):
            response += chunk

        cleaned = response.strip().lower()
        if "deep_research" in cleaned:
            intent = "deep_research"
        elif "search" in cleaned:
            intent = "search"
        else:
            intent = "chat"

        logger.info(f"Intent classified: {intent} (from '{cleaned}')")
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")

    return intent
