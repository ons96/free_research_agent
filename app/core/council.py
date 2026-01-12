import logging
import json
import asyncio
import pandas as pd
import os
from typing import AsyncGenerator, List, Dict
from app.core import llm

logger = logging.getLogger(__name__)

# Relative paths from app/core/
LEADERBOARD_PATH = "../../../llm-leaderboard/llm_aggregated_leaderboard.csv"
STATUS_PATH = "../../../llm-provider-tracker/provider_status.json"

# Map Model Families to Providers (simplified)
# This helps us guess if a model is "available" based on provider status
MODEL_PROVIDER_MAP = {
    "llama": ["Groq", "Cerebras", "OpenRouter", "G4F Local"],
    "mixtral": ["Groq", "OpenRouter", "G4F Local"],
    "gemma": ["Groq", "OpenRouter"],
    "claude": ["OpenRouter", "G4F Local"],
    "gpt": ["OpenRouter", "G4F Local"],
    "deepseek": ["OpenRouter", "G4F Local", "Groq"],
}


def get_available_providers() -> List[str]:
    """Reads provider_status.json and returns list of UP providers."""
    up_providers = []
    try:
        # Resolve absolute path for safety
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, STATUS_PATH)

        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                for p in data:
                    if p.get("status") == "UP":
                        up_providers.append(p.get("name"))
        else:
            # Fallback if tracker hasn't run
            logger.warning("Provider status not found, assuming G4F/OpenRouter UP")
            return ["G4F Local", "OpenRouter"]
    except Exception as e:
        logger.error(f"Error reading provider status: {e}")
        return ["G4F Local", "OpenRouter"]  # Safe fallback

    return up_providers


def get_best_models(category: str = "overall", n: int = 3) -> List[Dict[str, str]]:
    """
    Selects top N models from leaderboard that are likely available.
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, LEADERBOARD_PATH)

        if not os.path.exists(path):
            logger.warning(f"Leaderboard CSV not found at {path}. Using defaults.")
            return _get_default_models()

        df = pd.read_csv(path)

        # Filter for category if exists, else fallback to overall
        target_col = category if category in df.columns else "overall"

        # Sort by score desc (handle NaN)
        df = df.sort_values(by=target_col, ascending=False).dropna(subset=[target_col])

        available_providers = get_available_providers()
        selected_models = []

        for _, row in df.iterrows():
            model_name = str(row["model"])
            score = row[target_col]

            # Check availability
            is_available = False
            model_lower = model_name.lower()

            for family, providers in MODEL_PROVIDER_MAP.items():
                if family in model_lower:
                    # Check if ANY of the providers for this family are UP
                    # Fuzzy match provider names
                    for p_up in available_providers:
                        if any(prov.lower() in p_up.lower() for prov in providers):
                            is_available = True
                            break
                if is_available:
                    break

            # If we can't map it, assume G4F can handle popular ones (high score usually means popular)
            if not is_available and score > 50:
                is_available = True  # Optimistic fallback for top models

            if is_available:
                role = "General Expert"
                if "coding" in category:
                    role = "Senior Engineer"
                elif "reasoning" in category:
                    role = "Logician"
                elif "uncensored" in category:
                    role = "Unfiltered Analyst"

                selected_models.append(
                    {"model": model_name, "role": f"{role} (Score: {score:.1f})"}
                )

            if len(selected_models) >= n:
                break

        if not selected_models:
            return _get_default_models()

        return selected_models

    except Exception as e:
        logger.error(f"Error selecting models: {e}")
        return _get_default_models()


def _get_default_models():
    return [
        {"model": "gpt-4o", "role": "Fallback Expert 1"},
        {"model": "claude-3-opus", "role": "Fallback Expert 2"},
        {"model": "llama-3-70b-instruct", "role": "Fallback Expert 3"},
    ]


async def run_council_stream(query: str) -> AsyncGenerator[str, None]:
    """
    Queries multiple LLM experts and synthesizes their responses.
    """
    # Detect category intent (simple heuristic or use intent classifier)
    category = "overall"
    q_lower = query.lower()
    if "code" in q_lower or "function" in q_lower or "script" in q_lower:
        category = "coding"
    elif "calculate" in q_lower or "solve" in q_lower or "math" in q_lower:
        category = "math"
    elif (
        "sex" in q_lower
        or "nsfw" in q_lower
        or "uncensored" in q_lower
        or "jailbreak" in q_lower
    ):
        category = "uncensored"
    elif "reason" in q_lower or "logic" in q_lower:
        category = "reasoning"

    yield f"data: {json.dumps({'status': f'Selecting best models for {category} from leaderboard...'})}\n\n"

    experts = get_best_models(category=category, n=3)

    expert_names = ", ".join([e["model"] for e in experts])
    yield f"data: {json.dumps({'status': f'Council convened: {expert_names}'})}\n\n"

    async def query_expert(expert):
        model = expert["model"]
        role = expert["role"]
        try:
            full_resp = ""
            messages = [{"role": "user", "content": query}]
            # We rely on the router finding a provider that supports '*' (G4F) or exact match
            async for chunk in llm.get_completion(model, messages, stream=True):
                full_resp += chunk
            return f"### Expert: {role} ({model})\n{full_resp}\n"
        except Exception as e:
            return f"### Expert: {role} ({model})\n[Failed to respond: {e}]\n"

    # Parallel execution
    yield f"data: {json.dumps({'status': 'Consulting experts in parallel...'})}\n\n"

    tasks = [query_expert(e) for e in experts]
    results = await asyncio.gather(*tasks)

    full_transcript = "\n".join(results)

    yield f"data: {json.dumps({'status': 'Synthesizing council consensus...'})}\n\n"

    # Synthesis
    synthesis_prompt = (
        f"User Query: {query}\n\n"
        f"Council Responses:\n{full_transcript}\n\n"
        "You are the Council Chairperson. Synthesize the above responses into a single, authoritative answer. "
        "Point out any consensus or disagreement among the experts. "
        "Ensure the final answer uses the best information provided."
    )

    # Use best available model for synthesis (index 0 is usually best)
    synthesizer_model = experts[0]["model"]

    async for chunk in llm.get_completion(
        synthesizer_model, [{"role": "user", "content": synthesis_prompt}]
    ):
        yield chunk

    yield "data: [DONE]\n\n"
