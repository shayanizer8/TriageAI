import asyncio
import httpx
import logging
from typing import Any
from config.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "mistral-embed"
EMBEDDING_DIM = 1024


async def call_mistral_embed(texts: list[str]) -> list[list[float]]:
    """
    Call Mistral's embedding API via HTTP POST.
    Supports batching multiple texts.
    """
    if not texts:
        return []
    
    url = "https://api.mistral.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {settings.mistral_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts,
    }
    
    retries = 5
    sleep_time = 5.0
    while retries > 0:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    # Mistral returns data array. Sort by index to preserve input order.
                    sorted_data = sorted(data["data"], key=lambda x: x["index"])
                    return [x["embedding"] for x in sorted_data]
                elif response.status_code == 429:
                    logger.warning(
                        "Mistral embedding rate limit hit (429). Sleeping %.1f seconds before retrying.",
                        sleep_time,
                    )
                    await asyncio.sleep(sleep_time)
                    retries -= 1
                    sleep_time *= 2.0
                else:
                    response.raise_for_status()
        except Exception as exc:
            logger.warning("Error calling Mistral embedding API: %s. Retrying...", exc)
            if retries == 1:
                raise exc
            await asyncio.sleep(sleep_time)
            retries -= 1
            sleep_time *= 2.0
            
    raise RuntimeError("Failed to call Mistral embedding API after retries.")


async def embed_text(text: str) -> list[float]:
    """
    Embed a single string asynchronously.
    """
    embeddings = await call_mistral_embed([text])
    return embeddings[0]


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings efficiently by batching them into API calls.
    We partition into batches of 128 to stay within Mistral's request input limits.
    """
    BATCH_SIZE = 128
    embeddings: list[list[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        batch_embeddings = await call_mistral_embed(batch)
        embeddings.extend(batch_embeddings)
        
        # Brief sleep between large batches
        if i + BATCH_SIZE < len(texts):
            await asyncio.sleep(0.5)

    return embeddings


async def embed_query(query: str) -> list[float]:
    """
    Embed a query string.
    """
    embeddings = await call_mistral_embed([query])
    return embeddings[0]



