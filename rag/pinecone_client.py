"""
Pinecone client — upsert and query operations for the medical KB index.

Index schema per vector:
  id       : "<source>-<icd_code or row_id>"
  values   : list[float] (768-dim, text-embedding-004)
  metadata : {
      icd_code       : str,
      condition_name : str,
      symptom_keywords : str (comma-separated),
      urgency_hint   : int (1-5),
      source         : "icd10" | "kaggle"
  }
"""
from __future__ import annotations

from typing import Any
import logging
from pinecone import Pinecone, ServerlessSpec
from config.settings import get_settings
from rag.embedder import embed_query

settings = get_settings()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pinecone client singleton
# ---------------------------------------------------------------------------
_pc: Pinecone | None = None
_index: Any = None   # pinecone.Index


def _get_client() -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=settings.pinecone_api_key)
    return _pc


def get_index() -> Any:
    """Return (or lazily create) the Pinecone serverless index with 1024 dimensions."""
    global _index
    if _index is None:
        pc = _get_client()
        existing = [idx.name for idx in pc.list_indexes()]

        if settings.pinecone_index_name not in existing:
            logger.info("Creating Pinecone index '%s' (dim: 1024) ...", settings.pinecone_index_name)
            pc.create_index(
                name=settings.pinecone_index_name,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        else:
            desc = pc.describe_index(settings.pinecone_index_name)
            if desc.dimension != 1024:
                logger.info(
                    "Deleting existing Pinecone index '%s' because its dimension %d != 1024",
                    settings.pinecone_index_name,
                    desc.dimension,
                )
                pc.delete_index(settings.pinecone_index_name)
                logger.info("Recreating Pinecone index '%s' (dim: 1024) ...", settings.pinecone_index_name)
                pc.create_index(
                    name=settings.pinecone_index_name,
                    dimension=1024,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )

        _index = pc.Index(settings.pinecone_index_name)
    return _index



# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def upsert_vectors(vectors: list[dict]) -> None:
    """
    Upsert a list of {'id', 'values', 'metadata'} dicts.
    Called from ingest_kb.py; runs synchronously (in a script context).
    """
    index = get_index()
    # Pinecone recommends batches of ≤100
    BATCH_SIZE = 100
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i : i + BATCH_SIZE]
        index.upsert(vectors=batch)


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

async def query_medical_kb(
    symptom_text: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Embed the symptom text and return the top-k nearest conditions.

    Returns a list of dicts:
    [
        {
            "icd_code": "J06.9",
            "condition_name": "Acute upper respiratory infection",
            "symptom_keywords": "fever, cough, sore throat",
            "urgency_hint": 4,
            "similarity_score": 0.92,
        },
        ...
    ]
    """
    query_vector = await embed_query(symptom_text)
    index = get_index()

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )

    matches = []
    for match in results.get("matches", []):
        meta = match.get("metadata", {})
        matches.append(
            {
                "icd_code": meta.get("icd_code", ""),
                "condition_name": meta.get("condition_name", ""),
                "symptom_keywords": meta.get("symptom_keywords", "").split(","),
                "urgency_hint": int(meta.get("urgency_hint", 5)),
                "similarity_score": round(match.get("score", 0.0), 4),
            }
        )
    return matches
