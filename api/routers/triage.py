"""
Triage RAG endpoint — test the Pinecone medical KB directly via HTTP.

Useful for:
  - Verifying that KB ingestion worked correctly
  - Debugging symptom-to-condition matching
  - Integration tests without a live phone call
"""
from fastapi import APIRouter, HTTPException

from api.models import TriageQueryRequest, TriageQueryResponse, ICDMatchResponse
from rag.pinecone_client import query_medical_kb

router = APIRouter()


@router.post("/query", response_model=TriageQueryResponse)
async def query_triage_kb(payload: TriageQueryRequest):
    """
    Embed the symptom text and return the top-K matching ICD-10 conditions
    from Pinecone.

    Example request:
        {"symptom_text": "chest pain shortness of breath", "top_k": 5}
    """
    try:
        matches = await query_medical_kb(
            symptom_text=payload.symptom_text,
            top_k=payload.top_k,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"RAG query failed — check Pinecone connection: {exc}",
        )

    return TriageQueryResponse(
        query=payload.symptom_text,
        matches=[ICDMatchResponse(**m) for m in matches],
    )
