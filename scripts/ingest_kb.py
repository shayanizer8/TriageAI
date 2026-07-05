"""
KB Ingestion Script — run once to populate Pinecone with medical knowledge.

Sources:
  1. ICD-10 CSV (WHO) — download from: https://icd.who.int/browse10/Content/statichtml/ICD10Volume2_en_2019.htm
     Or use the simplified CSV from: https://github.com/kamilkosek/icd10/blob/main/icd10.csv
  2. Kaggle Disease-Symptom dataset — download from:
     https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset

Place downloaded files in: data/icd10.csv and data/disease_symptoms.csv
Then run: python scripts/ingest_kb.py

"""
from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from rag.embedder import embed_batch
from rag.pinecone_client import get_index, upsert_vectors
from config.settings import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
settings = get_settings()

# Urgency hints — map ICD-10 chapter to urgency score
ICD_CHAPTER_URGENCY: dict[str, int] = {
    "I": 2,   # Diseases of the circulatory system → urgent
    "J": 3,   # Diseases of the respiratory system → semi-urgent
    "K": 3,   # Diseases of the digestive system
    "S": 3,   # Injuries, poisoning → urgent
    "T": 2,   # Toxic effects → emergency
    "R": 3,   # Symptoms not elsewhere classified
    "Z": 5,   # Factors influencing health → routine
    "V": 2,   # External causes (accidents) → emergency
    "W": 2,
    "X": 1,   # Intentional self-harm → immediate
}


def get_urgency_hint(icd_code: str) -> int:
    """Derive a rough urgency score from the ICD-10 chapter letter."""
    chapter = icd_code[0].upper() if icd_code else "Z"
    return ICD_CHAPTER_URGENCY.get(chapter, 4)


def load_icd10_csv(path: Path) -> list[dict]:
    """
    Load ICD-10 CSV. Handles both headered and headerless CSV files.
    Deduplicates based on the 3-character category prefix to prevent exceeding Pinecone free tier limits.
    """
    logger.info("Loading ICD-10 CSV from: %s", path)
    rows = []
    try:
        # Load the CSV
        df = pd.read_csv(path, header=None, encoding="utf-8", on_bad_lines="skip")
        
        # Check if the first row is headers
        first_row_vals = [str(x).lower() for x in df.iloc[0].values]
        has_headers = any("code" in x or "desc" in x or "name" in x for x in first_row_vals)
        
        if has_headers:
            # Re-read with header
            df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
            df.columns = [c.lower().strip() for c in df.columns]
            code_col = next((c for c in df.columns if "code" in c), None)
            desc_col = next((c for c in df.columns if "desc" in c or "name" in c or "label" in c), None)
            category_col = df.columns[0]
        else:
            # Headerless (like the user's codes.csv)
            # column index 2 is the full code, column index 3 is description, column 0 is category block
            code_col = 2 if len(df.columns) >= 3 else 0
            desc_col = 3 if len(df.columns) >= 4 else 1
            category_col = 0

        if code_col is None or desc_col is None:
            logger.warning("Could not identify code/description columns in ICD-10 CSV.")
            return rows

        seen_blocks = set()
        for _, row in df.iterrows():
            code = str(row[code_col]).strip()
            name = str(row[desc_col]).strip()
            
            # Use 3-character block prefix for deduplication to fit Pinecone free tier
            category_block = str(row[category_col]).strip()[:3] if category_col in row else code[:3]
            
            if category_block in seen_blocks:
                continue
            seen_blocks.add(category_block)

            if code and name and len(name) > 3:
                rows.append({
                    "icd_code": code,
                    "condition_name": name,
                    "urgency_hint": get_urgency_hint(code),
                    "source": "icd10",
                })
    except Exception as exc:
        logger.error("Failed to load ICD-10 CSV: %s", exc)
    logger.info("Loaded %d deduplicated ICD-10 entries", len(rows))
    return rows


def load_kaggle_symptoms_csv(path: Path) -> list[dict]:
    """
    Load Kaggle disease-symptom CSV.
    Expected format: Disease | Symptom_1 | Symptom_2 | ... | Symptom_N
    Groups by disease and aggregates all unique symptoms to compress redundant rows.
    """
    logger.info("Loading Kaggle symptom CSV from: %s", path)
    rows = []
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
        df.columns = [c.lower().strip() for c in df.columns]

        disease_col = next((c for c in df.columns if "disease" in c or "condition" in c), None)
        symptom_cols = [c for c in df.columns if "symptom" in c]

        if not disease_col:
            logger.warning("No disease column found in Kaggle CSV. Columns: %s", df.columns.tolist())
            return rows

        # this groups all symptoms for the same disease together 
        grouped = {}
        for _, row in df.iterrows():
            disease = str(row[disease_col]).strip()
            if not disease:
                continue
            symptoms = [
                str(row[sc]).strip().replace("_", " ")
                for sc in symptom_cols
                if pd.notna(row.get(sc)) and str(row.get(sc)).strip()
            ]
            if disease not in grouped:
                grouped[disease] = set()
            grouped[disease].update(symptoms)
        
        # build rows from grouped symptoms  
        for disease, symptoms_set in grouped.items():
            if disease and symptoms_set:
                rows.append({
                    "icd_code": f"KAGGLE-{len(rows):04d}",
                    "condition_name": disease,
                    "symptom_keywords": ", ".join(sorted(list(symptoms_set))),
                    "urgency_hint": 4,  # default semi-urgent for Kaggle data
                    "source": "kaggle",
                })
    except Exception as exc:
        logger.error("Failed to load Kaggle CSV: %s", exc)
    logger.info("Loaded %d compressed Kaggle symptom entries", len(rows))
    return rows


def build_embed_text(row: dict) -> str:
    """Build a rich text string for embedding — condition + symptoms."""
    parts = [row["condition_name"]]
    if row.get("symptom_keywords"):
        parts.append(f"Symptoms: {row['symptom_keywords']}")
    if row.get("icd_code") and not row["icd_code"].startswith("KAGGLE"):
        parts.append(f"ICD-10: {row['icd_code']}")
    return ". ".join(parts)


async def ingest(data_dir: Path) -> None:
    """Main ingestion pipeline."""
    # 1. Check if already ingested
    index = get_index()
    stats = index.describe_index_stats()
    existing_count = stats.get("total_vector_count", 0)

    if existing_count > 1000:
        logger.info("Index already contains %d vectors — skipping ingestion.", existing_count)
        logger.info("To force re-ingest, delete the Pinecone index first.")
        return

    logger.info("Starting ingestion (existing vectors: %d)", existing_count)

    # 2. Load datasets
    all_rows: list[dict] = []

    icd10_path = data_dir / "icd10.csv"
    if not icd10_path.exists():
        icd10_path = data_dir / "codes.csv"

    if icd10_path.exists():
        all_rows.extend(load_icd10_csv(icd10_path))
    else:
        logger.warning("ICD-10 CSV not found at %s — skipping", icd10_path)

    kaggle_path = data_dir / "disease_symptoms.csv"
    if not kaggle_path.exists():
        kaggle_path = data_dir / "dataset.csv"

    if kaggle_path.exists():
        all_rows.extend(load_kaggle_symptoms_csv(kaggle_path))
    else:
        logger.warning("Kaggle CSV not found at %s — skipping", kaggle_path)

    if not all_rows:
        logger.error("No data loaded. Place CSVs in data/ directory and re-run.")
        return

    logger.info("Total rows to embed: %d", len(all_rows))

    # 3. Build text for embedding
    texts = [build_embed_text(row) for row in all_rows]

    # 4. Embed in batches with local cache
    cache_path = data_dir / "embeddings_cache.json"
    cache = {}
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            logger.info("Loaded %d cached embeddings from %s", len(cache), cache_path)
        except Exception as exc:
            logger.warning("Failed to load cache: %s", exc)

    missing_texts = [t for t in texts if t not in cache]
    logger.info("Total texts: %d, Cached: %d, Missing: %d", len(texts), len(texts) - len(missing_texts), len(missing_texts))

    if missing_texts:
        logger.info("Embedding %d missing texts in batches of 128...", len(missing_texts))
        BATCH_SIZE = 128
        for i in range(0, len(missing_texts), BATCH_SIZE):
            batch = missing_texts[i : i + BATCH_SIZE]
            
            # Embed this batch using embed_batch
            batch_embeddings = await embed_batch(batch)
            
            # Save to cache dictionary
            for t, emb in zip(batch, batch_embeddings):
                cache[t] = emb
                
            # Write cache to disk after every batch
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False)
            except Exception as exc:
                logger.warning("Failed to write cache: %s", exc)
                
            logger.info(
                "Processed batch %d/%d (Total cached: %d)",
                (i // BATCH_SIZE) + 1,
                ((len(missing_texts) - 1) // BATCH_SIZE) + 1,
                len(cache)
            )

            # Sleep between batches if not the last batch
            if i + BATCH_SIZE < len(missing_texts):
                logger.info("Sleeping 0.5s between batches...")
                await asyncio.sleep(0.5)

    # Reconstruct the list of embeddings for all texts in order
    embeddings = [cache[t] for t in texts]
    logger.info("Embedding complete.")

    # 5. Build Pinecone vectors
    vectors = []
    for i, (row, embedding) in enumerate(zip(all_rows, embeddings)):
        vector_id = f"{row['source']}-{row['icd_code'].replace('/', '-')}-{i}"
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {
                "id": vector_id,  # Added to prevent missing id issues
                "icd_code": row["icd_code"],
                "condition_name": row["condition_name"],
                "symptom_keywords": row.get("symptom_keywords", ""),
                "urgency_hint": row["urgency_hint"],
                "source": row["source"],
            },
        })

    # 6. Upsert to Pinecone
    logger.info("Upserting %d vectors to Pinecone index '%s' ...", len(vectors), settings.pinecone_index_name)
    upsert_vectors(vectors)

    # 7. Verify
    final_stats = index.describe_index_stats()
    logger.info(
        "Ingestion complete. Index now contains %d vectors.",
        final_stats.get("total_vector_count", 0),
    )


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    asyncio.run(ingest(data_dir))

