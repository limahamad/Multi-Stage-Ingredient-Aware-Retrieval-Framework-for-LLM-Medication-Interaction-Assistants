import json
import math
import os
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_ROOT = BASE_DIR / "rag_index"
EMBED_MODEL = "all-MiniLM-L6-v2"

STAGE1_DOCS = DATA_DIR / "stage1_normalization_docs.json"
STAGE2_DOCS = DATA_DIR / "stage2_ingredient_docs.json"
STAGE3_DOCS = DATA_DIR / "stage3_interaction_docs.json"


def simple_chunk(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    """
    Split long text into overlapping chunks.
    Keeps behavior similar to your original code.
    """
    text = " ".join(text.split())
    chunks: List[str] = []

    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)

    return chunks


def tokenize(text: str) -> List[str]:
    """Simple lowercase tokenizer shared by BM25."""
    return re.findall(r"\b\w+\b", text.lower())


def load_docs(path: Path) -> List[Dict]:
    """Load JSON documents from disk."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")

    normalized_docs: List[Dict] = []
    for item in data:
        if isinstance(item, dict):
            normalized_docs.append(item)
        elif isinstance(item, list):
            for nested_item in item:
                if not isinstance(nested_item, dict):
                    raise ValueError(
                        f"{path} contains a nested item that is not a JSON object: "
                        f"{type(nested_item).__name__}"
                    )
                normalized_docs.append(nested_item)
        else:
            raise ValueError(
                f"{path} contains an item that is not a JSON object: "
                f"{type(item).__name__}"
            )

    return normalized_docs


def build_bm25_index(chunks: List[Dict]) -> Dict:
    """Build BM25 statistics for lexical retrieval."""
    tokenized_docs = [tokenize(chunk["text"]) for chunk in chunks]
    doc_freqs: List[Dict[str, int]] = []
    doc_lengths: List[int] = []
    document_frequency = Counter()

    for tokens in tokenized_docs:
        frequencies = Counter(tokens)
        doc_freqs.append(dict(frequencies))
        doc_lengths.append(len(tokens))
        for token in frequencies:
            document_frequency[token] += 1

    total_docs = len(tokenized_docs)
    avgdl = sum(doc_lengths) / total_docs if total_docs else 0.0
    idf = {
        token: math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
        for token, freq in document_frequency.items()
    }

    return {
        "tokenized_docs": tokenized_docs,
        "doc_freqs": doc_freqs,
        "doc_lengths": doc_lengths,
        "avgdl": avgdl,
        "idf": idf,
        "k1": 1.5,
        "b": 0.75,
    }


def build_stage1_chunks(docs: List[Dict]) -> List[Dict]:
    """
    Stage 1: medication normalization.
    Each record should describe brand names, generic name, aliases, abbreviations.
    """
    chunks: List[Dict] = []

    for doc in docs:
        aliases = ", ".join(doc.get("aliases", []))
        abbreviations = ", ".join(doc.get("abbreviations", []))
        generic_name = doc.get("generic_name", "")
        brand_group = ", ".join(doc.get("brand_names", []))

        full_text = (
            f"canonical_generic_name: {generic_name}\n"
            f"brand_names: {brand_group}\n"
            f"aliases: {aliases}\n"
            f"abbreviations: {abbreviations}\n"
            f"description: {doc.get('description', '')}\n"
            f"source: {doc.get('source', '')}"
        )

        for i, chunk in enumerate(simple_chunk(full_text)):
            chunks.append(
                {
                    "chunk_id": f"{doc['id']}_chunk{i}",
                    "doc_id": doc["id"],
                    "stage": "normalization",
                    "title": generic_name,
                    "source": doc.get("source", ""),
                    "url": doc.get("url", ""),
                    "text": chunk,
                    "metadata": {
                        "generic_name": generic_name,
                        "brand_names": doc.get("brand_names", []),
                        "aliases": doc.get("aliases", []),
                        "abbreviations": doc.get("abbreviations", []),
                    },
                }
            )

    return chunks


def build_stage2_chunks(docs: List[Dict]) -> List[Dict]:
    """
    Stage 2: ingredient retrieval.
    Each record links a normalized medication to active ingredients.
    """
    chunks: List[Dict] = []

    for doc in docs:
        active_ingredients = ", ".join(doc.get("active_ingredients", []))
        strengths = ", ".join(doc.get("strengths", []))
        dosage_forms = ", ".join(doc.get("dosage_forms", []))

        full_text = (
            f"generic_name: {doc.get('generic_name', '')}\n"
            f"active_ingredients: {active_ingredients}\n"
            f"strengths: {strengths}\n"
            f"dosage_forms: {dosage_forms}\n"
            f"therapeutic_class: {doc.get('therapeutic_class', '')}\n"
            f"description: {doc.get('description', '')}\n"
            f"source: {doc.get('source', '')}"
        )

        for i, chunk in enumerate(simple_chunk(full_text)):
            chunks.append(
                {
                    "chunk_id": f"{doc['id']}_chunk{i}",
                    "doc_id": doc["id"],
                    "stage": "ingredients",
                    "title": doc.get("generic_name", ""),
                    "source": doc.get("source", ""),
                    "url": doc.get("url", ""),
                    "text": chunk,
                    "metadata": {
                        "generic_name": doc.get("generic_name", ""),
                        "active_ingredients": doc.get("active_ingredients", []),
                        "therapeutic_class": doc.get("therapeutic_class", ""),
                    },
                }
            )

    return chunks


def build_stage3_chunks(docs: List[Dict]) -> List[Dict]:
    """
    Stage 3: interaction retrieval.
    Each record describes ingredient-level or class-level interactions.
    """
    chunks: List[Dict] = []

    for doc in docs:
        ingredients = ", ".join(doc.get("ingredients_or_classes", []))
        evidence = " ".join(doc.get("evidence_points", []))
        recommended_actions = " ".join(doc.get("recommended_actions", []))

        full_text = (
            f"interaction_id: {doc.get('id', '')}\n"
            f"ingredients_or_classes: {ingredients}\n"
            f"severity: {doc.get('severity', '')}\n"
            f"interaction_type: {doc.get('interaction_type', '')}\n"
            f"summary: {doc.get('summary', '')}\n"
            f"mechanism: {doc.get('mechanism', '')}\n"
            f"evidence_points: {evidence}\n"
            f"recommended_actions: {recommended_actions}\n"
            f"source: {doc.get('source', '')}"
        )

        for i, chunk in enumerate(simple_chunk(full_text)):
            chunks.append(
                {
                    "chunk_id": f"{doc['id']}_chunk{i}",
                    "doc_id": doc["id"],
                    "stage": "interactions",
                    "title": doc.get("id", ""),
                    "source": doc.get("source", ""),
                    "url": doc.get("url", ""),
                    "text": chunk,
                    "metadata": {
                        "ingredients_or_classes": doc.get("ingredients_or_classes", []),
                        "severity": doc.get("severity", ""),
                        "interaction_type": doc.get("interaction_type", ""),
                    },
                }
            )

    return chunks


def encode_chunks(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """Create normalized dense embeddings."""
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings.astype(np.float32)


def save_stage_index(stage_name: str, chunks: List[Dict], embeddings: np.ndarray) -> None:
    """Save FAISS dense index + chunks + BM25."""
    stage_dir = INDEX_ROOT / stage_name
    os.makedirs(stage_dir, exist_ok=True)

    dim = embeddings.shape[1]
    dense_index = faiss.IndexFlatIP(dim)
    dense_index.add(embeddings)

    bm25_index = build_bm25_index(chunks)

    faiss.write_index(dense_index, str(stage_dir / "docs.index"))
    with open(stage_dir / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    with open(stage_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25_index, f)

    print(f"[{stage_name}] Built hybrid index with {len(chunks)} chunks.")


def build_all_indexes() -> None:
    """Build the three stage-specific indexes."""
    os.makedirs(INDEX_ROOT, exist_ok=True)
    model = SentenceTransformer(EMBED_MODEL)

    # Stage 1
    stage1_docs = load_docs(STAGE1_DOCS)
    stage1_chunks = build_stage1_chunks(stage1_docs)
    stage1_embeddings = encode_chunks(model, [c["text"] for c in stage1_chunks])
    save_stage_index("stage1_normalization", stage1_chunks, stage1_embeddings)

    # Stage 2
    stage2_docs = load_docs(STAGE2_DOCS)
    stage2_chunks = build_stage2_chunks(stage2_docs)
    stage2_embeddings = encode_chunks(model, [c["text"] for c in stage2_chunks])
    save_stage_index("stage2_ingredients", stage2_chunks, stage2_embeddings)

    # Stage 3
    stage3_docs = load_docs(STAGE3_DOCS)
    stage3_chunks = build_stage3_chunks(stage3_docs)
    stage3_embeddings = encode_chunks(model, [c["text"] for c in stage3_chunks])
    save_stage_index("stage3_interactions", stage3_chunks, stage3_embeddings)

    print("\nAll three RAG stage indexes were built successfully.")


if __name__ == "__main__":
    build_all_indexes()
