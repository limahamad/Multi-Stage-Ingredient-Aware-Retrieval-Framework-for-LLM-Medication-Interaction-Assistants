import json
import math
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import google.generativeai as genai
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
INDEX_ROOT = BASE_DIR / "rag_index"
EMBED_MODEL = "all-MiniLM-L6-v2"

GEN_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-flash-lite-latest")


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def bm25_score(query: str, bm25: Dict, doc_idx: int) -> float:
    """
    Compute BM25 score for one document.
    """
    query_tokens = tokenize(query)
    score = 0.0
    k1 = bm25["k1"]
    b = bm25["b"]
    avgdl = bm25["avgdl"] if bm25["avgdl"] > 0 else 1.0

    doc_freqs = bm25["doc_freqs"][doc_idx]
    doc_len = bm25["doc_lengths"][doc_idx]
    idf = bm25["idf"]

    for token in query_tokens:
        if token not in doc_freqs:
            continue
        f = doc_freqs[token]
        term_idf = idf.get(token, 0.0)
        denom = f + k1 * (1 - b + b * doc_len / avgdl)
        score += term_idf * ((f * (k1 + 1)) / denom)

    return score


def min_max_normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        return [1.0 for _ in values]
    return [(v - min_v) / (max_v - min_v) for v in values]


def get_google_api_key() -> str:
    """Read the Gemini API key using the same config style as Assignment 2."""
    return os.getenv("GOOGLE_API_KEY", "")


class HybridRetriever:
    """
    Dense + BM25 retriever for one stage.
    """

    def __init__(self, stage_dir: Path, model: SentenceTransformer):
        self.stage_dir = stage_dir
        self.model = model

        docs_index_path = stage_dir / "docs.index"
        chunks_path = stage_dir / "chunks.pkl"
        bm25_path = stage_dir / "bm25.pkl"
        missing_paths = [
            str(path)
            for path in (docs_index_path, chunks_path, bm25_path)
            if not path.exists()
        ]
        if missing_paths:
            raise FileNotFoundError(
                "Missing RAG index files for "
                f"{stage_dir.name}: {', '.join(missing_paths)}. "
                "Run `python build_three_stage_indexes.py` from the `Term_Project` directory "
                "to rebuild all stage indexes."
            )

        self.dense_index = faiss.read_index(str(docs_index_path))
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        with open(bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)

    def search(self, query: str, top_k: int = 5, alpha: float = 0.65) -> List[Dict]:
        """
        alpha weights dense retrieval, (1-alpha) weights BM25.
        """
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        dense_scores, dense_ids = self.dense_index.search(query_embedding, min(top_k * 4, len(self.chunks)))
        dense_scores = dense_scores[0].tolist()
        dense_ids = dense_ids[0].tolist()

        dense_candidates = []
        for idx, score in zip(dense_ids, dense_scores):
            if idx == -1:
                continue
            dense_candidates.append((idx, float(score)))

        bm25_candidates = []
        for idx in range(len(self.chunks)):
            score = bm25_score(query, self.bm25, idx)
            bm25_candidates.append((idx, score))

        # Keep stronger lexical candidates
        bm25_candidates = sorted(bm25_candidates, key=lambda x: x[1], reverse=True)[: max(top_k * 4, 10)]

        candidate_ids = {idx for idx, _ in dense_candidates} | {idx for idx, _ in bm25_candidates}

        dense_map = {idx: score for idx, score in dense_candidates}
        bm25_map = {idx: score for idx, score in bm25_candidates}

        dense_list = [dense_map.get(idx, 0.0) for idx in candidate_ids]
        bm25_list = [bm25_map.get(idx, 0.0) for idx in candidate_ids]

        dense_norm = min_max_normalize(dense_list)
        bm25_norm = min_max_normalize(bm25_list)

        merged = []
        for i, idx in enumerate(candidate_ids):
            hybrid = alpha * dense_norm[i] + (1 - alpha) * bm25_norm[i]
            chunk = self.chunks[idx]
            merged.append(
                {
                    "score": hybrid,
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "title": chunk["title"],
                    "stage": chunk["stage"],
                    "text": chunk["text"],
                    "source": chunk.get("source", ""),
                    "url": chunk.get("url", ""),
                    "metadata": chunk.get("metadata", {}),
                }
            )

        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged[:top_k]


def call_gemini(prompt: str, temperature: float = 0.1) -> str:
    """
    Generate text with Gemini using the same SDK pattern as Assignment 2.
    """
    api_key = get_google_api_key()
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEN_MODEL)
    response = model.generate_content(
        prompt,
        generation_config={"temperature": temperature},
    )

    text = getattr(response, "text", "").strip()
    if not text:
        raise RuntimeError(f"Unexpected Gemini response: {response!r}")
    return text


def safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Try to extract JSON from model output.
    """
    text = text.strip()

    # Remove markdown fences if present
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    return json.loads(text)


def build_context(results: List[Dict]) -> str:
    lines = []
    for i, item in enumerate(results, start=1):
        lines.append(
            f"[{i}] title={item['title']}\n"
            f"source={item['source']}\n"
            f"text={item['text']}\n"
        )
    return "\n".join(lines)


def extract_medications_stage1(
    user_question: str,
    stage1_results: List[Dict],
) -> Dict[str, Any]:
    """
    Stage 1: use retrieved normalization docs + Gemini
    to map user medication mentions to canonical generic names.
    """
    prompt = f"""
You are performing medication name normalization.

Task:
From the user's question, identify the medications mentioned and normalize them to canonical generic names
using ONLY the retrieved context below. Do not invent medications. If They are not found in the context, return the medications mentioned in the question. 

User question:
{user_question}

Retrieved normalization context:
{build_context(stage1_results)}

Return ONLY valid JSON with this schema:
{{
  "input_mentions": ["..."],
  "normalized_medications": [
    {{
      "input_mention": "...",
      "canonical_generic_name": "...",
      "confidence": 0.0,
      "matched_brand_or_alias": "..."
    }}
  ],
  "uncertain_mentions": ["..."]
}}
"""
    raw = call_gemini(prompt, temperature=0.0)
    return safe_json_loads(raw)


def retrieve_ingredient_records(
    retriever: HybridRetriever,
    normalized_medications: List[Dict[str, Any]],
    top_k_per_med: int = 3
) -> List[Dict]:
    """
    Stage 2: retrieve active ingredient docs for each normalized medication.
    """
    all_results: List[Dict] = []
    seen = set()

    for med in normalized_medications:
        generic = med.get("canonical_generic_name", "").strip()
        if not generic:
            continue

        query = f"{generic} active ingredients therapeutic class dosage form"
        results = retriever.search(query, top_k=top_k_per_med)

        for r in results:
            if r["chunk_id"] not in seen:
                seen.add(r["chunk_id"])
                all_results.append(r)

    return all_results


def extract_ingredients_stage2(
    normalized_data: Dict[str, Any],
    stage2_results: List[Dict],
) -> Dict[str, Any]:
    """
    Stage 2 reasoning with Gemini:
    attach active ingredients to each normalized medication.
    """
    normalized_json = json.dumps(normalized_data, indent=2)

    prompt = f"""
You are extracting active ingredients for normalized medications.

Use ONLY the normalized medication list and the retrieved ingredient context.
Do not invent missing ingredients. If ingredients are not found in the context return the normalized medication names as active ingredients. 

Normalized medications:
{normalized_json}

Retrieved ingredient context:
{build_context(stage2_results)}

Return ONLY valid JSON with this schema:
{{
  "medications": [
    {{
      "canonical_generic_name": "...",
      "active_ingredients": ["..."],
      "therapeutic_class": "...",
      "confidence": 0.0
    }}
  ]
}}
"""
    raw = call_gemini(prompt, temperature=0.0)
    return safe_json_loads(raw)


def retrieve_interaction_records(
    retriever: HybridRetriever,
    ingredient_data: Dict[str, Any],
    top_k: int = 6
) -> List[Dict]:
    """
    Stage 3: retrieve interaction records using ingredient pairs/classes.
    """
    meds = ingredient_data.get("medications", [])
    all_ingredients: List[str] = []

    for med in meds:
        all_ingredients.extend(med.get("active_ingredients", []))

    all_ingredients = sorted(set(i.strip() for i in all_ingredients if i.strip()))
    if len(all_ingredients) < 2:
        query = " ".join(all_ingredients) + " interaction contraindication"
    else:
        query = " interaction ".join(all_ingredients) + " severity mechanism recommended actions"

    return retriever.search(query, top_k=top_k)


def final_reasoning_stage3(
    user_question: str,
    normalized_data: Dict[str, Any],
    ingredient_data: Dict[str, Any],
    stage3_results: List[Dict]
) -> Dict[str, Any]:
    """
    Final answer:
    combine stage outputs and retrieved interaction evidence.
    """
    prompt = f"""
You are a medication safety assistant.

Use ONLY the retrieved context below to answer the question.
If the context does not contain enough information, say:
"I do not have enough information in the retrieved documents."

User question:
{user_question}

Stage 1 normalization:
{json.dumps(normalized_data, indent=2)}

Stage 2 ingredients:
{json.dumps(ingredient_data, indent=2)}

Stage 3 interaction evidence:
{build_context(stage3_results)}

Return ONLY valid JSON with this schema:
{{
  "decision": "Safe|Caution|Not Safe|Uncertain",
  "interaction_found": true,
  "severity": "none|mild|moderate|major|unknown",
  "short_answer": "...",
  "mechanism_summary": "...",
  "safety_advice": [
    "...",
    "..."
  ],
  "disclaimer": "This information does not replace professional medical advice.",
  "evidence_summary": [
    {{
      "source_title": "...",
      "reason_used": "..."
    }}
  ]
}}
"""
    raw = call_gemini(prompt, temperature=0.1)
    return safe_json_loads(raw)


def format_final_answer(final_output: Dict[str, Any]) -> str:
    """Convert the structured stage-3 response into a readable answer string."""
    safety_advice = final_output.get("safety_advice", [])
    evidence_summary = final_output.get("evidence_summary", [])

    lines = [
        f"Decision: {final_output.get('decision', 'Uncertain')}",
        f"Interaction Found: {final_output.get('interaction_found', False)}",
        f"Severity: {final_output.get('severity', 'unknown')}",
        "",
        f"Short Answer: {final_output.get('short_answer', '')}",
        "",
        f"Mechanism Summary: {final_output.get('mechanism_summary', '')}",
        "",
        "Safety Advice:",
    ]

    if safety_advice:
        lines.extend(f"- {item}" for item in safety_advice)
    else:
        lines.append("- No specific safety advice provided.")

    lines.extend(
        [
            "",
            f"Disclaimer: {final_output.get('disclaimer', '')}",
            "",
            "Evidence Summary:",
        ]
    )

    if evidence_summary:
        lines.extend(
            f"- {item.get('source_title', 'Unknown source')}: {item.get('reason_used', '')}"
            for item in evidence_summary
        )
    else:
        lines.append("- No evidence summary provided.")

    return "\n".join(lines).strip()


def run_three_stage_pipeline(
    user_question: str,
    stage1_retriever: HybridRetriever,
    stage2_retriever: HybridRetriever,
    stage3_retriever: HybridRetriever,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run the full three-stage RAG workflow and return all intermediate outputs."""
    if verbose:
        print("  Stage 1: retrieving normalization evidence...")
    stage1_results = stage1_retriever.search(user_question, top_k=5)
    if verbose:
        print(f"  Stage 1: retrieved {len(stage1_results)} chunks, asking Gemini...")
    stage1_output = extract_medications_stage1(user_question, stage1_results)

    normalized_meds = stage1_output.get("normalized_medications", [])
    if not normalized_meds:
        raise ValueError("Could not confidently normalize any medication names.")

    if verbose:
        print(
            f"  Stage 1: done, normalized {len(normalized_meds)} medication mention(s)."
        )
        print("  Stage 2: retrieving ingredient evidence...")
    stage2_results = retrieve_ingredient_records(
        stage2_retriever,
        normalized_meds,
        top_k_per_med=3,
    )
    if verbose:
        print(f"  Stage 2: retrieved {len(stage2_results)} chunks, asking Gemini...")
    stage2_output = extract_ingredients_stage2(stage1_output, stage2_results)

    if verbose:
        med_count = len(stage2_output.get("medications", []))
        print(f"  Stage 2: done, extracted ingredient data for {med_count} medication(s).")
        print("  Stage 3: retrieving interaction evidence...")
    stage3_results = retrieve_interaction_records(stage3_retriever, stage2_output, top_k=6)
    if verbose:
        print(f"  Stage 3: retrieved {len(stage3_results)} chunks, asking Gemini...")
    final_output = final_reasoning_stage3(
        user_question=user_question,
        normalized_data=stage1_output,
        ingredient_data=stage2_output,
        stage3_results=stage3_results,
    )
    if verbose:
        print("  Stage 3: done, final answer generated.")

    return {
        "query": user_question,
        "answer": format_final_answer(final_output),
        "stage1_results": stage1_results,
        "stage1_output": stage1_output,
        "stage2_results": stage2_results,
        "stage2_output": stage2_output,
        "stage3_results": stage3_results,
        "final_output": final_output,
    }


def pretty_print_analysis(
    user_question: str,
    stage1_output: Dict[str, Any],
    stage2_output: Dict[str, Any],
    final_output: Dict[str, Any]
) -> None:
    print("\n" + "=" * 80)
    print("USER QUESTION")
    print("=" * 80)
    print(user_question)

    print("\n" + "=" * 80)
    print("STAGE 1 - NORMALIZATION")
    print("=" * 80)
    print(json.dumps(stage1_output, indent=2, ensure_ascii=False))

    print("\n" + "=" * 80)
    print("STAGE 2 - INGREDIENT EXTRACTION")
    print("=" * 80)
    print(json.dumps(stage2_output, indent=2, ensure_ascii=False))

    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)
    print(json.dumps(final_output, indent=2, ensure_ascii=False))


def main() -> None:
    model = SentenceTransformer(EMBED_MODEL)

    stage1_retriever = HybridRetriever(INDEX_ROOT / "stage1_normalization", model)
    stage2_retriever = HybridRetriever(INDEX_ROOT / "stage2_ingredients", model)
    stage3_retriever = HybridRetriever(INDEX_ROOT / "stage3_interactions", model)

    print("Three-stage Gemini RAG medication assistant")
    print("Type 'exit' to quit.\n")

    while True:
        user_question = input("Ask a medication interaction question: ").strip()
        if not user_question:
            continue
        if user_question.lower() in {"exit", "quit"}:
            break

        try:
            pipeline_output = run_three_stage_pipeline(
                user_question=user_question,
                stage1_retriever=stage1_retriever,
                stage2_retriever=stage2_retriever,
                stage3_retriever=stage3_retriever,
                verbose=True,
            )

            pretty_print_analysis(
                user_question=user_question,
                stage1_output=pipeline_output["stage1_output"],
                stage2_output=pipeline_output["stage2_output"],
                final_output=pipeline_output["final_output"],
            )

        except Exception as exc:
            print(f"\nError: {exc}")


if __name__ == "__main__":
    main()
