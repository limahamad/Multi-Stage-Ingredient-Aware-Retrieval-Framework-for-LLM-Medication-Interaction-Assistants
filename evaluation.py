import csv
import json
import os
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sentence_transformers import SentenceTransformer

from app_three_stage import (
    EMBED_MODEL,
    INDEX_ROOT,
    GEN_MODEL,
    HybridRetriever,
    get_google_api_key,
    run_three_stage_pipeline,
)

BASE_DIR = Path(__file__).resolve().parent
TESTS_PATH = BASE_DIR / "testcases.json"
MANUAL_RATINGS_OUTPUT_PATH = BASE_DIR / "evaluation_manual_ratings.csv"
AUTOMATED_RAGAS_OUTPUT_PATH = BASE_DIR / "evaluation_ragas_metrics.csv"
RAGAS_EVAL_PROVIDER = "groq"
RAGAS_EVAL_MODEL = "llama-3.1-8b-instant"
RAGAS_EVAL_FALLBACKS = [
    ("openai", "models/gemini-flash-lite-latest"),
    ("openai", "gemini-2.0-flash"),
    ("openai", "gemini-1.5-flash"),
]

MANUAL_DIMENSIONS = ["accuracy", "safety", "clarity"]
AUTOMATED_RAGAS_DIMENSIONS = [
    "faithfulness",
    "context_relevance",
    "context_utilization",
    "response_groundedness",
]
EVALUATION_LEVELS = ["pipeline", "stage1", "stage2", "stage3"]


def load_test_cases() -> List[Dict]:
    """Read the evaluation prompts from the local testcases file."""
    with open(TESTS_PATH, "r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload.get("test_cases", [])


def load_retrievers() -> Tuple[HybridRetriever, HybridRetriever, HybridRetriever]:
    """Initialize the three retrievers once for the full evaluation run."""
    model = SentenceTransformer(EMBED_MODEL)
    stage1_retriever = HybridRetriever(INDEX_ROOT / "stage1_normalization", model)
    stage2_retriever = HybridRetriever(INDEX_ROOT / "stage2_ingredients", model)
    stage3_retriever = HybridRetriever(INDEX_ROOT / "stage3_interactions", model)
    return stage1_retriever, stage2_retriever, stage3_retriever


def collect_all_retrieved_contexts(output: Dict) -> List[str]:
    """Aggregate retrieved text chunks from all three stages for ragas."""
    contexts: List[str] = []
    seen = set()

    for key in ("stage1_results", "stage2_results", "stage3_results"):
        for result in output.get(key, []):
            chunk_id = result.get("chunk_id", "")
            text = result.get("text", "").strip()
            if not text:
                continue
            dedupe_key = chunk_id or text
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            contexts.append(text)

    return contexts


def collect_stage_retrieved_contexts(output: Dict, stage_name: str) -> List[str]:
    """Collect retrieved chunks for one stage only."""
    contexts: List[str] = []
    seen = set()

    for result in output.get(f"{stage_name}_results", []):
        chunk_id = result.get("chunk_id", "")
        text = result.get("text", "").strip()
        if not text:
            continue
        dedupe_key = chunk_id or text
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        contexts.append(text)

    return contexts


def stringify_stage_output(payload: Any) -> str:
    """Convert structured stage outputs to stable text for stage-level ragas scoring."""
    if isinstance(payload, str):
        return payload.strip()
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_stage_evaluation_views(output: Dict) -> Dict[str, Dict[str, Any]]:
    """Create pipeline-level and stage-level views for ragas scoring."""
    pipeline_contexts = collect_all_retrieved_contexts(output)
    stage1_contexts = collect_stage_retrieved_contexts(output, "stage1")
    stage2_contexts = collect_stage_retrieved_contexts(output, "stage2")
    stage3_contexts = collect_stage_retrieved_contexts(output, "stage3")

    return {
        "pipeline": {
            "query": output["query"],
            "answer": output["answer"],
            "retrieved_contexts": pipeline_contexts,
        },
        "stage1": {
            "query": output["query"],
            "answer": stringify_stage_output(output.get("stage1_output", {})),
            "retrieved_contexts": stage1_contexts,
        },
        "stage2": {
            "query": output["query"],
            "answer": stringify_stage_output(output.get("stage2_output", {})),
            "retrieved_contexts": stage2_contexts,
        },
        "stage3": {
            "query": output["query"],
            "answer": stringify_stage_output(output.get("final_output", {})),
            "retrieved_contexts": stage3_contexts,
        },
    }


def ragas_is_available() -> bool:
    """Check whether ragas is installed before attempting automated metrics."""
    return find_spec("ragas") is not None


def get_groq_api_key() -> str:
    """Read the Groq API key for temporary non-Gemini evaluation."""
    return os.getenv("GROQ_API_KEY", "")


def get_ragas_eval_targets() -> List[Tuple[str, str]]:
    """Return the primary provider/model pair followed by unique fallbacks."""
    ordered_targets = [(RAGAS_EVAL_PROVIDER, RAGAS_EVAL_MODEL), *RAGAS_EVAL_FALLBACKS]
    unique_targets: List[Tuple[str, str]] = []
    for provider_name, model_name in ordered_targets:
        target = (provider_name, model_name)
        if provider_name and model_name and target not in unique_targets:
            unique_targets.append(target)
    return unique_targets


def build_ragas_scorers(provider_name: str, model_name: str) -> Dict[str, object]:
    """Create ragas scorers for the configured provider/model pair."""
    if not ragas_is_available():
        raise ImportError(
            "ragas is not installed. Install it with: pip install ragas litellm openai"
        )

    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        ContextRelevance,
        ContextUtilization,
        Faithfulness,
        ResponseGroundedness,
    )

    if provider_name == "groq":
        api_key = get_groq_api_key()
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured, so Groq-based ragas metrics cannot run."
            )
        import instructor
        from groq import AsyncGroq
        from ragas.llms.base import InstructorLLM, InstructorModelArgs

        ragas_client = instructor.from_groq(
            AsyncGroq(api_key=api_key),
            mode=instructor.Mode.JSON,
        )
        ragas_llm = InstructorLLM(
            client=ragas_client,
            model=model_name,
            provider=provider_name,
            model_args=InstructorModelArgs(),
        )
    elif provider_name == "openai":
        api_key = get_google_api_key()
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is not configured, so Gemini-based ragas metrics cannot run."
            )
        from openai import AsyncOpenAI

        ragas_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        ragas_llm = llm_factory(
            model_name,
            provider=provider_name,
            client=ragas_client,
        )
    else:
        raise ValueError(f"Unsupported ragas evaluation provider: {provider_name}")

    return {
        "faithfulness": Faithfulness(llm=ragas_llm),
        "context_relevance": ContextRelevance(llm=ragas_llm),
        "context_utilization": ContextUtilization(llm=ragas_llm),
        "response_groundedness": ResponseGroundedness(llm=ragas_llm),
    }


def score_with_ragas(metric_name: str, scorer: object, output: Dict) -> float:
    """Evaluate one answer with a single ragas metric."""
    payload = {"retrieved_contexts": output["retrieved_contexts"]}

    if metric_name in {"faithfulness", "context_relevance", "context_utilization"}:
        payload["user_input"] = output["query"]
    if metric_name in {
        "faithfulness",
        "context_utilization",
        "response_groundedness",
    }:
        payload["response"] = output["answer"]

    score = scorer.score(**payload)
    return float(getattr(score, "value", score))


def prompt_for_rating(label: str) -> int:
    """Ask for a manual score from 1 to 5."""
    while True:
        raw = input(f"Rate {label} from 1-5: ").strip()
        if raw in {"1", "2", "3", "4", "5"}:
            return int(raw)
        print("Please enter one of: 1, 2, 3, 4, 5")


def run_manual_evaluation(test_cases: List[Dict], outputs: Dict[str, Dict]) -> List[Dict]:
    """Prompt the evaluator to score each answer for accuracy, safety, and clarity."""
    rows: List[Dict] = []

    print("\n" + "=" * 80)
    print("MANUAL EVALUATION")
    print("=" * 80)

    for case in test_cases:
        output = outputs.get(case["id"])
        if not output:
            continue

        print("\n" + "-" * 80)
        print(f"Case: {case['id']} | Category: {case.get('category', '')}")
        print(f"Question: {case.get('input', '')}")
        print(f"Notes: {case.get('notes', '')}")
        print("\nAnswer:")
        print(output["answer"])
        print("\nWaiting for manual ratings...")

        row = {
            "id": case["id"],
            "category": case.get("category", ""),
            "query": case.get("input", ""),
            "notes": case.get("notes", ""),
        }
        for metric in MANUAL_DIMENSIONS:
            row[metric] = prompt_for_rating(metric)
        rows.append(row)

    return rows


def run_automated_ragas_evaluation(
    test_cases: List[Dict],
    outputs: Dict[str, Dict],
) -> List[Dict]:
    """Evaluate generated answers with automated ragas metrics."""
    last_error: Exception | None = None

    for provider_name, model_name in get_ragas_eval_targets():
        scorers = build_ragas_scorers(provider_name, model_name)
        rows: List[Dict] = []

        print(
            f"\nRunning automated ragas evaluation with {provider_name}: {model_name}"
        )
        try:
            for case in test_cases:
                output = outputs.get(case["id"])
                if not output:
                    continue

                row = {
                    "id": case["id"],
                    "category": case.get("category", ""),
                    "query": output["query"],
                }
                evaluation_views = build_stage_evaluation_views(output)
                for level_name in EVALUATION_LEVELS:
                    level_output = evaluation_views[level_name]
                    for metric_name, scorer in scorers.items():
                        row[f"{level_name}_{metric_name}"] = score_with_ragas(
                            metric_name, scorer, level_output
                        )
                rows.append(row)
            return rows
        except Exception as exc:
            last_error = exc
            print(
                f"Model {provider_name}/{model_name} failed for ragas evaluation: {exc}"
            )

    if last_error is not None:
        raise last_error
    raise RuntimeError("No ragas evaluation models were configured.")


def save_manual_ratings(rows: List[Dict]) -> Path:
    """Persist the manual evaluation scores to CSV."""
    fieldnames = ["id", "category", "query", "notes"] + MANUAL_DIMENSIONS
    with open(MANUAL_RATINGS_OUTPUT_PATH, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return MANUAL_RATINGS_OUTPUT_PATH


def save_ragas_metrics(rows: List[Dict]) -> Path:
    """Persist the automated ragas metrics to CSV."""
    metric_columns = [
        f"{level_name}_{metric_name}"
        for level_name in EVALUATION_LEVELS
        for metric_name in AUTOMATED_RAGAS_DIMENSIONS
    ]
    fieldnames = ["id", "category", "query"] + metric_columns
    with open(AUTOMATED_RAGAS_OUTPUT_PATH, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return AUTOMATED_RAGAS_OUTPUT_PATH


def generate_outputs(test_cases: List[Dict]) -> Dict[str, Dict]:
    """Run the three-stage pipeline on every test case and cache the outputs in memory."""
    outputs: Dict[str, Dict] = {}
    stage1_retriever, stage2_retriever, stage3_retriever = load_retrievers()

    print(f"Using generation model: {GEN_MODEL}")
    for case in test_cases:
        case_id = case["id"]
        query = case.get("input", "").strip()
        if not query:
            continue

        print(f"Running {case_id}: {query}")
        try:
            outputs[case_id] = run_three_stage_pipeline(
                user_question=query,
                stage1_retriever=stage1_retriever,
                stage2_retriever=stage2_retriever,
                stage3_retriever=stage3_retriever,
                verbose=True,
            )
            print(f"Completed {case_id}")
        except Exception as exc:
            print(f"Skipping {case_id} due to error: {exc}")

    return outputs


def main() -> None:
    test_cases = load_test_cases()
    if not test_cases:
        raise ValueError(f"No test cases found in {TESTS_PATH}")

    outputs = generate_outputs(test_cases)
    if not outputs:
        raise RuntimeError("No evaluation outputs were generated.")

    # print("\nGeneration finished. Starting manual rating...")
    # manual_rows = run_manual_evaluation(test_cases, outputs)
    # manual_path = save_manual_ratings(manual_rows)
    # print(f"\nSaved manual ratings to: {manual_path}")

    try:
        print("\nManual rating finished. Starting automated ragas scoring...")
        ragas_rows = run_automated_ragas_evaluation(test_cases, outputs)
        ragas_path = save_ragas_metrics(ragas_rows)
        print(f"Saved automated ragas metrics to: {ragas_path}")
    except Exception as exc:
        print(f"Automated ragas evaluation was skipped: {exc}")


if __name__ == "__main__":
    main()
