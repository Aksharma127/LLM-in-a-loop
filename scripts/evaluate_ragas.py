"""
RAGAS-based evaluation script — replaces keyword-matching with real
groundedness/relevance/precision metrics.

Judge LLM: Groq strong model (via the same provider config as the app).
Judge embeddings: local MiniLM (same one the app already uses for
retrieval), so eval adds zero new infra and zero new API cost beyond
the judge LLM calls.

Metrics:
- faithfulness        — is every claim in the answer supported by the
                         retrieved context? (hallucination detector)
- answer_relevancy    — does the answer actually address the question?
- context_precision   — of the retrieved chunks, how many were relevant?
- context_recall      — did retrieval find what it needed to?
                         (only computed for cases with a ground_truth)

Usage:
    python -m scripts.evaluate_ragas
    python -m scripts.evaluate_ragas --api-base http://localhost:8000
    python -m scripts.evaluate_ragas --out results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_settings  # noqa: E402

# ── Eval cases ───────────────────────────────────────────────────
# Same as the original eval set, extended with optional ground_truth.
# Add ground_truth wherever you can — it unlocks context_recall, which
# is the metric that tells you whether retrieval is even finding the
# right material, as opposed to just not contradicting it.

EVAL_CASES: list[dict[str, Any]] = [
    {
        "query": "What is LangGraph?",
        "expected_route": "retrieve",
        "ground_truth": (
            "LangGraph is a library for building stateful, multi-actor "
            "applications with LLMs, built on top of LangChain, that "
            "supports cyclic graphs rather than only DAGs."
        ),
    },
    {
        "query": "How does hybrid search work?",
        "expected_route": "retrieve",
        "ground_truth": (
            "Hybrid search combines dense vector (semantic) search with "
            "sparse keyword search such as BM25, merging the two result "
            "sets using a fusion method like Reciprocal Rank Fusion (RRF)."
        ),
    },
    {
        "query": "What is 2 + 2?",
        "expected_route": "direct",
        "ground_truth": "4",
    },
    {
        "query": "What is RAG and how does it enhance LLM responses?",
        "expected_route": "retrieve",
        "ground_truth": (
            "Retrieval-Augmented Generation (RAG) retrieves relevant "
            "external context and provides it to an LLM at generation "
            "time, grounding the answer in that context instead of "
            "relying solely on parametric knowledge, which reduces "
            "hallucination."
        ),
    },
]


# ── Step 1: call the running API for each case ────────────────────

async def collect_responses(
    api_base: str, cases: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Hit the live /chat endpoint for every eval case."""
    collected = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        for case in cases:
            try:
                resp = await client.post(
                    f"{api_base}/api/v1/chat",
                    json={"message": case["query"]},
                )
                resp.raise_for_status()
                data = resp.json()

                contexts = [
                    src.get("full_text") or src.get("text", "")
                    for src in data.get("sources", [])
                ]

                collected.append(
                    {
                        "query": case["query"],
                        "answer": data.get("answer", ""),
                        "contexts": contexts or [""],  # ragas needs non-empty
                        "ground_truth": case.get("ground_truth"),
                        "expected_route": case.get("expected_route"),
                        "actual_route": data.get("route"),
                        "critic_verdict": data.get("critic_verdict"),
                    }
                )
                print(f"  collected: {case['query'][:60]}")

            except Exception as e:
                print(f"  FAILED: {case['query'][:60]} — {e}")
                collected.append(
                    {
                        "query": case["query"],
                        "answer": "",
                        "contexts": [""],
                        "ground_truth": case.get("ground_truth"),
                        "expected_route": case.get("expected_route"),
                        "actual_route": None,
                        "critic_verdict": None,
                        "error": str(e),
                    }
                )
    return collected


# ── Step 2: build a RAGAS dataset and score it ─────────────────────

def build_ragas_judge():
    """
    Wrap the app's own Groq strong model + local MiniLM embeddings as
    the RAGAS judge, so evaluation reuses exactly what's already
    configured — no separate OpenAI key, no new infra.
    """
    from langchain_groq import ChatGroq
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from langchain_huggingface import HuggingFaceEmbeddings

    settings = get_settings()

    judge_llm = LangchainLLMWrapper(
        ChatGroq(
            model=settings.strong_model,
            api_key=settings.groq_api_key,
            temperature=0.0,
        )
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=settings.embedding_model)
    )
    return judge_llm, judge_embeddings


def run_ragas_scoring(records: list[dict[str, Any]]) -> dict[str, Any]:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    judge_llm, judge_embeddings = build_ragas_judge()

    # context_recall requires ground_truth; split records so cases
    # without one don't get silently scored as 0 / crash the metric.
    has_gt = [r for r in records if r.get("ground_truth")]
    no_gt = [r for r in records if not r.get("ground_truth")]

    results_by_query: dict[str, dict[str, float]] = {}

    def _to_dataset(recs: list[dict[str, Any]]) -> Dataset:
        return Dataset.from_dict(
            {
                "question": [r["query"] for r in recs],
                "answer": [r["answer"] for r in recs],
                "contexts": [r["contexts"] for r in recs],
                **(
                    {"ground_truth": [r["ground_truth"] for r in recs]}
                    if recs and recs[0].get("ground_truth")
                    else {}
                ),
            }
        )

    if has_gt:
        ds = _to_dataset(has_gt)
        scored = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=judge_llm,
            embeddings=judge_embeddings,
        )
        df = scored.to_pandas()
        for i, r in enumerate(has_gt):
            results_by_query[r["query"]] = df.iloc[i].to_dict()

    if no_gt:
        ds = _to_dataset(no_gt)
        scored = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=judge_llm,
            embeddings=judge_embeddings,
        )
        df = scored.to_pandas()
        for i, r in enumerate(no_gt):
            results_by_query[r["query"]] = df.iloc[i].to_dict()

    return results_by_query


# ── Step 3: combine with routing accuracy and print a report ──────

def build_report(
    records: list[dict[str, Any]], ragas_scores: dict[str, dict[str, float]]
) -> dict[str, Any]:
    rows = []
    for r in records:
        scores = ragas_scores.get(r["query"], {})
        route_correct = r["actual_route"] == r["expected_route"]

        row = {
            "query": r["query"],
            "expected_route": r["expected_route"],
            "actual_route": r["actual_route"],
            "route_correct": route_correct,
            "critic_verdict": r["critic_verdict"],
            "faithfulness": scores.get("faithfulness"),
            "answer_relevancy": scores.get("answer_relevancy"),
            "context_precision": scores.get("context_precision"),
            "context_recall": scores.get("context_recall"),
        }
        rows.append(row)

        status = "OK" if route_correct and (row["faithfulness"] or 0) >= 0.7 else "CHECK"
        print(f"\n[{status}] {r['query']}")
        print(f"  route: {r['actual_route']} (expected {r['expected_route']})")
        for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
            val = row[metric]
            if val is not None:
                print(f"  {metric:<18} {val:.2f}")

    def _avg(key: str) -> float | None:
        vals = [row[key] for row in rows if row[key] is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "total_cases": len(rows),
        "route_accuracy": sum(1 for r in rows if r["route_correct"]) / len(rows),
        "avg_faithfulness": _avg("faithfulness"),
        "avg_answer_relevancy": _avg("answer_relevancy"),
        "avg_context_precision": _avg("context_precision"),
        "avg_context_recall": _avg("context_recall"),
    }

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<24} {v:.2%}" if "accuracy" not in k else f"  {k:<24} {v:.2f}")
        else:
            print(f"  {k:<24} {v}")

    return {"results": rows, "summary": summary}


# ── Entry point ─────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--out", default=None, help="Optional path to write JSON results")
    args = parser.parse_args()

    print("Collecting responses from the running API...")
    records = await collect_responses(args.api_base, EVAL_CASES)

    print("\nScoring with RAGAS (this calls the judge LLM — a few seconds per case)...")
    ragas_scores = run_ragas_scoring(records)

    report = build_report(records, ragas_scores)

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
        print(f"\nWrote results to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
