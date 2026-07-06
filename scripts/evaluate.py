"""
Evaluation script — score retrieval and answer quality.

Uses simple custom evaluation (RAGAS optional).

Usage:
    python -m scripts.evaluate
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Test cases ───────────────────────────────────────────────

EVAL_CASES = [
    {
        "query": "What is LangGraph?",
        "expected_keywords": ["stateful", "multi-actor", "LangChain", "cyclic"],
        "expected_route": "retrieve",
    },
    {
        "query": "How does hybrid search work?",
        "expected_keywords": ["dense", "BM25", "RRF", "rank"],
        "expected_route": "retrieve",
    },
    {
        "query": "What is 2 + 2?",
        "expected_keywords": ["4"],
        "expected_route": "direct",
    },
    {
        "query": "What is RAG and how does it enhance LLM responses?",
        "expected_keywords": ["retrieval", "generation", "grounding"],
        "expected_route": "retrieve",
    },
]


async def evaluate(api_base: str = "http://localhost:8000") -> dict:
    """Run evaluation against the running API."""
    results = []

    async with httpx.AsyncClient(timeout=120.0) as client:
        for case in EVAL_CASES:
            try:
                resp = await client.post(
                    f"{api_base}/api/v1/chat",
                    json={"message": case["query"]},
                )
                data = resp.json()

                # Score: keyword coverage
                answer_lower = data.get("answer", "").lower()
                keywords_found = sum(
                    1 for kw in case["expected_keywords"]
                    if kw.lower() in answer_lower
                )
                keyword_score = keywords_found / len(case["expected_keywords"])

                # Route accuracy
                route_correct = data.get("route") == case["expected_route"]

                # Critic verdict
                verdict = data.get("critic_verdict", "unknown")

                result = {
                    "query": case["query"],
                    "keyword_coverage": keyword_score,
                    "route_correct": route_correct,
                    "actual_route": data.get("route"),
                    "expected_route": case["expected_route"],
                    "critic_verdict": verdict,
                    "answer_length": len(data.get("answer", "")),
                    "sources_count": len(data.get("sources", [])),
                    "passed": keyword_score >= 0.5 and route_correct,
                }
                results.append(result)

                status = "✅" if result["passed"] else "❌"
                print(f"{status} {case['query'][:50]}")
                print(f"   Keywords: {keyword_score:.0%} | Route: {'✓' if route_correct else '✗'} | Critic: {verdict}")

            except Exception as e:
                print(f"❌ {case['query'][:50]} — Error: {e}")
                results.append({
                    "query": case["query"],
                    "error": str(e),
                    "passed": False,
                })

    # Summary
    passed = sum(1 for r in results if r.get("passed"))
    total = len(results)
    avg_keyword = sum(r.get("keyword_coverage", 0) for r in results) / total if total else 0

    summary = {
        "total_cases": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": f"{passed/total:.0%}" if total else "N/A",
        "avg_keyword_coverage": f"{avg_keyword:.0%}",
    }

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed ({summary['pass_rate']})")
    print(f"Avg keyword coverage: {summary['avg_keyword_coverage']}")

    return {"results": results, "summary": summary}


if __name__ == "__main__":
    asyncio.run(evaluate())
