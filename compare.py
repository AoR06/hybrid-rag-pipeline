"""Retrieval comparison: BM25 vs Dense vs Hybrid vs Hybrid+Rerank.

Usage:
  python compare.py data/paper --eval eval_questions.json
  python compare.py data/paper --auto 30        # LLM generates questions from random chunks

eval JSON format: [{"question": "...", "keywords": ["exact phrase from answer", ...]}, ...]
(a chunk is relevant if it contains ANY keyword; items may use "chunk_id" instead)
"""
import argparse
import json
import random

import matplotlib.pyplot as plt
import pandas as pd

from rag_core import HybridIndex, timed
from llm import call_llm

MODES = ["bm25", "dense", "hybrid", "hybrid+rerank"]


def auto_questions(idx: HybridIndex, n: int) -> list[dict]:
    random.seed(7)
    pool = [c for c in idx.chunks if len(c.text) > 400]
    items = []
    for c in random.sample(pool, min(n, len(pool))):
        q = call_llm(
            f"Write ONE specific question that is answered by this passage. "
            f"Do not copy long phrases from it. Output only the question.\n\n{c.text}",
            system="You write evaluation questions for retrieval systems.",
        ).strip()
        items.append({"question": q, "chunk_id": c.id})
    return items


def is_rel(item: dict, chunk) -> bool:
    if "chunk_id" in item and chunk.id == item["chunk_id"]:
        return True
    return any(k.lower() in chunk.text.lower() for k in item.get("keywords", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("store")
    ap.add_argument("--eval")
    ap.add_argument("--auto", type=int)
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()

    idx = HybridIndex.load(a.store)
    items = auto_questions(idx, a.auto) if a.auto else json.load(open(a.eval))
    if a.auto:
        json.dump(items, open("eval_auto.json", "w"), indent=2)

    rows, per_q = [], []
    for it in items:
        for m in MODES:
            hits, ms = timed(idx.search, it["question"], m, a.k)
            ranks = [h.rank for h in hits if is_rel(it, h.chunk)]
            first = min(ranks) if ranks else None
            rows.append({"mode": m, "hit": first is not None, "rr": 1 / first if first else 0.0, "ms": ms})
            per_q.append({"question": it["question"], "mode": m, "first_relevant_rank": first})

    df = pd.DataFrame(rows).groupby("mode").agg(
        **{f"Hit@{a.k}": ("hit", "mean"), f"MRR@{a.k}": ("rr", "mean"), "latency_ms": ("ms", "mean")}
    ).reindex(MODES).round(3)
    print(df.to_markdown())
    df.to_csv("comparison_results.csv")
    pd.DataFrame(per_q).to_csv("comparison_per_question.csv", index=False)
    open("comparison_results.md", "w").write(df.to_markdown())

    ax = df[[f"Hit@{a.k}", f"MRR@{a.k}"]].plot(kind="bar", figsize=(7, 4), rot=0)
    ax.set_title(f"Retrieval comparison ({len(items)} questions)")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig("comparison_chart.png", dpi=150)
    print("Saved comparison_results.csv/.md, comparison_per_question.csv, comparison_chart.png")


if __name__ == "__main__":
    main()
