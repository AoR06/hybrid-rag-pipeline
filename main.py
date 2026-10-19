"""CLI:  python main.py ingest paper.pdf   |   python main.py ask data/paper "question" [--mode ...]"""
import argparse
from pathlib import Path

from rag_core import HybridIndex
from llm import answer

p = argparse.ArgumentParser()
sub = p.add_subparsers(dest="cmd", required=True)
a = sub.add_parser("ingest"); a.add_argument("pdf")
b = sub.add_parser("ask"); b.add_argument("store"); b.add_argument("question")
b.add_argument("--mode", default="hybrid+rerank", choices=["bm25", "dense", "hybrid", "hybrid+rerank"])
b.add_argument("--k", type=int, default=5)
args = p.parse_args()

if args.cmd == "ingest":
    idx = HybridIndex.build(args.pdf)
    print(f"Indexed {len(idx.chunks)} chunks -> {idx.dir}")
else:
    idx = HybridIndex.load(args.store)
    hits = idx.search(args.question, args.mode, args.k)
    out = answer(args.question, hits)
    print("\n" + out["answer"] + "\n\nSources:")
    for s in out["sources"]:
        print(f"  [{s['tag']}] {s['file']} - p.{s['page']} - {s['section']}")
