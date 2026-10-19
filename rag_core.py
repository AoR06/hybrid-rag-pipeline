"""Core of the hybrid RAG pipeline: ingestion, BM25, dense, hybrid (RRF), reranking."""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import chromadb
import fitz  # PyMuPDF
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
STOPWORDS = set(
    "a an the and or of to in on for with is are was were be been by as at from that this "
    "it its into than then these those which who whom we our their can may not no".split()
)

_models: dict = {}


def get_embedder() -> SentenceTransformer:
    if "emb" not in _models:
        _models["emb"] = SentenceTransformer(EMBED_MODEL)
    return _models["emb"]


def get_reranker() -> CrossEncoder:
    if "rr" not in _models:
        _models["rr"] = CrossEncoder(RERANK_MODEL, max_length=512)
    return _models["rr"]


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    id: int
    text: str
    page: int
    section: str
    source: str


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int
    mode: str


# --------------------------------------------------------------------------- #
# PDF ingestion: text + page numbers + section headings
# --------------------------------------------------------------------------- #
HEADING_RE = re.compile(r"^(\d+(\.\d+)*\.?\s+\S|[IVX]+\.\s+\S|abstract|references|introduction|conclusion)", re.I)


def extract_pages(pdf_path: str) -> list[tuple[int, list[tuple[str, str]]]]:
    """Return [(page_no, [(section, text), ...]), ...]. Headings detected via font size/bold."""
    doc = fitz.open(pdf_path)
    sizes: Counter = Counter()
    for page in doc:
        for b in page.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                for s in line["spans"]:
                    sizes[round(s["size"])] += len(s["text"])
    body = sizes.most_common(1)[0][0] if sizes else 10

    pages, section = [], "Front matter"
    for pno, page in enumerate(doc, start=1):
        segs: list[tuple[str, str]] = []
        for b in page.get_text("dict")["blocks"]:
            para = ""
            for line in b.get("lines", []):
                spans = line["spans"]
                text = "".join(s["text"] for s in spans).strip()
                if not text:
                    continue
                size = max(s["size"] for s in spans)
                bold = any(s["flags"] & 16 for s in spans)
                is_head = len(text) < 120 and (
                    size >= body + 1.5 or (bold and len(text) < 70 and HEADING_RE.match(text))
                )
                if is_head:
                    if para.strip():
                        segs.append((section, para.strip()))
                        para = ""
                    section = text
                    continue
                if para.endswith("-"):
                    para = para[:-1] + text  # de-hyphenate line breaks
                else:
                    para += (" " if para else "") + text
            if para.strip():
                segs.append((section, para.strip()))
        pages.append((pno, segs))
    return pages


def chunk_pages(pages, source: str, size: int = 900, overlap: int = 150) -> list[Chunk]:
    """Chunks never cross page boundaries, so every citation has an exact page."""
    chunks: list[Chunk] = []
    for pno, segs in pages:
        merged: list[tuple[str, str]] = []
        for sec, txt in segs:  # merge consecutive paragraphs of the same section
            if merged and merged[-1][0] == sec:
                merged[-1] = (sec, merged[-1][1] + "\n" + txt)
            else:
                merged.append((sec, txt))
        for sec, text in merged:
            start = 0
            while start < len(text):
                end = min(start + size, len(text))
                if end < len(text):
                    cut = max(text.rfind(". ", start + size // 2, end), text.rfind(" ", start + size // 2, end))
                    if cut > start:
                        end = cut + 1
                piece = text[start:end].strip()
                if len(piece) > 40:
                    chunks.append(Chunk(len(chunks), piece, pno, sec, source))
                if end >= len(text):
                    break
                start = max(end - overlap, start + 1)
    return chunks


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #
class HybridIndex:
    def __init__(self, store_dir: str | Path):
        self.dir = Path(store_dir)
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None
        self.col = None

    # ---- build / load ---- #
    @classmethod
    def build(cls, pdf_path: str, store_dir: str | Path | None = None, **chunk_kw) -> "HybridIndex":
        pdf = Path(pdf_path)
        store = Path(store_dir or f"data/{pdf.stem}")
        store.mkdir(parents=True, exist_ok=True)
        idx = cls(store)
        idx.chunks = chunk_pages(extract_pages(str(pdf)), pdf.name, **chunk_kw)
        (store / "chunks.json").write_text(json.dumps([asdict(c) for c in idx.chunks]))

        client = chromadb.PersistentClient(path=str(store / "chroma"))
        try:
            client.delete_collection("chunks")
        except Exception:
            pass
        idx.col = client.create_collection("chunks", metadata={"hnsw:space": "cosine"})
        embs = get_embedder().encode([c.text for c in idx.chunks], batch_size=32,
                                     normalize_embeddings=True, show_progress_bar=True)
        for i in range(0, len(idx.chunks), 500):  # chroma batch limit safety
            sl = slice(i, i + 500)
            idx.col.add(
                ids=[str(c.id) for c in idx.chunks[sl]],
                embeddings=embs[sl].tolist(),
                documents=[c.text for c in idx.chunks[sl]],
                metadatas=[{"page": c.page, "section": c.section} for c in idx.chunks[sl]],
            )
        idx._build_bm25()
        return idx

    @classmethod
    def load(cls, store_dir: str | Path) -> "HybridIndex":
        idx = cls(store_dir)
        idx.chunks = [Chunk(**d) for d in json.loads((idx.dir / "chunks.json").read_text())]
        idx.col = chromadb.PersistentClient(path=str(idx.dir / "chroma")).get_collection("chunks")
        idx._build_bm25()
        return idx

    def _build_bm25(self):
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])

    # ---- retrievers ---- #
    def bm25_search(self, query: str, k: int = 5) -> list[Hit]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [Hit(self.chunks[i], float(scores[i]), r + 1, "bm25") for r, i in enumerate(top)]

    def dense_search(self, query: str, k: int = 5) -> list[Hit]:
        q = get_embedder().encode([BGE_QUERY_PREFIX + query], normalize_embeddings=True)[0]
        res = self.col.query(query_embeddings=[q.tolist()], n_results=k)
        ids = [int(i) for i in res["ids"][0]]
        sims = [1 - d for d in res["distances"][0]]  # cosine distance -> similarity
        return [Hit(self.chunks[i], float(s), r + 1, "dense") for r, (i, s) in enumerate(zip(ids, sims))]

    def hybrid_search(self, query: str, k: int = 5, pool: int = 20,
                      w_bm25: float = 1.0, w_dense: float = 1.0, rrf_k: int = 60) -> list[Hit]:
        """Reciprocal Rank Fusion of BM25 and dense rankings."""
        fused: dict[int, float] = {}
        for w, hits in ((w_bm25, self.bm25_search(query, pool)), (w_dense, self.dense_search(query, pool))):
            for h in hits:
                fused[h.chunk.id] = fused.get(h.chunk.id, 0.0) + w / (rrf_k + h.rank)
        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [Hit(self.chunks[i], s, r + 1, "hybrid") for r, (i, s) in enumerate(top)]

    def rerank(self, query: str, hits: list[Hit], k: int = 5, min_score: float | None = None) -> list[Hit]:
        """Cross-encoder scoring of (query, passage) pairs; optional sigmoid-score threshold."""
        if not hits:
            return []
        logits = get_reranker().predict([(query, h.chunk.text) for h in hits])
        probs = 1 / (1 + np.exp(-np.asarray(logits)))
        order = np.argsort(probs)[::-1]
        out = [Hit(hits[i].chunk, float(probs[i]), 0, "hybrid+rerank") for i in order]
        if min_score is not None:
            out = [h for h in out if h.score >= min_score] or out[:1]  # never return nothing
        out = out[:k]
        for r, h in enumerate(out):
            h.rank = r + 1
        return out

    def search(self, query: str, mode: str = "hybrid+rerank", k: int = 5, pool: int = 20,
               min_score: float | None = None) -> list[Hit]:
        if mode == "bm25":
            return self.bm25_search(query, k)
        if mode == "dense":
            return self.dense_search(query, k)
        if mode == "hybrid":
            return self.hybrid_search(query, k, pool)
        if mode == "hybrid+rerank":
            return self.rerank(query, self.hybrid_search(query, pool, pool), k, min_score)
        raise ValueError(f"unknown mode {mode}")


def timed(fn, *a, **kw):
    t = time.perf_counter()
    out = fn(*a, **kw)
    return out, (time.perf_counter() - t) * 1000
