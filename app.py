"""Streamlit UI:  streamlit run app.py"""
from pathlib import Path

import streamlit as st

from rag_core import HybridIndex, timed
from llm import answer

st.set_page_config(page_title="Hybrid RAG with Citations", page_icon="📚", layout="wide")
st.title("📚 Hybrid RAG: BM25 + Dense + Reranker")


@st.cache_resource(show_spinner="Building index (first time only)...")
def get_index(pdf_path: str) -> HybridIndex:
    store = Path("data") / Path(pdf_path).stem
    return HybridIndex.load(store) if (store / "chunks.json").exists() else HybridIndex.build(pdf_path)


with st.sidebar:
    up = st.file_uploader("Upload a PDF", type="pdf")
    provider = st.selectbox("LLM provider", ["groq", "gemini"])
    mode = st.selectbox("Retrieval mode", ["hybrid+rerank", "hybrid", "dense", "bm25"])
    k = st.slider("Passages sent to LLM", 2, 10, 5)

if not up:
    st.info("Upload a technical PDF or research paper to start.")
    st.stop()

Path("data/uploads").mkdir(parents=True, exist_ok=True)
pdf_path = Path("data/uploads") / up.name
pdf_path.write_bytes(up.getvalue())
idx = get_index(str(pdf_path))
st.sidebar.success(f"{len(idx.chunks)} chunks indexed")

tab_qa, tab_cmp = st.tabs(["Ask", "Compare retrievers"])

with tab_qa:
    q = st.text_input("Your question")
    if q:
        hits, ms = timed(idx.search, q, mode, k)
        with st.spinner("Generating answer..."):
            out = answer(q, hits, provider)
        st.markdown(out["answer"])
        st.caption(f"Retrieval: {mode} in {ms:.0f} ms")
        st.subheader("Sources")
        for s in out["sources"]:
            with st.expander(f"[{s['tag']}] {s['file']} - page {s['page']} - {s['section']}"):
                st.write(s["text"])

with tab_cmp:
    cq = st.text_input("Query to compare", key="cmp")
    if cq:
        cols = st.columns(4)
        for col, m in zip(cols, ["bm25", "dense", "hybrid", "hybrid+rerank"]):
            hits, ms = timed(idx.search, cq, m, 5)
            col.markdown(f"**{m}**  \n{ms:.0f} ms")
            for h in hits:
                col.markdown(f"- p.{h.chunk.page} · {h.chunk.section[:35]} · `{h.score:.3f}`")
                col.caption(h.chunk.text[:160] + "...")
