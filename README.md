# Production Hybrid RAG Pipeline with Source Citations
SkillCraft Technology – Task 01

## Architecture
PDF -> PyMuPDF (text + page no. + section headings) -> page-bounded chunks (~900 chars, 150 overlap)
  -> BM25 (rank_bm25)  +  Dense (bge-small-en-v1.5 in ChromaDB, cosine)
  -> Reciprocal Rank Fusion (top 20) -> Cross-encoder rerank (bge-reranker-base) -> top 5
  -> LLM (Groq Llama-3.3-70B or Gemini Flash) -> answer with [S#] tags -> mapped to file / page / section

Design choices worth mentioning in your report:
- Chunks never cross page boundaries, so every citation has an exact page.
- RRF fuses ranks, not raw scores (BM25 and cosine scores are not comparable).
- Reranking is applied only to a small candidate pool, since cross-encoders are slow.
- The prompt forces citation tags and an explicit "not found" fallback, to reduce hallucination.

## Run
    pip install -r requirements.txt
    cp .env.example .env        # add GROQ_API_KEY or GEMINI_API_KEY
    python main.py ingest paper.pdf
    python main.py ask data/paper "What is the main contribution?"
    streamlit run app.py

## Retrieval comparison (BM25 vs Dense vs Hybrid)
    python compare.py data/paper --auto 30                    # LLM-generated questions
    python compare.py data/paper --eval eval_questions.json   # your own hand-written set
Outputs: comparison_results.md/.csv, comparison_chart.png, comparison_per_question.csv
Hand-written questions (20+, with keyword ground truth) are more credible than auto ones; mix both.
Include in the report: exact-term queries (BM25 wins), paraphrased queries (dense wins),
and cases where hybrid + rerank beats both.

## Deploy the UI (optional)
Hugging Face Spaces (Streamlit SDK) or Streamlit Community Cloud: push this folder to GitHub,
add GROQ_API_KEY / GEMINI_API_KEY in the platform's Secrets.
