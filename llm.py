"""Answer synthesis with citations using free providers (Groq or Google AI Studio)."""
from __future__ import annotations

import os
import re

from dotenv import load_dotenv

from rag_core import Hit

load_dotenv()

SYSTEM = """You are a careful research assistant. Answer ONLY using the numbered context passages.
Rules:
- Cite every factual sentence with the passage tag(s), e.g. [S1] or [S2][S3].
- Never cite a passage that does not support the sentence.
- If the context does not contain the answer, say: "I could not find this in the document."
- Be concise and precise."""


def build_prompt(question: str, hits: list[Hit]) -> str:
    ctx = "\n\n".join(
        f"[S{i}] (Section: {h.chunk.section} | Page: {h.chunk.page})\n{h.chunk.text}"
        for i, h in enumerate(hits, 1)
    )
    return f"Context:\n{ctx}\n\nQuestion: {question}\n\nAnswer with citations:"


def call_llm(prompt: str, provider: str | None = None, system: str = SYSTEM) -> str:
    provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower()
    if provider == "groq":
        from groq import Groq
        r = Groq(api_key=os.environ["GROQ_API_KEY"]).chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return r.choices[0].message.content
    if provider == "gemini":
        from google import genai
        from google.genai import types
        r = genai.Client(api_key=os.environ["GEMINI_API_KEY"]).models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system, temperature=0.1),
        )
        return r.text
    raise ValueError("LLM_PROVIDER must be 'groq' or 'gemini'")


def answer(question: str, hits: list[Hit], provider: str | None = None) -> dict:
    text = call_llm(build_prompt(question, hits), provider)
    cited = sorted({int(n) for n in re.findall(r"\[S(\d+)\]", text) if 0 < int(n) <= len(hits)})
    sources = [
        {"tag": f"S{n}", "page": hits[n - 1].chunk.page, "section": hits[n - 1].chunk.section,
         "file": hits[n - 1].chunk.source, "text": hits[n - 1].chunk.text}
        for n in cited
    ]
    return {"answer": text, "sources": sources}
