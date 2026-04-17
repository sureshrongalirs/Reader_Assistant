"""
rag.py
------
Pure RAG retrieval layer — no CrewAI here.
Used by the agent layer and can be tested independently.
"""

import os
import logging
from dataclasses import dataclass, field

from llama_index.core import Settings
from llama_index.llms.groq import Groq

from src.config.settings import settings
from src.ingestion.ingestion import get_or_create_index, get_embed_model

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    raw_answer: str
    source_files: list[str] = field(default_factory=list)
    source_pages: list[str] = field(default_factory=list)
    has_answer: bool = True


def _get_groq_api_key() -> str:
    """
    Always read the Groq API key fresh at call time.

    Priority:
      1. settings object (may have been updated by app.py after secrets load)
      2. environment variable (set by app.py from st.secrets)
    This avoids the stale-singleton problem where the settings object
    was created at import time before Streamlit secrets were injected.
    """
    key = settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Add it to your .env file (local) or Streamlit secrets (cloud)."
        )
    return key


def rag_query(query: str, top_k: int = 5) -> RAGResult:
    """
    Retrieve top_k relevant chunks and generate an answer.
    top_k=5 gives better cross-document coverage than default 3.
    """
    api_key = _get_groq_api_key()
    embed_model = get_embed_model()

    # Always configure LlamaIndex settings fresh — never rely on a cached LLM
    Settings.embed_model = embed_model
    Settings.llm = Groq(
        model=settings.MODEL_NAME,
        temperature=settings.MODEL_TEMPERATURE,
        api_key=api_key,                  # ← explicit key, not relying on env
    )

    index = get_or_create_index()

    # Build query engine with the explicit LLM so it never falls back to OpenAI
    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        llm=Settings.llm,                 # ← pass LLM explicitly to engine too
    )
    response = query_engine.query(query)

    raw_answer = response.response or ""

    # Extract source metadata
    source_files, source_pages = [], []
    for meta in getattr(response, "metadata", {}).values():
        fname = meta.get("file_name", "")
        page  = meta.get("page_label", meta.get("page_number", ""))
        if fname and fname not in source_files:
            source_files.append(fname)
        if page:
            source_pages.append(f"{fname} p.{page}" if fname else f"p.{page}")

    has_answer = bool(raw_answer.strip()) and "empty" not in raw_answer.lower()

    logger.info(f"RAG → len={len(raw_answer)}  sources={source_files}")
    return RAGResult(
        raw_answer=raw_answer,
        source_files=source_files,
        source_pages=source_pages,
        has_answer=has_answer,
    )
