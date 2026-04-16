"""
rag.py
------
Pure RAG retrieval layer — no CrewAI here.
Used by the agent layer and can be tested independently.
"""

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


def rag_query(query: str, top_k: int = 5) -> RAGResult:
    """
    Retrieve top_k relevant chunks and generate an answer.
    top_k=5 gives better cross-document coverage than default 3.
    """
    embed_model = get_embed_model()
    Settings.embed_model = embed_model
    Settings.llm = Groq(
        model=settings.MODEL_NAME,
        temperature=settings.MODEL_TEMPERATURE,
        api_key=settings.GROQ_API_KEY,
    )

    index = get_or_create_index()
    query_engine = index.as_query_engine(similarity_top_k=top_k)
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
