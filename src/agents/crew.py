"""
crew.py
-------
CrewAI agents and tasks for different analytical query types:
  - qa          : factual Q&A with citations
  - summarise   : document / topic summarisation
  - compare     : cross-document comparison
  - extract     : numbers, tables, structured data
"""

import logging
from typing import Literal
from pydantic import BaseModel

from crewai import Agent, Task, Crew, LLM

from src.config.settings import settings
from src.agents.rag import rag_query, RAGResult

logger = logging.getLogger(__name__)

QueryType = Literal["qa", "summarise", "compare", "extract"]


# ── Output schema ─────────────────────────────────────────────────────────────

class AnalyticalAnswer(BaseModel):
    answer: str
    sources: list[str]
    query_type: str
    tool_used: str
    rationale: str
    confidence: str   # "high" | "medium" | "low"


# ── LLM factory ───────────────────────────────────────────────────────────────

def _get_llm() -> LLM:
    return LLM(
        model=f"groq/{settings.MODEL_NAME}",
        temperature=settings.MODEL_TEMPERATURE,
    )


# ── Agent definitions ─────────────────────────────────────────────────────────

def _qa_agent(llm: LLM) -> Agent:
    return Agent(
        role="Document QA Specialist",
        llm=llm,
        goal="Answer user questions accurately using only retrieved document context, always citing sources.",
        backstory=(
            "You are a meticulous research analyst who reads document passages carefully "
            "and provides precise, well-cited answers. You never speculate beyond the evidence."
        ),
        verbose=False,
    )


def _summarise_agent(llm: LLM) -> Agent:
    return Agent(
        role="Document Summarisation Expert",
        llm=llm,
        goal="Produce clear, structured summaries of document content, highlighting key points.",
        backstory=(
            "You are an expert at distilling complex documents into clear, readable summaries. "
            "You preserve important details while making content accessible."
        ),
        verbose=False,
    )


def _compare_agent(llm: LLM) -> Agent:
    return Agent(
        role="Cross-Document Analyst",
        llm=llm,
        goal="Compare and contrast information across multiple documents, identifying agreements, contradictions, and unique insights.",
        backstory=(
            "You specialise in synthesis — reading across multiple sources to find patterns, "
            "differences, and connections that aren't visible when reading one document at a time."
        ),
        verbose=False,
    )


def _extract_agent(llm: LLM) -> Agent:
    return Agent(
        role="Data Extraction Specialist",
        llm=llm,
        goal="Extract structured data — numbers, statistics, dates, tables — from document context accurately.",
        backstory=(
            "You are a data analyst who pulls precise figures, statistics, and structured information "
            "from documents. You present extracted data in clean, readable formats."
        ),
        verbose=False,
    )


# ── Task builders ─────────────────────────────────────────────────────────────

def _build_task(agent: Agent, query: str, rag_result: RAGResult,
                query_type: str, chat_history: str) -> Task:

    type_instructions = {
        "qa": "Answer the question directly and precisely. Cite the source documents.",
        "summarise": "Produce a structured summary with key points as bullet points or sections.",
        "compare": "Create a structured comparison. Use a table or side-by-side format where helpful.",
        "extract": "Extract all relevant numbers, statistics, dates, or structured data. Present in a clean list or table.",
    }

    instruction = type_instructions.get(query_type, type_instructions["qa"])

    return Task(
        agent=agent,
        name=f"{query_type.title()} Task",
        description=f"""
User query: "{query}"
Query type: {query_type}
Prior conversation: {chat_history}

Retrieved document context (use ONLY this — do not fabricate):
---
{rag_result.raw_answer}
---
Source files available: {rag_result.source_files}

Task instruction: {instruction}

If the retrieved context does not contain sufficient information, clearly state:
"The available documents do not contain enough information to answer this query."

Always include which source files your answer draws from.
        """,
        expected_output=(
            'Structured JSON with keys: answer (string), sources (list of filenames), '
            'query_type (string), tool_used (string), rationale (string), confidence ("high"/"medium"/"low")'
        ),
        output_pydantic=AnalyticalAnswer,
    )


# ── Main entry point ──────────────────────────────────────────────────────────

def detect_query_type(query: str) -> QueryType:
    """Heuristic query type detection."""
    q = query.lower()
    if any(w in q for w in ["compare", "difference", "vs", "versus", "contrast", "similar", "both"]):
        return "compare"
    if any(w in q for w in ["summarise", "summarize", "summary", "overview", "brief", "what is about"]):
        return "summarise"
    if any(w in q for w in ["how many", "number", "count", "total", "percent", "%", "statistic",
                             "data", "figure", "table", "extract", "list all"]):
        return "extract"
    return "qa"


def run_analytical_query(
    query: str,
    chat_history: list[dict],
    query_type: QueryType | None = None,
) -> dict:
    """
    Full pipeline: RAG retrieval → agent selection → structured answer.

    Args:
        query: User's natural language question.
        chat_history: List of {"role": ..., "content": ...} dicts.
        query_type: Override auto-detection if desired.

    Returns:
        dict matching AnalyticalAnswer schema.
    """
    # Step 1: Auto-detect query type
    if query_type is None:
        query_type = detect_query_type(query)
    logger.info(f"Query type: {query_type}")

    # Step 2: RAG — more chunks for compare/extract
    top_k = 7 if query_type in ("compare", "extract") else 5
    rag_result = rag_query(query, top_k=top_k)

    if not rag_result.has_answer:
        return {
            "answer": "The available documents do not contain information relevant to your query.",
            "sources": [],
            "query_type": query_type,
            "tool_used": "RAG (no match)",
            "rationale": "Retrieval returned no relevant context.",
            "confidence": "low",
        }

    # Step 3: Select agent
    llm = _get_llm()
    agent_map = {
        "qa": _qa_agent,
        "summarise": _summarise_agent,
        "compare": _compare_agent,
        "extract": _extract_agent,
    }
    agent = agent_map[query_type](llm)

    # Step 4: Format chat history
    history_str = (
        "\n".join(f"{m['role'].title()}: {m['content']}" for m in chat_history[-6:])
        if chat_history else "none"
    )

    # Step 5: Run crew
    task = _build_task(agent, query, rag_result, query_type, history_str)
    crew = Crew(agents=[agent], tasks=[task], verbose=False)

    try:
        result = crew.kickoff().to_dict()
    except Exception as e:
        logger.error(f"Crew error: {e}")
        result = {
            "answer": rag_result.raw_answer,
            "sources": rag_result.source_files,
            "tool_used": "RAG (crew fallback)",
            "rationale": f"Agent error: {e}",
            "confidence": "medium",
        }

    result["query_type"] = query_type
    if not result.get("sources"):
        result["sources"] = rag_result.source_files

    return result
