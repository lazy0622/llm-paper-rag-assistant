from langchain_core.prompts import PromptTemplate

from app.config import settings
from app.services.llm_client import chat_completion


REWRITE_PROMPT = PromptTemplate.from_template(
    """You are a retrieval query rewriting assistant for an academic paper RAG system.

Rewrite the user question into one concise English retrieval query.
Keep key technical terms, paper method names, and abbreviations.
Do not answer the question.
Do not add explanations.
Return only the rewritten query.

User question:
{question}
"""
)


def rewrite_query_for_retrieval(question: str, run_id: str | None = None) -> str:
    """Rewrite a user question for retrieval while keeping the original question for answer generation."""
    if not settings.enable_query_rewrite:
        return question

    normalized_question = question.strip()
    if not normalized_question:
        return question

    try:
        prompt = REWRITE_PROMPT.format(question=normalized_question)
        model = settings.query_rewrite_model or None
        rewritten = chat_completion(prompt, run_id=run_id, model=model).strip()
    except Exception:
        # Query rewrite is an optimization, not the main product path. If the
        # local model or gateway fails, retrieval should still use the raw question.
        return question

    rewritten = _clean_rewrite_output(rewritten)
    if not rewritten:
        return question
    return rewritten


def _clean_rewrite_output(text: str) -> str:
    first_line = text.strip().splitlines()[0].strip()
    first_line = first_line.strip("\"'`：: ")
    if len(first_line) > 300:
        first_line = first_line[:300].rsplit(" ", 1)[0]
    return first_line
