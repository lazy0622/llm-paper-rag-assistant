from typing import Literal

from langchain_core.prompts import PromptTemplate

from app.services.llm_client import chat_completion


AnswerStyle = Literal["brief", "standard", "detailed"]


GROUNDED_PROMPT = PromptTemplate.from_template(
    """你是一个严谨的论文知识库问答助手。

请只依据【已检索资料】回答问题，不要编造资料中不存在的信息。
如果资料中没有明确依据，请直接说明“资料中没有明确依据”，并给出还需要补充的资料类型。

【已检索资料】
{context}

【用户问题】
{question}

【回答要求】
{style_instruction}
无论哪种详细程度，都必须在每个主要事实结论后使用结构化引用标记 [S1]、[S2] 等，并同时写出文件名和页码；如果依据不足，明确说明不确定点。只能使用已检索资料中存在的标记。
"""
)

FALLBACK_PROMPT = PromptTemplate.from_template(
    """你是一个严谨的大模型应用学习助手。

知识库中没有找到能够直接回答用户问题的明确依据。
请基于你的通用知识回答，但必须清楚说明：以下内容为本地模型通用知识补充，不代表知识库来源。

【用户问题】
{question}

【回答要求】
1. 开头先写：知识库中没有找到明确依据。以下内容为本地模型通用知识补充。
2. {style_instruction}
3. 不要编造文件名、页码或引用来源。
"""
)


def answer_with_context(
    question: str,
    chunks: list[dict],
    answer_style: AnswerStyle = "standard",
    run_id: str | None = None,
) -> str:
    """Build a grounded prompt and call the local LLM."""
    context = _format_context(chunks)
    style_instruction = _answer_style_instruction(answer_style)
    # 用 LangChain PromptTemplate 管理变量化 Prompt；检索和模式判别仍由业务代码控制。
    prompt = GROUNDED_PROMPT.format(
        context=context,
        question=question,
        style_instruction=style_instruction,
    )
    return chat_completion(prompt, run_id=run_id)


def answer_with_fallback(question: str, answer_style: AnswerStyle = "standard", run_id: str | None = None) -> str:
    """Answer with the local model's general knowledge when the knowledge base is insufficient."""
    style_instruction = _fallback_style_instruction(answer_style)
    # 兜底 Prompt 同样模板化，但不接收 sources，避免把模型常识伪装成知识库依据。
    prompt = FALLBACK_PROMPT.format(
        question=question,
        style_instruction=style_instruction,
    )
    return chat_completion(prompt, run_id=run_id)


def _answer_style_instruction(answer_style: AnswerStyle) -> str:
    if answer_style == "brief":
        return "1. 用 2-4 句话回答，先给结论，再给 1 条最关键依据。"
    if answer_style == "detailed":
        return "1. 按“背景、核心原理、关键步骤、优缺点、应用场景、依据来源”展开回答。"
    return "1. 按“结论、核心解释、依据来源、不确定点”回答，长度适中。"


def _fallback_style_instruction(answer_style: AnswerStyle) -> str:
    if answer_style == "brief":
        return "用 2-4 句话简洁回答，适合快速理解。"
    if answer_style == "detailed":
        return "按“概念定义、核心原理、工作流程、优缺点、应用场景、学习建议”详细展开。"
    return "用适合初学者理解的标准长度回答，包含定义、核心思路和典型场景。"


def needs_fallback(answer: str) -> bool:
    """Detect whether the strict RAG answer says the evidence is insufficient."""
    meaningful_lines = [
        line.strip()
        for line in answer.splitlines()
        if line.strip()
        and line.strip() not in {"【回答】", "[回答]", "回答："}
        and not line.strip().startswith("###")
    ]
    answer_head = meaningful_lines[0] if meaningful_lines else ""
    answer_head_lower = answer_head.lower()
    markers = [
        "资料中没有明确依据",
        "资料中没有明确提到",
        "资料中没有明确说明",
        "资料中没有明确提及",
        "资料中没有关于",
        "当前提供的资料中没有关于",
        "知识库中没有找到明确依据",
        "未检索到相关资料",
        "依据不足",
        "无法判断",
        "无法从现有资料",
        "资料不足以回答",
        "检索到的资料不足以回答",
        "does not provide enough evidence",
        "does not mention",
        "not enough evidence",
        "insufficient evidence",
    ]
    # 只把“开头就拒答/证据不足”的回答切到 fallback，避免把回答末尾的局限说明误判成失败。
    return any(marker in answer_head_lower for marker in markers)


def is_strict_knowledge_question(question: str) -> bool:
    """Detect user intent that should not be answered with model-only fallback."""
    normalized_question = question.strip().lower()
    strict_markers = [
        "只根据知识库",
        "只依据知识库",
        "只根据资料",
        "只依据资料",
        "只根据文档",
        "只依据文档",
        "根据已检索资料",
        "根据这些论文",
        "这些论文有没有证明",
        "论文有没有证明",
        "文中是否提到",
        "资料中是否提到",
        "文档中是否提到",
        "是否提到",
        "有没有证明",
    ]
    return any(marker in normalized_question for marker in strict_markers)


def is_metadata_question(question: str) -> bool:
    """Detect questions about the knowledge base itself instead of document content."""
    normalized_question = question.strip().lower()
    metadata_markers = [
        "知识库包含哪些",
        "知识库有哪些",
        "知识库里有哪些",
        "知识库里面有哪些",
        "知识库有多少",
        "知识库里有多少",
        "知识库里面有多少",
        "知识库多少个",
        "知识库几个",
        "有哪些论文",
        "包含哪些论文",
        "有哪些文件",
        "包含哪些文件",
        "多少文件",
        "多少个文件",
        "几个文件",
        "分别是什么",
        "当前入库",
        "已经入库",
        "入库了什么",
        "收录了哪些",
        "文件清单",
        "文档列表",
    ]
    return any(marker in normalized_question for marker in metadata_markers)


def _format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "未检索到相关资料。"

    parts = []
    for index, chunk in enumerate(chunks, start=1):
        page = chunk.get("page")
        page_text = f"p.{page}" if page else "page unknown"
        # 引用信息放进 prompt，模型回答时更容易带上来源。
        citation_id = f"S{index}"
        title = chunk.get("document_title") or chunk.get("file_name", "unknown")
        section = chunk.get("section") or "unknown section"
        parts.append(
            f"[{citation_id}] {title} | {chunk.get('file_name', 'unknown')} {page_text} "
            f"section={section} chunk={chunk.get('chunk_id')}\n{chunk.get('content', '')}"
        )
    return "\n\n".join(parts)
