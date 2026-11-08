"""生成层：有 Key 走大模型，没 Key 走原文摘录。

演示必须「没有 API Key 也能跑」，否则面试官 clone 下来第一步就卡住。
但两种模式的答案都带出处，行为一致。
"""
from __future__ import annotations

import re
from typing import Any

import httpx

from .config import settings

SYSTEM_PROMPT = (
    "你是银行内部知识库助手。规则：\n"
    "1. 只能依据【资料】回答，不许使用资料以外的知识；\n"
    "2. 【资料】里没有答案时，直接回答「资料里没有找到相关内容」；\n"
    "3. 每个结论后面用 [1][2] 标注来源编号；\n"
    "4. 中文回答，200 字以内，不要复述问题。"
)

_SENT = re.compile(r"(?<=[。！？；!?;\n])")


def is_configured() -> bool:
    return bool(settings.llm_base_url and settings.llm_api_key)


def build_context(hits: list[dict[str, Any]]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(f"[{i}] 来源：{hit['doc_name']}\n{hit['text']}")
    return "\n\n".join(blocks)


def _question_terms(question: str) -> set[str]:
    from .embedding import tokenize

    return set(tokenize(question))


def _strip_headings(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    body = [ln for ln in lines if ln and not ln.startswith("#")]
    return " ".join(body)


def _snippet(text: str, question: str, max_chars: int = 220) -> str:
    """在块里找关键词密度最高的一段原文，而不是猜「哪一句是答案」。

    早期版本按句子打分会挑中标题或相邻小节，因为标题往往包含问题里的所有词。
    改成滑窗取最相关的一段，至少保证引用的是真原文、上下文完整。
    """
    body = _strip_headings(text) or text.strip()
    if len(body) <= max_chars:
        return body

    terms = _question_terms(question)
    best_pos, best_hit = 0, -1
    for pos in range(0, max(len(body) - max_chars, 0) + 1, 40):
        window = body[pos : pos + max_chars]
        hit = len(terms & _question_terms(window))
        if hit > best_hit:
            best_pos, best_hit = pos, hit

    head = "…" if best_pos > 0 else ""
    tail = "…" if best_pos + max_chars < len(body) else ""
    return f"{head}{body[best_pos:best_pos + max_chars].strip()}{tail}"


def extractive_answer(question: str, hits: list[dict[str, Any]]) -> str:
    parts = ["（未配置大模型，以下为原文摘录）"]
    for i, hit in enumerate(hits[:2], start=1):
        parts.append(f"根据资料[{i}]（{hit['doc_name']}）：{_snippet(hit['text'], question)}")
    return "\n\n".join(parts)


def answer(question: str, hits: list[dict[str, Any]]) -> tuple[str, str]:
    """返回 (答案, 模式)。模式：llm / extractive。"""
    if not hits:
        return "资料里没有找到相关内容。", "refuse"
    if not is_configured():
        return extractive_answer(question, hits), "extractive"

    user_prompt = f"【资料】\n{build_context(hits)}\n\n【问题】{question}"
    try:
        with httpx.Client(headers={"Authorization": f"Bearer {settings.llm_api_key}"}, timeout=settings.llm_timeout) as client:
            resp = client.post(
                f"{settings.llm_base_url.rstrip('/')}/chat/completions",
                json={
                    "model": settings.llm_model,
                    "temperature": 0.2,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                },
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
        return text, "llm"
    except Exception as exc:  # 模型不可用要降级，不能让整个问答挂掉
        fallback = extractive_answer(question, hits)
        return f"{fallback}\n\n（大模型调用失败：{type(exc).__name__}，已降级为摘录）", "extractive"
