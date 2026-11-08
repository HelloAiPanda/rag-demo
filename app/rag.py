"""RAG 编排：入库、问答、评测。"""
from __future__ import annotations

import json
import time
from typing import Any

from . import llm
from .chunking import chunk_text, extract_text
from .config import EVAL_PATH, UPLOAD_DIR, settings
from .store import Index, sha256_bytes


class RagService:
    def __init__(self) -> None:
        self.index = Index()

    # ---------- 入库 ----------
    def ingest(self, filename: str, raw: bytes) -> dict[str, Any]:
        if not raw:
            raise ValueError("文件是空的")
        sha = sha256_bytes(raw)

        existing = self.index.find_by_sha(sha)
        if existing:
            return {
                "deduped": True,
                "doc_id": existing.id,
                "filename": existing.name,
                "chunks": len(existing.chunks),
                "message": "同样的内容已经在库里，跳过（按文件指纹去重）",
            }

        text = extract_text(filename, raw)
        if len(text) < 20:
            raise ValueError("解析出来的文本太短，可能不是有效文档")

        texts = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        doc = self.index.add_document(filename, sha, texts)

        try:
            (UPLOAD_DIR / f"{doc.id}_{filename}").write_bytes(raw)
        except OSError:
            pass  # 存原件只是方便排查，失败不影响索引

        return {
            "deduped": False,
            "doc_id": doc.id,
            "filename": filename,
            "chunks": len(doc.chunks),
            "chars": len(text),
        }

    # ---------- 问答 ----------
    def ask(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        question = (question or "").strip()
        if not question:
            raise ValueError("问题不能为空")

        started = time.perf_counter()
        k = top_k or settings.top_k
        hits = self.index.search(question, top_k=k)

        # 安全阀：最高覆盖率低于阈值就拒答，不要硬编
        kept = [h for h in hits if h["score"] >= settings.min_coverage]
        if not kept:
            return {
                "answer": "资料里没有找到相关内容。",
                "mode": "refuse",
                "sources": [],
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }

        # 摘录模式只保留跟第一名接近的片段，弱相关的会带进噪声
        if not llm.is_configured():
            best = kept[0]["score"]
            kept = [h for h in kept if h["score"] >= best * 0.7]

        text, mode = llm.answer(question, kept[:2])
        return {
            "answer": text,
            "mode": mode,
            "sources": kept,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    # ---------- 评测 ----------
    def evaluate(self) -> dict[str, Any]:
        if not EVAL_PATH.exists():
            raise FileNotFoundError(f"缺少评测集：{EVAL_PATH}")
        rows = [
            json.loads(line)
            for line in EVAL_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        results, hits = [], 0
        for row in rows:
            started = time.perf_counter()
            resp = self.ask(row["question"])
            elapsed = round((time.perf_counter() - started) * 1000, 1)
            answer = resp["answer"]
            keywords = row.get("keywords", [])
            ok = any(kw in answer for kw in keywords) if keywords else resp["mode"] != "refuse"
            hits += int(ok)
            results.append(
                {
                    "question": row["question"],
                    "ok": ok,
                    "mode": resp["mode"],
                    "top_score": resp["sources"][0]["score"] if resp["sources"] else 0,
                    "latency_ms": elapsed,
                    "answer": answer[:180],
                }
            )
        total = len(results) or 1
        return {
            "total": len(results),
            "hit": hits,
            "hit_rate": round(hits / total, 3),
            "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / total, 1),
            "results": results,
        }


service = RagService()
