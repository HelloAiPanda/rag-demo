"""向量索引：内存里算，落盘成 JSON。

为什么不用 Chroma？
演示规模是几百个 chunk，numpy 一次矩阵乘法就是微秒级；
引入 Chroma 要多背一个依赖、一套生命周期。这里把接口留成 add/remove/search，
以后要换 Milvus 只改这一个文件。
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import INDEX_PATH
from .embedding import get_embedder, tokenize


@dataclass
class Chunk:
    id: str
    text: str
    vec: np.ndarray


@dataclass
class Doc:
    id: str
    name: str
    sha: str
    added_at: float
    chunks: list[Chunk] = field(default_factory=list)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class Index:
    def __init__(self) -> None:
        self._docs: dict[str, Doc] = {}
        self._lock = threading.RLock()
        self._matrix: np.ndarray | None = None
        self._rows: list[tuple[str, Chunk]] = []
        self._tokens: list[Counter] = []
        self._df: Counter = Counter()
        self._avgdl: float = 1.0
        self.embedder = get_embedder()
        self.load()

    # ---------- 读写 ----------
    def load(self) -> None:
        if not INDEX_PATH.exists():
            return
        try:
            payload = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for d in payload.get("docs", []):
            chunks = [
                Chunk(id=c["id"], text=c["text"], vec=np.array(c["vec"], dtype=np.float32))
                for c in d["chunks"]
            ]
            self._docs[d["id"]] = Doc(
                id=d["id"], name=d["name"], sha=d["sha"], added_at=d["added_at"], chunks=chunks
            )
        self._rebuild()

    def save(self) -> None:
        payload = {
            "version": 1,
            "docs": [
                {
                    "id": d.id,
                    "name": d.name,
                    "sha": d.sha,
                    "added_at": d.added_at,
                    "chunks": [
                        {"id": c.id, "text": c.text, "vec": [round(float(x), 6) for x in c.vec]}
                        for c in d.chunks
                    ],
                }
                for d in self._docs.values()
            ],
        }
        tmp = INDEX_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(INDEX_PATH)

    def _rebuild(self) -> None:
        rows: list[tuple[str, Chunk]] = []
        for doc in self._docs.values():
            for chunk in doc.chunks:
                rows.append((doc.id, chunk))
        self._rows = rows
        self._matrix = (
            np.vstack([c.vec for _, c in rows]) if rows else np.zeros((0, self.embedder.dim), np.float32)
        )
        # BM25 需要的词频、文档频率、平均长度
        self._tokens = [Counter(tokenize(c.text)) for _, c in rows]
        self._df = Counter()
        for tf in self._tokens:
            self._df.update(tf.keys())
        total = sum(sum(tf.values()) for tf in self._tokens)
        self._avgdl = (total / len(self._tokens)) if self._tokens else 1.0

    # ---------- 业务 ----------
    def find_by_sha(self, sha: str) -> Doc | None:
        for doc in self._docs.values():
            if doc.sha == sha:
                return doc
        return None

    def add_document(self, name: str, sha: str, texts: list[str]) -> Doc:
        with self._lock:
            doc_id = f"doc_{int(time.time() * 1000)}_{len(self._docs)}"
            vectors = self.embedder.encode_many(texts)
            chunks = [
                Chunk(id=f"{doc_id}#{i}", text=t, vec=vectors[i].astype(np.float32))
                for i, t in enumerate(texts)
            ]
            doc = Doc(id=doc_id, name=name, sha=sha, added_at=time.time(), chunks=chunks)
            self._docs[doc_id] = doc
            self._rebuild()
            self.save()
            return doc

    def remove(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id not in self._docs:
                return False
            del self._docs[doc_id]
            self._rebuild()
            self.save()
            return True

    def _idf(self, term: str) -> float:
        n = len(self._rows)
        df = self._df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, question: str, top_k: int = 3) -> list[dict[str, Any]]:
        """BM25 排序 + 关键词覆盖率 + 向量分（辅助）。

        为什么不用纯余弦？
        中文短问句只有十来个 token，扣掉停用词后有效词很少；
        对上一个 500 字的块，余弦会被块的长度稀释，正相关问题只有 0.1 左右，
        和噪声分不开。所以改成：
          - 排序：BM25（自带长度归一和 IDF）
          - 拒答：IDF 加权覆盖率（问题里的关键词有多少真的在这块里出现过）
          - 向量分只作为参考展示，不参与阈值判断
        """
        with self._lock:
            if not self._rows:
                return []
            qtokens = Counter(tokenize(question))
            if not qtokens:
                return []
            qterms = list(qtokens)
            idf = {t: self._idf(t) for t in qterms}
            total_idf = sum(idf.values()) or 1.0

            qvec = self.embedder.encode(question).astype(np.float32)
            vec_scores = self._matrix @ qvec

            k1, b = 1.5, 0.75
            ranked: list[tuple[float, float, float, int]] = []
            for i, tf in enumerate(self._tokens):
                dl = sum(tf.values()) or 1
                bm25 = 0.0
                covered = 0.0
                for term in qterms:
                    freq = tf.get(term, 0)
                    if not freq:
                        continue
                    covered += idf[term]
                    denom = freq + k1 * (1 - b + b * dl / self._avgdl)
                    bm25 += idf[term] * freq * (k1 + 1) / denom
                ranked.append((bm25, covered / total_idf, float(vec_scores[i]), i))

            ranked.sort(key=lambda x: (-x[0], -x[1]))
            hits: list[dict[str, Any]] = []
            for bm25, coverage, vscore, idx in ranked[: max(1, top_k)]:
                doc_id, chunk = self._rows[idx]
                hits.append(
                    {
                        "doc_id": doc_id,
                        "doc_name": self._docs[doc_id].name,
                        "chunk_id": chunk.id,
                        "score": round(coverage, 4),
                        "bm25": round(bm25, 4),
                        "vector": round(vscore, 4),
                        "text": chunk.text,
                    }
                )
            return hits

    def stats(self) -> dict[str, Any]:
        return {
            "docs": len(self._docs),
            "chunks": len(self._rows),
            "embedder": getattr(self.embedder, "name", "unknown"),
            "dim": int(self.embedder.dim),
        }

    def list_docs(self) -> list[dict[str, Any]]:
        return [
            {
                "id": d.id,
                "name": d.name,
                "chunks": len(d.chunks),
                "added_at": d.added_at,
            }
            for d in sorted(self._docs.values(), key=lambda x: x.added_at, reverse=True)
        ]
