"""向量化：默认用本地 hashing 向量，零依赖、零下载、确定性。

为什么不默认用 sentence-transformers？
- 装 torch 要 2GB，面试演示机器上不一定跑得动；
- 演示需要「谁都能 clone 下来 30 秒跑通」。
所以把 embedding 抽象成一个函数，默认 hash，配置了远端就切远端。
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import numpy as np

from .config import settings

_LATIN = re.compile(r"[a-z0-9_]{2,}")
_CJK = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens: list[str] = list(_LATIN.findall(text))
    cjk_runs = re.findall(r"[\u4e00-\u9fff]+", text)
    for run in cjk_runs:
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def _bucket(token: str, dim: int) -> int:
    # 用 crc32 而不是内置 hash：跨进程稳定，索引重建后向量不变
    h = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little") % dim


class HashEmbedder:
    """字符二元组 + 次线性词频 + L2 归一化。"""

    name = "hash"

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim or settings.embed_dim

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token, count in Counter(tokenize(text)).items():
            vec[_bucket(token, self.dim)] += 1.0 + math.log(count)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def encode_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self.encode(t) for t in texts])


class RemoteEmbedder:
    """OpenAI 兼容的 /embeddings 接口。"""

    name = "remote"

    def __init__(self) -> None:
        import httpx

        if not settings.embed_base_url or not settings.embed_api_key:
            raise ValueError("EMBED_MODE=remote 需要配置 EMBED_BASE_URL / EMBED_API_KEY")
        self._client = httpx.Client(
            base_url=settings.embed_base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {settings.embed_api_key}"},
            timeout=settings.llm_timeout,
        )
        self.dim = settings.embed_dim

    def encode_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        resp = self._client.post(
            "/embeddings",
            json={"model": settings.embed_model, "input": texts},
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda x: x["index"])
        mat = np.array([d["embedding"] for d in data], dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms
        self.dim = mat.shape[1]
        return mat

    def encode(self, text: str) -> np.ndarray:
        return self.encode_many([text])[0]


def get_embedder():
    if settings.embed_mode == "remote":
        return RemoteEmbedder()
    return HashEmbedder()


def cosine_scores(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return matrix @ query
