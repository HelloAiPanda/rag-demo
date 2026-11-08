"""核心逻辑的单测：切块、去重、检索、拒答。

跑：.venv\\Scripts\\python -m pytest -q
"""
from __future__ import annotations

import numpy as np
import pytest

from app import store
from app.chunking import chunk_text, split_paragraphs
from app.embedding import HashEmbedder, tokenize
from app.llm import extractive_answer


def test_chinese_bigram_tokenize():
    tokens = tokenize("流式对话")
    assert "流式" in tokens and "对话" in tokens


def test_chunk_size_and_overlap():
    text = "\n\n".join(["这是一段测试文本，用来验证切块逻辑。" * 8] * 6)
    chunks = chunk_text(text, size=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # 相邻块应当有重叠内容
    assert chunks[0][-20:] in chunks[1]


def test_embedding_is_normalized_and_stable():
    e = HashEmbedder(dim=256)
    v1, v2 = e.encode("银行知识库"), e.encode("银行知识库")
    assert np.allclose(v1, v2)
    assert abs(float(np.linalg.norm(v1)) - 1.0) < 1e-5


def test_search_returns_relevant_chunk(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "INDEX_PATH", tmp_path / "index.json")
    idx = store.Index()
    idx.add_document(
        "kb.md",
        "sha1",
        ["流式对话不能把事务挂在模型流上，否则连接池会被打满。", "声纹聚类的默认阈值是 0.42。"],
    )
    hits = idx.search("为什么连接池会被打满", top_k=1)
    assert hits and "连接池" in hits[0]["text"]
    assert hits[0]["score"] > 0


def test_duplicate_document_is_deduplicated(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "INDEX_PATH", tmp_path / "index.json")
    idx = store.Index()
    doc = idx.add_document("a.txt", "same-sha", ["内容一"])
    assert idx.find_by_sha("same-sha") is doc
    assert idx.find_by_sha("other") is None


def test_extractive_answer_has_citation():
    hits = [
        {
            "doc_name": "kb.md",
            "score": 0.5,
            "text": "检索到的最佳片段相似度低于阈值时，应当直接回答没有找到相关内容，不要编造。",
        }
    ]
    answer = extractive_answer("检索不到怎么办", hits)
    assert "资料[1]" in answer
    assert "没有找到" in answer


def test_split_paragraphs_drops_blank():
    assert split_paragraphs("第一段\n\n\n第二段\n") == ["第一段", "第二段"]
