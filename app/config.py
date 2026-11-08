"""配置：全部走环境变量，默认值保证零配置就能跑。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_PATH = DATA_DIR / "index.json"
EVAL_PATH = BASE_DIR / "eval" / "questions.jsonl"


@dataclass(frozen=True)
class Settings:
    # 切片
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "350"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    # 检索
    top_k: int = int(os.getenv("TOP_K", "4"))
    # 拒答阈值：问题关键词的 IDF 加权覆盖率。
    # 实测：本领域问题 0.25–0.87，无关问题 0.0，取 0.2 留足余量。
    min_coverage: float = float(os.getenv("MIN_COVERAGE", os.getenv("MIN_SCORE", "0.2")))
    # 向量
    embed_mode: str = os.getenv("EMBED_MODE", "hash")  # hash | remote
    embed_dim: int = int(os.getenv("EMBED_DIM", "512"))
    embed_base_url: str = os.getenv("EMBED_BASE_URL", "")
    embed_api_key: str = os.getenv("EMBED_API_KEY", "")
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    # 生成
    llm_base_url: str = os.getenv("LLM_BASE_URL", "")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_timeout: float = float(os.getenv("LLM_TIMEOUT", "60"))


settings = Settings()

for _d in (DATA_DIR, UPLOAD_DIR, EVAL_PATH.parent):
    _d.mkdir(parents=True, exist_ok=True)
