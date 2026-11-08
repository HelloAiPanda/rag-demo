"""FastAPI 入口：上传、提问、评测、文档管理。"""
from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import llm
from .config import BASE_DIR, settings
from .rag import service

WEB_DIR = BASE_DIR / "web"

app = FastAPI(title="RAG Demo · 银行知识库问答", version="1.0.0")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    top_k: int | None = Field(default=None, ge=1, le=10)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/stats")
def stats() -> dict:
    return {
        **service.index.stats(),
        "llm": "on" if llm.is_configured() else "off",
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
        "min_coverage": settings.min_coverage,
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    raw = await file.read()
    try:
        return service.ingest(file.filename or "unnamed.txt", raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/ask")
def ask(body: AskIn) -> dict:
    try:
        return service.ask(body.question, body.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/docs")
def list_docs() -> dict:
    return {"docs": service.index.list_docs()}


@app.delete("/api/docs/{doc_id}")
def delete_doc(doc_id: str) -> dict:
    if not service.index.remove(doc_id):
        raise HTTPException(status_code=404, detail="文档不存在")
    return {"deleted": doc_id}


@app.post("/api/eval")
def run_eval() -> dict:
    try:
        return service.evaluate()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
