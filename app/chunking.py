"""文本切块：先按段落分，再按字数打包，带重叠。

为什么不用固定窗口硬切？
中文没有空格，硬切会把一句话砍成两半，检索命中后模型读到半句，答案就会跑偏。
所以先按空行分段，段落超长再按句号切，最后才按字数打包。
"""
from __future__ import annotations

import re

_SENT_SPLIT = re.compile(r"(?<=[。！？；!?;\n])")


def split_paragraphs(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def split_sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(paragraph) if s.strip()]


def chunk_text(text: str, size: int = 500, overlap: int = 100) -> list[str]:
    """返回切好的块。size / overlap 以字符计（中文按字）。"""
    if size <= 0:
        raise ValueError("size must be positive")
    overlap = max(0, min(overlap, size // 2))

    units: list[str] = []
    for para in split_paragraphs(text):
        if len(para) <= size:
            units.append(para)
            continue
        for sent in split_sentences(para):
            if len(sent) <= size:
                units.append(sent)
            else:
                # 超长句：退化成按字数滑窗
                for i in range(0, len(sent), size):
                    units.append(sent[i : i + size])

    chunks: list[str] = []
    buf = ""
    for unit in units:
        # 标题就是天然的切块边界：一个小节一块，避免把两节内容混进同一个块。
        # 混块的后果很直接——问 A 小节，检索命中的块里前一半是 B 小节，
        # 摘录出来的就是 B 的内容。
        if unit.startswith("#") and buf:
            chunks.append(buf)
            buf = unit
            continue
        candidate = f"{buf}\n{unit}".strip() if buf else unit
        if len(candidate) <= size:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap else ""
            buf = f"{tail}\n{unit}".strip() if tail else unit
        else:
            buf = unit
        # 重叠后仍超长，强制切
        while len(buf) > size:
            chunks.append(buf[:size])
            buf = buf[size - overlap :] if overlap else buf[size:]

    if buf:
        chunks.append(buf)
    return chunks


def clean_text(text: str) -> str:
    """去掉多余空白和页眉页脚常见噪声。"""
    text = text.replace("\u00a0", " ").replace("\ufeff", "")
    text = re.sub(r"[ \t]+", " ", text)
    # markdown 标题单独成段：否则标题会和正文粘成一块，切片边界错位
    text = re.sub(r"\n(?=#{1,6}\s)", "\n\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text(filename: str, raw: bytes) -> str:
    """txt / md 直接解码，pdf 走 pypdf。"""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        import io

        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(raw))
        pages = [(p.extract_text() or "") for p in reader.pages]
        return clean_text("\n\n".join(pages))
    for enc in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return clean_text(raw.decode(enc))
        except UnicodeDecodeError:
            continue
    raise ValueError("无法识别文件编码，请转成 UTF-8 或 PDF")
