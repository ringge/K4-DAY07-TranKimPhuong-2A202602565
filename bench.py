"""Benchmark retrieval trên bộ 5 chính sách thương mại điện tử của nhóm.

Mỗi thành viên chỉ thay đổi dòng ``CHUNKER = ...`` để kết quả giữa các
chiến lược vẫn có thể so sánh công bằng.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Callable, TextIO

from dotenv import load_dotenv

from src.chunking import FixedSizeChunker, RecursiveChunker, SentenceChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    GEMINI_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    GeminiEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "ecommerce"
CACHE_PATH = ROOT / ".cache" / "bench_openai_embeddings.json"
RESULT_PATH = ROOT / "ket_qua_benchmark.txt"

# Dùng đúng 5 tài liệu trong Data Inventory của REPORT_NHOM mục 1.
CORPUS_FILES = (
    "chinh-sach-ban-hang-seller-pyloherb.md",
    "doi-tra-hang-hiclean.md",
    "doi-tra-hang-smarthomekit.md",
    "doi-tra-hang-thegioinem.md",
    "doi-tra-hang-xtrend.md",
)

BENCHMARKS = (
    {
        "query": (
            "Đối với người bán hoặc đại lý của PyLoHerb, chi phí vận chuyển "
            "và bốc dỡ được hỗ trợ như thế nào?"
        ),
        "gold": (
            "PyLoHerb hỗ trợ vận chuyển đơn hàng từ công ty đến đối tác bán "
            "hàng; chi phí bốc dỡ mỗi bên chịu một đầu."
        ),
        "metadata_filter": {"audience": "seller"},
    },
    {
        "query": (
            "Theo chính sách HiClean, khách hàng đổi hàng từ ngày 04 đến "
            "ngày 07 phải chịu mức phí bao nhiêu?"
        ),
        "gold": (
            "Khách hàng chịu 20% phí đổi hàng, phí vận chuyển, lắp đặt và "
            "phần chênh lệch giá trị hàng hóa nếu có."
        ),
        "metadata_filter": None,
    },
    {
        "query": (
            "Với đơn hàng COD tại Smart HomeKit, người mua phải cung cấp "
            "thông tin gì để nhận tiền hoàn trả?"
        ),
        "gold": "Người mua phải cung cấp thông tin tài khoản ngân hàng.",
        "metadata_filter": None,
    },
    {
        "query": (
            "Theo chính sách 102 ngày ngủ thử Tatana của Thế Giới Nệm, "
            "thời gian xử lý hoàn tiền là bao lâu?"
        ),
        "gold": (
            "Trong vòng 14 ngày làm việc kể từ khi Thế Giới Nệm nhận lại "
            "sản phẩm và xác nhận sản phẩm đạt điều kiện."
        ),
        "metadata_filter": None,
    },
    {
        "query": (
            "Khách mua hàng online tại XTREND được yêu cầu đổi trả trong "
            "thời gian bao lâu?"
        ),
        "gold": "Trong vòng 07 ngày kể từ khi nhận hàng.",
        "metadata_filter": None,
    },
)


class HeadingChunker:
    """Tách theo heading cấp 2, rồi recursive nếu một section quá dài."""

    def __init__(self, chunk_size: int = 500) -> None:
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []

        headings = list(re.finditer(r"(?m)^##(?!#)\s+.+$", text))
        if not headings:
            return RecursiveChunker(chunk_size=self.chunk_size).chunk(text)

        chunks: list[str] = []
        preamble = text[: headings[0].start()].strip()
        if preamble:
            chunks.extend(RecursiveChunker(chunk_size=self.chunk_size).chunk(preamble))

        for index, match in enumerate(headings):
            end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
            heading = match.group().strip()
            content = text[match.end() : end].strip()
            section = f"{heading}\n\n{content}" if content else heading

            if len(section) <= self.chunk_size:
                chunks.append(section)
                continue

            child_size = max(1, self.chunk_size - len(heading) - 2)
            child_chunks = RecursiveChunker(chunk_size=child_size).chunk(content)
            chunks.extend(f"{heading}\n\n{part}" for part in child_chunks)

        return chunks


# DÒNG DUY NHẤT MỖI THÀNH VIÊN THAY ĐỔI ĐỂ CHỌN CHIẾN LƯỢC CHUNKING.
CHUNKER = HeadingChunker(chunk_size=500)


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Tách YAML frontmatter dạng key/value đơn giản khỏi nội dung Markdown."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    try:
        closing_index = next(
            index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"
        )
    except StopIteration:
        return {}, text

    metadata: dict[str, str] = {}
    for line in lines[1:closing_index]:
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"\'')

    content = "\n".join(lines[closing_index + 1 :]).strip()
    return metadata, content


def load_chunked_documents() -> list[Document]:
    """Đọc corpus, bỏ frontmatter khỏi content và tạo một Document mỗi chunk."""
    documents: list[Document] = []
    for filename in CORPUS_FILES:
        path = DATA_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy tài liệu benchmark: {path}")

        frontmatter, content = parse_frontmatter(path.read_text(encoding="utf-8"))
        for index, chunk in enumerate(CHUNKER.chunk(content)):
            documents.append(
                Document(
                    id=f"{path.stem}#{index}",
                    content=chunk,
                    metadata={
                        **frontmatter,
                        "doc_id": path.stem,
                        "source": str(path.relative_to(ROOT)),
                        "chunk_index": index,
                    },
                )
            )
    return documents


class CachedEmbedder:
    """Cache embedding theo hash(model + nội dung), chủ yếu để tránh tốn OpenAI API."""

    def __init__(
        self,
        embedder: Callable[[str], list[float]],
        model_name: str,
        cache_path: Path,
    ) -> None:
        self.embedder = embedder
        self.model_name = model_name
        self.cache_path = cache_path
        self._backend_name = f"{model_name} (disk cache)"
        self.cache = self._load_cache()

    def _load_cache(self) -> dict[str, list[float]]:
        if not self.cache_path.is_file():
            return {}
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def __call__(self, text: str) -> list[float]:
        key = hashlib.sha256(f"{self.model_name}\0{text}".encode("utf-8")).hexdigest()
        if key not in self.cache:
            self.cache[key] = self.embedder(text)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(self.cache, ensure_ascii=False), encoding="utf-8"
            )
        return self.cache[key]


class TeeWriter:
    """Ghi cùng một output ra terminal và file kết quả."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def select_embedder() -> Callable[[str], list[float]]:
    """Chọn backend giống main.py; mặc định mock để bench chạy không cần API key."""
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        model = os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL)
        try:
            return LocalEmbedder(model_name=model)
        except Exception as exc:
            print(f"Không tải được local embedder ({exc}); dùng mock embedder.")
    elif provider == "openai":
        model = os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL)
        try:
            embedder = OpenAIEmbedder(model_name=model)
            return CachedEmbedder(embedder, model, CACHE_PATH)
        except Exception as exc:
            print(f"Không khởi tạo được OpenAI embedder ({exc}); dùng mock embedder.")
    elif provider == "gemini":
        model = os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL)
        try:
            return GeminiEmbedder(model_name=model)
        except Exception as exc:
            print(f"Không khởi tạo được Gemini embedder ({exc}); dùng mock embedder.")
    elif provider != "mock":
        print(f"EMBEDDING_PROVIDER={provider!r} không hợp lệ; dùng mock embedder.")
    return _mock_embed


def run_benchmark() -> None:
    """Nạp chunks, chạy 5 query và in kết quả benchmark."""
    documents = load_chunked_documents()
    embedder = select_embedder()
    store = EmbeddingStore(collection_name="ecommerce_benchmark", embedding_fn=embedder)
    store.add_documents(documents)

    print("=== Ecommerce Retrieval Benchmark ===")
    print(f"Chunker: {CHUNKER.__class__.__name__}")
    print(f"Embedding: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")
    print(f"Đã nạp {store.get_collection_size()} chunks từ {len(CORPUS_FILES)} files.")

    for query_index, benchmark in enumerate(BENCHMARKS, start=1):
        query = str(benchmark["query"])
        metadata_filter = benchmark["metadata_filter"]
        results = store.search_with_filter(
            query,
            top_k=3,
            metadata_filter=metadata_filter,
        )

        print(f"\n[{query_index}] Query: {query}")
        print(f"    Filter: {metadata_filter or '{}'}")
        print(f"    Gold: {benchmark['gold']}")
        for rank, result in enumerate(results, start=1):
            doc_id = result["metadata"].get("doc_id", "?")
            preview = " ".join(result["content"].split())[:180]
            print(
                f"    Top-{rank}: score={result['score']:.4f} "
                f"doc_id={doc_id} chunk_id={result['id']}"
            )
            print(f"           {preview}")


def main() -> int:
    load_dotenv(dotenv_path=ROOT / ".env", override=False)

    with RESULT_PATH.open("w", encoding="utf-8") as result_file:
        with redirect_stdout(TeeWriter(sys.stdout, result_file)):
            run_benchmark()

    print(f"Đã lưu kết quả vào: {RESULT_PATH.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
