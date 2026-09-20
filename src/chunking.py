from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        # Split after sentence-ending punctuation so the punctuation remains
        # attached to the sentence it belongs to.
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
            if sentence.strip()
        ]

        return [
            " ".join(sentences[start : start + self.max_sentences_per_chunk])
            for start in range(0, len(sentences), self.max_sentences_per_chunk)
        ]


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if not current_text:
            return []
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # Once no semantic separator remains, fall back to hard character
        # boundaries so that even an unbroken string can still be chunked.
        if not remaining_separators or remaining_separators[0] == "":
            return [
                current_text[start : start + self.chunk_size]
                for start in range(0, len(current_text), self.chunk_size)
            ]

        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]

        # If this separator does not occur, try the next, finer boundary.
        if separator not in current_text:
            return self._split(current_text, next_separators)

        # Keep separators attached to the preceding piece.  Besides retaining
        # punctuation/paragraph breaks, this guarantees that concatenating the
        # resulting chunks recreates the original text exactly.
        pieces: list[str] = []
        start = 0
        while True:
            separator_at = current_text.find(separator, start)
            if separator_at == -1:
                break
            end = separator_at + len(separator)
            pieces.append(current_text[start:end])
            start = end
        if start < len(current_text):
            pieces.append(current_text[start:])

        split_pieces: list[str] = []
        for piece in pieces:
            if len(piece) > self.chunk_size:
                split_pieces.extend(self._split(piece, next_separators))
            elif piece:
                split_pieces.append(piece)

        # Recombine adjacent small pieces up to the requested size.  Recursive
        # splitting alone tends to create many tiny, low-context chunks.
        chunks: list[str] = []
        pending = ""
        for piece in split_pieces:
            if pending and len(pending) + len(piece) > self.chunk_size:
                chunks.append(pending)
                pending = piece
            else:
                pending += piece
        if pending:
            chunks.append(pending)

        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    magnitude_a = math.sqrt(_dot(vec_a, vec_a))
    magnitude_b = math.sqrt(_dot(vec_b, vec_b))
    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0

    return _dot(vec_a, vec_b) / (magnitude_a * magnitude_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        # Keep the usual 50-character overlap where possible while ensuring
        # that even very small positive chunk sizes produce a valid step.
        overlap = min(50, chunk_size - 1)
        strategy_chunks = {
            "fixed_size": FixedSizeChunker(
                chunk_size=chunk_size, overlap=overlap
            ).chunk(text),
            "by_sentences": SentenceChunker().chunk(text),
            "recursive": RecursiveChunker(chunk_size=chunk_size).chunk(text),
        }

        comparison: dict = {}
        for strategy_name, chunks in strategy_chunks.items():
            count = len(chunks)
            comparison[strategy_name] = {
                "count": count,
                "avg_length": (
                    sum(len(chunk) for chunk in chunks) / count if count else 0.0
                ),
                "chunks": chunks,
            }

        return comparison
