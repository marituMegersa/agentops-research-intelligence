"""Document ingestion, parsing, and text chunking utilities."""

from __future__ import annotations

import re
from typing import List, Optional
from app.retrieval.models import Chunk, Document


class TextSplitter:
    """Splits continuous text into overlapping chunks respecting structural separators."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "; ", ", ", " "]

    def _approx_token_count(self, text: str) -> int:
        """Estimate token count based on whitespace and punctuation words."""
        return len(re.findall(r"\w+|[^\w\s]", text))

    def split_text(self, text: str) -> List[str]:
        """Split a string into chunks with overlap."""
        text = text.strip()
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        splits = self._split_with_separators(text, self.separators)
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for piece in splits:
            piece_len = len(piece)
            if current_len + piece_len > self.chunk_size and current_chunk:
                merged = "".join(current_chunk).strip()
                if merged:
                    chunks.append(merged)

                # Maintain overlap from current chunk buffer
                overlap_buffer: List[str] = []
                overlap_len = 0
                for item in reversed(current_chunk):
                    if overlap_len + len(item) <= self.chunk_overlap:
                        overlap_buffer.insert(0, item)
                        overlap_len += len(item)
                    else:
                        break
                current_chunk = overlap_buffer
                current_len = overlap_len

            current_chunk.append(piece)
            current_len += piece_len

        if current_chunk:
            final_chunk = "".join(current_chunk).strip()
            if final_chunk and (not chunks or final_chunk != chunks[-1]):
                chunks.append(final_chunk)

        return chunks

    def _split_with_separators(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text by priority separators."""
        if not separators:
            step = max(1, self.chunk_size - self.chunk_overlap)
            return [text[i : i + step] for i in range(0, len(text), step)]

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep not in text:
            return self._split_with_separators(text, remaining_seps)

        parts = text.split(sep)
        result: List[str] = []
        for i, part in enumerate(parts):
            suffix = sep if i < len(parts) - 1 else ""
            full_part = part + suffix
            if len(full_part) > self.chunk_size and remaining_seps:
                result.extend(self._split_with_separators(full_part, remaining_seps))
            else:
                result.append(full_part)
        return result


class DocumentIngestor:
    """Ingests raw documents and produces indexed chunks."""

    def __init__(self, text_splitter: Optional[TextSplitter] = None):
        self.text_splitter = text_splitter or TextSplitter()

    def ingest_document(self, document: Document) -> List[Chunk]:
        """Process a Document instance and return an ordered list of Chunks."""
        raw_chunks = self.text_splitter.split_text(document.content)
        chunks: List[Chunk] = []

        for idx, text in enumerate(raw_chunks):
            token_count = self.text_splitter._approx_token_count(text)
            chunk = Chunk(
                document_id=document.id,
                text=text,
                chunk_index=idx,
                token_count=token_count,
                metadata={
                    **document.metadata,
                    "title": document.title,
                    "chunk_index": idx,
                    "total_chunks": len(raw_chunks),
                },
            )
            chunks.append(chunk)

        return chunks
