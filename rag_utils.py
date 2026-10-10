"""Reusable building blocks for the RAG notebooks.

The module keeps ingestion, both retrieval strategies, and answer generation
out of the small question-only notebooks. The two retrievers deliberately
share the same ``ChunkRecord`` objects so their rankings are comparable.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import pymupdf
from google.genai import types


DEFAULT_SYSTEM_INSTRUCTION = (
    "Você é um assistente que responde perguntas sobre uma coleção de documentos. "
    "Responda somente com base nos trechos fornecidos. Se eles não forem suficientes, "
    "diga isso explicitamente, sem adivinhar. Cite cada afirmação factual usando "
    "exatamente o identificador de trecho apresentado entre colchetes."
)


def iter_pdfs(root: Path) -> list[Path]:
    """Recursively find every PDF under ``root``, sorted reproducibly."""
    return sorted(path for path in root.rglob("*.pdf") if path.is_file())


def extract_pages(pdf_path: Path) -> list[str]:
    """Extract one plain-text string per PDF page.

    Ligatures (e.g. "ﬁ") are expanded to plain letters so that tokens such as
    "significa" match between documents and queries.
    """
    flags = pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP
    with pymupdf.open(pdf_path) as document:
        return [page.get_text("text", flags=flags) for page in document]


def _split_paragraphs(text: str) -> list[str]:
    text = re.sub(r"[ \t]+\n", "\n", text)
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Split text into fixed-size character windows with a fixed overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    normalized_text = "\n\n".join(_split_paragraphs(text))
    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(normalized_text):
        chunk = normalized_text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(normalized_text):
            break
        start += step
    return chunks


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """A small language-agnostic tokenizer used by the TF-IDF retriever."""
    return _TOKEN_RE.findall(text.casefold())


@dataclass(frozen=True)
class ChunkRecord:
    id: str
    text: str
    source_path: str
    category: str
    paper: str
    page: int
    chunk_index: int

    @property
    def citation(self) -> str:
        return f"{self.paper}, p. {self.page}, trecho {self.chunk_index}"

    @property
    def metadata(self) -> dict[str, str | int]:
        return {
            "source_path": self.source_path,
            "category": self.category,
            "paper": self.paper,
            "page": self.page,
            "chunk_index": self.chunk_index,
        }


def build_chunk_records(
    pdf_path: Path,
    data_root: Path,
    pages: Sequence[str],
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[ChunkRecord]:
    """Chunk every page of a PDF and attach source/citation metadata."""
    relative_path = pdf_path.relative_to(data_root)
    category = str(relative_path.parent) if relative_path.parent != Path(".") else "root"
    paper = pdf_path.stem

    records: list[ChunkRecord] = []
    for page_number, page_text in enumerate(pages, start=1):
        for chunk_index, chunk in enumerate(chunk_text(page_text, chunk_size, overlap)):
            raw_id = f"{relative_path}::p{page_number}::c{chunk_index}::{chunk}"
            records.append(
                ChunkRecord(
                    id=hashlib.sha1(raw_id.encode("utf-8")).hexdigest(),
                    text=chunk,
                    source_path=str(relative_path),
                    category=category,
                    paper=paper,
                    page=page_number,
                    chunk_index=chunk_index,
                )
            )
    return records


def load_corpus(
    data_root: Path, chunk_size: int = 1000, overlap: int = 150
) -> list[ChunkRecord]:
    """Discover, extract, and chunk every PDF in ``data_root``."""
    records: list[ChunkRecord] = []
    for pdf_path in iter_pdfs(data_root):
        records.extend(
            build_chunk_records(
                pdf_path,
                data_root,
                extract_pages(pdf_path),
                chunk_size,
                overlap,
            )
        )
    return records


class GeminiEmbedder:
    """Batching/retry adapter for Gemini document and query embeddings."""

    def __init__(
        self,
        client: Any,
        model: str = "gemini-embedding-001",
        batch_size: int = 32,
        max_retries: int = 5,
    ) -> None:
        self.client = client
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries

    def embed(self, texts: Sequence[str], task_type: str) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            for attempt in range(self.max_retries):
                try:
                    response = self.client.models.embed_content(
                        model=self.model,
                        contents=batch,
                        config=types.EmbedContentConfig(task_type=task_type),
                    )
                    embeddings.extend(item.values for item in response.embeddings)
                    break
                except Exception:
                    if attempt == self.max_retries - 1:
                        raise
                    time.sleep(2**attempt)
        return embeddings

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, query: str) -> list[float]:
        return self.embed([query], task_type="RETRIEVAL_QUERY")[0]


def index_dense_collection(
    collection: Any,
    records: Sequence[ChunkRecord],
    embedder: GeminiEmbedder,
) -> dict[str, int]:
    """Embed and upsert records not already present in a Chroma collection."""
    existing_ids = set(collection.get(include=[])["ids"])
    pending = [record for record in records if record.id not in existing_ids]

    for start in range(0, len(pending), embedder.batch_size):
        batch = pending[start : start + embedder.batch_size]
        collection.upsert(
            ids=[record.id for record in batch],
            embeddings=embedder.embed_documents([record.text for record in batch]),
            documents=[record.text for record in batch],
            metadatas=[record.metadata for record in batch],
        )

    return {
        "total": len(records),
        "already_indexed": len(records) - len(pending),
        "indexed_now": len(pending),
    }


def records_from_collection(collection: Any) -> list[ChunkRecord]:
    """Load all chunk records stored in a Chroma collection."""
    payload = collection.get(include=["documents", "metadatas"])
    documents = payload.get("documents") or []
    metadatas = payload.get("metadatas") or []
    records = []
    for chunk_id, document, metadata in zip(payload["ids"], documents, metadatas):
        metadata = metadata or {}
        records.append(
            ChunkRecord(
                id=chunk_id,
                text=document or "",
                source_path=str(metadata.get("source_path", "")),
                category=str(metadata.get("category", "root")),
                paper=str(metadata.get("paper", "document")),
                page=int(metadata.get("page", 0)),
                chunk_index=int(metadata.get("chunk_index", 0)),
            )
        )
    return sorted(
        records,
        key=lambda item: (item.source_path, item.page, item.chunk_index, item.id),
    )


@dataclass(frozen=True)
class RetrievalHit:
    rank: int
    score: float
    score_name: str
    record: ChunkRecord


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 5) -> list[RetrievalHit]: ...


class DenseRetriever:
    """Dense retrieval using Gemini embeddings and Chroma cosine distance."""

    def __init__(self, collection: Any, embedder: GeminiEmbedder) -> None:
        self.collection = collection
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalHit]:
        count = self.collection.count()
        if count == 0:
            return []
        result = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=min(k, count),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        ids = result["ids"][0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for rank, (chunk_id, document, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances), start=1
        ):
            metadata = metadata or {}
            record = ChunkRecord(
                id=chunk_id,
                text=document or "",
                source_path=str(metadata.get("source_path", "")),
                category=str(metadata.get("category", "root")),
                paper=str(metadata.get("paper", "document")),
                page=int(metadata.get("page", 0)),
                chunk_index=int(metadata.get("chunk_index", 0)),
            )
            hits.append(
                RetrievalHit(
                    rank=rank,
                    score=1.0 - float(distance),
                    score_name="cosine_similarity",
                    record=record,
                )
            )
        return hits


class TfidfRetriever:
    """Sparse TF-IDF retrieval with L2 normalization and cosine similarity."""

    def __init__(self, records: Sequence[ChunkRecord]) -> None:
        if not records:
            raise ValueError("TF-IDF needs at least one indexed chunk")
        self.records = list(records)
        token_counts = [Counter(tokenize(record.text)) for record in self.records]

        document_frequency: Counter[str] = Counter()
        for counts in token_counts:
            document_frequency.update(counts.keys())

        document_count = len(self.records)
        self.idf = {
            term: math.log((1 + document_count) / (1 + frequency)) + 1.0
            for term, frequency in document_frequency.items()
        }
        self.norms: list[float] = []
        self.postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for document_index, counts in enumerate(token_counts):
            weights = {term: count * self.idf[term] for term, count in counts.items()}
            norm = math.sqrt(sum(weight * weight for weight in weights.values()))
            self.norms.append(norm)
            for term, weight in weights.items():
                self.postings[term].append((document_index, weight))

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalHit]:
        query_counts = Counter(tokenize(query))
        query_weights = {
            term: count * self.idf[term]
            for term, count in query_counts.items()
            if term in self.idf
        }
        query_norm = math.sqrt(sum(weight * weight for weight in query_weights.values()))
        if query_norm == 0:
            return []

        dot_products: dict[int, float] = defaultdict(float)
        for term, query_weight in query_weights.items():
            for document_index, document_weight in self.postings[term]:
                dot_products[document_index] += query_weight * document_weight

        scored = [
            (dot / (query_norm * self.norms[index]), index)
            for index, dot in dot_products.items()
            if self.norms[index] > 0
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            RetrievalHit(
                rank=rank,
                score=score,
                score_name="tfidf_cosine_similarity",
                record=self.records[index],
            )
            for rank, (score, index) in enumerate(scored[:k], start=1)
        ]


def hits_as_rows(hits: Iterable[RetrievalHit]) -> list[dict[str, Any]]:
    """Convert retrieval hits to display-friendly dictionaries."""
    return [
        {
            "rank": hit.rank,
            hit.score_name: round(hit.score, 4),
            "paper": hit.record.paper,
            "page": hit.record.page,
            "chunk": hit.record.chunk_index,
            "snippet": (
                f"{hit.record.text[:220]}…"
                if len(hit.record.text) > 220
                else hit.record.text
            ),
        }
        for hit in hits
    ]


def build_context(hits: Sequence[RetrievalHit]) -> str:
    """Format retrieved chunks with unambiguous labels for grounded citation."""
    return "\n\n---\n\n".join(
        f"[{hit.record.citation}]\n{hit.record.text}" for hit in hits
    )


@dataclass(frozen=True)
class QueryResult:
    question: str
    hits: list[RetrievalHit]
    answer: str


class RAGAssistant:
    """Combine any compatible retriever with Gemini answer generation."""

    def __init__(
        self,
        client: Any,
        retriever: Retriever,
        generation_model: str = "gemini-flash-latest",
        system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
    ) -> None:
        self.client = client
        self.retriever = retriever
        self.generation_model = generation_model
        self.system_instruction = system_instruction

    def ask(self, question: str, k: int = 5) -> QueryResult:
        hits = self.retriever.retrieve(question, k=k)
        if not hits:
            return QueryResult(
                question=question,
                hits=[],
                answer="Não encontrei trechos relevantes para responder à pergunta.",
            )
        prompt = (
            f"Trechos de contexto:\n\n{build_context(hits)}"
            f"\n\n---\n\nPergunta: {question}\n\nResposta:"
        )
        response = self.client.models.generate_content(
            model=self.generation_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.system_instruction,
                temperature=0,
            ),
        )
        return QueryResult(question=question, hits=hits, answer=response.text or "")


def answer_without_retrieval(
    client: Any,
    question: str,
    generation_model: str = "gemini-flash-latest",
) -> str:
    """Generate the comparison baseline without access to retrieved context."""
    response = client.models.generate_content(
        model=generation_model,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=(
                "Responda à pergunta usando apenas o seu conhecimento interno. "
                "Não finja ter consultado documentos e não invente citações."
            ),
            temperature=0,
        ),
    )
    return response.text or ""
