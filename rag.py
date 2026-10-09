import os
from pathlib import Path

import chromadb
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm.auto import tqdm

import sys
sys.path.append(str(Path.cwd().parent))
import rag_utils

load_dotenv(Path.cwd().parent / ".env")

API_KEY = os.environ.get("GEMINI_API_KEY")
assert API_KEY, "Set GEMINI_API_KEY in a .env file at the project root (see .env.example)."

client = genai.Client(api_key=API_KEY)

# --- Config -------------------------------------------------------------
DATA_DIR = Path.cwd().parent / "data" / "Papers"
CHROMA_DIR = Path.cwd().parent / "chroma_db"
COLLECTION_NAME = "papers"

EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-flash-latest"  # alias for the current default Gemini flash model

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBED_BATCH_SIZE = 32
TOP_K = 5

print(f"Data dir: {DATA_DIR} (exists: {DATA_DIR.exists()})")
print(f"Chroma dir: {CHROMA_DIR}")

# ## 2. Ingestion
# 
# Discover PDFs → extract text per page → chunk → embed (`task_type="RETRIEVAL_DOCUMENT"`) → store in ChromaDB.
# 
# Re-running this cell is idempotent: if the collection already holds the expected number of chunks, ingestion is skipped. To force a full rebuild, delete the `chroma_db/` folder (or the collection) first.


import time


def embed_texts(texts: list[str], task_type: str, max_retries: int = 5) -> list[list[float]]:
    """Embed a list of texts in batches with the given Gemini task type.

    Wraps each batch call with a small retry/backoff on top of the SDK's own
    retries, since transient connection resets can occasionally exceed those.
    """
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        for attempt in range(max_retries):
            try:
                response = client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                all_embeddings.extend(e.values for e in response.embeddings)
                break
            except Exception as exc:
                if attempt == max_retries - 1:
                    raise
                wait = 2**attempt
                print(f"  embed batch failed ({exc!r}), retrying in {wait}s...")
                time.sleep(wait)
    return all_embeddings


chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

pdf_paths = rag_utils.iter_pdfs(DATA_DIR)
print(f"Found {len(pdf_paths)} PDFs under {DATA_DIR}")

all_records = []
for pdf_path in tqdm(pdf_paths, desc="Extracting + chunking"):
    pages = rag_utils.extract_pages(pdf_path)
    all_records.extend(
        rag_utils.build_chunk_records(pdf_path, DATA_DIR, pages, CHUNK_SIZE, CHUNK_OVERLAP)
    )

print(f"Built {len(all_records)} chunks from {len(pdf_paths)} PDFs")

# Skip chunks already present in the collection (makes ingestion resumable
# after a transient failure, and idempotent on repeated runs).
existing_ids = set(collection.get(include=[])["ids"])
pending = [r for r in all_records if r.id not in existing_ids]
print(f"{len(all_records) - len(pending)} chunks already indexed, {len(pending)} pending")

for i in tqdm(range(0, len(pending), EMBED_BATCH_SIZE), desc="Embedding + indexing"):
    batch = pending[i : i + EMBED_BATCH_SIZE]
    embeddings = embed_texts([r.text for r in batch], task_type="RETRIEVAL_DOCUMENT")
    collection.upsert(
        ids=[r.id for r in batch],
        embeddings=embeddings,
        documents=[r.text for r in batch],
        metadatas=[
            {
                "source_path": r.source_path,
                "category": r.category,
                "paper": r.paper,
                "page": r.page,
                "chunk_index": r.chunk_index,
            }
            for r in batch
        ],
    )

print(f"Collection '{COLLECTION_NAME}' now has {collection.count()} chunks")

# ## 3. Retrieval (ranking only — no generation yet)
# 
# Embed the query with `task_type="RETRIEVAL_QUERY"` and inspect the ranked chunks Chroma returns before any LLM call happens.


def retrieve(query: str, k: int = TOP_K) -> tuple[pd.DataFrame, list[str]]:
    """Return (ranking_dataframe, full_chunk_texts) for a query.

    The dataframe is for display (truncated snippets); full_chunk_texts keeps
    the untruncated chunk text in the same order, for feeding to generation.
    """
    query_embedding = embed_texts([query], task_type="RETRIEVAL_QUERY")[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    full_docs = results["documents"][0]
    rows = []
    for rank, (doc, meta, dist) in enumerate(
        zip(full_docs, results["metadatas"][0], results["distances"][0]), start=1
    ):
        rows.append(
            {
                "rank": rank,
                "similarity": round(1 - dist, 4),
                "paper": meta["paper"],
                "category": meta["category"],
                "page": meta["page"],
                "snippet": (doc[:220] + "…") if len(doc) > 220 else doc,
            }
        )
    return pd.DataFrame(rows), full_docs


sample_query = "How is oral reading fluency measured automatically?"
ranking_df, _ = retrieve(sample_query, k=TOP_K)
ranking_df

# ## 4. Generation (full RAG)
# 
# Build a grounded prompt from the retrieved chunks and ask Gemini to answer using only that context, citing the source paper and page.


SYSTEM_INSTRUCTION = (
    "You are a research assistant answering questions about a set of academic papers. "
    "Answer ONLY using the provided context excerpts. If the context does not contain "
    "enough information to answer, say so explicitly instead of guessing. "
    "Cite every claim with the paper title and page number, like (Paper Title, p. 3)."
)


def build_context(df: pd.DataFrame, full_docs: list[str]) -> str:
    blocks = []
    for row, doc in zip(df.itertuples(), full_docs):
        blocks.append(
            f"[Source: {row.paper}, page {row.page}, category: {row.category}]\n{doc}"
        )
    return "\n\n---\n\n".join(blocks)


def generate_answer(question: str, ranking_df: pd.DataFrame, full_docs: list[str]) -> str:
    context = build_context(ranking_df, full_docs)
    prompt = f"Context excerpts:\n\n{context}\n\n---\n\nQuestion: {question}\n\nAnswer:"
    response = client.models.generate_content(
        model=GEN_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
    )
    return response.text


df, full_docs = retrieve(sample_query, k=TOP_K)
answer = generate_answer(sample_query, df, full_docs)
print(answer)

# ## 5. Convenience wrapper — ask anything


def rag_query(question: str, k: int = TOP_K, show_ranking: bool = True):
    df, full_docs = retrieve(question, k)
    if show_ranking:
        display(df)
    answer = generate_answer(question, df, full_docs)
    print("\n--- Answer ---\n")
    print(answer)
    return answer


_ = rag_query("What machine learning approaches have been used to predict reading fluency?")


_ = rag_query("What speech enhancement or pre-processing techniques are discussed for improving ASR on children's speech?")


