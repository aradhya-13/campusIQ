"""Vector store: ChromaDB with a local embedding model (all-MiniLM-L6-v2).

The embedding model is downloaded automatically on first use (~80 MB).
"""
from collections import Counter

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from .config import CHROMA_DIR

_client = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = _client.get_or_create_collection(
    name="documents",
    embedding_function=DefaultEmbeddingFunction(),
    metadata={"hnsw:space": "cosine"},
)

_BATCH = 200


def replace_document(source: str, records: list[tuple[int, str]]) -> None:
    """Store chunks for `source`, replacing any previous version of it."""
    _collection.delete(where={"source": source})
    for start in range(0, len(records), _BATCH):
        batch = records[start : start + _BATCH]
        _collection.add(
            ids=[f"{source}::{start + i}" for i in range(len(batch))],
            documents=[text for _, text in batch],
            metadatas=[
                {"source": source, "page": page, "chunk": start + i}
                for i, (page, _) in enumerate(batch)
            ],
        )


def delete_document(source: str) -> None:
    _collection.delete(where={"source": source})


def list_documents() -> list[dict]:
    metas = _collection.get(include=["metadatas"])["metadatas"] or []
    counts = Counter(m["source"] for m in metas)
    return [{"name": name, "chunks": n} for name, n in sorted(counts.items())]


def query(text: str, k: int) -> list[dict]:
    total = _collection.count()
    if total == 0:
        return []
    res = _collection.query(
        query_texts=[text],
        n_results=min(k, total),
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append(
            {
                "source": meta["source"],
                "page": meta["page"],
                "text": doc,
                "score": round(1 - dist, 3),  # cosine similarity
            }
        )
    return hits
