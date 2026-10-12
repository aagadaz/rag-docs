import hashlib
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.config import settings

COLLECTION_NAME = "docs"
BATCH_SIZE = 256


# ---------- embeddings ----------

@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    # Les modèles e5 exigent un préfixe : "passage: " pour les documents indexés,
    # "query: " pour les questions. Sans lui, la qualité de recherche chute.
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True, "prompt": "passage: "},
        query_encode_kwargs={"normalize_embeddings": True, "prompt": "query: "},
    )


# ---------- base vectorielle ----------

def get_vectorstore(
    embeddings: Embeddings | None = None,
    persist_directory: str | None = None,
) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=persist_directory or settings.chroma_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )


def _chunk_ids(chunks: list[Document]) -> list[str]:
    """Un identifiant stable par chunk : hash de (source, position dans la source)."""
    positions: dict[str, int] = {}
    ids = []
    for chunk in chunks:
        source = chunk.metadata["source"]
        position = positions.get(source, 0)
        positions[source] = position + 1
        key = f"{source}|{position}"
        ids.append(hashlib.sha256(key.encode("utf-8")).hexdigest()[:32])
    return ids


def index_chunks(store: Chroma, chunks: list[Document]) -> int:
    """Indexe des chunks. Réindexer une source remplace ses anciens chunks."""
    if not chunks:
        return 0

    for source in {chunk.metadata["source"] for chunk in chunks}:
        delete_source(store, source)

    ids = _chunk_ids(chunks)
    for start in range(0, len(chunks), BATCH_SIZE):
        store.add_documents(chunks[start:start + BATCH_SIZE], ids=ids[start:start + BATCH_SIZE])

    return len(chunks)


def delete_source(store: Chroma, source: str) -> None:
    store.delete(where={"source": source})


def list_sources(store: Chroma) -> dict[str, int]:
    """Renvoie {source: nombre de chunks}, trié par nom."""
    metadatas = store.get(include=["metadatas"])["metadatas"]
    counts: dict[str, int] = {}
    for metadata in metadatas:
        counts[metadata["source"]] = counts.get(metadata["source"], 0) + 1
    return dict(sorted(counts.items()))


def search(store: Chroma, query: str, k: int = settings.top_k) -> list[tuple[Document, float]]:
    """Les k chunks les plus proches de la question, avec leur score (1 = identique)."""
    return store.similarity_search_with_relevance_scores(query, k=k)
