import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag.vectorstore import delete_source, get_vectorstore, index_chunks, list_sources, search


@pytest.fixture
def store(tmp_path):
    # Faux embeddings : vecteurs déterministes, ni modèle à télécharger ni GPU.
    return get_vectorstore(embeddings=DeterministicFakeEmbedding(size=32), persist_directory=str(tmp_path))


def _chunks(source, n):
    return [
        Document(page_content=f"{source} passage {i}", metadata={"source": source, "file_type": "md", "chunk_index": i})
        for i in range(n)
    ]


def test_index_and_list_sources(store):
    index_chunks(store, _chunks("a.md", 3) + _chunks("b.md", 2))
    assert list_sources(store) == {"a.md": 3, "b.md": 2}


def test_reindexing_same_source_creates_no_duplicates(store):
    index_chunks(store, _chunks("a.md", 3))
    index_chunks(store, _chunks("a.md", 3))
    assert list_sources(store) == {"a.md": 3}


def test_reindexing_shorter_document_removes_old_chunks(store):
    index_chunks(store, _chunks("a.md", 5))
    index_chunks(store, _chunks("a.md", 2))
    assert list_sources(store) == {"a.md": 2}


def test_delete_source(store):
    index_chunks(store, _chunks("a.md", 2) + _chunks("b.md", 2))
    delete_source(store, "a.md")
    assert list_sources(store) == {"b.md": 2}


def test_search_returns_documents_with_scores(store):
    index_chunks(store, _chunks("a.md", 3))
    results = search(store, "a.md passage 1", k=2)
    assert len(results) == 2
    doc, score = results[0]
    assert doc.metadata["source"] == "a.md"
    assert isinstance(score, float)


def test_store_persists_on_disk(tmp_path):
    embeddings = DeterministicFakeEmbedding(size=32)
    index_chunks(get_vectorstore(embeddings, str(tmp_path)), _chunks("a.md", 2))
    reopened = get_vectorstore(embeddings, str(tmp_path))
    assert list_sources(reopened) == {"a.md": 2}


def test_pdf_pages_with_same_chunk_index_do_not_collide(store):
    # chunk_index repart de 0 à chaque page : les identifiants ne doivent pas se chevaucher
    chunks = [
        Document(page_content=f"page {p}", metadata={"source": "a.pdf", "file_type": "pdf", "page": p, "chunk_index": 0})
        for p in (1, 2, 3)
    ]
    index_chunks(store, chunks)
    assert list_sources(store) == {"a.pdf": 3}
