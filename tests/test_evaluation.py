from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from evaluation.evaluate_retrieval import (
    Question, compute_metrics, evaluate, first_relevant_rank, format_report, is_relevant, load_questions,
)
from rag.vectorstore import get_vectorstore, index_chunks

Q = Question("comment grouper ?", "groupby.html", ("Aggregation",))


def _doc(source, section):
    return Document(page_content="x", metadata={"source": source, "section": section})


def test_relevance_needs_right_source_and_section():
    assert is_relevant(_doc("groupby.html", "Group by > Aggregation > Named aggregation"), Q)
    assert not is_relevant(_doc("merging.html", "Aggregation"), Q)
    assert not is_relevant(_doc("groupby.html", "Filtration"), Q)


def test_first_relevant_rank():
    docs = [_doc("groupby.html", "Filtration"), _doc("groupby.html", "Aggregation")]
    assert first_relevant_rank(docs, Q) == 2
    assert first_relevant_rank(docs[:1], Q) is None


def test_metrics():
    metrics = compute_metrics([1, 2, None, 4])
    assert metrics["hit_rate"] == 0.75
    assert metrics["mrr"] == (1 + 0.5 + 0.25) / 4


def test_question_file_is_valid():
    questions = load_questions()
    assert len(questions) >= 20
    assert all(q.sections for q in questions)


def test_evaluate_end_to_end(tmp_path):
    store = get_vectorstore(DeterministicFakeEmbedding(size=32), str(tmp_path))
    index_chunks(store, [
        Document(page_content="comment grouper ?", metadata={"source": "groupby.html", "section": "Aggregation", "chunk_index": 0}),
        Document(page_content="autre sujet", metadata={"source": "merging.html", "section": "concat()", "chunk_index": 0}),
    ])
    results = evaluate(store, [Q], {"sim": lambda s, q: s.similarity_search(q, k=2)})
    assert results["sim"]["hit_rate"] == 1.0
    assert "| sim | 100% |" in format_report(results, [Q], 2)


def test_llm_retrievers_with_fake_llm(tmp_path):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from evaluation.evaluate_retrieval import llm_retrievers

    store = get_vectorstore(DeterministicFakeEmbedding(size=32), str(tmp_path))
    index_chunks(store, [
        Document(page_content="groupby mean", metadata={"source": "groupby.html", "section": "Aggregation", "chunk_index": 0}),
    ])
    llm = FakeListChatModel(responses=["groupby mean"] * 3)
    retrievers = llm_retrievers(llm)
    results = evaluate(store, [Q], retrievers)
    assert set(results) == {"Réécriture LLM (EN + API)", "HyDE", "Réécriture + MMR λ=0.7 (application)"}
    assert results["HyDE"]["hit_rate"] == 1.0


def test_english_variant_uses_question_en(tmp_path):
    store = get_vectorstore(DeterministicFakeEmbedding(size=32), str(tmp_path))
    index_chunks(store, [
        Document(page_content="how to group", metadata={"source": "groupby.html", "section": "Aggregation", "chunk_index": 0}),
    ])
    q = Question("comment grouper ?", "groupby.html", ("Aggregation",), "how to group")
    seen = []
    def spy(store, text):
        seen.append(text)
        return store.similarity_search(text, k=1)
    evaluate(store, [q], {"spy": spy}, english=True)
    assert seen == ["how to group"]


def test_every_question_has_an_english_version():
    assert all(q.question_en for q in load_questions())
