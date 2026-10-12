import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag.chains import _group_texts, format_context, source_label
from rag.service import RAGService
from rag.vectorstore import get_vectorstore


def _service(tmp_path, responses):
    store = get_vectorstore(DeterministicFakeEmbedding(size=32), str(tmp_path / "db"))
    return RAGService(store=store, llm=FakeListChatModel(responses=responses), length_function=len)


@pytest.fixture
def doc_file(tmp_path):
    path = tmp_path / "guide.md"
    path.write_text("# Install\n\nUtilise pip install pandas.\n\n# Groupby\n\nUtilise df.groupby('A').mean().", encoding="utf-8")
    return path


def test_ingest_reports_success_and_failure(tmp_path, doc_file):
    service = _service(tmp_path, ["ok"])
    report = service.ingest([str(doc_file), str(tmp_path / "absent.pdf")])
    assert report[0].startswith("✅ guide.md")
    assert report[1].startswith("❌ absent.pdf")
    assert "guide.md" in service.sources()


def test_ask_without_documents_does_not_call_llm(tmp_path):
    service = _service(tmp_path, [])
    answer, documents = service.ask("Comment grouper ?")
    assert "Aucun document" in answer
    assert documents == []


def test_ask_streams_answer_with_sources(tmp_path, doc_file):
    service = _service(tmp_path, ["Utilise groupby [1]."])
    service.ingest([str(doc_file)])
    steps = list(service.ask_stream("Comment grouper ?"))
    assert len(steps) > 1                      # la réponse arrive par morceaux
    answer, documents = steps[-1]
    assert answer == "Utilise groupby [1]."
    assert documents and documents[0].metadata["source"] == "guide.md"


def test_follow_up_question_is_reformulated_first(tmp_path, doc_file):
    # 1re réponse du faux LLM = question reformulée, 2e = réponse finale
    service = _service(tmp_path, ["Comment calculer la moyenne avec groupby ?", "Avec .mean() [1]."])
    service.ingest([str(doc_file)])
    history = [{"role": "user", "content": "Comment grouper ?"}, {"role": "assistant", "content": "Avec groupby."}]
    answer, _ = service.ask("et pour la moyenne ?", history=history)
    assert answer == "Avec .mean() [1]."


def test_summarize_known_and_unknown_source(tmp_path, doc_file):
    service = _service(tmp_path, ["Résumé du guide."])
    service.ingest([str(doc_file)])
    assert service.summarize("guide.md") == "Résumé du guide."
    assert "inconnue" in service.summarize("autre.pdf")


def test_delete_source(tmp_path, doc_file):
    service = _service(tmp_path, [])
    service.ingest([str(doc_file)])
    service.delete("guide.md")
    assert service.sources() == {}


def test_source_label_and_context_numbering():
    from langchain_core.documents import Document
    doc = Document(page_content="texte", metadata={"source": "m.pdf", "page": 12, "section": "Installation"})
    assert source_label(doc.metadata) == "m.pdf, p. 12 — Installation"
    assert format_context([doc]).startswith("[1] Source : m.pdf, p. 12 — Installation")


def test_group_texts_respects_budget():
    groups = _group_texts(["a" * 40, "b" * 40, "c" * 40], max_chars=100)
    assert len(groups) == 2


def test_gradio6_history_format_is_understood():
    from rag.service import _to_messages
    history = [
        {"role": "user", "content": [{"type": "text", "text": "Comment grouper ?"}]},
        {"role": "assistant", "content": "Avec groupby."},
    ]
    messages = _to_messages(history)
    assert [m.content for m in messages] == ["Comment grouper ?", "Avec groupby."]


def test_summarize_long_document_uses_map_reduce():
    from rag.chains import summarize_texts
    llm = FakeListChatModel(responses=["partiel"] * 3 + ["résumé final"])
    result = summarize_texts(llm, ["x" * 80, "y" * 80, "z" * 80], max_chars=100)
    assert result == "résumé final"
