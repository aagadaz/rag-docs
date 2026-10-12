from collections.abc import Iterator
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from rag.chains import build_answer_chain, build_condense_chain, format_context, get_llm, summarize_texts
from rag.config import settings
from rag.loaders import load_source
from rag.splitter import split_documents
from rag.vectorstore import delete_source, get_vectorstore, index_chunks, list_sources

MAX_HISTORY_MESSAGES = 6


def _text_of(content) -> str:
    """Gradio 6 stocke le contenu d'un message comme une liste de blocs [{'type': 'text', 'text': ...}]."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(block.get("text", "") for block in content if isinstance(block, dict))
    return ""


def _to_messages(history: list[dict] | None) -> list[BaseMessage]:
    """Convertit l'historique Gradio [{'role': 'user', 'content': ...}, ...] en messages LangChain."""
    messages = []
    for message in (history or [])[-MAX_HISTORY_MESSAGES:]:
        text = _text_of(message.get("content"))
        if not text:
            continue
        cls = HumanMessage if message.get("role") == "user" else AIMessage
        messages.append(cls(content=text))
    return messages


def _reading_order(metadata: dict) -> tuple:
    return (
        metadata.get("page", 0),
        metadata.get("slide", 0),
        metadata.get("row_start", 0),
        metadata.get("chunk_index", 0),
    )


class RAGService:
    """Façade unique utilisée par l'interface : ingérer, interroger, résumer."""

    def __init__(self, store=None, llm=None, length_function=None):
        self.store = store or get_vectorstore()
        self.llm = llm or get_llm()
        self.length_function = length_function
        self.answer_chain = build_answer_chain(self.llm)
        self.condense_chain = build_condense_chain(self.llm)

    # ----- gestion des documents -----

    def ingest(self, sources: list[str]) -> list[str]:
        report = []
        for source in sources:
            name = source if source.startswith(("http://", "https://")) else Path(source).name
            try:
                chunks = split_documents(load_source(source), length_function=self.length_function)
                count = index_chunks(self.store, chunks)
                report.append(f"✅ {name} : {count} passages indexés")
            except Exception as error:
                report.append(f"❌ {name} : {error}")
        return report

    def sources(self) -> dict[str, int]:
        return list_sources(self.store)

    def delete(self, source: str) -> None:
        delete_source(self.store, source)

    # ----- questions / réponses -----

    def retrieve(self, question: str, source: str | None = None) -> list[Document]:
        if not self.sources():
            return []
        return self.store.max_marginal_relevance_search(
            question,
            k=settings.top_k,
            fetch_k=20,
            lambda_mult=0.7,  # 1 = pertinence pure, 0 = diversité pure
            filter={"source": source} if source else None,
        )

    def ask_stream(
        self, question: str, history: list[dict] | None = None, source: str | None = None
    ) -> Iterator[tuple[str, list[Document]]]:
        messages = _to_messages(history)
        standalone = self.condense_chain.invoke({"history": messages, "question": question}) if messages else question

        documents = self.retrieve(standalone, source)
        if not documents:
            yield "Aucun document indexé pour l'instant : ajoute d'abord une documentation.", []
            return

        answer = ""
        for piece in self.answer_chain.stream({
            "history": messages,
            "context": format_context(documents),
            "question": question,
        }):
            answer += piece
            yield answer, documents

    def ask(self, question: str, history: list[dict] | None = None, source: str | None = None):
        answer, documents = "", []
        for answer, documents in self.ask_stream(question, history, source):
            pass
        return answer, documents

    # ----- résumé -----

    def summarize(self, source: str) -> str:
        data = self.store.get(where={"source": source}, include=["documents", "metadatas"])
        if not data["documents"]:
            return f"Source inconnue : {source}"
        ordered = sorted(zip(data["metadatas"], data["documents"]), key=lambda pair: _reading_order(pair[0]))
        return summarize_texts(self.llm, [text for _, text in ordered])
