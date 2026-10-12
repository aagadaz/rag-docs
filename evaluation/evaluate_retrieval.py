"""Évaluation de la recherche : hit rate@k et MRR@k sur un jeu de questions annotées.

Usage, depuis la racine du projet :
    python -m evaluation.evaluate_retrieval             # indexe le corpus si besoin, puis évalue
    python -m evaluation.evaluate_retrieval --rebuild   # force la réindexation du corpus
"""
import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

HERE = Path(__file__).parent
QUESTIONS_FILE = HERE / "questions.jsonl"
CORPUS_FILE = HERE / "corpus.txt"
DB_DIR = HERE / "chroma_eval"
REPORT_FILE = HERE / "results.md"
K = 4

Retriever = Callable[[Chroma, str], list[Document]]

RETRIEVERS: dict[str, Retriever] = {
    "Similarité pure": lambda store, q: store.similarity_search(q, k=K),
    "MMR λ=0.5": lambda store, q: store.max_marginal_relevance_search(q, k=K, fetch_k=20, lambda_mult=0.5),
    "MMR λ=0.7 (réglage actuel)": lambda store, q: store.max_marginal_relevance_search(q, k=K, fetch_k=20, lambda_mult=0.7),
}


def llm_retrievers(llm=None) -> dict[str, Retriever]:
    """Configurations qui appellent le LLM avant la recherche (une requête par question)."""
    from rag.chains import build_hyde_chain, build_rewrite_chain, get_llm

    llm = llm or get_llm(max_tokens=300)
    rewrite, hyde = build_rewrite_chain(llm), build_hyde_chain(llm)

    def rewritten(store, q):
        return store.similarity_search(rewrite.invoke({"history": [], "question": q}), k=K)

    def application(store, q):
        # exactement le pipeline de RAGService : réécriture puis MMR λ=0.7
        query = rewrite.invoke({"history": [], "question": q})
        return store.max_marginal_relevance_search(query, k=K, fetch_k=20, lambda_mult=0.7)

    def hypothetical(store, q):
        passage = hyde.invoke({"question": q})
        # le passage hypothétique est un "document" : on l'encode avec le préfixe passage:
        vector = store.embeddings.embed_documents([passage])[0]
        return store.similarity_search_by_vector(vector, k=K)

    return {
        "Réécriture LLM (EN + API)": rewritten,
        "HyDE": hypothetical,
        "Réécriture + MMR λ=0.7 (application)": application,
    }


# ---------- données ----------

@dataclass(frozen=True)
class Question:
    question: str
    source: str
    sections: tuple[str, ...]
    question_en: str | None = None


def load_questions(path: Path = QUESTIONS_FILE) -> list[Question]:
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            data = json.loads(line)
            questions.append(Question(
                data["question"], data["source"], tuple(data["sections"]), data.get("question_en"),
            ))
    return questions


def load_corpus(path: Path = CORPUS_FILE) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


# ---------- métriques ----------

def is_relevant(doc: Document, question: Question) -> bool:
    """Un passage est pertinent s'il vient de la bonne source et d'une des sections attendues."""
    if doc.metadata.get("source") != question.source:
        return False
    section = doc.metadata.get("section", "").lower()
    return any(expected.lower() in section for expected in question.sections)


def first_relevant_rank(docs: list[Document], question: Question) -> int | None:
    """Rang (1, 2, 3...) du premier passage pertinent, ou None s'il n'y en a aucun."""
    for rank, doc in enumerate(docs, start=1):
        if is_relevant(doc, question):
            return rank
    return None


def compute_metrics(ranks: list[int | None]) -> dict[str, float]:
    n = len(ranks)
    return {
        "hit_rate": sum(rank is not None for rank in ranks) / n,
        "mrr": sum(1 / rank for rank in ranks if rank is not None) / n,
    }


# ---------- évaluation ----------

def evaluate(
    store: Chroma,
    questions: list[Question],
    retrievers: dict[str, Retriever] = RETRIEVERS,
    english: bool = False,
) -> dict:
    """english=True pose la version anglaise de chaque question : même recherche, seule la langue change."""
    results = {}
    for name, retrieve in retrievers.items():
        ranks = []
        for q in questions:
            text = q.question_en if english else q.question
            ranks.append(first_relevant_rank(retrieve(store, text), q))
        results[name] = {"ranks": ranks, **compute_metrics(ranks)}
    return results


def build_index(store: Chroma, corpus: list[str]) -> int:
    from rag.loaders import load_source
    from rag.splitter import split_documents
    from rag.vectorstore import index_chunks

    total = 0
    for source in corpus:
        total += index_chunks(store, split_documents(load_source(source)))
        print(f"  indexé : {source}")
    return total


def format_report(results: dict, questions: list[Question], n_chunks: int) -> str:
    lines = [
        "# Évaluation de la recherche",
        "",
        f"{len(questions)} questions annotées · corpus : {n_chunks} chunks · k = {K}",
        "",
        "| Configuration | Hit rate@4 | MRR@4 |",
        "|---|---|---|",
    ]
    for name, r in results.items():
        lines.append(f"| {name} | {r['hit_rate']:.0%} | {r['mrr']:.2f} |")

    lines += ["", "## Détail par question", "", "| Question | " + " | ".join(results) + " |",
              "|---|" + "---|" * len(results)]
    for i, q in enumerate(questions):
        cells = [str(r["ranks"][i]) if r["ranks"][i] else "✗" for r in results.values()]
        lines.append(f"| {q.question} | " + " | ".join(cells) + " |")
    lines += ["", "Rang du premier passage pertinent (1 = en tête) ; ✗ = absent des 4 passages."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="réindexer le corpus")
    parser.add_argument("--llm", action="store_true", help="tester aussi les stratégies qui appellent Claude")
    args = parser.parse_args()

    from rag.vectorstore import get_vectorstore, list_sources

    store = get_vectorstore(persist_directory=str(DB_DIR))
    if args.rebuild or not list_sources(store):
        print("Indexation du corpus d'évaluation…")
        build_index(store, load_corpus())

    questions = load_questions()
    n_chunks = sum(list_sources(store).values())
    retrievers = dict(RETRIEVERS)
    if args.llm:
        retrievers.update(llm_retrievers())
    results = evaluate(store, questions, retrievers)
    if all(q.question_en for q in questions):
        english = evaluate(store, questions, {"Similarité pure": RETRIEVERS["Similarité pure"]}, english=True)
        results["Similarité pure, questions en anglais"] = english["Similarité pure"]

    report = format_report(results, questions, n_chunks)
    REPORT_FILE.write_text(report, encoding="utf-8")

    print()
    for name, r in results.items():
        print(f"{name:<40} hit rate@4 = {r['hit_rate']:.0%}   MRR@4 = {r['mrr']:.2f}")
    print(f"\nRapport détaillé : {REPORT_FILE}")


if __name__ == "__main__":
    main()
