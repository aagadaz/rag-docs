import re
from functools import lru_cache
from typing import Callable

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from rag.config import settings

LengthFn = Callable[[str], int]

CODE_LANGUAGES = {
    "python": Language.PYTHON,
    "javascript": Language.JS,
    "jsx": Language.JS,
    "typescript": Language.TS,
    "tsx": Language.TS,
    "java": Language.JAVA,
    "go": Language.GO,
    "rust": Language.RUST,
    "c": Language.C,
    "cpp": Language.CPP,
}
ROW_BASED = {"csv", "xlsx"}
DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]
MARKDOWN_SEPARATORS = RecursiveCharacterTextSplitter.get_separators_for_language(Language.MARKDOWN)
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


# ---------- mesure de la longueur en tokens ----------

@lru_cache(maxsize=1)
def token_length() -> LengthFn:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
    tokenizer.model_max_length = 10**9   # on compte des tokens, on ne fait pas tourner le modèle

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count


# ---------- découpage en sections Markdown ----------

def _split_sections(text: str) -> list[tuple[str, str]]:
    """Coupe un texte aux titres Markdown, sans jamais couper dans un bloc de code.

    Renvoie une liste de (fil d'Ariane, contenu), par ex. ("Installation > Exemple", "...").
    """
    sections = []
    path: list[str] = []
    current: list[str] = []
    in_code = False

    def flush():
        body = "\n".join(current).strip("\n")
        if body.strip():
            sections.append((" > ".join(path), body))
        current.clear()

    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_code = not in_code

        match = None if in_code else HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            path[:] = path[:level - 1] + [match.group(2)]
            continue

        current.append(line)

    flush()
    return sections


# ---------- choix des séparateurs ----------

def _separators_for(metadata: dict) -> list[str]:
    if metadata.get("file_type") == "code":
        language = CODE_LANGUAGES.get(metadata.get("language", ""))
        if language is not None:
            return RecursiveCharacterTextSplitter.get_separators_for_language(language)
        return DEFAULT_SEPARATORS

    if metadata.get("file_type") in ROW_BASED:
        return ["\n", " ", ""]

    return MARKDOWN_SEPARATORS


def _make_splitter(separators, chunk_size, chunk_overlap, length_function):
    return RecursiveCharacterTextSplitter(
        separators=separators,
        is_separator_regex=True,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
        strip_whitespace=False,
    )


# ---------- point d'entrée ----------

def split_documents(
    documents: list[Document],
    length_function: LengthFn | None = None,
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> list[Document]:
    length_function = length_function or token_length()
    chunks = []

    for doc in documents:
        separators = _separators_for(doc.metadata)
        is_structured = doc.metadata.get("file_type") not in ROW_BASED | {"code"}

        if is_structured:
            sections = _split_sections(doc.page_content)
        else:
            sections = [("", doc.page_content)]

        index = 0
        for breadcrumb, body in sections:
            prefix = f"[{breadcrumb}]\n" if breadcrumb else ""
            budget = max(chunk_size - length_function(prefix), chunk_size // 2)
            splitter = _make_splitter(separators, budget, chunk_overlap, length_function)

            for text in splitter.split_text(body):
                text = text.strip("\n").rstrip()
                if is_structured:
                    text = text.lstrip()
                if not text.strip():
                    continue
                metadata = {**doc.metadata, "chunk_index": index}
                if breadcrumb:
                    metadata["section"] = breadcrumb
                chunks.append(Document(page_content=prefix + text, metadata=metadata))
                index += 1

    return chunks