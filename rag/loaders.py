import re
import unicodedata

from pathlib import Path
from typing import Callable
from pypdf import PdfReader

from langchain_core.documents import Document

LoaderFn = Callable[[Path], list[Document]]
_REGISTRY: dict[str, LoaderFn] = {}


class UnsupportedFormatError(ValueError):
    pass


def register(*extensions: str):
    def decorator(fn: LoaderFn) -> LoaderFn:
        for ext in extensions:
            _REGISTRY[ext.lower()] = fn
        return fn
    return decorator


@register(".txt", ".md")
def _load_text(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [Document(
        page_content=text,
        metadata={"source": path.name, "file_type": path.suffix.lower().lstrip(".")},
    )]

def _clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)   # "ﬁ" -> "fi"
    text = re.sub(r"-\n(\w)", r"\1", text)       # "configu-\nration" -> "configuration"
    text = re.sub(r"[ \t]+", " ", text)          # espaces multiples -> un seul
    text = re.sub(r"\n{3,}", "\n\n", text)       # lignes vides en série -> une seule
    return text.strip()

@register(".pdf")

def _load_pdf(path: Path) -> list[Document]:
    reader = PdfReader(path)
    documents=[]

    for page_number, page in enumerate(reader.pages, start=1):
        text = _clean_text(page.extract_text() or "")

        if not text:
            continue
    documents.append(Document(
        page_content=text,
        metadata={
            "source":path.name,
            "file_type":"pdf",
            "page":page_number
        },
    ))

    if not documents:
        raise ValueError(
            f"Aucun texte extractible dans {path.name}."
            "C'est probablement un pdf scanné (images),l'OCR n'est pas supporté."
        )
    return documents




def load_source(source: str | Path) -> list[Document]:
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(path)
    loader = _REGISTRY.get(path.suffix.lower())
    if loader is None:
        raise UnsupportedFormatError(f"Format non supporté : {path.suffix}")
    return loader(path)