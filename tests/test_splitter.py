from langchain_core.documents import Document

from rag.splitter import _split_sections, split_documents

CHUNK_SIZE = 300


def _split(*docs):
    # len = longueur en caractères : pas besoin de télécharger le tokenizer pour tester
    return split_documents(list(docs), length_function=len, chunk_size=CHUNK_SIZE, chunk_overlap=30)


def test_sections_follow_headings_and_ignore_comments_in_code():
    text = "# A\n\nIntro.\n\n## B\n\n```\n# pas un titre\nx = 1\n```\n\n# C\n\nFin."
    sections = _split_sections(text)
    assert [breadcrumb for breadcrumb, _ in sections] == ["A", "A > B", "C"]
    assert "# pas un titre" in sections[1][1]


def test_chunks_respect_size_and_keep_metadata():
    doc = Document(page_content="Phrase de test. " * 200, metadata={"source": "a.pdf", "file_type": "pdf", "page": 3})
    chunks = _split(doc)
    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert len(chunk.page_content) <= CHUNK_SIZE
        assert chunk.metadata["source"] == "a.pdf"
        assert chunk.metadata["page"] == 3
        assert chunk.metadata["chunk_index"] == i


def test_chunks_carry_their_section_title():
    doc = Document(page_content="# Installation\n\n## Pip\n\n" + "Texte. " * 100, metadata={"source": "a.md", "file_type": "md"})
    for chunk in _split(doc):
        assert chunk.page_content.startswith("[Installation > Pip]")
        assert chunk.metadata["section"] == "Installation > Pip"


def test_small_code_block_is_never_cut():
    code = "```\nfrom docx import Document\n\nd = Document()\nfor p in d.paragraphs:\n    print(p.text)\n```"
    doc = Document(page_content="# Exemple\n\n" + code, metadata={"source": "a.md", "file_type": "md"})
    chunks = _split(doc)
    assert len(chunks) == 1
    assert "    print(p.text)" in chunks[0].page_content


def test_python_code_keeps_indentation():
    code = "def f():\n    return 1\n\n\ndef g():\n" + "    x = 1\n" * 60
    doc = Document(page_content=code, metadata={"source": "a.py", "file_type": "code", "language": "python"})
    chunks = _split(doc)
    assert chunks[0].page_content.startswith("def f():")
    assert all("x = 1" not in c.page_content or "    x = 1" in c.page_content for c in chunks)