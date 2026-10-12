from pathlib import Path
import json
import re

INPUT_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 400
OVERLAP = 80

def normalize_text(text):
    text = text.replace("\r\n", "\n")

    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[#\]\([^)]+\)", "", text)

    text = re.sub(r"\b(In|Out) \[\d+\]:", "", text)
    text = re.sub(r"\.{3,}:?", "", text)

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()

def create_chunks(text, chunk_size =CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end == len(words):
            break
        start = end - overlap
    return chunks

documents = []

for file_path in INPUT_DIR.rglob("*.txt"):
    raw_text = file_path.read_text(encoding="utf-8")
    text = normalize_text(raw_text)

    if not text:
        continue
    chunks = create_chunks(text)

    for chunk_id, chunk in enumerate(chunks):
        documents.append({
            "id" : f"{file_path.stem}-{chunk_id}",
            "source": str(file_path),
            "text": chunk,
        })

    output_file = OUTPUT_DIR / "chunks.json"
    output_file.write_text(
        json.dumps(documents, ensure_ascii =False, indent=2),
        encoding = "utf-8",
    )

    print(f"{len(documents)} passage créés.")
    print(f"Fichier enregistré : {output_file}")
