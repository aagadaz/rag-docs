from pathlib import Path
import pickle

import faiss
import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

INDEX_DIR = Path("indexes")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = INDEX_DIR / "uploaded_docs.faiss"
METADATA_PATH = INDEX_DIR / "uploaded_metadata.pkl"


def build_index(documents):
    if not documents:
        raise ValueError("Aucun passage à indexer.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Appareil utilisé pour les embeddings : {device}")

    model = SentenceTransformer(MODEL_NAME, device=device)

    texts = [document["text"] for document in documents]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    with METADATA_PATH.open("wb") as file:
        pickle.dump(documents, file)

    return len(documents)
