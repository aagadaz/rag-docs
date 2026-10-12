from pathlib import Path
import json
import pickle

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = Path("data/processed/chunks.json")
INDEX_DIR  = Path("indexes")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

with CHUNKS_FILE.open("r", encoding="utf-8") as file:
    documents = json.load(file)

texts = [document["text"] for document in documents]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Appareil utilisé: {device}")

if device =="cuda":
    print(f"GPU : {torch.cuda.get_device_name(0)}")

model = SentenceTransformer(MODEL_NAME, device=device)

embeddings = model.encode(
    texts,
    batch_size = 32,
    show_progress_bar = True,
    normalize_embeddings = True,
    convert_to_numpy = True,
)

embeddings = embeddings.astype("float32")
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index,str(INDEX_DIR/ "docs.faiss"))

with (INDEX_DIR / "metadata.pk1").open("wb") as file:
    pickle.dump(documents,file)
print(f"{len(documents)} passages indexés.")
print(f"Dimension des vecteurs: {embeddings.shape[1]}")