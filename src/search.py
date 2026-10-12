from pathlib import Path
import pickle

import faiss
import torch
from sentence_transformers import SentenceTransformer

INDEX_PATH = Path("indexes/docs.faiss")
METADATA_PATH = Path("indexes/metadata.pkl")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class SemanticSearch:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Appareil utilisé pour la recherche : {self.device}")

        self.model = SentenceTransformer(
            MODEL_NAME,
            device=self.device,
        )

        self.index = faiss.read_index(str(INDEX_PATH))

        with METADATA_PATH.open("rb") as file:
            self.documents = pickle.load(file)

    def search(self, question, top_k=5):
        query_embedding = self.model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        scores, ids = self.index.search(query_embedding, top_k)

        results = []

        for score, document_id in zip(scores[0], ids[0]):
            if document_id == -1:
                continue

            document = self.documents[document_id]

            results.append({
                "score": float(score),
                "source": document["source"],
                "text": document["text"],
            })

        return results


if __name__ == "__main__":
    search_engine = SemanticSearch()

    question = input("Pose ta question : ")
    results = search_engine.search(question, top_k=3)

    print("\nRésultats :\n")

    for number, result in enumerate(results, start=1):
        print(f"--- Résultat {number} | score={result['score']:.4f} ---")
        print(f"Source : {result['source']}")
        print(result["text"][:1000])
        print()