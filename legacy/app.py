import pickle

import faiss
import gradio as gr
import requests
import torch

from pathlib import Path
from sentence_transformers import SentenceTransformer

from upload_index import process_files
from build_uploaded_index import build_index, MODEL_NAME
from query_llm import check_ollama, stream_answer, DEFAULT_MODEL


INDEX_PATH = Path("indexes/uploaded_docs.faiss")
METADATA_PATH = Path("indexes/uploaded_metadata.pkl")

MIN_SCORE = 0.25

embedding_model = None
index = None
documents = []


def load_embedding_model():
    global embedding_model

    if embedding_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Chargement du modèle d'embeddings sur : {device}")
        embedding_model = SentenceTransformer(MODEL_NAME, device=device)

    return embedding_model


def load_existing_index():
    global index, documents

    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        return "Aucun index trouvé. Uploade des documents pour commencer."

    index = faiss.read_index(str(INDEX_PATH))

    with METADATA_PATH.open("rb") as file:
        documents = pickle.load(file)

    load_embedding_model()

    return f"Index existant chargé : {len(documents)} passages."


def index_uploaded_files(files):
    global index, documents

    if not files:
        return "Aucun fichier sélectionné."

    file_paths = [
        file.name if hasattr(file, "name") else file
        for file in files
    ]

    documents = process_files(file_paths)

    if not documents:
        return "Aucun texte exploitable trouvé dans les fichiers."

    number_of_chunks = build_index(documents)

    load_embedding_model()
    index = faiss.read_index(str(INDEX_PATH))

    return (
        f"Index construit avec succès.\n\n"
        f"- Fichiers reçus : {len(file_paths)}\n"
        f"- Passages indexés : {number_of_chunks}"
    )


def retrieve(question, top_k):
    query_embedding = load_embedding_model().encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, ids = index.search(query_embedding, int(top_k))

    passages = []

    for score, document_id in zip(scores[0], ids[0]):
        if document_id == -1 or score < MIN_SCORE:
            continue

        document = documents[document_id]

        passages.append({
            "score": float(score),
            "source": document["source"],
            "text": document["text"],
        })

    return passages


def format_sources(passages):
    blocks = []

    for number, passage in enumerate(passages, start=1):
        blocks.append(
            f"**[{number}] {passage['source']}** — score {passage['score']:.4f}\n\n"
            f"> {passage['text'][:500]}…"
        )

    return "\n\n---\n\n".join(blocks)


def answer_question(question, top_k, ollama_model):
    if index is None:
        yield "Commence par uploader des documents et construire l'index.", ""
        return

    if not question or not question.strip():
        yield "Écris une question.", ""
        return

    question = question.strip()
    passages = retrieve(question, top_k)

    if not passages:
        yield (
            "Aucun passage suffisamment pertinent trouvé dans tes documents.",
            "",
        )
        return

    sources = format_sources(passages)
    available, _ = check_ollama()

    if not available:
        yield (
            "Ollama n'est pas accessible sur http://localhost:11434.\n\n"
            "Lance `ollama serve`, puis réessaie. "
            "En attendant, voici les passages trouvés.",
            sources,
        )
        return

    answer = "_Génération en cours…_"
    yield answer, sources

    answer = ""

    try:
        for piece in stream_answer(question, passages, model_name=ollama_model):
            answer += piece
            yield answer, sources

    except requests.RequestException as error:
        yield (
            f"{answer}\n\n---\n\nLa génération s'est interrompue : {error}",
            sources,
        )


ollama_available, installed_models = check_ollama()
model_choices = installed_models or [DEFAULT_MODEL]


with gr.Blocks(title="RAG - Documents personnels") as demo:
    gr.Markdown(
        """
        # Assistant RAG documentaire

        Uploade tes documentations, construis l'index,
        puis pose tes questions. La réponse est générée
        localement à partir de tes seuls documents.
        """
    )

    with gr.Accordion("Documents et index", open=True):
        files = gr.File(
            label="Documents à indexer",
            file_count="multiple",
            file_types=[".txt", ".md", ".pdf", ".docx", ".json",".csv", ".xlsx"],
            type="filepath",
        )

        index_button = gr.Button("Construire l'index", variant="primary")
        index_status = gr.Markdown()

    question = gr.Textbox(
        label="Question",
        placeholder="Pose une question sur tes documents…",
        lines=3,
    )

    with gr.Row():
        top_k = gr.Slider(
            minimum=1,
            maximum=10,
            value=5,
            step=1,
            label="Passages utilisés",
        )

        ollama_model = gr.Dropdown(
            choices=model_choices,
            value=model_choices[0],
            label="Modèle Ollama",
        )

    ask_button = gr.Button("Répondre", variant="primary")

    gr.Markdown("## Réponse")
    answer_output = gr.Markdown()

    with gr.Accordion("Passages sources", open=False):
        sources_output = gr.Markdown()

    index_button.click(
        fn=index_uploaded_files,
        inputs=files,
        outputs=index_status,
    )

    ask_button.click(
        fn=answer_question,
        inputs=[question, top_k, ollama_model],
        outputs=[answer_output, sources_output],
    )

    question.submit(
        fn=answer_question,
        inputs=[question, top_k, ollama_model],
        outputs=[answer_output, sources_output],
    )

    demo.load(fn=load_existing_index, outputs=index_status)


if __name__ == "__main__":
    demo.launch()
