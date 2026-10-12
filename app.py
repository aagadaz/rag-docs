import gradio as gr

from rag.chains import source_label
from rag.loaders import supported_extensions
from rag.service import RAGService

ALL_SOURCES = "Toutes les sources"

service = RAGService()


# ---------- helpers d'affichage ----------

def _source_choices() -> list[str]:
    return [ALL_SOURCES] + list(service.sources())


def _sources_table() -> str:
    sources = service.sources()
    if not sources:
        return "_Aucun document indexé._"
    lines = ["| Document | Passages |", "|---|---|"]
    lines += [f"| {name} | {count} |" for name, count in sources.items()]
    return "\n".join(lines)


def _format_passages(documents) -> str:
    blocks = []
    for number, doc in enumerate(documents, start=1):
        excerpt = doc.page_content[:600]
        blocks.append(f"**[{number}] {source_label(doc.metadata)}**\n\n```\n{excerpt}\n```")
    return "\n\n".join(blocks)


def _selected(source: str) -> str | None:
    return None if source in (None, "", ALL_SOURCES) else source


# ---------- actions ----------

def add_documents(files, urls):
    sources = list(files or [])
    sources += [line.strip() for line in (urls or "").splitlines() if line.strip()]
    if not sources:
        return "Ajoute au moins un fichier ou une URL.", gr.update(), _sources_table(), ""

    report = "\n".join(service.ingest(sources))
    return report, gr.update(choices=_source_choices(), value=ALL_SOURCES), _sources_table(), ""


def delete_document(source):
    if _selected(source) is None:
        return "Choisis un document précis à supprimer.", gr.update(), _sources_table()
    service.delete(source)
    return f"🗑️ {source} supprimé.", gr.update(choices=_source_choices(), value=ALL_SOURCES), _sources_table()


def respond(message, history, source):
    if not message or not message.strip():
        yield history, "", gr.update()
        return

    previous = list(history or [])
    history = previous + [{"role": "user", "content": message}, {"role": "assistant", "content": "…"}]
    yield history, "", gr.update()

    try:
        for answer, documents in service.ask_stream(message, previous, _selected(source)):
            history[-1] = {"role": "assistant", "content": answer}
            yield history, "", _format_passages(documents)
    except Exception as error:
        history[-1] = {"role": "assistant", "content": f"⚠️ Erreur : {error}"}
        yield history, "", gr.update()


def summarize(source):
    if _selected(source) is None:
        yield "Choisis d'abord un document dans la liste."
        return
    yield f"⏳ Résumé de **{source}** en cours…"
    try:
        yield service.summarize(source)
    except Exception as error:
        yield f"⚠️ Erreur : {error}"


# ---------- interface ----------

with gr.Blocks(title="RAG Doc Assistant") as demo:
    gr.Markdown("# 📚 RAG Doc Assistant\nPose tes questions sur ta documentation technique, ou demande-en un résumé.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Documents")
            files = gr.File(
                label="Fichiers",
                file_count="multiple",
                file_types=supported_extensions(),
                type="filepath",
            )
            urls = gr.Textbox(label="URLs (une par ligne)", lines=2, placeholder="https://pandas.pydata.org/docs/...")
            add_button = gr.Button("Indexer", variant="primary")
            ingest_report = gr.Markdown()
            sources_table = gr.Markdown(_sources_table())

            source = gr.Dropdown(choices=_source_choices(), value=ALL_SOURCES, label="Limiter à un document")
            with gr.Row():
                summarize_button = gr.Button("Résumer ce document")
                delete_button = gr.Button("Supprimer", variant="stop")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation", height=520)
            question = gr.Textbox(placeholder="Ex. : comment calculer une moyenne par groupe avec pandas ?", show_label=False)
            with gr.Accordion("Passages utilisés", open=False):
                passages = gr.Markdown()
            with gr.Accordion("Résumé", open=False):
                summary = gr.Markdown()

    add_button.click(add_documents, [files, urls], [ingest_report, source, sources_table, urls])
    delete_button.click(delete_document, [source], [ingest_report, source, sources_table])
    question.submit(respond, [question, chatbot, source], [chatbot, question, passages])
    summarize_button.click(summarize, [source], [summary])


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
