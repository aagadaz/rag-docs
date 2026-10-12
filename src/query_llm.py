import json

import requests


OLLAMA_URL = "127.0.0.1 http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3.2:3b"
REQUEST_TIMEOUT = 600
MAX_CONTEXT_CHARACTERS = 6000

SYSTEM_PROMPT = """Tu es un assistant documentaire technique.

Tu réponds uniquement à partir des passages de documentation fournis.

Règles :
- si les passages ne contiennent pas la réponse, dis-le clairement et n'invente rien ;
- cite les numéros des passages utilisés, par exemple [1] ou [3] ;
- réponds toujours en français, même si les passages sont en anglais ;
- si un passage contient du code, reprends-le dans un bloc de code ;
- reste concis et factuel, sans reformuler la question.
"""


def check_ollama():
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        response.raise_for_status()
    except requests.RequestException:
        return False, []

    installed_models = [
        entry["name"]
        for entry in response.json().get("models", [])
    ]

    return True, installed_models


def build_context(passages, max_characters=MAX_CONTEXT_CHARACTERS):
    blocks = []
    total_length = 0

    for number, passage in enumerate(passages, start=1):
        block = (
            f"[{number}] Source : {passage['source']}\n"
            f"{passage['text']}"
        )

        if total_length + len(block) > max_characters:
            break

        blocks.append(block)
        total_length += len(block)

    return "\n\n".join(blocks)


def build_payload(question, passages, model_name, temperature, stream):
    context = build_context(passages)

    user_prompt = (
        "Passages de documentation :\n\n"
        f"{context}\n\n"
        "Question :\n"
        f"{question}\n\n"
        "Réponse :"
    )

    return {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": stream,
        "options": {
            "temperature": temperature,
            "num_ctx": 8192,
        },
    }


def generate_answer(question, passages, model_name=DEFAULT_MODEL, temperature=0.2):
    payload = build_payload(
        question,
        passages,
        model_name,
        temperature,
        stream=False,
    )

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    return response.json()["message"]["content"]


def stream_answer(question, passages, model_name=DEFAULT_MODEL, temperature=0.2):
    payload = build_payload(
        question,
        passages,
        model_name,
        temperature,
        stream=True,
    )

    with requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        stream=True,
        timeout=REQUEST_TIMEOUT,
    ) as response:
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            chunk = json.loads(line)
            piece = chunk.get("message", {}).get("content", "")

            if piece:
                yield piece

            if chunk.get("done"):
                break


if __name__ == "__main__":
    available, installed_models = check_ollama()

    if not available:
        print("Ollama n'est pas accessible sur 127.0.0.1 http://127.0.0.1:11434")
        print("Lance l'application Ollama ou 'ollama serve'.")
    else:
        print("Ollama est accessible.")
        print("Modèles installés :", installed_models)
