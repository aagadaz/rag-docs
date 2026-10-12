from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str
    chat_model: str
    embedding_model: str
    chroma_dir: str
    chunk_size: int = 400
    chunk_overlap: int = 60
    top_k: int = 4


def _load_settings() -> Settings:
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY manquante. Copie .env.example en .env "
            "et renseigne ta clé (platform.claude.com > API Keys)."
        )

    return Settings(
        anthropic_api_key=api_key,
        chat_model=os.getenv("CHAT_MODEL", "claude-haiku-5-5"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
        chroma_dir=os.getenv("CHROMA_DIR", "chroma_db"),
    )


settings = _load_settings()