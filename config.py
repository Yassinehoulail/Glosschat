"""Configuration centralisée du RAG."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a conventional boolean environment variable."""
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Paramètres applicatifs chargés depuis les variables d'environnement."""

    glossary_path: Path = Path("data/glossaries_for_retrieval.json")
    file_metadata_path: Path = Path("data/file_metadata.json")
    vector_path: str = os.getenv("RAG_VECTOR_PATH", "data/rag_vector_store")
    collection_name: str = os.getenv("RAG_COLLECTION_NAME", "kpi_glossaries")
    llm_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    embedding_model: str = os.getenv(
        "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"
    )
    retrieval_k: int = int(os.getenv("RAG_RETRIEVAL_K", "10"))
    retrieval_candidate_k: int = int(os.getenv("RAG_RETRIEVAL_CANDIDATE_K", "50"))
    use_reranker: bool = _env_flag("RAG_USE_RERANKER")
    # Global intents use deterministic exhaustive processing. Full-catalogue
    # vector retrieval remains an explicit diagnostic option.
    exhaustive_search: bool = _env_flag("RAG_EXHAUSTIVE_SEARCH")


def validate_api_key() -> None:
    """Valide la clé et accepte les deux noms usuels de variable Gemini."""
    google_key = os.getenv("GOOGLE_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not google_key and not gemini_key:
        raise RuntimeError(
            "Définissez GOOGLE_API_KEY (ou GEMINI_API_KEY) dans le fichier .env."
        )
    if not google_key and gemini_key:
        os.environ["GOOGLE_API_KEY"] = gemini_key


settings = Settings()
