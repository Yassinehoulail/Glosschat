"""Création et alimentation du vector store Chroma."""

from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import settings, validate_api_key
from loader import load_documents


@lru_cache(maxsize=1)
def create_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Construit le modèle d'embeddings Gemini officiel."""
    validate_api_key()
    return GoogleGenerativeAIEmbeddings(model=settings.embedding_model)


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """Ouvre (ou crée) la collection Chroma persistante."""
    return Chroma(
        collection_name=settings.collection_name,
        persist_directory=settings.vector_path,
        embedding_function=create_embeddings(),
    )


def rebuild_vector_store(batch_size: int = 100) -> int:
    """Reconstruit complètement l'index à partir des JSON."""
    documents = load_documents()
    store = get_vector_store()
    store.delete_collection()
    get_vector_store.cache_clear()
    store = get_vector_store()

    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        store.add_documents(batch, ids=[str(document.id) for document in batch])
        print(f"Indexés : {min(start + batch_size, len(documents))}/{len(documents)}")
    return len(documents)
