"""Configuration du SelfQueryRetriever officiel de LangChain."""

from __future__ import annotations

from functools import lru_cache

from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.query_constructors.chroma import ChromaTranslator
from langchain_core.structured_query import StructuredQuery
from langchain_google_genai import ChatGoogleGenerativeAI

from config import settings, validate_api_key
from vector_store import get_vector_store

DOCUMENT_CONTENT_DESCRIPTION = (
    "Une fiche de KPI provenant d'un glossaire métier. Son contenu contient le nom, "
    "la définition, la formule et les autres informations disponibles sur ce KPI."
)

METADATA_FIELD_INFO = [
    AttributeInfo(
        name="glossary_name",
        description="Nom exact du glossaire ou du client, par exemple BP, Orange ou ag2r.",
        type="string",
    ),
    AttributeInfo(
        name="kpi_name",
        description="Nom exact du KPI, par exemple DMC, DMT ou Appels reçus SVI.",
        type="string",
    ),
]


@lru_cache(maxsize=1)
def create_llm() -> ChatGoogleGenerativeAI:
    """Construit le même LLM Gemini pour self-query et génération."""
    validate_api_key()
    return ChatGoogleGenerativeAI(model=settings.llm_model, temperature=0)


@lru_cache(maxsize=1)
def create_self_query_retriever() -> SelfQueryRetriever:
    """Crée le retriever officiel; LangChain produit et traduit tous les filtres."""
    return SelfQueryRetriever.from_llm(
        llm=create_llm(),
        vectorstore=get_vector_store(),
        document_contents=DOCUMENT_CONTENT_DESCRIPTION,
        metadata_field_info=METADATA_FIELD_INFO,
        structured_query_translator=ChromaTranslator(),
        search_kwargs={"k": settings.retrieval_k},
        enable_limit=True,
        verbose=True,
    )


def build_structured_query(question: str) -> StructuredQuery:
    """Expose la StructuredQuery officielle à des fins de diagnostic."""
    retriever = create_self_query_retriever()
    query = retriever.query_constructor.invoke({"query": question})
    if not isinstance(query, StructuredQuery):
        raise TypeError("Le constructeur LangChain n'a pas produit de StructuredQuery.")
    return query
