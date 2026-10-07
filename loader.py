"""Chargement des deux fichiers JSON en documents LangChain."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from config import settings

KPI_NAME_ALIASES = {
    "kpi",
    "kpi s",  # KPI’s / KPI's after normalization
    "nom kpi",
    "nom du kpi",
    "nom de kpi",
    "nom indicateur",
    "nom de l indicateur",
    "libelle kpi",
}


def normalize(value: Any) -> str:
    """Normalise les libellés issus de feuilles Excel hétérogènes."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def repair_encoding(value: Any) -> Any:
    """Repair common UTF-8/Latin-1 mojibake in imported glossary values."""
    if isinstance(value, dict):
        return {repair_encoding(key): repair_encoding(item) for key, item in value.items()}
    if isinstance(value, list):
        return [repair_encoding(item) for item in value]
    if not isinstance(value, str) or not any(marker in value for marker in ("Ã", "Â", "â")):
        return value
    try:
        return value.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


def find_kpi_name(record: dict[str, Any]) -> str | None:
    """Retrouve le nom du KPI parmi les variantes de colonnes sources."""
    for key, value in record.items():
        if normalize(key) in KPI_NAME_ALIASES and value not in (None, ""):
            return str(value).strip()
    return None


def _glossary_name_from_file(item: dict[str, Any]) -> str:
    name = str(item.get("original_name") or item.get("local_filename") or "")
    name = Path(name).stem
    name = re.sub(r"(?i)^glossaire_kpis?_", "", name)
    name = re.sub(r"(?i)^glossaire_", "", name)
    name = re.sub(r"_[A-Za-z0-9_-]{20,}$", "", name)
    return name.replace("_", " ").strip()


def load_documents() -> list[Document]:
    """Crée un document par KPI et joint les métadonnées du fichier source."""
    glossaries = repair_encoding(json.loads(settings.glossary_path.read_text(encoding="utf-8")))
    files = json.loads(settings.file_metadata_path.read_text(encoding="utf-8"))
    source_by_glossary = {
        normalize(_glossary_name_from_file(item)): item for item in files
    }

    documents: list[Document] = []
    for glossary in glossaries:
        glossary_name = str(glossary.get("glossary_name") or "unknown")
        source = source_by_glossary.get(normalize(glossary_name), {})
        for record in glossary.get("kpis", []):
            if not isinstance(record, dict):
                continue
            kpi_name = find_kpi_name(record)
            if not kpi_name:
                continue

            values = {
                str(key): value
                for key, value in record.items()
                if value not in (None, "")
            }
            page_content = (
                f"Glossaire : {glossary_name}\n"
                f"KPI : {kpi_name}\n"
                f"Informations : {json.dumps(values, ensure_ascii=False, default=str)}"
            )
            document_id = hashlib.sha256(
                f"{glossary_name}|{kpi_name}|{record.get('sheet_name')}|"
                f"{record.get('row_number')}".encode()
            ).hexdigest()
            documents.append(
                Document(
                    id=document_id,
                    page_content=page_content,
                    metadata={
                        "glossary_name": glossary_name,
                        "kpi_name": kpi_name,
                        "sheet_name": str(record.get("sheet_name") or ""),
                        "row_number": int(record.get("row_number") or 0),
                        "source_file": str(source.get("local_filename") or ""),
                        "source_url": str(source.get("web_view_link") or ""),
                    },
                )
            )
    return documents
