"""Extract KPI records from downloaded Excel glossaries into one JSON file."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from datetime import date, datetime
from pathlib import Path
from typing import Any

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

from hardcoded import KPI_SHEETS
from ingestion.config import (
    GLOSSARY_KPI_JSON_PATH,
    KPI_JSON_PATH,
    METADATA_PATH,
    RETRIEVAL_GLOSSARY_JSON_PATH,
)

LOGGER = logging.getLogger(__name__)

# Column names are normalized before comparison, so accents and case do not matter.
KPI_ALIASES = {
    "kpi",
    "kpi s",
    "nom kpi",
    "nom du kpi",
    "nom de kpi",
    "indicateur",
    "nom indicateur",
    "nom de l indicateur",
    "libelle kpi",
}
DEFINITION_ALIASES = {
    "definition",
    "definition kpi",
    "description",
    "description kpi",
    "libelle",
}
FORMULA_ALIASES = {
    "formule",
    "formula",
    "formule de calcul",
    "formule calcul",
    "calcul",
    "mode de calcul",
}
SOURCE_ALIASES = {
    "source",
    "systeme source",
    "source system",
    "source de donnees",
}
IGNORED_SHEETS = {
    "readme",
    "read me",
    "lisez moi",
    "notice",
    "sommaire",
    "recette",
    "suivi recette",
    "explore 360",
    "sources",
    "sources de donnees",
}

METADATA_FIELDS = {"file_id", "source_file", "modified_time"}
ORIGINAL_COLUMNS_KEY = "_original_columns"
# ``KPI's`` normalizes to ``kpi s`` (the apostrophe becomes a separator).
KPI_SHEET_NAMES = {"kpi", "kpis", "kpi s"}


class KpiExtractionError(RuntimeError):
    """Raised when KPI extraction cannot start or save its result."""


def normalize_header(value: Any) -> str:
    """Normalize an Excel header for simple and reliable comparisons."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.casefold()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def clean_value(value: Any) -> Any:
    """Convert an Excel value to a clean JSON-compatible value."""
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, str):
        cleaned = " ".join(value.strip().split())
        return cleaned or None
    return value


def preserve_excel_value(value: Any) -> Any:
    """Convert a value for JSON without altering its textual representation."""
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def glossary_name_from_filename(filename: str) -> str:
    """Create a readable glossary name from an Excel filename."""
    name = Path(filename).stem
    name = re.sub(r"(?i)^glossaire_kpis?_", "", name)
    name = re.sub(r"(?i)^glossaire_", "", name)
    name = re.sub(r"(?i)_kpis?$", "", name)
    return name.replace("_", " ").strip()


def find_column(headers: list[str], aliases: set[str]) -> int | None:
    """Return the position of the first header matching an alias."""
    for index, header in enumerate(headers):
        if header in aliases:
            return index
    return None


def find_header(
    worksheet: Any,
    maximum_rows: int = 30,
) -> tuple[int, dict[str, int | None]] | None:
    """Find the KPI table header and the useful column positions."""
    for row_number, row in enumerate(
        worksheet.iter_rows(min_row=1, max_row=maximum_rows, values_only=True),
        start=1,
    ):
        headers = [normalize_header(value) for value in row]
        kpi_column = find_column(headers, KPI_ALIASES)
        if kpi_column is None:
            continue

        columns = {
            "kpi_name": kpi_column,
            "definition": find_column(headers, DEFINITION_ALIASES),
            "formula": find_column(headers, FORMULA_ALIASES),
            "source_system": find_column(headers, SOURCE_ALIASES),
        }
        return row_number, columns
    return None


def find_kpi_sheet_header(
    worksheet: Any,
    maximum_rows: int = 30,
    sample_rows: int = 500,
) -> tuple[int, dict[str, int | None]] | None:
    """Find a KPI header, including source sheets with an unnamed KPI column.

    Some source glossaries (notably RETAF) have a genuine ``KPIs`` sheet but
    leave the KPI-name header cell empty.  In that case, identify a table row
    from its definition/formula columns and select the populated unnamed
    column as the KPI-name column.
    """
    header = find_header(worksheet, maximum_rows)
    if header is not None:
        return header
    if normalize_header(worksheet.title) not in KPI_SHEET_NAMES:
        return None

    rows = list(
        worksheet.iter_rows(
            min_row=1,
            max_row=maximum_rows,
            values_only=True,
        )
    )
    for row_number, row in enumerate(rows, start=1):
        headers = [normalize_header(value) for value in row]
        definition = find_column(headers, DEFINITION_ALIASES)
        formula = find_column(headers, FORMULA_ALIASES)
        source = find_column(headers, SOURCE_ALIASES)
        if definition is None or formula is None:
            continue

        unnamed_columns = [index for index, value in enumerate(row) if value is None]
        if not unnamed_columns:
            continue
        counts = {index: 0 for index in unnamed_columns}
        for data_row in worksheet.iter_rows(
            min_row=row_number + 1,
            max_row=row_number + sample_rows,
            values_only=True,
        ):
            for index in unnamed_columns:
                if index < len(data_row) and clean_value(data_row[index]) is not None:
                    counts[index] += 1
        kpi_column = max(counts, key=counts.get)
        if counts[kpi_column] == 0:
            continue
        return row_number, {
            "kpi_name": kpi_column,
            "definition": definition,
            "formula": formula,
            "source_system": source,
        }
    return None


def value_at(
    row: tuple[Any, ...],
    column: int | None,
    *,
    preserve_text: bool = False,
) -> Any:
    """Safely read one value from an Excel row."""
    if column is None or column >= len(row):
        return None
    if preserve_text:
        return preserve_excel_value(row[column])
    return clean_value(row[column])


def original_columns(
    worksheet: Any,
    header_row: int,
    kpi_column: int | None = None,
) -> list[tuple[int, str]]:
    """Return every named Excel column, keeping duplicate names unique."""
    header_values = next(
        worksheet.iter_rows(
            min_row=header_row,
            max_row=header_row,
            values_only=True,
        )
    )
    columns = []
    occurrences: dict[str, int] = {}
    unnamed_with_data: set[int] = set()
    unnamed_indexes = {
        index for index, value in enumerate(header_values) if value is None
    }
    if unnamed_indexes:
        for row in worksheet.iter_rows(min_row=header_row + 1, values_only=True):
            for index in unnamed_indexes - unnamed_with_data:
                if index < len(row) and preserve_excel_value(row[index]) is not None:
                    unnamed_with_data.add(index)
            if unnamed_with_data == unnamed_indexes:
                break
    for index, value in enumerate(header_values):
        name = str(value).strip() if value is not None else ""
        if not name and index == kpi_column:
            name = "Nom KPI"
        elif not name and index in unnamed_with_data:
            name = f"Unnamed column {get_column_letter(index + 1)}"
        if not name:
            continue
        occurrences[name] = occurrences.get(name, 0) + 1
        if occurrences[name] > 1:
            name = f"{name} ({occurrences[name]})"
        columns.append((index, name))
    return columns


def extract_sheet_records(
    worksheet: Any,
    file_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract all KPI rows from one worksheet."""
    if normalize_header(worksheet.title) in IGNORED_SHEETS:
        return []

    header = find_kpi_sheet_header(worksheet)
    if header is None:
        return []

    header_row, columns = header
    source_columns = original_columns(
        worksheet,
        header_row,
        columns["kpi_name"],
    )
    records = []

    # Read every row below the detected header.
    for row_number, row in enumerate(
        worksheet.iter_rows(min_row=header_row + 1, values_only=True),
        start=header_row + 1,
    ):
        kpi_name = value_at(row, columns["kpi_name"])
        if kpi_name is None:
            continue

        record = {
            "kpi_name": str(kpi_name),
            "definition": value_at(row, columns["definition"]),
            # KPI formulas are source data: keep their exact Excel text.
            "formula": value_at(
                row,
                columns["formula"],
                preserve_text=True,
            ),
            "source_system": value_at(row, columns["source_system"]),
            "glossary_name": glossary_name_from_filename(
                str(file_metadata.get("original_name") or file_metadata["local_filename"])
            ),
            "sheet_name": worksheet.title,
            "modified_time": file_metadata.get("modified_time"),
            "file_id": file_metadata.get("file_id"),
            "source_file": file_metadata.get("local_filename"),
            "row_number": row_number,
            ORIGINAL_COLUMNS_KEY: {
                name: preserve_excel_value(row[index]) if index < len(row) else None
                for index, name in source_columns
            },
        }
        records.append(record)

    return records


def selected_worksheets(workbook: Any, glossary_name: str) -> list[Any]:
    """Order a glossary's sheets without excluding any potential KPI data.

    A sheet named KPI/KPIs is processed first. When none exists, the
    per-glossary fallback declared in :mod:`hardcoded` is processed first.
    Every remaining sheet is still examined afterwards so an unconfigured
    source can never silently disappear from the extracted KPI data.
    """
    kpi_sheets = [
        worksheet
        for worksheet in workbook.worksheets
        if normalize_header(worksheet.title) in KPI_SHEET_NAMES
    ]
    configured_names = next(
        (
            sheet_names
            for configured_glossary, sheet_names in KPI_SHEETS.items()
            if normalize_header(configured_glossary) == normalize_header(glossary_name)
        ),
        (),
    )
    configured_titles = {normalize_header(name) for name in configured_names}
    configured_sheets = [
        worksheet
        for worksheet in workbook.worksheets
        if normalize_header(worksheet.title) in configured_titles
    ]
    preferred = kpi_sheets or configured_sheets
    preferred_ids = {id(worksheet) for worksheet in preferred}
    return [
        *preferred,
        *(worksheet for worksheet in workbook.worksheets if id(worksheet) not in preferred_ids),
    ]


def load_metadata(metadata_path: str = METADATA_PATH) -> list[dict[str, Any]]:
    """Load and validate the file metadata catalog."""
    path = Path(metadata_path)
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise KpiExtractionError(f"Metadata file not found: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise KpiExtractionError(f"Could not read metadata: {path}") from error

    if not isinstance(content, list):
        raise KpiExtractionError("File metadata must be a JSON list.")
    return [item for item in content if isinstance(item, dict)]


def extract_all_kpis(
    metadata: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract KPI records from every Excel file listed in the metadata."""
    records = []
    processed_files = 0
    missing_files = []
    failed_files = []
    sheets_with_kpis = 0

    for item in metadata:
        local_path = Path(str(item.get("local_path") or ""))
        if local_path.suffix.lower() != ".xlsx":
            continue
        if not local_path.is_file():
            missing_files.append(local_path.as_posix())
            continue

        try:
            workbook = load_workbook(
                local_path,
                read_only=True,
                data_only=False,
            )
            processed_files += 1
            glossary_name = glossary_name_from_filename(
                str(item.get("original_name") or item["local_filename"])
            )
            for worksheet in selected_worksheets(workbook, glossary_name):
                sheet_records = extract_sheet_records(worksheet, item)
                if sheet_records:
                    sheets_with_kpis += 1
                    records.extend(sheet_records)
            workbook.close()
        except Exception as error:
            failed_files.append(
                {
                    "local_path": local_path.as_posix(),
                    "error": type(error).__name__,
                }
            )
            LOGGER.warning("Could not extract %s", local_path.as_posix())

    report = {
        "metadata_entries": len(metadata),
        "processed_excel_files": processed_files,
        "sheets_with_kpis": sheets_with_kpis,
        "kpi_records": len(records),
        "missing_files": missing_files,
        "failed_files": failed_files,
    }
    return records, report


def save_json(data: Any, output_path: str) -> None:
    """Save data as formatted UTF-8 JSON using a temporary file."""
    path = Path(output_path)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(path)
    except (OSError, TypeError, ValueError) as error:
        temporary_path.unlink(missing_ok=True)
        raise KpiExtractionError(f"Could not save JSON: {path}") from error


def group_kpis_by_glossary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group original Excel rows without repeating file-level metadata."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        glossary_name = str(record.get("glossary_name") or "MISSING")
        source_values = record.get(ORIGINAL_COLUMNS_KEY)
        if isinstance(source_values, dict):
            glossary_record = dict(source_values)
        else:
            glossary_record = {
                key: value
                for key, value in record.items()
                if key not in METADATA_FIELDS
                and key not in {"glossary_name", ORIGINAL_COLUMNS_KEY}
            }

        glossary_record["sheet_name"] = record.get("sheet_name")
        glossary_record["row_number"] = record.get("row_number")
        grouped.setdefault(glossary_name, []).append(glossary_record)

    return [
        {
            "glossary_name": glossary_name,
            "kpi_count": len(kpis),
            "kpis": kpis,
        }
        for glossary_name, kpis in sorted(
            grouped.items(),
            key=lambda item: item[0].casefold(),
        )
    ]


def filter_glossaries_for_retrieval(
    glossaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep direct KPI sheets, or configured fallback sheets per glossary."""
    filtered = []
    for glossary in glossaries:
        records = [record for record in glossary.get("kpis", []) if isinstance(record, dict)]
        direct = [
            record for record in records
            if normalize_header(record.get("sheet_name")) in KPI_SHEET_NAMES
        ]
        if direct:
            kpis = direct
        else:
            configured_names = next(
                (
                    sheet_names for configured_glossary, sheet_names in KPI_SHEETS.items()
                    if normalize_header(configured_glossary)
                    == normalize_header(glossary.get("glossary_name"))
                ),
                (),
            )
            configured_titles = {normalize_header(name) for name in configured_names}
            kpis = [
                record for record in records
                if normalize_header(record.get("sheet_name")) in configured_titles
            ]
        if kpis:
            filtered.append({**glossary, "kpi_count": len(kpis), "kpis": kpis})
    return filtered


def build_kpi_json(
    metadata_path: str = METADATA_PATH,
    output_path: str = KPI_JSON_PATH,
    glossary_output_path: str = GLOSSARY_KPI_JSON_PATH,
    retrieval_output_path: str = RETRIEVAL_GLOSSARY_JSON_PATH,
) -> dict[str, Any]:
    """Run extraction and save complete and retrieval-scoped KPI JSON files."""
    metadata = load_metadata(metadata_path)
    records, report = extract_all_kpis(metadata)
    flat_records = [
        {key: value for key, value in record.items() if key != ORIGINAL_COLUMNS_KEY}
        for record in records
    ]
    save_json(flat_records, output_path)
    glossaries = group_kpis_by_glossary(records)
    save_json(glossaries, glossary_output_path)
    retrieval_glossaries = filter_glossaries_for_retrieval(glossaries)
    save_json(retrieval_glossaries, retrieval_output_path)
    report["glossaries"] = len(glossaries)
    report["retrieval_glossaries"] = len(retrieval_glossaries)
    report["retrieval_records"] = sum(
        int(glossary["kpi_count"]) for glossary in retrieval_glossaries
    )
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    extraction_report = build_kpi_json()
    print(json.dumps(extraction_report, ensure_ascii=False, indent=2))
