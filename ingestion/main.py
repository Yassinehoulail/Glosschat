"""Command-line entry point for Google Drive ingestion."""

from __future__ import annotations

import logging
import sys

from ingestion.config import (
    GLOSSARY_KPI_JSON_PATH,
    KPI_JSON_PATH,
    METADATA_PATH,
    RETRIEVAL_GLOSSARY_JSON_PATH,
)
from ingestion.downloader import DriveDownloader, MetadataWriteError
from ingestion.drive_client import DriveClient, DriveClientError
from ingestion.excel_extractor import KpiExtractionError, build_kpi_json

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure concise console logging without sensitive values."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def print_summary(
    discovered: int,
    downloaded: int,
    ignored: int,
    failed: int,
    metadata_saved: bool,
) -> None:
    """Print the final ingestion totals."""
    print(f"Files discovered: {discovered}")
    print(f"Files downloaded: {downloaded}")
    print(f"Files ignored: {ignored}")
    print(f"Files failed: {failed}")
    if metadata_saved:
        print(f"Metadata saved: {METADATA_PATH}")
    else:
        print("Metadata saved: no")


def main() -> int:
    """Run authentication, discovery, download, and metadata creation."""
    configure_logging()

    try:
        client = DriveClient()
        client.authenticate()
        discovered_files = client.list_files()
    except DriveClientError as exc:
        LOGGER.error("%s", exc)
        print_summary(0, 0, 0, 0, False)
        return 1

    if client.service is None:
        LOGGER.error("Google Drive service was not initialized.")
        print_summary(len(discovered_files), 0, 0, 0, False)
        return 1

    try:
        downloader = DriveDownloader(client.service)
    except OSError as exc:
        LOGGER.error(
            "Cannot create the download directory: %s",
            type(exc).__name__,
        )
        print_summary(len(discovered_files), 0, 0, 0, False)
        return 1

    metadata, ignored_count, failed_count = downloader.ingest(discovered_files)
    metadata_saved = False
    try:
        downloader.save_metadata(metadata)
        metadata_saved = True
    except MetadataWriteError as exc:
        LOGGER.error("%s", exc)

    LOGGER.info(
        "Ingestion complete: discovered=%d, downloaded=%d, ignored=%d, failed=%d",
        len(discovered_files),
        len(metadata),
        ignored_count,
        failed_count,
    )
    print_summary(
        discovered=len(discovered_files),
        downloaded=len(metadata),
        ignored=ignored_count,
        failed=failed_count,
        metadata_saved=metadata_saved,
    )

    if not metadata_saved:
        return 1

    # Create the consolidated KPI JSON after saving file metadata.
    try:
        report = build_kpi_json()
        print(f"KPI records created: {report['kpi_records']}")
        print(f"KPI JSON saved: {KPI_JSON_PATH}")
        print(f"Glossaries created: {report['glossaries']}")
        print(f"Glossary KPI JSON saved: {GLOSSARY_KPI_JSON_PATH}")
        print(f"Retrieval glossary JSON saved: {RETRIEVAL_GLOSSARY_JSON_PATH}")
    except KpiExtractionError as exc:
        LOGGER.error("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
