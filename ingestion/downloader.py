"""Download supported files and write ingestion metadata."""

from __future__ import annotations

import io
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from googleapiclient.discovery import Resource
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

from ingestion.config import (
    DOWNLOAD_DIRECTORY,
    GOOGLE_SHEETS_EXPORT_MIME_TYPE,
    GOOGLE_SHEETS_MIME_TYPE,
    METADATA_PATH,
    SUPPORTED_MIME_TYPES,
)

LOGGER = logging.getLogger(__name__)

INVALID_FILENAME_CHARACTERS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


class MetadataWriteError(RuntimeError):
    """Raised when the metadata file cannot be written."""


def safe_filename(filename: str) -> str:
    """Return a filename safe for Windows and POSIX."""
    cleaned = INVALID_FILENAME_CHARACTERS.sub("_", filename).strip(" .")
    if not cleaned:
        return "unnamed_file"
    if Path(cleaned).stem.upper() in WINDOWS_RESERVED_NAMES:
        return f"_{cleaned}"
    return cleaned


def sha256(file_path: Path) -> str:
    """Return the SHA-256 checksum of a file."""
    checksum = hashlib.sha256()
    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


class DriveDownloader:
    """Download supported Drive files without aborting on per-file failures."""

    def __init__(
        self,
        service: Resource,
        download_directory: str = DOWNLOAD_DIRECTORY,
        metadata_path: str = METADATA_PATH,
    ) -> None:
        self.service = service
        self.download_directory = Path(download_directory)
        self.metadata_path = Path(metadata_path)
        self.download_directory.mkdir(parents=True, exist_ok=True)
        self.previous_metadata: dict[str, dict[str, Any]] = {}
        try:
            saved_metadata = json.loads(
                self.metadata_path.read_text(encoding="utf-8")
            )
            if isinstance(saved_metadata, list):
                self.previous_metadata = {
                    str(item["file_id"]): item
                    for item in saved_metadata
                    if isinstance(item, dict)
                    and item.get("file_id")
                    and item.get("local_filename")
                }
        except FileNotFoundError:
            pass
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("Existing metadata could not be read.")

    def _local_filename(self, drive_file: dict[str, Any]) -> str:
        """Return one deterministic filename for a Drive file ID."""
        file_id = str(drive_file["id"])
        original_name = str(drive_file.get("name") or "unnamed_file")
        mime_type = str(drive_file.get("mimeType") or "")
        required_suffix = SUPPORTED_MIME_TYPES[mime_type]

        if mime_type == GOOGLE_SHEETS_MIME_TYPE:
            base_name = original_name
            if base_name.lower().endswith(required_suffix):
                base_name = base_name[: -len(required_suffix)]
            filename = f"{base_name}{required_suffix}"
        else:
            filename = original_name

        filename = safe_filename(filename)
        path = Path(filename)
        safe_file_id = safe_filename(file_id)
        return f"{path.stem}_{safe_file_id}{path.suffix}"

    def _download_one(
        self,
        drive_file: dict[str, Any],
        destination: Path,
    ) -> None:
        file_id = str(drive_file["id"])
        mime_type = str(drive_file["mimeType"])

        if mime_type == GOOGLE_SHEETS_MIME_TYPE:
            request = self.service.files().export_media(
                fileId=file_id,
                mimeType=GOOGLE_SHEETS_EXPORT_MIME_TYPE,
            )
        else:
            request = self.service.files().get_media(
                fileId=file_id,
                supportsAllDrives=True,
            )

        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        # Write to a temporary file, then replace the destination atomically.
        temporary_destination = destination.with_suffix(
            f"{destination.suffix}.part"
        )
        temporary_destination.write_bytes(buffer.getvalue())
        temporary_destination.replace(destination)

    def ingest(
        self,
        discovered_files: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int, int]:
        """Download supported files and return metadata, ignored, and failed counts."""
        metadata: list[dict[str, Any]] = []
        ignored_count = 0
        failed_count = 0

        for drive_file in discovered_files:
            name = str(drive_file.get("name") or "unnamed_file")
            mime_type = str(drive_file.get("mimeType") or "")

            if mime_type not in SUPPORTED_MIME_TYPES:
                ignored_count += 1
                LOGGER.info(
                    "Ignoring unsupported file: %s (MIME type: %s)",
                    name,
                    mime_type or "unknown",
                )
                continue

            local_filename = self._local_filename(drive_file)
            destination = self.download_directory / local_filename
            operation = (
                "Exporting"
                if mime_type == GOOGLE_SHEETS_MIME_TYPE
                else "Downloading"
            )
            LOGGER.info("%s %s as %s", operation, name, local_filename)

            try:
                self._download_one(drive_file, destination)
            except Exception as exc:
                failed_count += 1
                if isinstance(exc, HttpError):
                    status = getattr(exc.resp, "status", None)
                    reason = (
                        "permission denied"
                        if status == 403
                        else f"Google Drive API error (HTTP {status or 'unknown'})"
                    )
                else:
                    reason = type(exc).__name__
                LOGGER.error("Failed to download %s: %s", name, reason)
                destination.with_suffix(
                    f"{destination.suffix}.part"
                ).unlink(missing_ok=True)
                # Keep the last valid local copy in the active catalogue when
                # refreshing this Drive file fails.
                previous = self.previous_metadata.get(str(drive_file["id"]))
                if previous:
                    previous_path = Path(str(previous.get("local_path") or ""))
                    if previous_path.is_file():
                        metadata.append(previous)
                        LOGGER.warning(
                            "Keeping previous local copy for %s: %s",
                            name,
                            previous_path.as_posix(),
                        )
                elif destination.is_file():
                    metadata.append(
                        {
                            "file_id": drive_file["id"],
                            "original_name": name,
                            "local_filename": local_filename,
                            "mime_type": mime_type,
                            "modified_time": drive_file.get("modifiedTime"),
                            "web_view_link": drive_file.get("webViewLink"),
                            "local_path": destination.as_posix(),
                            "file_extension": destination.suffix.lower(),
                            "file_size_bytes": destination.stat().st_size,
                            "drive_size_bytes": (
                                int(drive_file["size"])
                                if str(drive_file.get("size", "")).isdigit()
                                else None
                            ),
                            "sha256": sha256(destination),
                            "downloaded_at": datetime.fromtimestamp(
                                destination.stat().st_mtime,
                                timezone.utc,
                            ).isoformat(),
                        }
                    )
                    LOGGER.warning(
                        "Keeping existing stable local copy for %s: %s",
                        name,
                        destination.as_posix(),
                    )
                continue

            local_path = destination.as_posix()
            file_size = destination.stat().st_size
            metadata.append(
                {
                    "file_id": drive_file["id"],
                    "original_name": name,
                    "local_filename": local_filename,
                    "mime_type": mime_type,
                    "modified_time": drive_file.get("modifiedTime"),
                    "web_view_link": drive_file.get("webViewLink"),
                    "local_path": local_path,
                    "file_extension": destination.suffix.lower(),
                    "file_size_bytes": file_size,
                    "drive_size_bytes": (
                        int(drive_file["size"])
                        if str(drive_file.get("size", "")).isdigit()
                        else None
                    ),
                    "sha256": sha256(destination),
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            LOGGER.info("Successfully saved %s", local_path)

        return metadata, ignored_count, failed_count

    def save_metadata(self, metadata: list[dict[str, Any]]) -> None:
        """Write metadata as formatted UTF-8 JSON."""
        try:
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self.metadata_path.with_suffix(
                f"{self.metadata_path.suffix}.tmp"
            )
            temporary_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            temporary_path.replace(self.metadata_path)
        except (OSError, TypeError, ValueError) as exc:
            raise MetadataWriteError(
                f"Unable to save metadata to {self.metadata_path}"
            ) from exc

        LOGGER.info("Metadata saved to %s", self.metadata_path.as_posix())
