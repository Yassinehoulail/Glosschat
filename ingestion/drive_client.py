"""Authentication and file discovery for the Google Drive API."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from google.auth.exceptions import GoogleAuthError
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import Resource, build
from googleapiclient.errors import HttpError

from ingestion.config import CREDENTIALS_PATH, DRIVE_SCOPES

LOGGER = logging.getLogger(__name__)

FILE_FIELDS = "id,name,mimeType,modifiedTime,size,webViewLink"


class DriveClientError(RuntimeError):
    """Raised when Drive authentication or file discovery cannot continue."""


class DriveClient:
    """A minimal read-only Google Drive client."""

    def __init__(
        self,
        credentials_path: str = CREDENTIALS_PATH,
        scopes: list[str] | None = None,
    ) -> None:
        self.credentials_path = Path(credentials_path)
        self.scopes = scopes or DRIVE_SCOPES
        self.service: Resource | None = None

    def authenticate(self) -> None:
        """Load service-account credentials and construct the Drive service."""
        if not self.credentials_path.is_file():
            raise DriveClientError(
                f"Credential file not found: {self.credentials_path}"
            )

        try:
            credentials = Credentials.from_service_account_file(
                str(self.credentials_path),
                scopes=self.scopes,
            )
            self.service = build(
                "drive",
                "v3",
                credentials=credentials,
                cache_discovery=False,
            )
        except (GoogleAuthError, ValueError, OSError) as exc:
            raise DriveClientError(
                "Unable to load the service-account credentials."
            ) from exc
        except Exception as exc:
            raise DriveClientError(
                "Unable to initialize the Google Drive API client."
            ) from exc

        LOGGER.info("Service-account credentials loaded successfully.")

    def list_files(self) -> list[dict[str, Any]]:
        """Return every non-trashed file accessible to the service account."""
        if self.service is None:
            raise DriveClientError(
                "The Drive client is not authenticated. Call authenticate() first."
            )

        files: list[dict[str, Any]] = []
        page_token: str | None = None

        try:
            while True:
                response = (
                    self.service.files()
                    .list(
                        q="trashed = false",
                        pageSize=1000,
                        pageToken=page_token,
                        fields=f"nextPageToken,files({FILE_FIELDS})",
                        includeItemsFromAllDrives=True,
                        supportsAllDrives=True,
                    )
                    .execute()
                )

                files.extend(response.get("files", []))
                page_token = response.get("nextPageToken")
                if not page_token:
                    break
        except HttpError as exc:
            status = getattr(exc.resp, "status", None)
            if status in {401, 403}:
                message = (
                    "Google Drive authentication or permission error while "
                    "listing files."
                )
            else:
                message = "Google Drive API error while listing files."
            raise DriveClientError(message) from exc
        except (GoogleAuthError, OSError) as exc:
            raise DriveClientError(
                "Authentication or network error while listing Drive files."
            ) from exc

        LOGGER.info("Discovered %d non-trashed Drive files.", len(files))
        if not files:
            LOGGER.warning("No accessible non-trashed files were found.")
        return files
