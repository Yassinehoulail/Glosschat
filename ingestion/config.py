"""Configuration for the Google Drive ingestion pipeline."""

CREDENTIALS_PATH = "cred/data_gov_app_credentials.json"
DOWNLOAD_DIRECTORY = "data/raw"
METADATA_PATH = "data/file_metadata.json"
KPI_JSON_PATH = "data/kpi_records.json"
GLOSSARY_KPI_JSON_PATH = "data/glossaries_with_kpis.json"
RETRIEVAL_GLOSSARY_JSON_PATH = "data/glossaries_for_retrieval.json"

DRIVE_SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly"
]

GOOGLE_SHEETS_MIME_TYPE = "application/vnd.google-apps.spreadsheet"
EXCEL_MIME_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
CSV_MIME_TYPE = "text/csv"
PDF_MIME_TYPE = "application/pdf"
DOCX_MIME_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)

SUPPORTED_MIME_TYPES = {
    GOOGLE_SHEETS_MIME_TYPE: ".xlsx",
    EXCEL_MIME_TYPE: ".xlsx",
    CSV_MIME_TYPE: ".csv",
    PDF_MIME_TYPE: ".pdf",
    DOCX_MIME_TYPE: ".docx",
}

GOOGLE_SHEETS_EXPORT_MIME_TYPE = EXCEL_MIME_TYPE
