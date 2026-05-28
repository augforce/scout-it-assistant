"""Google Drive client: OAuth + file listing/fetching for indexable folders."""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
ROOT = Path(__file__).resolve().parent.parent
CREDENTIALS_FILE = ROOT / "credentials.json"
TOKEN_FILE = ROOT / "token.json"

# Mime types we know how to extract text from
GOOGLE_DOC = "application/vnd.google-apps.document"
GOOGLE_SHEET = "application/vnd.google-apps.spreadsheet"
GOOGLE_FOLDER = "application/vnd.google-apps.folder"
PDF = "application/pdf"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

INDEXABLE_MIMES = {GOOGLE_DOC, GOOGLE_SHEET, PDF, DOCX, XLSX}


def get_service():
    """Authenticate and return a Drive API service. Opens a browser on first run."""
    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_FILE.write_text(creds.to_json())
        # Token holds a long-lived refresh token; keep it readable only by owner.
        TOKEN_FILE.chmod(0o600)

    return build("drive", "v3", credentials=creds, cache_discovery=False)


def list_files_in_folder(service, folder_id: str, recursive: bool = True) -> list[dict]:
    """List all files (recursively by default) inside a Drive folder.

    Returns dicts with: id, name, mimeType, modifiedTime, webViewLink.
    Skips folders themselves and files with mime types we can't index.
    """
    results: list[dict] = []
    page_token = None
    fields = "nextPageToken, files(id, name, mimeType, modifiedTime, webViewLink, parents)"
    query = f"'{folder_id}' in parents and trashed = false"

    while True:
        resp = (
            service.files()
            .list(
                q=query,
                fields=fields,
                pageToken=page_token,
                pageSize=200,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        for f in resp.get("files", []):
            if f["mimeType"] == GOOGLE_FOLDER:
                if recursive:
                    results.extend(list_files_in_folder(service, f["id"], recursive=True))
            elif f["mimeType"] in INDEXABLE_MIMES:
                results.append(f)
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return results


# Hard cap on per-file size at parse time. A malicious 200 MB PDF in an indexed
# folder could OOM-kill the indexer process while pypdf parses; this bounds memory
# pressure before that happens. The 50 MB default is generous for normal docs and
# tight enough to refuse obvious zip-bomb / blob payloads.
MAX_FILE_BYTES = 50 * 1024 * 1024


def download_file(
    service,
    file_id: str,
    mime_type: str,
    max_retries: int = 4,
    max_bytes: int = MAX_FILE_BYTES,
) -> bytes:
    """Download a file's bytes. Google-native files are exported to a portable format.
    Retries with exponential backoff on transient 5xx errors from Drive. Raises
    ValueError if the file exceeds max_bytes mid-download."""
    for attempt in range(max_retries):
        try:
            if mime_type == GOOGLE_DOC:
                request = service.files().export_media(fileId=file_id, mimeType="text/plain")
            elif mime_type == GOOGLE_SHEET:
                # Export as XLSX (not CSV) so multi-tab sheets keep every tab and
                # the downstream extractor can read row-by-row with header awareness.
                request = service.files().export_media(fileId=file_id, mimeType=XLSX)
            else:
                request = service.files().get_media(fileId=file_id, supportsAllDrives=True)

            buf = io.BytesIO()
            downloader = MediaIoBaseDownload(buf, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
                if buf.tell() > max_bytes:
                    raise ValueError(
                        f"File {file_id} exceeds size cap ({max_bytes} bytes); skipped"
                    )
            return buf.getvalue()
        except HttpError as e:
            if e.resp.status in (500, 502, 503, 504) and attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue
            raise


def get_file_metadata(service, file_id: str) -> dict:
    """Fetch metadata for a single file by ID."""
    return (
        service.files()
        .get(
            fileId=file_id,
            fields="id, name, mimeType, modifiedTime, webViewLink",
            supportsAllDrives=True,
        )
        .execute()
    )


def web_view_url(file_id: str, mime_type: str) -> str:
    """Build a user-friendly URL that opens the file in Google's web UI.

    The URL ends up in an `href` in the source sidebar. We hardcode the
    `https://` prefix here so a future refactor that ever shifts to a Drive
    API-supplied URL (which could in principle be attacker-controlled metadata)
    can't sneak a `javascript:` scheme into the template — the assertion
    locks the invariant down."""
    if mime_type == GOOGLE_DOC:
        url = f"https://docs.google.com/document/d/{file_id}"
    elif mime_type == GOOGLE_SHEET:
        url = f"https://docs.google.com/spreadsheets/d/{file_id}"
    else:
        url = f"https://drive.google.com/file/d/{file_id}/view"
    assert url.startswith("https://"), f"unexpected URL scheme: {url}"
    return url


def _folder_ids_from_env() -> list[str]:
    raw = os.environ.get("DRIVE_FOLDER_IDS", "")
    return [x.strip() for x in raw.split(",") if x.strip()]


if __name__ == "__main__":
    # Smoke test: auth, then list files in each configured folder.
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")

    folder_ids = _folder_ids_from_env()
    if not folder_ids:
        raise SystemExit("DRIVE_FOLDER_IDS not set in .env")

    svc = get_service()
    print("Auth OK. Listing files...\n")
    total = 0
    for fid in folder_ids:
        files = list_files_in_folder(svc, fid)
        print(f"Folder {fid}: {len(files)} indexable files")
        for f in files[:10]:
            print(f"  - {f['name']}  ({f['mimeType']})")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more")
        total += len(files)
    print(f"\nTotal indexable: {total}")
