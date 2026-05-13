"""Build / refresh the local Chroma index from configured Drive folders."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

from src.drive import (
    _folder_ids_from_env,
    download_file,
    get_file_metadata,
    get_service,
    list_files_in_folder,
    web_view_url,
)
from src.extractors import extract_text

ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "drive_files"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBED_MODEL = "all-MiniLM-L6-v2"


@dataclass
class SyncResult:
    new: int = 0
    changed: int = 0
    unchanged: int = 0
    deleted: int = 0
    chunks_written: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph then sentence boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            # Prefer a clean break: paragraph > sentence > word
            for sep in ("\n\n", "\n", ". ", " "):
                idx = text.rfind(sep, start + size // 2, end)
                if idx != -1:
                    end = idx + len(sep)
                    break
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def get_collection(reset: bool = False):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def _stored_file_state(collection) -> dict[str, tuple[str, list[str]]]:
    """Map every file currently in the collection to (modified_time, [chunk_ids])."""
    res = collection.get(include=["metadatas"])
    out: dict[str, tuple[str, list[str]]] = {}
    for chunk_id, meta in zip(res["ids"], res["metadatas"]):
        fid = meta["file_id"]
        mtime = meta.get("modified_time", "")
        if fid not in out:
            out[fid] = (mtime, [])
        out[fid][1].append(chunk_id)
    return out


def _write_file_chunks(collection, f: dict) -> int:
    """Download + extract + chunk + upsert. Returns chunk count (0 if empty)."""
    service_local = f.pop("_service")  # smuggled in by caller to avoid re-auth
    data = download_file(service_local, f["id"], f["mimeType"])
    text = extract_text(data, f["mimeType"])
    chunks = chunk_text(text)
    if not chunks:
        return 0
    ids = [f"{f['id']}::{j}" for j in range(len(chunks))]
    metadatas = [
        {
            "file_id": f["id"],
            "file_name": f["name"],
            "mime_type": f["mimeType"],
            "web_view_url": web_view_url(f["id"], f["mimeType"]),
            "modified_time": f.get("modifiedTime", ""),
            "chunk_index": j,
        }
        for j in range(len(chunks))
    ]
    collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
    return len(chunks)


def _process_targeted(
    service, collection, files: list[dict], verbose: bool = True
) -> SyncResult:
    """Index a fixed list of files (--file-ids path). Clears prior chunks for each
    target file before writing so a re-index with fewer chunks doesn't leave stale
    rows behind. Still no cross-corpus deletions (those need a global view)."""
    result = SyncResult()
    if verbose:
        print(f"Indexing {len(files)} files into '{COLLECTION_NAME}'...\n")
    for i, f in enumerate(files, 1):
        try:
            collection.delete(where={"file_id": f["id"]})
            f["_service"] = service
            n = _write_file_chunks(collection, f)
            result.chunks_written += n
            result.new += 1
            if verbose:
                label = f"{n:>3} chunks" if n else "  empty  "
                print(f"  [{i}/{len(files)}] {label}  {f['name']}")
        except Exception as e:
            result.skipped.append((f["name"], str(e)))
            if verbose:
                print(f"  [{i}/{len(files)}] SKIP {f['name']}: {e}")
    return result


def _process_incremental(
    service, collection, files: list[dict], allow_deletions: bool, verbose: bool = True
) -> SyncResult:
    """Compare Drive listing against stored state, only process new/changed files."""
    stored = _stored_file_state(collection)
    current_ids = {f["id"] for f in files}

    new_files: list[dict] = []
    changed_files: list[dict] = []
    unchanged = 0
    for f in files:
        if f["id"] not in stored:
            new_files.append(f)
        elif stored[f["id"]][0] != f.get("modifiedTime", ""):
            changed_files.append(f)
        else:
            unchanged += 1

    deletable: list[tuple[str, list[str]]] = []
    if allow_deletions:
        for fid, (_, chunk_ids) in stored.items():
            if fid not in current_ids:
                deletable.append((fid, chunk_ids))

    result = SyncResult(
        new=len(new_files),
        changed=len(changed_files),
        unchanged=unchanged,
        deleted=len(deletable),
    )

    if verbose:
        print(
            f"Plan: {result.new} new, {result.changed} changed, "
            f"{result.unchanged} unchanged, {result.deleted} to delete.\n"
        )

    if not new_files and not changed_files and not deletable:
        if verbose:
            print("Index is already in sync. Nothing to do.")
        return result

    for fid, chunk_ids in deletable:
        collection.delete(ids=chunk_ids)
        if verbose:
            print(f"  -    deleted {len(chunk_ids)} chunks (file_id={fid})")

    to_process = [(f, False) for f in new_files] + [(f, True) for f in changed_files]
    for i, (f, is_changed) in enumerate(to_process, 1):
        if is_changed:
            collection.delete(ids=stored[f["id"]][1])
        try:
            f["_service"] = service
            n = _write_file_chunks(collection, f)
            result.chunks_written += n
            if verbose:
                marker = "~" if is_changed else "+"
                label = f"{n:>3} chunks" if n else "  empty  "
                print(f"  [{i}/{len(to_process)}] {marker} {label}  {f['name']}")
        except Exception as e:
            result.skipped.append((f["name"], str(e)))
            if verbose:
                print(f"  [{i}/{len(to_process)}] SKIP {f['name']}: {e}")

    if verbose:
        print(
            f"\nDone. {result.chunks_written} new/updated chunks written; "
            f"{result.unchanged} files unchanged."
        )
    return result


def index_folders(
    folder_ids: list[str] | None = None,
    file_ids: list[str] | None = None,
    limit: int | None = None,
    reset: bool = False,
    scoped: bool = False,
    verbose: bool = True,
) -> SyncResult:
    service = get_service()
    collection = get_collection(reset=reset)

    if file_ids:
        files = [get_file_metadata(service, fid) for fid in file_ids]
        return _process_targeted(service, collection, files, verbose=verbose)

    files: list[dict] = []
    for fid in folder_ids or []:
        files.extend(list_files_in_folder(service, fid))
    if limit:
        files = files[:limit]

    # Deletions only when we have a global view: a full sweep of all configured folders
    # AND not a scoped --folder-ids run AND not a reset (reset wipes everything anyway).
    allow_deletions = not scoped and not reset and limit is None
    return _process_incremental(
        service, collection, files, allow_deletions=allow_deletions, verbose=verbose
    )


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Index Drive folder(s) into Chroma.")
    parser.add_argument("--limit", type=int, help="Index only the first N files (for testing).")
    parser.add_argument("--reset", action="store_true", help="Delete the existing collection first.")
    parser.add_argument(
        "--file-ids",
        help="Comma-separated Drive file IDs to (re)index instead of walking folders.",
    )
    parser.add_argument(
        "--folder-ids",
        help="Comma-separated folder IDs to index, overriding DRIVE_FOLDER_IDS from .env.",
    )
    args = parser.parse_args()

    if args.file_ids:
        file_ids = [x.strip() for x in args.file_ids.split(",") if x.strip()]
        index_folders(file_ids=file_ids, reset=args.reset)
    else:
        if args.folder_ids:
            folder_ids = [x.strip() for x in args.folder_ids.split(",") if x.strip()]
            scoped = True
        else:
            folder_ids = _folder_ids_from_env()
            scoped = False
        if not folder_ids:
            raise SystemExit("DRIVE_FOLDER_IDS not set in .env (or pass --folder-ids)")
        index_folders(
            folder_ids=folder_ids, limit=args.limit, reset=args.reset, scoped=scoped
        )
