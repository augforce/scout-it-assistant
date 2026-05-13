# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Local-only webapp that answers IT-support questions by RAG over a chosen set of Google Drive folders. FastAPI + HTMX server-rendered UI, Chroma file-based vector store, local `sentence-transformers` embeddings, Claude Sonnet 4.6 for generation, Google Drive API v3 for source documents.

## Commands

All commands assume the repo's venv (`.venv/`) is activated.

```bash
# Install deps
pip install -r requirements.txt

# Full sync: walk every folder in DRIVE_FOLDER_IDS, upsert new/changed, delete files
# that are gone from Drive. Safe to run repeatedly.
python -m src.indexer

# Index just a few files (test): --limit caps the listing; --reset wipes the collection first
python -m src.indexer --limit 5 --reset

# Re-index specific files by Drive ID (skips folder walk; no deletions)
python -m src.indexer --file-ids <id1>,<id2>

# Index a scoped subset of folders (overrides DRIVE_FOLDER_IDS); no deletions in scoped mode
python -m src.indexer --folder-ids <folder_id1>,<folder_id2>

# Run the web app (http://localhost:8000)
uvicorn src.main:app --reload

# Smoke-test Drive auth + folder listing
python -m src.drive

# One-shot CLI query (bypasses the UI)
python -m src.rag "is Slack approved?"
```

There is no test suite, linter, or build step configured.

## Required setup before anything runs

- `.env` with `ANTHROPIC_API_KEY=...` and `DRIVE_FOLDER_IDS=id1,id2` (folder IDs from Drive URLs).
- `credentials.json` at repo root: OAuth client of type **Desktop app**, Google Drive API enabled, your account added as a test user on the OAuth consent screen.
- First indexer/drive run opens a browser for OAuth and writes `token.json`. All three files are gitignored.

## Architecture

The pipeline has two phases: **build the index** (offline) and **answer questions** (online). They share the same Chroma collection on disk at `chroma_db/` (`COLLECTION_NAME = "drive_files"`).

### Indexing pipeline (`src/indexer.py`)

`index_folders()` is the single entry point — same function is used by the CLI and by the `/sync` HTTP endpoint, so a sync from the UI updates the same Chroma client the server is already querying (no restart needed).

Two execution paths:
- **`_process_incremental`** (default folder walk): diffs Drive listing against `modified_time` stored in chunk metadata; re-indexes only new/changed files; deletes chunks for files that disappeared from Drive — but only when the run has a *global view* (`allow_deletions = not scoped and not reset and limit is None`). Scoped/limited runs never delete, because the listing is partial.
- **`_process_targeted`** (`--file-ids`): re-indexes the listed files. Deletes existing chunks for each target file *before* upserting so a re-index that produces fewer chunks than the previous version doesn't leave stale rows behind. Still no cross-corpus deletions (those need a global view of all folders).

Chunk IDs are deterministic: `f"{file_id}::{chunk_index}"`. This is what makes re-indexing a changed file idempotent — delete-then-upsert by the old chunk IDs cleanly replaces the file's footprint.

Per-chunk metadata carries `file_id`, `file_name`, `mime_type`, `web_view_url`, `modified_time`, `chunk_index`. `modified_time` drives the incremental diff; `web_view_url` is what becomes the clickable citation in the UI, so it has to be on every chunk (not just on a sources table).

Text extraction (`src/extractors.py`) is a mime-type dispatch: Google Docs → text export, Google Sheets → **XLSX export** (then through `_xlsx_to_text`, same path as native `.xlsx`), PDFs via `pypdf`, `.docx` via `python-docx`. The XLSX-not-CSV export for Google Sheets is deliberate: CSV export only returns the first tab, while XLSX preserves every tab. Any new mime type needs both an entry in `INDEXABLE_MIMES` (in `src/drive.py`) and a handler in `extract_text`.

`_xlsx_to_text` formats every data row as `Col1: val | Col2: val | ...` using a detected header row, so each row stays self-describing even after the 800-char chunker splits the sheet across chunk boundaries. The header is picked by "most-populated row in the first 5 rows," then sanity-checked: every non-empty cell must be ≤50 chars and digit-free. If the check fails (e.g. a freeform sheet that has no real header — common for summary/pivot tabs), the sheet falls back to plain tab-separated rows so the data is still indexed without bogus column tags.

### Retrieval + generation (`src/rag.py`)

`ask(question)` does **hybrid retrieval** then a single Claude call. Four retrieval modes stack into one chunk list in this order, dedup'd by `chunk_id`. The order matters: source numbering follows first appearance, so earlier modes get lower source numbers (which is what the UI sidebar reads top-down).

1. **List-question detour.** If the question matches any pattern in `LIST_QUESTION_PATTERNS` (enumeration / "how many" / "approved software" / "is X approved" / "can we use X" phrasing), the retriever first dumps **every chunk** from files whose name matches `MASTER_LIST_NAME_PATTERNS` (e.g. "Approved Applications", "Laptop Inventory"). The point is to give Claude the full enumeration so it can categorize or count against the entire list using its world knowledge — semantic top-k alone would only surface a subset. List mode also raises the per-question chunk cap from `MAX_CHUNKS_TO_LLM` (25) to `MAX_CHUNKS_FOR_LIST_QUESTION` (200). The cap is generous because a single large reference doc (e.g. a 73-chunk inventory sheet) shouldn't crowd out the other master-list files.
2. **Filename match.** `_filename_match_chunks` scores each file by how many of `_extract_entities(question)` tokens appear (case-insensitive substring) in its name; files scoring ≥ `MIN_FILENAME_MATCHES` (2) are surfaced, top `MAX_FILENAME_MATCH_FILES` (3) by score, up to `MAX_CHUNKS_PER_FILENAME_MATCH` (5) chunks each. This handles "pull up the X and Y draft" / "where is the email template" style lookups where the user is naming the doc and the body text doesn't echo the question. The 2-token threshold is what keeps "Is Slack approved?" from sweeping in every Slack-named file.
3. **Semantic search.** Top-`SEMANTIC_K` (12) Chroma nearest neighbours by cosine.
4. **Keyword lookup.** Up to 4 entity tokens from `_extract_entities`; each looked up via Chroma's `where_document={"$contains": ...}` in multiple casings (Chroma's `$contains` is case-sensitive).

Sources are numbered by **first appearance of each unique `file_id`** — multiple chunks from the same file share one source number. The system prompt tells Claude that repeated `[N]` means same file. The UI rewrites `[N]` in the answer into anchor links to the source sidebar (`linkify_citations` in `src/main.py`).

When extending retrieval: keep the dedup-by-`chunk_id` invariant, and remember the `cap` is applied at the end — anything added past the cap is silently dropped, so order matters. File names are NOT embedded or indexed for `$contains` — only chunk body text is. Filename match (step 2) is the only path that reads file names; if you need name-based recall, that's the place to extend.

### Web layer (`src/main.py`)

Three routes, all HTMX-driven, server-renders Jinja partials:
- `GET /` → `index.html` (full page).
- `POST /ask` → `answer.html` (partial; HTMX swaps it into the page).
- `POST /sync` → `sync_result.html` (partial). Runs `index_folders` in-process, so vectors written here are immediately visible to subsequent `/ask` calls.

`ANONYMIZED_TELEMETRY=False` is set before importing `chromadb` (both in `main.py` and `indexer.py`) — keep that ordering or Chroma will phone home.

### Security middleware (`src/main.py`)

One `@app.middleware("http")` (`security_middleware`) wraps every request and does three things in a fixed order. On POSTs:

1. **Origin check.** `request.headers.get("origin") or ""` must be in `ALLOWED_ORIGINS = {"http://localhost:8000", "http://127.0.0.1:8000"}`. Foreign, missing, empty, and spoofed Origins all 403. Modern browsers always include `Origin` on same-origin POSTs from HTMX, so this is the conservative posture (don't relax the `if origin not in ALLOWED_ORIGINS` check to also accept empty — that was the prior bug).
2. **Per-IP rate limit.** `_rate_limit_ok(ip)` allows up to `RATE_LIMIT_MAX = 60` requests per `RATE_LIMIT_WINDOW_SEC = 60` per `request.client.host`. Sliding window backed by a `deque[float]` of monotonic timestamps per IP. State is in-process — restarting uvicorn clears all buckets. Over the cap returns 429. Belt-and-braces against a runaway local script burning Anthropic spend.

On every response (POST or GET):

3. **Hardening headers** — `Content-Security-Policy` (the `CSP` constant: `default-src 'self'; script-src 'self' https://unpkg.com; ...; frame-ancestors 'none'`), `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, `X-Frame-Options: DENY`. The CSP forbids inline scripts, which is why page-scoped JS lives in `static/scout.js` (referenced with `<script src>`) and not in `<script>` tags inside templates. If you add new external assets, add them to the CSP origin lists.

Two more pieces complete the picture:

- **XSS-safe answer rendering** — `linkify_citations` in `src/main.py:50-60` calls `markupsafe.escape(text)` *first*, then runs the `[N]` regex substitution on the escaped string, then wraps in `Markup(...)`. The escape must happen before the substitution: a document indexed from Drive could otherwise persuade Claude to echo raw `<script>` or `<img onerror>` into the answer, and the existing `| safe` in `answer.html` would render it live. Tests: hostile text → entity-encoded; citation pills still rendered.
- **Question-length cap** — `MAX_QUESTION_LENGTH = 1000` checked in `ask_endpoint`; 400 above. Prevents prompt-blowup and incidentally bounds the worst-case Claude input size.

### Defenses outside the web layer

- **Secret-file modes.** `.env`, `credentials.json`, `token.json` are 0600 on disk. `src/drive.py:get_service` re-applies `TOKEN_FILE.chmod(0o600)` after every OAuth refresh, so a future refresh can't reintroduce 644.
- **Download size cap.** `MAX_FILE_BYTES = 50 MB` in `src/drive.py:download_file`. Checked mid-stream (`if buf.tell() > max_bytes`) so a malicious 200 MB PDF can't OOM the indexer process before any parser sees the bytes. Oversize raises `ValueError`, which the caller logs in the skipped-files list.
- **OAuth scope** is `drive.readonly` — minimal for this app.
- **Uvicorn default bind** is `127.0.0.1`. Don't add `--host 0.0.0.0` to the start command — doing so exposes everything above to the LAN.

## Conventions worth knowing

- `chunk_id` format `"{file_id}::{idx}"` is parsed in `_fetch_all_chunks_for_files` to sort master-list chunks back into document order. Don't change the separator without updating that parse.
- The hardcoded `MASTER_LIST_NAME_PATTERNS` / `LIST_QUESTION_PATTERNS` / `ENTITY_STOPWORDS` / `MIN_FILENAME_MATCHES` in `src/rag.py` are tuned for an IT-support corpus. They're the right knobs to turn when retrieval quality is off — not the embedding model or `SEMANTIC_K`.
- Embeddings use `all-MiniLM-L6-v2` locally via `sentence-transformers`; no API call for embedding. Swapping to Voyage AI is noted as an upgrade path in the README.
- The CSP in `src/main.py` allows `script-src 'self' https://unpkg.com` — if you add another external script source (analytics, a font loader that ships JS, etc.) it'll be blocked until you extend the CSP. Same applies to `style-src` and `font-src`. Don't add `'unsafe-inline'` to `script-src` as a shortcut; that re-opens the XSS path that `linkify_citations` was hardened against.
- `docs/scout-security-audit.pdf` is the latest static security audit. There's a reusable project subagent at `.claude/agents/scout-security-auditor.md` — ask Claude for a fresh security review to regenerate.
