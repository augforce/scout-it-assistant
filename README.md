# Scout

**A personal IT-support assistant.** Ask a question, Scout searches a chosen set of Google Drive folders, and answers with clickable citations back to the source documents.



https://github.com/user-attachments/assets/25ac01e7-2953-462d-8c9a-2c3258b2eff2



Scout was built as a single-user assistant to answer day-to-day IT-support questions against an existing Drive-based knowledge base — approved-software lists, hardware inventory, onboarding/offboarding runbooks, vendor procedures, and so on. It runs entirely on the operator's machine, binds to `localhost`, and uses OAuth read-only access to Drive. There is no shared deployment.

## Highlights

- **Hybrid retrieval** — list-question detection (e.g. *"is X approved?"*, *"how many laptops do we have?"*), filename match for *"pull up the X draft"* style lookups, semantic top-k, and case-insensitive keyword fallback. Source numbering follows first-appearance order so the UI sidebar matches the answer.
- **Drive-native extraction** — Google Docs via text export, Google Sheets via **XLSX** export (preserves every tab, not just the first), PDFs via `pypdf`, `.docx` via `python-docx`. Sheets are formatted row-wise as `Col1: val | Col2: val | …` so each row stays self-describing across chunk boundaries.
- **Idempotent incremental sync** — chunk IDs are deterministic (`{file_id}::{idx}`), so a re-index of a changed file cleanly replaces its previous chunks. Deletions only run when the sync has a global view of the corpus (full folder walk, no `--limit`, no `--reset`).
- **Server-rendered UI** — FastAPI + HTMX. No JS bundler, no build step. Sync runs in-process, so vectors written by `/sync` are immediately queryable by the next `/ask`.
- **Local-first** — embeddings via `sentence-transformers` (no API call). Only the answer-generation step talks to Anthropic.
- **Hardened for local use** — strict-Origin CSRF check, per-IP rate limit, CSP without `unsafe-inline`, XSS-safe answer rendering (escape → linkify → mark-safe), question-length cap, 50 MB per-file download ceiling, OAuth scope locked to `drive.readonly`, secret files force-chmod'd to 0600 after every refresh.

## Stack

| Layer | Choice |
| --- | --- |
| Backend | Python 3.14 + FastAPI |
| LLM | Claude Sonnet 4.6 via the Anthropic SDK |
| Frontend | HTML + HTMX (server-rendered partials, no build step) |
| Vector store | Chroma, file-based (`chroma_db/`) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`), local |
| Drive access | Google Drive API v3, OAuth 2.0 desktop-app credentials, `drive.readonly` scope |

Indexed file types: Google Docs, Google Sheets, PDFs, Word/Office docs.

## Repo layout

```
.
├── src/
│   ├── main.py          # FastAPI app, HTMX routes, security middleware
│   ├── drive.py         # Drive OAuth client, listing, download (with 50 MB cap)
│   ├── extractors.py    # mime-type → text dispatch (Docs/Sheets/PDF/DOCX/XLSX)
│   ├── indexer.py       # chunking, embedding, Chroma upsert; incremental + targeted modes
│   └── rag.py           # hybrid retrieval, Claude call, citation formatting
├── templates/           # index / answer / sync_result Jinja partials
├── static/              # scout.js + style.css (CSP-friendly: no inline script)
├── requirements.txt
├── .env.example         # copy to .env and fill in
├── CLAUDE.md            # working-with-the-codebase notes (architecture, conventions)
└── README.md
```

## Setup

You need three things before the first run:

1. **Anthropic API key** — https://console.anthropic.com → goes in `.env` as `ANTHROPIC_API_KEY=`.
2. **Google OAuth credentials** — at https://console.cloud.google.com:
   - Create a project, enable the **Google Drive API**.
   - OAuth consent screen → External → add your account as a test user.
   - Credentials → OAuth client ID → **Desktop app** → download as `credentials.json` in the repo root.
3. **Drive folder IDs** — open each folder in the browser; the ID is the path segment after `/folders/`. Comma-separate in `.env` as `DRIVE_FOLDER_IDS=id1,id2`.

Then:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # fill in ANTHROPIC_API_KEY + DRIVE_FOLDER_IDS

python -m src.indexer       # one-time build of the vector store (opens browser for OAuth on first run)
uvicorn src.main:app --reload
# → http://localhost:8000
```

`.env`, `credentials.json`, `token.json`, and `chroma_db/` are gitignored and chmod-0600'd at runtime.

## Common operations

```bash
# Full sync — walks every folder in DRIVE_FOLDER_IDS, upserts new/changed,
# deletes chunks for files that disappeared from Drive. Idempotent.
python -m src.indexer

# Re-index only a few files by Drive ID (skips the folder walk; no deletions)
python -m src.indexer --file-ids <id1>,<id2>

# Scope to specific folders for this run only (no deletions in scoped mode)
python -m src.indexer --folder-ids <folder_id1>,<folder_id2>

# Smoke-test Drive auth + folder listing
python -m src.drive

# CLI query (bypasses the web UI)
python -m src.rag "is Slack approved?"
```

The `/sync` button in the header runs the same `index_folders()` entry point in-process, so a re-sync from the UI is visible to subsequent `/ask` calls without restarting uvicorn.

## How retrieval works

`ask(question)` runs four retrieval modes and merges them, deduplicated by chunk ID. Order matters because source numbering follows first appearance:

1. **List-question detour.** If the question matches an enumeration pattern (*"approved software"*, *"how many"*, *"is X approved"*, *"can we use Y"*), Scout pulls **every chunk** from files whose names match the master-list patterns (e.g. *Approved Applications*, *Laptop Inventory*) and raises the per-question chunk cap from 25 to 200. Lets the model categorize/count against the full list instead of a semantic-top-k subset.
2. **Filename match.** Scores each file by how many entity tokens from the question appear in its name; files at the threshold are surfaced top-N. Handles *"pull up the onboarding doc"* style lookups where the user names the doc and the body doesn't echo the question.
3. **Semantic search.** Top-K Chroma nearest neighbours by cosine.
4. **Keyword lookup.** Entity tokens looked up via Chroma's `$contains` in multiple casings (Chroma is case-sensitive on `$contains`).

The system prompt tells Claude that repeated `[N]` citations mean the same file; the UI rewrites every `[N]` into an anchor link to the source sidebar. Citations are XSS-safe: the answer is HTML-escaped *first*, then the `[N]` regex substitution runs on the escaped string.

For deeper architecture notes — security middleware, indexing internals, retrieval tuning knobs — see [`CLAUDE.md`](./CLAUDE.md).

## Project status & handoff

Scout is a **local-only**, single-operator tool. It is not deployed to a server, has no shared multi-tenant version, and is intentionally bound to `127.0.0.1` (do not add `--host 0.0.0.0` to the uvicorn command — it would expose the app to the LAN).

It was built to be **handoff-ready**: the next operator filling the same support role can clone this repo on their own machine, follow the *Setup* steps above, point `DRIVE_FOLDER_IDS` at the shared knowledge-base folders they have access to, and have an answer-with-citations workflow running in minutes. No infrastructure team, no shared service, no operations dependency.

If a future operator wants to extend the corpus to new file types, the entry points are `INDEXABLE_MIMES` in `src/drive.py` and the `extract_text` dispatch in `src/extractors.py`. If retrieval quality drifts, the tuning knobs are `MASTER_LIST_NAME_PATTERNS` / `LIST_QUESTION_PATTERNS` / `MIN_FILENAME_MATCHES` in `src/rag.py` — not the embedding model or `SEMANTIC_K`.

## Security posture (summary)

Scout is single-user, runs on `localhost`, and trusts only itself. The hardening focuses on (a) keeping secrets off disk in world-readable form, (b) protecting the local HTTP surface from drive-by browser attacks, and (c) preventing untrusted document content from breaking out of the answer-rendering path.

- **Secret hygiene** — `.env`, `credentials.json`, `token.json` are gitignored and force-chmod'd to 0600 (re-applied after every OAuth refresh).
- **HTTP surface** — strict-Origin CSRF check on POSTs, per-IP sliding-window rate limit (60/min), CSP with no `unsafe-inline` for `script-src`, `frame-ancestors 'none'`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`.
- **Content safety** — `linkify_citations` escapes the model's answer *before* injecting citation anchors, so a hostile document indexed from Drive can't smuggle `<script>` or `<img onerror>` into the page.
- **Resource bounds** — 50 MB per-file download cap (checked mid-stream), 1000-char question cap, 200-chunk hard ceiling per question.
- **OAuth scope** — `drive.readonly` only.

Full architectural rationale for each control lives in `CLAUDE.md`.

## License

MIT (or your preferred OSS license — add a `LICENSE` file before going public if you fork).
