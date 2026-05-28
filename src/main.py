"""FastAPI + HTMX webapp for the Drive RAG agent."""

from __future__ import annotations

import os
import re
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape
from starlette.responses import Response

from src.drive import _folder_ids_from_env
from src.indexer import get_collection, index_folders
from src.rag import (
    Chunk,
    Source,
    _filter_to_cited_sources,
    _retrieve,
    stream_answer,
    validate_citations,
)

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Local app, single user — we only ever expect POSTs from the UI itself.
ALLOWED_ORIGINS = {"http://localhost:8000", "http://127.0.0.1:8000"}
MAX_QUESTION_LENGTH = 1000

# Simple in-process rate limit on POST routes. Belt-and-braces against a
# runaway local script hammering /ask and burning Anthropic spend.
RATE_LIMIT_WINDOW_SEC = 60
RATE_LIMIT_MAX = 60  # requests per window per remote IP
_rate_buckets: dict[str, deque[float]] = {}

# Streaming bridge: POST /ask stashes the retrieved chunks here under a uuid4
# request_id. The browser then GETs /ask/stream/{id} to stream Claude's reply
# token-by-token, and finally GETs /ask/finalize/{id} to fetch the validated +
# linkified HTML and the cited-sources sidebar. State is single-process and
# evicted after PENDING_TTL_SEC; restarting uvicorn clears it.
PENDING_TTL_SEC = 300


@dataclass
class _Pending:
    question: str
    chunks: list[Chunk]
    sources: list[Source]
    created_at: float
    accumulated: str = ""
    streamed: bool = False  # True after /ask/stream has been invoked once


_pending: dict[str, _Pending] = {}


def _evict_stale_pending() -> None:
    """Drop entries older than the TTL. Called lazily on every read so we never
    grow unboundedly even if the user abandons mid-stream."""
    cutoff = monotonic() - PENDING_TTL_SEC
    stale = [k for k, v in _pending.items() if v.created_at < cutoff]
    for k in stale:
        _pending.pop(k, None)


def _rate_limit_ok(ip: str) -> bool:
    now = monotonic()
    bucket = _rate_buckets.setdefault(ip, deque())
    while bucket and now - bucket[0] > RATE_LIMIT_WINDOW_SEC:
        bucket.popleft()
    if not bucket:
        # opportunistic prune: free the empty deque so the dict doesn't accrete
        # one entry per ever-seen IP over very long uptimes
        _rate_buckets.pop(ip, None)
        bucket = _rate_buckets.setdefault(ip, deque())
    if len(bucket) >= RATE_LIMIT_MAX:
        return False
    bucket.append(now)
    return True

# CSP whitelist: same-origin everything, plus unpkg for HTMX and Google Fonts for
# the brand typography. Inline <script> is blocked — keep page-scoped JS in
# static/scout.js and reference it with a <script src> tag.
CSP = (
    "default-src 'self'; "
    "script-src 'self' https://unpkg.com; "
    "style-src 'self' https://fonts.googleapis.com 'unsafe-inline'; "
    "font-src 'self' https://fonts.gstatic.com; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "frame-ancestors 'none'"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm the sentence-transformers model + Chroma client so the first
    # user query doesn't pay the 1–3 s cold-start cost mid-request.
    try:
        get_collection().query(query_texts=["warm"], n_results=1)
    except Exception:
        # The collection may be empty on a fresh install; that's fine — the
        # embedder still gets JIT'd by the query call.
        pass
    yield


app = FastAPI(title="Drive RAG Agent", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "templates")

CITATION_RE = re.compile(r"\[(\d+)\]")


def linkify_citations(text: str) -> Markup:
    """Escape the LLM's output for safe HTML rendering, then turn '[N]' into
    clickable citation chips. The escape MUST happen before the regex substitution:
    a malicious document indexed from Drive could otherwise persuade Claude to emit
    raw <script> or <img onerror> tags that would land verbatim in the answer pane.

    By the time this runs, validate_citations() has already normalized any grouped
    "[1, 2]" forms into "[1][2]", so the strict single-digit regex matches them."""
    safe = str(escape(text))
    linked = CITATION_RE.sub(
        r'<a href="#source-\1" class="cite">[\1]</a>',
        safe,
    )
    return Markup(linked.replace("\n", "<br>"))


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    if request.method == "POST":
        # CSRF / cross-origin abuse: state-changing requests must come from the UI.
        # We require a positive match (an empty or missing Origin is rejected too —
        # browsers always include Origin on POSTs from the same origin, so this is
        # the conservative posture).
        origin = request.headers.get("origin") or ""
        if origin not in ALLOWED_ORIGINS:
            return Response("Forbidden origin", status_code=403)
        # Cheap per-IP rate limit to keep a runaway local client from burning spend.
        ip = request.client.host if request.client else "unknown"
        if not _rate_limit_ok(ip):
            return Response("Rate limit exceeded", status_code=429)
    response = await call_next(request)
    # Hardening headers applied to every response.
    response.headers["Content-Security-Policy"] = CSP
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    return response


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask", response_class=HTMLResponse)
async def ask_endpoint(request: Request, question: str = Form(...)):
    if len(question) > MAX_QUESTION_LENGTH:
        raise HTTPException(status_code=400, detail="Question too long.")
    _evict_stale_pending()

    chunks, sources = await _retrieve(question)
    request_id = uuid.uuid4().hex
    _pending[request_id] = _Pending(
        question=question, chunks=chunks, sources=sources, created_at=monotonic()
    )

    return templates.TemplateResponse(
        "answer.html",
        {
            "request": request,
            "question": question,
            "request_id": request_id,
            "has_results": bool(chunks),
        },
    )


@app.get("/ask/stream/{request_id}")
async def ask_stream(request_id: str):
    _evict_stale_pending()
    pending = _pending.get(request_id)
    if pending is None:
        raise HTTPException(status_code=404, detail="Unknown or expired request")
    if pending.streamed:
        # Idempotency / replay protection. The browser only calls this once.
        raise HTTPException(status_code=410, detail="Already streamed")
    pending.streamed = True

    async def gen():
        async for delta in stream_answer(pending.question, pending.chunks, pending.sources):
            pending.accumulated += delta
            yield delta

    # text/plain (not text/event-stream) — the client uses fetch + ReadableStream
    # rather than EventSource, which keeps the protocol trivial and avoids the
    # SSE retry/event-id machinery we don't need.
    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")


@app.get("/ask/finalize/{request_id}")
async def ask_finalize(request_id: str):
    _evict_stale_pending()
    pending = _pending.get(request_id)
    if pending is None:
        raise HTTPException(status_code=404, detail="Unknown or expired request")
    if not pending.streamed:
        raise HTTPException(status_code=425, detail="Stream not started")

    cleaned, warnings = validate_citations(pending.accumulated, pending.sources)
    filtered_sources = _filter_to_cited_sources(cleaned, pending.sources)
    answer_html = str(linkify_citations(cleaned))
    sources_html = templates.get_template("_sources.html").render(sources=filtered_sources)

    _pending.pop(request_id, None)

    return JSONResponse({
        "answer_html": answer_html,
        "sources_html": sources_html,
        "warnings": warnings,
    })


@app.post("/sync", response_class=HTMLResponse)
async def sync_endpoint(request: Request):
    """Run an incremental sync against all configured Drive folders.

    Same Python process as the server, so writes update the in-memory Chroma
    state directly — no restart needed for queries to see the new vectors.
    """
    folder_ids = _folder_ids_from_env()
    if not folder_ids:
        result_html = "<div class='sync-result error'>DRIVE_FOLDER_IDS not set in .env</div>"
        return HTMLResponse(result_html)

    result = index_folders(folder_ids=folder_ids, verbose=False)
    return templates.TemplateResponse(
        "sync_result.html",
        {"request": request, "result": result},
    )
