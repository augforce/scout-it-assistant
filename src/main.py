"""FastAPI + HTMX webapp for the Drive RAG agent."""

from __future__ import annotations

import os
import re
from collections import deque
from pathlib import Path
from time import monotonic

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape
from starlette.responses import Response

from src.drive import _folder_ids_from_env
from src.indexer import index_folders
from src.rag import ask

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


def _rate_limit_ok(ip: str) -> bool:
    now = monotonic()
    bucket = _rate_buckets.setdefault(ip, deque())
    while bucket and now - bucket[0] > RATE_LIMIT_WINDOW_SEC:
        bucket.popleft()
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

app = FastAPI(title="Drive RAG Agent")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "templates")

CITATION_RE = re.compile(r"\[(\d+)\]")


def linkify_citations(text: str) -> Markup:
    """Escape the LLM's output for safe HTML rendering, then turn '[N]' into
    clickable citation chips. The escape MUST happen before the regex substitution:
    a malicious document indexed from Drive could otherwise persuade Claude to emit
    raw <script> or <img onerror> tags that would land verbatim in the answer pane."""
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
    answer = await ask(question)
    return templates.TemplateResponse(
        "answer.html",
        {
            "request": request,
            "question": question,
            "answer_html": linkify_citations(answer.text),
            "sources": answer.sources,
        },
    )


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
