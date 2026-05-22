"""Retrieval + Claude call. Returns the answer plus the source files cited."""

from __future__ import annotations

import asyncio
import functools
import os
import re
from dataclasses import dataclass
from pathlib import Path

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from src.indexer import get_collection

ROOT = Path(__file__).resolve().parent.parent
MODEL = "claude-sonnet-4-6"
SEMANTIC_K = 12
KEYWORD_K_PER_ENTITY = 5
MAX_CHUNKS_TO_LLM = 25
MAX_CHUNKS_FOR_LIST_QUESTION = 200
MAX_TOKENS = 1500

# Filename-match retrieval: when a question's entity tokens hit 2+ words in a file's
# name, the user is likely asking for that file by name ("pull up the X and Y draft").
# Require >=2 matches so single-word questions like "is Slack approved?" don't pull in
# every Slack-named doc.
MIN_FILENAME_MATCHES = 2
MAX_FILENAME_MATCH_FILES = 3
MAX_CHUNKS_PER_FILENAME_MATCH = 5

# Documents that are flat lists/registries. When a question asks for an enumeration
# or category, we pull in EVERY chunk of these docs so Claude can categorize using
# its world knowledge. Match is case-insensitive substring on the file name.
MASTER_LIST_NAME_PATTERNS = [
    "approved applications",
    "approved desktop software",
    "quarterly software risk",
    "currently installed software",
    "laptop inventory",
]

# Heuristics for detecting "give me a view of the whole list" style questions.
# Any pattern matching triggers full master-list inclusion.
LIST_QUESTION_PATTERNS = [
    # "what/which/list/show all chat tools..."
    re.compile(
        r"\b(?:what|which|list|show|all|any|enumerate|name|tell\s+me\s+(?:the|about))\b"
        r".{0,80}?\b"
        r"(?:tools?|software|apps?|applications?|programs?|ides?|editors?|"
        r"browsers?|extensions?|utilities|clients?|services?|managers?|viewers?)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    # Counting questions: "how many IDEs..."
    re.compile(r"\bhow\s+many\b", re.IGNORECASE),
    # Explicit reference to the master list itself
    re.compile(
        r"\b(?:approved|disapproved|denied|banned|rejected)\s+"
        r"(?:software|applications?|list|apps?|programs?)\b",
        re.IGNORECASE,
    ),
    # "are there [any] ... tools/software/..."
    re.compile(
        r"\b(?:are\s+there|do\s+we\s+have)\b.{0,80}?\b"
        r"(?:tools?|software|apps?|applications?|programs?|ides?|editors?|"
        r"browsers?|extensions?|utilities|clients?|services?|managers?|viewers?)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    # Single-item approval questions: "is Slack approved?", "has Notion been allowed?".
    # One-line list entries don't score well on semantic search, so route these
    # through the master-list dump too.
    re.compile(
        r"\b(?:is|are|was|were|has|have|had|can|could|may|might|should|do|does|did)\b"
        r".{1,80}?\b"
        r"(?:approved|allowed|permitted|sanctioned|authorized|"
        r"banned|blocked|denied|disapproved|prohibited|rejected)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    # Permission-by-usage phrasing: "can we use X?", "should I install Y?".
    # Anchored to start-of-question so "how can I use X" / "where can I install X"
    # style usage questions don't match.
    re.compile(
        r"(?:^|[.?!]\s+)(?:can|could|may|might|should)\s+\w+\s+"
        r"(?:use|install|run|adopt|deploy|access)\b",
        re.IGNORECASE,
    ),
]

# Stopwords for entity extraction — words that should NOT trigger a keyword lookup.
# Lowercased; matched case-insensitively.
ENTITY_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "can", "could", "may", "might", "should", "would",
    "will", "shall", "have", "has", "had",
    "i", "you", "we", "they", "he", "she", "it", "me", "us", "them", "my",
    "your", "our", "their", "his", "her", "its",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "to", "of", "in", "on", "at", "by", "for", "with", "from", "about", "as",
    "into", "onto", "over", "under", "between", "through",
    "and", "or", "but", "if", "else", "than", "then", "so", "not", "no", "yes",
    "this", "that", "these", "those",
    "approved", "approve", "approval", "software", "application", "applications",
    "app", "apps", "use", "using", "used", "tool", "tools", "list", "lists",
    "policy", "policies", "document", "documents",
}

SYSTEM_PROMPT = """You are an internal IT support assistant. Answer questions about \
company software, policies, and procedures using ONLY the provided document excerpts.

Rules:
- Cite every fact with bracketed source numbers like [1], [2]. Multiple sources: [1][3].
- For approval-style questions ("is X approved?"), give a clear YES / NO / UNCLEAR up front, then explain.
- If the documents don't contain the answer, say so plainly — do not guess.
- The same source number may appear on multiple excerpts; that means the excerpts came from the same file.
- When asked to enumerate or categorize items (e.g., "what chat tools are approved?"), be EXHAUSTIVE:
  scan every excerpt for items that fit the category, using your world knowledge to classify them
  (e.g., Slack and Microsoft Teams are chat tools; VS Code and IntelliJ are IDEs; Photoshop is photo
  editing). List every matching item you can find.
- For "pull up", "find", "show me", "where is" style requests for a specific document, identify the
  matching source(s) by name and point the user to the source number — do not refuse just because
  the body excerpts don't form a self-contained answer. A short summary of what the doc contains is
  helpful, but the link itself is the goal.
- Spreadsheet rows (lines of the form "Col1: val | Col2: val | ...") are individual records.
  For counting questions: scan EVERY row top-to-bottom, list every matching row, then state the
  total. Do not skim or estimate. Treat date strings like "2024-03-01", "March 2024", and
  "Sept 2024" as equivalent for year-based filters. When the same record appears in multiple sheet
  tabs (e.g. "# Sheet: MacBooks" and "# Sheet: Sheet5"), dedupe by the primary identifier
  (employee name, serial number, etc.) before counting.
- Keep answers concise. Bullet points when listing things.
"""


@dataclass
class Chunk:
    chunk_id: str
    file_id: str
    file_name: str
    mime_type: str
    web_view_url: str
    content: str


@dataclass
class Source:
    number: int
    file_id: str
    file_name: str
    mime_type: str
    web_view_url: str


@dataclass
class Answer:
    text: str
    sources: list[Source]


def _extract_entities(question: str) -> list[str]:
    """Pull likely product/software names out of the question for substring lookup.
    Keeps tokens that aren't generic stopwords. Order-preserving, deduped."""
    cleaned = re.sub(r"[^\w\s.-]", " ", question)
    seen: set[str] = set()
    out: list[str] = []
    for tok in cleaned.split():
        if len(tok) < 3:
            continue
        if tok.lower() in ENTITY_STOPWORDS:
            continue
        key = tok.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(tok)
    return out


def _chunk_from_result(chunk_id: str, doc: str, meta: dict) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        file_id=meta["file_id"],
        file_name=meta["file_name"],
        mime_type=meta["mime_type"],
        web_view_url=meta["web_view_url"],
        content=doc,
    )


def _is_list_question(question: str) -> bool:
    return any(p.search(question) for p in LIST_QUESTION_PATTERNS)


def _master_list_file_ids_from_metas(metas: list[dict]) -> set[str]:
    """Identify file IDs whose names match the master-list patterns from pre-fetched metadata."""
    out: set[str] = set()
    for meta in metas:
        name_lower = meta["file_name"].lower()
        if any(p in name_lower for p in MASTER_LIST_NAME_PATTERNS):
            out.add(meta["file_id"])
    return out


def _fetch_all_chunks_for_files(collection, file_ids: set[str]) -> list[Chunk]:
    if not file_ids:
        return []
    res = collection.get(
        where={"file_id": {"$in": list(file_ids)}},
        include=["documents", "metadatas"],
    )
    chunks = [
        _chunk_from_result(cid, doc, meta)
        for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])
    ]
    # Order within each file by chunk_index so list rows read coherently
    chunks.sort(
        key=lambda c: (
            c.file_name,
            int(c.chunk_id.rsplit("::", 1)[-1]) if "::" in c.chunk_id else 0,
        )
    )
    return chunks


def _filename_match_chunks_from_metas(
    collection, entities: list[str], metas: list[dict]
) -> list[Chunk]:
    """Chunks from files whose name contains >= MIN_FILENAME_MATCHES entity tokens.
    Targets 'pull up the X and Y draft' style lookups where the file is identifiable
    by name but its body text doesn't echo the question well.
    Uses pre-fetched metadata to avoid a second full table scan."""
    if len(entities) < MIN_FILENAME_MATCHES:
        return []
    lowered = [e.lower() for e in entities]

    file_score: dict[str, int] = {}
    for meta in metas:
        fid = meta["file_id"]
        if fid in file_score:
            continue
        name_lower = meta["file_name"].lower()
        score = sum(1 for e in lowered if e in name_lower)
        if score >= MIN_FILENAME_MATCHES:
            file_score[fid] = score

    if not file_score:
        return []

    top_files = sorted(file_score, key=lambda f: -file_score[f])[:MAX_FILENAME_MATCH_FILES]
    all_chunks = _fetch_all_chunks_for_files(collection, set(top_files))

    by_file: dict[str, list[Chunk]] = {}
    for c in all_chunks:
        by_file.setdefault(c.file_id, []).append(c)

    out: list[Chunk] = []
    for fid in top_files:
        out.extend(by_file.get(fid, [])[:MAX_CHUNKS_PER_FILENAME_MATCH])
    return out


async def _retrieve(question: str) -> tuple[list[Chunk], list[Source]]:
    """Hybrid retrieval: semantic + keyword + filename-match + (for list-style questions) full master lists."""
    collection = get_collection()
    chunks: list[Chunk] = []
    seen: set[str] = set()
    list_mode = _is_list_question(question)
    cap = MAX_CHUNKS_FOR_LIST_QUESTION if list_mode else MAX_CHUNKS_TO_LLM
    entities = _extract_entities(question)

    # Single metadata scan shared by both list-mode and filename-match paths.
    # Skip entirely when neither path needs it (short questions with < 2 entity tokens).
    needs_meta_scan = list_mode or len(entities) >= MIN_FILENAME_MATCHES
    all_metas: list[dict] = []
    if needs_meta_scan:
        all_metas = collection.get(include=["metadatas"])["metadatas"]

    # For list/category questions, dump the entire master list docs first so Claude
    # has the full enumeration to categorize against.
    if list_mode:
        master_ids = _master_list_file_ids_from_metas(all_metas)
        for chunk in _fetch_all_chunks_for_files(collection, master_ids):
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            chunks.append(chunk)

    # Filename-match: if multiple entity tokens collide on a single file's name, that
    # file is almost certainly the one the user is asking about. Prioritize it before
    # semantic/keyword so it lands a low source number.
    for chunk in _filename_match_chunks_from_metas(collection, entities, all_metas):
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        chunks.append(chunk)

    sem = collection.query(query_texts=[question], n_results=SEMANTIC_K)
    for cid, doc, meta in zip(sem["ids"][0], sem["documents"][0], sem["metadatas"][0]):
        if cid in seen:
            continue
        seen.add(cid)
        chunks.append(_chunk_from_result(cid, doc, meta))

    # Keyword lookups: Chroma's $contains is case-sensitive; try a couple casings.
    # All variants are dispatched concurrently via run_in_executor.
    loop = asyncio.get_running_loop()
    variants: list[str] = []
    for entity in entities[:4]:
        variants.extend({entity, entity.lower(), entity.title()})

    async def _kw(variant: str) -> dict:
        try:
            return await loop.run_in_executor(
                None,
                functools.partial(
                    collection.get,
                    where_document={"$contains": variant},
                    include=["documents", "metadatas"],
                    limit=KEYWORD_K_PER_ENTITY,
                ),
            )
        except Exception:
            return {"ids": [], "documents": [], "metadatas": []}

    for kw in await asyncio.gather(*[_kw(v) for v in variants]):
        for cid, doc, meta in zip(kw["ids"], kw["documents"], kw["metadatas"]):
            if cid in seen:
                continue
            seen.add(cid)
            chunks.append(_chunk_from_result(cid, doc, meta))

    chunks = chunks[:cap]

    # Number sources by first appearance of each unique file
    sources: list[Source] = []
    file_to_num: dict[str, int] = {}
    for c in chunks:
        if c.file_id not in file_to_num:
            file_to_num[c.file_id] = len(sources) + 1
            sources.append(
                Source(
                    number=file_to_num[c.file_id],
                    file_id=c.file_id,
                    file_name=c.file_name,
                    mime_type=c.mime_type,
                    web_view_url=c.web_view_url,
                )
            )
    return chunks, sources


def _build_context(chunks: list[Chunk], sources: list[Source]) -> str:
    file_to_num = {s.file_id: s.number for s in sources}
    blocks = []
    for c in chunks:
        num = file_to_num[c.file_id]
        blocks.append(f"[{num}] From: {c.file_name}\n{c.content}")
    return "\n\n---\n\n".join(blocks)


# Matches "[3]" and "[3, 7]" / "[3,7]" forms — any digit run inside a bracket pair.
_CITATION_RE = re.compile(r"\[([\d,\s]+)\]")


def _filter_to_cited_sources(text: str, sources: list[Source]) -> list[Source]:
    """Reduce the source list to only files Claude actually cited in the answer.
    Reference docs that were dumped into context by master-list / filename-match
    rules but never cited inline are dropped from the UI."""
    cited: set[int] = set()
    for inner in _CITATION_RE.findall(text):
        for token in inner.split(","):
            token = token.strip()
            if token.isdigit():
                cited.add(int(token))
    return [s for s in sources if s.number in cited]


async def ask(question: str) -> Answer:
    load_dotenv(ROOT / ".env")
    chunks, sources = await _retrieve(question)

    if not chunks:
        return Answer(text="No matching documents found in the index.", sources=[])

    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    user_content = (
        f"Documents:\n\n{_build_context(chunks, sources)}\n\n"
        f"---\n\nQuestion: {question}"
    )

    resp = await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    return Answer(text=text, sources=_filter_to_cited_sources(text, sources))


if __name__ == "__main__":
    import asyncio
    import sys

    q = " ".join(sys.argv[1:]) or "What is in this knowledge base?"
    answer = asyncio.run(ask(q))
    print(answer.text)
    print("\nSources:")
    for s in answer.sources:
        print(f"  [{s.number}] {s.file_name} -> {s.web_view_url}")
