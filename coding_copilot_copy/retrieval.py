"""Hybrid codebase retrieval: syntactic chunks, embeddings, FTS, RRF.

Industry-shaped pipeline used by Copilot/Cursor-style agents:
chunk -> embed -> vector search ∪ lexical search -> reciprocal rank fusion.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pgvector.psycopg import register_vector
from psycopg import Connection
from psycopg.rows import dict_row

from coding_copilot_copy.workspace import (
    WorkspaceAccessError,
    _is_restricted_name,
    _is_skipped_dir,
    _resolve_repo_root,
)

EMBEDDING_DIM = 1536
DEFAULT_HYBRID_RESULTS = 8
MAX_HYBRID_RESULTS = 20
_RRF_K = 60
_MAX_INDEX_FILE_BYTES = 512_000
_INDEX_SUFFIXES = frozenset(
    {
        ".java",
        ".kt",
        ".py",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".go",
        ".rs",
        ".md",
        ".xml",
        ".yml",
        ".yaml",
        ".properties",
        ".gradle",
        ".json",
        ".txt",
    }
)
_HEADER = re.compile(
    r"^\s*(?:"
    r"@(?:SystemMessage|UserMessage|AiService)\b"
    r"|(?:public|private|protected|internal|export|async|static|final|open)*\s*"
    r"(?:class|interface|enum|record|def|function|fn|func)\s+\w+"
    r")",
    re.MULTILINE,
)


class RetrievalError(Exception):
    """Raised when indexing or hybrid search fails."""


EmbedFn = Callable[[list[str]], list[list[float]]]


def _database_url(explicit: str | None = None) -> str:
    load_dotenv()
    url = explicit or os.environ.get("DATABASE_URL")
    if not url:
        raise RetrievalError("DATABASE_URL is not set")
    return url


def embeddings_from_env() -> OpenAIEmbeddings:
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RetrievalError("OPENAI_API_KEY is not set")
    kwargs: dict = {
        "model": os.environ.get("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small",
        "api_key": api_key,
    }
    url = os.environ.get("OPENAI_URL")
    if url:
        kwargs["base_url"] = url
    return OpenAIEmbeddings(**kwargs)


def _embed_fn_from_env() -> EmbedFn:
    client = embeddings_from_env()

    def _embed(texts: list[str]) -> list[list[float]]:
        return client.embed_documents(texts)

    return _embed


def chunk_source(rel_path: str, text: str) -> list[dict[str, Any]]:
    """Split a file into syntactic-ish chunks (headers / size), with path metadata."""
    lines = text.splitlines()
    if not lines:
        return []
    bounds = [0]
    for i, line in enumerate(lines):
        if i == 0:
            continue
        if _HEADER.search(line) and i - bounds[-1] >= 4:
            bounds.append(i)
    bounds.append(len(lines))
    chunks: list[dict[str, Any]] = []
    for start, end in zip(bounds, bounds[1:]):
        while end - start > 120:
            mid = start + 120
            chunks.append(_chunk(rel_path, lines, start, mid))
            start = mid
        if start < end:
            chunks.append(_chunk(rel_path, lines, start, end))
    return chunks


def _chunk(rel_path: str, lines: list[str], start: int, end: int) -> dict[str, Any]:
    body = "\n".join(lines[start:end]).strip("\n")
    return {
        "path": rel_path,
        "start_line": start + 1,
        "end_line": end,
        "content": body,
    }


def _file_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _iter_indexable_files(repo_root: Path) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if _is_skipped_dir(path, repo_root):
            continue
        if path.suffix.lower() not in _INDEX_SUFFIXES:
            continue
        rel = path.relative_to(repo_root).as_posix()
        if any(_is_restricted_name(part) for part in Path(rel).parts):
            continue
        if path.stat().st_size > _MAX_INDEX_FILE_BYTES:
            continue
        found.append((rel, path))
    return found


def _connect(database_url: str | None = None) -> Connection:
    conn = Connection.connect(_database_url(database_url), autocommit=True)
    with conn.cursor() as cur:
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except Exception as exc:
            raise RetrievalError(
                "Postgres is missing pgvector. Recreate the DB with image "
                "pgvector/pgvector:pg16 (docker compose down && docker compose up -d)."
            ) from exc
    register_vector(conn)
    return conn


def ensure_schema(conn: Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS code_chunks (
                id bigserial PRIMARY KEY,
                repo_root text NOT NULL,
                path text NOT NULL,
                start_line int NOT NULL,
                end_line int NOT NULL,
                file_hash text NOT NULL,
                content text NOT NULL,
                embedding vector({EMBEDDING_DIM}) NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS code_chunks_repo_path
            ON code_chunks (repo_root, path)
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS code_chunks_embedding_hnsw
            ON code_chunks USING hnsw (embedding vector_cosine_ops)
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS code_chunks_fts
            ON code_chunks
            USING gin (to_tsvector('simple', path || ' ' || content))
            """
        )


def ensure_index(
    repo_path: str | Path,
    *,
    database_url: str | None = None,
    embed: EmbedFn | None = None,
) -> str:
    """Incrementally (re)embed changed files for ``repo_path``."""
    repo_root = _resolve_repo_root(repo_path)
    embed_fn = embed or _embed_fn_from_env()
    files = _iter_indexable_files(repo_root)
    repo_key = repo_root.as_posix()

    with _connect(database_url) as conn:
        ensure_schema(conn)
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT DISTINCT path, file_hash FROM code_chunks WHERE repo_root = %s",
                (repo_key,),
            )
            stored = {row["path"]: row["file_hash"] for row in cur.fetchall()}

        current_paths: set[str] = set()
        embedded_files = 0
        for rel, path in files:
            current_paths.add(rel)
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            digest = _file_hash(text)
            if stored.get(rel) == digest:
                continue
            pieces = chunk_source(rel, text)
            if not pieces:
                continue
            vectors = embed_fn([p["content"] for p in pieces])
            if len(vectors) != len(pieces):
                raise RetrievalError("embedding count does not match chunk count")
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM code_chunks WHERE repo_root = %s AND path = %s",
                    (repo_key, rel),
                )
                for piece, vector in zip(pieces, vectors):
                    cur.execute(
                        """
                        INSERT INTO code_chunks
                            (repo_root, path, start_line, end_line, file_hash, content, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            repo_key,
                            piece["path"],
                            piece["start_line"],
                            piece["end_line"],
                            digest,
                            piece["content"],
                            vector,
                        ),
                    )
            embedded_files += 1

        stale = set(stored) - current_paths
        if stale:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM code_chunks WHERE repo_root = %s AND path = ANY(%s)",
                    (repo_key, list(stale)),
                )

    return (
        f"Indexed {repo_key}: {len(current_paths)} files, "
        f"{embedded_files} re-embedded, {len(stale)} removed"
    )


def _rrf(rank_lists: list[list[tuple[str, dict[str, Any]]]]) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    payload: dict[str, dict[str, Any]] = {}
    for ranked in rank_lists:
        for rank, (key, item) in enumerate(ranked, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)
            payload.setdefault(key, item)
    ordered = sorted(scores, key=lambda k: scores[k], reverse=True)
    results = []
    for key in ordered:
        item = dict(payload[key])
        item["score"] = round(scores[key], 6)
        results.append(item)
    return results


def hybrid_search(
    repo_path: str | Path,
    query: str,
    *,
    max_results: int = DEFAULT_HYBRID_RESULTS,
    database_url: str | None = None,
    embed: EmbedFn | None = None,
) -> str:
    """Semantic ∪ lexical search with reciprocal rank fusion."""
    if not query.strip():
        raise WorkspaceAccessError("Query cannot be empty")
    repo_root = _resolve_repo_root(repo_path)
    repo_key = repo_root.as_posix()
    limited = min(max_results, MAX_HYBRID_RESULTS)
    fetch = max(limited * 4, 20)
    embed_fn = embed or _embed_fn_from_env()
    query_vec = embed_fn([query.strip()])[0]

    with _connect(database_url) as conn:
        ensure_schema(conn)
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT path, start_line, end_line, content,
                       embedding <=> %s::vector AS distance
                FROM code_chunks
                WHERE repo_root = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_vec, repo_key, query_vec, fetch),
            )
            semantic_rows = list(cur.fetchall())
            try:
                cur.execute(
                    """
                    SELECT path, start_line, end_line, content,
                           ts_rank(
                               to_tsvector('simple', path || ' ' || content),
                               websearch_to_tsquery('simple', %s)
                           ) AS rank
                    FROM code_chunks
                    WHERE repo_root = %s
                      AND to_tsvector('simple', path || ' ' || content)
                          @@ websearch_to_tsquery('simple', %s)
                    ORDER BY rank DESC
                    LIMIT %s
                    """,
                    (query, repo_key, query, fetch),
                )
                lexical_rows = list(cur.fetchall())
            except Exception:
                lexical_rows = []

    def _key(row: dict[str, Any]) -> str:
        return f"{row['path']}:{row['start_line']}-{row['end_line']}"

    fused = _rrf(
        [
            [(_key(row), row) for row in semantic_rows],
            [(_key(row), row) for row in lexical_rows],
        ]
    )[:limited]
    if not fused:
        return "No hybrid search results. Try codebase_search with a shorter phrase, or search_code for an exact token."

    lines = [f"Hybrid search results for '{query}'"]
    for row in fused:
        snippet = " ".join(str(row["content"]).split())[:240]
        lines.append(
            f"{row['path']}:{row['start_line']}-{row['end_line']}: {snippet}"
        )
    return "\n".join(lines)
