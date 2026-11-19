"""HTTP API around run_agent / resume_agent for the chat UI."""

from __future__ import annotations

import os
import queue
import threading
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from coding_copilot_copy.agent import (
    AgentError,
    AgentTurn,
    DEFAULT_MAX_TOOL_CALLS,
    default_thread_id,
    resume_agent,
    run_agent,
)
from coding_copilot_copy.model import ModelError, model_from_env
from coding_copilot_copy.workspace import WorkspaceAccessError, _resolve_repo_root

load_dotenv()

app = FastAPI(title="coding-copilot", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: dict[str, dict[str, Any]] = {}
_model = None


class CreateSessionBody(BaseModel):
    repo_path: str | None = Field(
        default=None,
        description="Local repo root. Defaults to CODING_COPILOT_REPO.",
    )
    thread_id: str | None = None
    max_tool_calls: int = Field(default=DEFAULT_MAX_TOOL_CALLS, ge=1)


class MessageBody(BaseModel):
    content: str


class ResumeBody(BaseModel):
    action: str


def _get_model():
    global _model
    if _model is None:
        _model = model_from_env()
    return _model


def _default_repo() -> str | None:
    raw = os.environ.get("CODING_COPILOT_REPO")
    return raw.strip() if raw and raw.strip() else None


def serialize_turn(turn: AgentTurn, traces: list[str] | None = None) -> dict[str, Any]:
    status = "interrupted" if turn.interrupted else "completed"
    payload = turn.payload if isinstance(turn.payload, dict) else None
    return {
        "status": status,
        "thread_id": turn.thread_id,
        "message": {"role": "assistant", "content": turn.text},
        "interrupt": payload,
        "trace": list(traces or []),
    }


def _session_or_404(session_id: str) -> dict[str, Any]:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return session


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/config")
def config() -> dict[str, Any]:
    return {"default_repo": _default_repo()}


@app.post("/v1/sessions")
def create_session(body: CreateSessionBody) -> dict[str, Any]:
    raw = (body.repo_path or _default_repo() or "").strip()
    if not raw:
        raise HTTPException(
            status_code=400,
            detail="repo_path is required (or set CODING_COPILOT_REPO)",
        )
    try:
        repo = _resolve_repo_root(raw)
    except WorkspaceAccessError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    thread_id = body.thread_id or f"web:{uuid.uuid4()}"
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "repo": repo,
        "thread_id": thread_id,
        "max_tool_calls": body.max_tool_calls,
    }
    return {
        "session_id": session_id,
        "thread_id": thread_id,
        "repo_root": repo.as_posix(),
        "max_tool_calls": body.max_tool_calls,
    }


@app.get("/v1/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    session = _session_or_404(session_id)
    repo: Path = session["repo"]
    return {
        "session_id": session_id,
        "thread_id": session["thread_id"],
        "repo_root": repo.as_posix(),
        "max_tool_calls": session["max_tool_calls"],
    }


def _run_message(session: dict[str, Any], content: str, on_trace) -> AgentTurn:
    if not content.strip():
        raise AgentError("question cannot be empty")
    return run_agent(
        session["repo"],
        content,
        _get_model(),
        max_tool_calls=session["max_tool_calls"],
        thread_id=session["thread_id"],
        memory=True,
        on_trace=on_trace,
    )


def _run_resume(session: dict[str, Any], action: str, on_trace) -> AgentTurn:
    value = action.strip().lower()
    if value not in {"approve", "reject", "continue", "stop"}:
        raise AgentError("action must be approve, reject, continue, or stop")
    return resume_agent(
        session["repo"],
        _get_model(),
        action=value,
        max_tool_calls=session["max_tool_calls"],
        thread_id=session["thread_id"],
        on_trace=on_trace,
    )


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, (AgentError, ModelError, WorkspaceAccessError)):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


@app.post("/v1/sessions/{session_id}/messages")
def post_message(session_id: str, body: MessageBody) -> dict[str, Any]:
    session = _session_or_404(session_id)
    traces: list[str] = []
    try:
        turn = _run_message(session, body.content, traces.append)
    except Exception as exc:
        raise _http_error(exc) from exc
    return serialize_turn(turn, traces)


@app.post("/v1/sessions/{session_id}/resume")
def post_resume(session_id: str, body: ResumeBody) -> dict[str, Any]:
    session = _session_or_404(session_id)
    traces: list[str] = []
    try:
        turn = _run_resume(session, body.action, traces.append)
    except Exception as exc:
        raise _http_error(exc) from exc
    return serialize_turn(turn, traces)


def _sse_run(work) -> StreamingResponse:
    events: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def on_trace(line: str) -> None:
        events.put({"event": "trace", "line": line})

    def worker() -> None:
        try:
            turn = work(on_trace)
            events.put({"event": "done", "turn": serialize_turn(turn, [])})
        except Exception as exc:
            err = _http_error(exc)
            events.put({"event": "error", "detail": err.detail, "status": err.status_code})
        events.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def generate():
        import json

        while True:
            item = events.get()
            if item is None:
                break
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/v1/sessions/{session_id}/messages/stream")
def stream_message(session_id: str, body: MessageBody) -> StreamingResponse:
    session = _session_or_404(session_id)

    def work(on_trace):
        return _run_message(session, body.content, on_trace)

    return _sse_run(work)


@app.post("/v1/sessions/{session_id}/resume/stream")
def stream_resume(session_id: str, body: ResumeBody) -> StreamingResponse:
    session = _session_or_404(session_id)

    def work(on_trace):
        return _run_resume(session, body.action, on_trace)

    return _sse_run(work)


def main() -> None:
    load_dotenv()
    uvicorn.run(
        "coding_copilot_copy.server:app",
        host=os.environ.get("COPILOT_HOST", "127.0.0.1"),
        port=int(os.environ.get("COPILOT_PORT", "8765")),
        reload=False,
    )


if __name__ == "__main__":
    main()
