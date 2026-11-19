"""LangGraph tool-calling agent.

Graph shape: ``agent ⇄ tools_read ⇄ write_review ⇄ tool_limit → agent → END``.

Read tools run immediately. Writes pause at ``interrupt`` until approve/reject.
Hitting the per-round tool budget pauses at ``tool_limit`` until continue/stop.
Continue resets the counter and keeps tools bound; stop answers once without
tools. The next user question always starts a fresh budget.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Iterator, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from coding_copilot_copy.prompts import SYSTEM_PROMPT
from coding_copilot_copy.retrieval import RetrievalError, ensure_index
from coding_copilot_copy.tools import WRITE_KIND_BY_TOOL, WRITE_TOOL_NAMES, make_repo_tools
from coding_copilot_copy.workspace import (
    WorkspaceAccessError,
    apply_create,
    apply_delete,
    apply_replace,
    _resolve_repo_root,
)


DEFAULT_MAX_TOOL_CALLS = 25


class AgentError(Exception):
    """Raised when the agent cannot run a question."""


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tool_calls_used: int
    pending_edits: list[dict[str, Any]]


@dataclass
class AgentTurn:
    text: str
    interrupted: bool = False
    payload: dict[str, Any] | None = None
    thread_id: str = ""


TraceFn = Callable[[str], None]
_TRACE_TOOL_CHARS = 500
_TRACE_THINK_CHARS = 800


def build_agent_graph(
    model: BaseChatModel,
    tools: list[BaseTool],
    repo: Path,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    checkpointer: PostgresSaver | None = None,
):
    """Build the compiled agent graph."""
    tools_by_name = {tool.name: tool for tool in tools}
    model_with_tools = model.bind_tools(tools)

    def agent_node(state: AgentState) -> dict[str, Any]:
        used = int(state.get("tool_calls_used", 0))
        runnable = model if used >= max_tool_calls else model_with_tools
        response = runnable.invoke(state["messages"])
        return {"messages": [response]}

    def tools_read_node(state: AgentState) -> dict[str, Any]:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {"pending_edits": []}

        used = int(state.get("tool_calls_used", 0))
        output: list[AnyMessage] = []
        pending: list[dict[str, Any]] = []

        for tool_call in last_message.tool_calls:
            call_id = str(tool_call["id"])
            name = str(tool_call["name"])
            args = tool_call.get("args") or {}

            if used >= max_tool_calls:
                output.append(
                    ToolMessage(
                        content=(
                            f"Tool {name} was not run: this round hit the "
                            f"{max_tool_calls} tool-call pause (like Cursor's 25). "
                            "The user can type continue to keep using tools, or stop "
                            "to answer with what you already have. Tools are not banned."
                        ),
                        tool_call_id=call_id,
                    )
                )
                continue

            used += 1
            if name in WRITE_TOOL_NAMES:
                if checkpointer is None:
                    output.append(
                        ToolMessage(
                            content=(
                                f"Error: {name} requires Postgres memory "
                                "(checkpointer). Re-run with --memory."
                            ),
                            tool_call_id=call_id,
                        )
                    )
                    continue
                tool = tools_by_name.get(name)
                if tool is None:
                    output.append(
                        ToolMessage(
                            content=f"Error: unknown tool {name}",
                            tool_call_id=call_id,
                        )
                    )
                    continue
                preview = tool.invoke(args)
                preview_text = preview if isinstance(preview, str) else str(preview)
                if preview_text.startswith("Error"):
                    output.append(ToolMessage(content=preview_text, tool_call_id=call_id))
                    continue
                pending.append(
                    {
                        "kind": WRITE_KIND_BY_TOOL[name],
                        "tool_call_id": call_id,
                        "path": str(args.get("path") or ""),
                        "old_string": str(args.get("old_string") or ""),
                        "new_string": str(args.get("new_string") or ""),
                        "content": str(args.get("content") or ""),
                        "reason": str(args.get("reason") or ""),
                        "preview": preview_text,
                    }
                )
                continue

            tool = tools_by_name.get(name)
            if tool is None:
                content = f"Error: unknown tool {name}"
            else:
                try:
                    raw = tool.invoke(args)
                    content = raw if isinstance(raw, str) else str(raw)
                except Exception as exc:
                    content = f"Error: {exc}"
            output.append(ToolMessage(content=content, tool_call_id=call_id))

        return {
            "messages": output,
            "tool_calls_used": used,
            "pending_edits": pending,
        }

    def write_review_node(state: AgentState) -> dict[str, Any]:
        pending = list(state.get("pending_edits") or [])
        if not pending:
            return {"pending_edits": []}

        decision = interrupt(
            {
                "type": "edit_approval",
                "prompt": "Type approve to apply the write(s), or reject to skip.",
                "edits": [
                    {
                        "kind": item.get("kind") or "replace",
                        "path": item["path"],
                        "reason": item.get("reason") or "",
                        "preview": item["preview"],
                    }
                    for item in pending
                ],
            }
        )
        approved = _is_approved(decision)
        output: list[AnyMessage] = []
        for item in pending:
            call_id = str(item["tool_call_id"])
            if not approved:
                output.append(
                    ToolMessage(
                        content=(
                            "User rejected this edit. The file was not changed. "
                            "Acknowledge the rejection and wait for new instructions."
                        ),
                        tool_call_id=call_id,
                    )
                )
                continue
            try:
                applied = _apply_pending_write(repo, item)
                content = (
                    "STATUS: APPLIED. The filesystem was updated. "
                    "Tell the user these writes are done. Do not say the edit is "
                    "still pending or waiting for approve/reject.\n"
                    f"{applied}"
                )
            except WorkspaceAccessError as exc:
                content = f"Error applying write: {exc}"
            except Exception as exc:
                content = f"Error applying write: {exc}"
            output.append(ToolMessage(content=content, tool_call_id=call_id))
        return {"messages": output, "pending_edits": []}

    def tool_limit_node(state: AgentState) -> dict[str, Any]:
        used = int(state.get("tool_calls_used", 0))
        if used < max_tool_calls:
            return {}
        decision = interrupt(
            {
                "type": "tool_limit",
                "used": used,
                "limit": max_tool_calls,
                "prompt": (
                    f"Paused after {used} tool calls (limit {max_tool_calls}). "
                    "Type continue to keep going, or stop to answer with what you have."
                ),
            }
        )
        if _wants_continue(decision):
            return {"tool_calls_used": 0}
        return {}

    def after_agent(state: AgentState) -> Literal["tools_read", "__end__"]:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools_read"
        return END

    def after_tools_read(
        state: AgentState,
    ) -> Literal["write_review", "tool_limit", "agent"]:
        if state.get("pending_edits"):
            return "write_review"
        if int(state.get("tool_calls_used", 0)) >= max_tool_calls:
            return "tool_limit"
        return "agent"

    def after_write_review(
        state: AgentState,
    ) -> Literal["tool_limit", "agent"]:
        if int(state.get("tool_calls_used", 0)) >= max_tool_calls:
            return "tool_limit"
        return "agent"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools_read", tools_read_node)
    graph.add_node("write_review", write_review_node)
    graph.add_node("tool_limit", tool_limit_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent)
    graph.add_conditional_edges("tools_read", after_tools_read)
    graph.add_conditional_edges("write_review", after_write_review)
    graph.add_edge("tool_limit", "agent")
    return graph.compile(checkpointer=checkpointer)


def _apply_pending_write(repo: Path, item: dict[str, Any]) -> str:
    kind = str(item.get("kind") or "replace")
    path = str(item.get("path") or "")
    if kind == "create":
        applied = apply_create(repo, path, str(item.get("content") or ""))
        return f"Created.\n{applied}"
    if kind == "delete":
        applied = apply_delete(repo, path)
        return f"Deleted.\n{applied}"
    applied = apply_replace(
        repo,
        path,
        str(item.get("old_string") or ""),
        str(item.get("new_string") or ""),
    )
    return f"Edit applied.\n{applied}"


def _is_approved(decision: Any) -> bool:
    if decision is True:
        return True
    if isinstance(decision, str):
        return decision.strip().lower() in {"approve", "approved", "yes", "y", "a"}
    if isinstance(decision, dict):
        action = str(decision.get("action") or decision.get("value") or "")
        return action.strip().lower() in {"approve", "approved", "yes", "y", "a"}
    return False


def _wants_continue(decision: Any) -> bool:
    if isinstance(decision, str):
        return decision.strip().lower() in {"continue", "c", "resume", "go"}
    if isinstance(decision, dict):
        action = str(decision.get("action") or decision.get("value") or "")
        return action.strip().lower() in {"continue", "c", "resume", "go"}
    return False


def _message_text(message: AnyMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or ""))
        return "".join(parts).strip()
    return str(content)


def format_interrupt_payload(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "Paused; resume to continue."
    if payload.get("type") == "tool_limit":
        prompt = str(
            payload.get("prompt")
            or "Tool-call pause. Type continue to keep going, or stop to answer."
        )
        return f"{prompt}\n\ncontinue / stop"
    lines = [str(payload.get("prompt") or "Approve or reject the proposed edit.")]
    for item in payload.get("edits") or []:
        path = item.get("path") or "(unknown path)"
        reason = item.get("reason") or ""
        preview = item.get("preview") or ""
        header = f"\n--- {path} ---"
        if reason:
            header += f"\n{reason}"
        lines.append(f"{header}\n{preview}".rstrip())
    lines.append("\napprove / reject")
    return "\n".join(lines)


def _interrupt_payload(graph, config: dict[str, Any]) -> dict[str, Any] | None:
    snapshot = graph.get_state(config)
    values: list[Any] = []
    for task in snapshot.tasks:
        for item in task.interrupts:
            values.append(item.value)
    if not values:
        return None
    first = values[0]
    return first if isinstance(first, dict) else {"value": first}


def _turn_from_graph(graph, config: dict[str, Any], thread_id: str) -> AgentTurn:
    payload = _interrupt_payload(graph, config)
    if payload is not None:
        return AgentTurn(
            text=format_interrupt_payload(payload),
            interrupted=True,
            payload=payload,
            thread_id=thread_id,
        )
    snapshot = graph.get_state(config)
    messages = (snapshot.values or {}).get("messages") or []
    last = messages[-1] if messages else None
    text = _message_text(last) if last is not None else ""
    return AgentTurn(text=text, interrupted=False, thread_id=thread_id)


@contextmanager
def _postgres_checkpointer(conn_string: str) -> Iterator[PostgresSaver]:
    try:
        with PostgresSaver.from_conn_string(conn_string) as saver:
            saver.setup()
            yield saver
    except Exception as exc:
        raise AgentError(
            f"could not connect to Postgres checkpointer: {exc}"
        ) from exc


def default_thread_id(repo: Path) -> str:
    return f"repo:{repo.as_posix()}"


def _database_url(explicit: str | None) -> str:
    load_dotenv()
    url = explicit or os.environ.get("DATABASE_URL")
    if not url:
        raise AgentError("DATABASE_URL is not set")
    return url


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _reasoning_text(message: AnyMessage) -> str:
    extra = getattr(message, "additional_kwargs", None) or {}
    for key in ("reasoning", "reasoning_content", "thinking"):
        value = extra.get(key)
        if value:
            return str(value)
    content = getattr(message, "content", "")
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in {"reasoning", "thinking"}:
                parts.append(str(block.get("text") or block.get("thinking") or ""))
        return "\n".join(p for p in parts if p)
    return ""


def _tool_args_text(args: Any) -> str:
    if not isinstance(args, dict):
        return str(args)
    compact = {}
    for key, value in args.items():
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        compact[key] = _clip(str(text), 120)
    return json.dumps(compact, ensure_ascii=False)


def emit_trace(update: dict[str, Any], on_trace: TraceFn) -> None:
    """Print one LangGraph stream chunk: tool calls, thinking, tool results."""
    interrupts = update.get("__interrupt__")
    if interrupts:
        first = interrupts[0] if isinstance(interrupts, (list, tuple)) else interrupts
        value = getattr(first, "value", first)
        kind = value.get("type") if isinstance(value, dict) else None
        if kind == "tool_limit":
            on_trace("⏸ interrupt: tool limit — continue / stop")
        else:
            on_trace("⏸ interrupt: waiting for approve / reject")
        return
    for node, delta in update.items():
        if node.startswith("__"):
            continue
        on_trace(f"→ {node}")
        if not isinstance(delta, dict):
            continue
        for message in delta.get("messages") or []:
            if isinstance(message, AIMessage):
                thinking = _reasoning_text(message)
                if thinking:
                    on_trace(f"  think: {_clip(thinking, _TRACE_THINK_CHARS)}")
                spoken = _message_text(message)
                if spoken and not message.tool_calls:
                    on_trace(f"  say: {_clip(spoken, _TRACE_THINK_CHARS)}")
                elif spoken and message.tool_calls:
                    on_trace(f"  note: {_clip(spoken, 240)}")
                for call in message.tool_calls or []:
                    name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
                    args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
                    on_trace(f"  tool {name}({_tool_args_text(args)})")
            elif isinstance(message, ToolMessage):
                on_trace(
                    f"  ← {_clip(_message_text(message), _TRACE_TOOL_CHARS)}"
                )
        pending = delta.get("pending_edits") or []
        if pending:
            on_trace(f"  pending edits: {len(pending)}")


def _run_with_graph(
    graph,
    config: dict[str, Any] | None,
    payload: dict[str, Any] | Command,
    thread_id: str,
    on_trace: TraceFn | None = None,
) -> AgentTurn:
    last_text = ""
    kwargs: dict[str, Any] = {"stream_mode": "updates"}
    if config is not None:
        kwargs["config"] = config
    for chunk in graph.stream(payload, **kwargs):
        if isinstance(chunk, dict):
            if on_trace:
                emit_trace(chunk, on_trace)
            for delta in chunk.values():
                if not isinstance(delta, dict):
                    continue
                for message in delta.get("messages") or []:
                    if isinstance(message, AIMessage) and not message.tool_calls:
                        last_text = _message_text(message)
    if config is not None:
        return _turn_from_graph(graph, config, thread_id)
    return AgentTurn(text=last_text, thread_id=thread_id)


def run_agent(
    repo_path: str | Path,
    question: str,
    model: BaseChatModel,
    *,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    thread_id: str | None = None,
    database_url: str | None = None,
    memory: bool = True,
    on_trace: TraceFn | None = None,
) -> AgentTurn:
    """Resolve the repo, bind workspace tools, and run one user turn."""
    if not question.strip():
        raise AgentError("question cannot be empty")
    repo = _resolve_repo_root(repo_path)
    tools = make_repo_tools(repo)
    thread = thread_id or default_thread_id(repo)

    if not memory:
        graph = build_agent_graph(model, tools, repo, max_tool_calls)
        return _run_with_graph(
            graph,
            None,
            _new_turn_payload(question, has_history=False),
            thread,
            on_trace=on_trace,
        )

    db_url = _database_url(database_url)
    try:
        ensure_index(repo, database_url=db_url)
    except RetrievalError as exc:
        raise AgentError(str(exc)) from exc
    with _postgres_checkpointer(db_url) as checkpointer:
        graph = build_agent_graph(
            model, tools, repo, max_tool_calls, checkpointer=checkpointer
        )
        config = {"configurable": {"thread_id": thread}}
        snapshot = graph.get_state(config)
        if _interrupt_payload(graph, config) is not None:
            raise AgentError(
                "this thread is paused; resume with continue/stop or "
                "approve/reject before asking again"
            )
        has_history = bool((snapshot.values or {}).get("messages"))
        return _run_with_graph(
            graph,
            config,
            _new_turn_payload(question, has_history=has_history),
            thread,
            on_trace=on_trace,
        )


def resume_agent(
    repo_path: str | Path,
    model: BaseChatModel,
    *,
    action: str,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    thread_id: str | None = None,
    database_url: str | None = None,
    on_trace: TraceFn | None = None,
) -> AgentTurn:
    """Resume a graph paused on edit approval or the tool-call budget."""
    repo = _resolve_repo_root(repo_path)
    tools = make_repo_tools(repo)
    thread = thread_id or default_thread_id(repo)
    with _postgres_checkpointer(_database_url(database_url)) as checkpointer:
        graph = build_agent_graph(
            model, tools, repo, max_tool_calls, checkpointer=checkpointer
        )
        config = {"configurable": {"thread_id": thread}}
        if _interrupt_payload(graph, config) is None:
            raise AgentError("this thread is not paused")
        return _run_with_graph(
            graph,
            config,
            Command(resume={"action": action}),
            thread,
            on_trace=on_trace,
        )


def _new_turn_payload(question: str, *, has_history: bool) -> dict[str, Any]:
    messages: list[AnyMessage] = []
    if not has_history:
        messages.append(SystemMessage(content=SYSTEM_PROMPT))
    messages.append(HumanMessage(content=question))
    return {
        "messages": messages,
        "tool_calls_used": 0,
        "pending_edits": [],
    }
