"""Command line: ask one question or chat about a local repository."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

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

app = typer.Typer(add_completion=False, no_args_is_help=True)

RepoOption = Annotated[
    Path,
    typer.Option(
        "--repo",
        help="仓库根目录",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
]
MaxToolCallsOption = Annotated[
    int,
    typer.Option(
        "--max-tool-calls",
        min=1,
        help="每轮工具次数上限，打满后暂停等 continue / stop（默认 25）",
    ),
]
ThreadOption = Annotated[
    str | None,
    typer.Option("--thread", help="会话 id，默认按仓库路径隔离"),
]
MemoryOption = Annotated[
    bool,
    typer.Option("--memory/--no-memory", help="是否把对话写入 Postgres checkpointer"),
]


@app.callback()
def _root() -> None:
    """Local coding assistant."""


def _on_trace(line: str) -> None:
    typer.secho(line, fg="cyan")


def _print_turn(turn: AgentTurn) -> None:
    if turn.interrupted:
        kind = (turn.payload or {}).get("type")
        label = "tool limit" if kind == "tool_limit" else "edit review"
        typer.secho(f"← {label}", fg="yellow")
    else:
        typer.secho("← answer", fg="green")
    typer.echo(turn.text)


def _interrupt_kind(turn: AgentTurn) -> str:
    return str((turn.payload or {}).get("type") or "edit_approval")


def _resolve_interrupt_action(raw: str, kind: str) -> str | None:
    value = raw.strip().lower()
    if kind == "tool_limit":
        if value in {"continue", "c", "resume", "go"}:
            return "continue"
        if value in {"stop", "s"}:
            return "stop"
        return None
    if value in {"approve", "approved", "yes", "y", "a"}:
        return "approve"
    if value in {"reject", "rejected", "no", "n", "r"}:
        return "reject"
    return None


def _resume_action_from_user(raw: str) -> str | None:
    value = raw.strip().lower()
    mapping = {
        "continue": "continue",
        "c": "continue",
        "resume": "continue",
        "go": "continue",
        "stop": "stop",
        "s": "stop",
        "approve": "approve",
        "approved": "approve",
        "a": "approve",
        "reject": "reject",
        "rejected": "reject",
        "r": "reject",
    }
    return mapping.get(value)


def _handle_interrupts(
    turn: AgentTurn,
    repo: Path,
    model,
    *,
    max_tool_calls: int,
) -> AgentTurn:
    while turn.interrupted:
        _print_turn(turn)
        kind = _interrupt_kind(turn)
        prompt_label = "limit" if kind == "tool_limit" else "edit"
        hint = (
            "Type continue or stop."
            if kind == "tool_limit"
            else "Type approve or reject."
        )
        try:
            choice = typer.prompt(prompt_label)
        except typer.Abort:
            raise AgentError("paused turn cancelled") from None
        action = _resolve_interrupt_action(choice, kind)
        if action is None:
            typer.echo(hint)
            continue
        turn = resume_agent(
            repo,
            model,
            action=action,
            max_tool_calls=max_tool_calls,
            thread_id=turn.thread_id,
            on_trace=_on_trace,
        )
    return turn


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="要问这个仓库的问题")],
    repo: RepoOption = Path("."),
    max_tool_calls: MaxToolCallsOption = DEFAULT_MAX_TOOL_CALLS,
    thread: ThreadOption = None,
    memory: MemoryOption = True,
) -> None:
    """问一轮并退出。写入等 approve/reject；工具打满等 continue/stop。"""
    try:
        model = model_from_env()
        turn = run_agent(
            repo,
            question,
            model,
            max_tool_calls=max_tool_calls,
            thread_id=thread,
            memory=memory,
            on_trace=_on_trace,
        )
        turn = _handle_interrupts(
            turn, repo, model, max_tool_calls=max_tool_calls
        )
    except (AgentError, ModelError, WorkspaceAccessError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    _print_turn(turn)


@app.command()
def chat(
    repo: RepoOption = Path("."),
    max_tool_calls: MaxToolCallsOption = DEFAULT_MAX_TOOL_CALLS,
    thread: ThreadOption = None,
    memory: MemoryOption = True,
) -> None:
    """多轮对话。写入等 approve/reject；工具打满等 continue/stop。"""
    try:
        model = model_from_env()
        repo_root = _resolve_repo_root(repo)
    except (ModelError, WorkspaceAccessError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    session = thread or default_thread_id(repo_root)
    typer.echo(f"repo: {repo_root}")
    typer.echo(f"thread: {session}")
    typer.echo("Type a question. quit / exit / Ctrl-C to stop.")

    while True:
        try:
            question = typer.prompt("you")
        except typer.Abort:
            typer.echo("")
            break
        if question.strip().lower() in {"q", "quit", "exit"}:
            break
        if not question.strip():
            continue
        resume_action = _resume_action_from_user(question)
        if resume_action is not None:
            try:
                turn = resume_agent(
                    repo_root,
                    model,
                    action=resume_action,
                    max_tool_calls=max_tool_calls,
                    thread_id=session,
                    on_trace=_on_trace,
                )
                turn = _handle_interrupts(
                    turn, repo_root, model, max_tool_calls=max_tool_calls
                )
            except (AgentError, WorkspaceAccessError) as exc:
                if "not paused" not in str(exc):
                    typer.echo(str(exc), err=True)
                    continue
            else:
                _print_turn(turn)
                continue
        try:
            turn = run_agent(
                repo_root,
                question,
                model,
                max_tool_calls=max_tool_calls,
                thread_id=session,
                memory=memory,
                on_trace=_on_trace,
            )
            turn = _handle_interrupts(
                turn, repo_root, model, max_tool_calls=max_tool_calls
            )
        except (AgentError, WorkspaceAccessError) as exc:
            typer.echo(str(exc), err=True)
            continue
        _print_turn(turn)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
