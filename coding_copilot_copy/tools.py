"""
wrapped the functions in workplace.py into langgraph tools which allows the agent to use them.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from coding_copilot_copy.retrieval import RetrievalError, ensure_index, hybrid_search
from coding_copilot_copy.workspace import (
    WorkspaceAccessError,
    glob_files,
    preview_create,
    preview_delete,
    preview_replace,
    read_file,
    search_repo,
)

SEARCH_DESCRIPTION = (
    "Exact substring search. Good when you already know a token "
    "(DailyNewsAgent, SystemMessage). For vague questions prefer codebase_search."
)

CODEBASE_DESCRIPTION = (
    "Hybrid semantic + lexical search. Good first move for natural-language "
    "questions. Returns path:line ranges; follow with read_file on useful hits."
)

GLOB_DESCRIPTION = (
    "Find files by glob on the relative path, e.g. **/*Agent*.java or **/*.py."
)

READ_DESCRIPTION = (
    "Read a numbered slice of a text file by relative path. At most 200 lines. "
    "Read before editing an existing file."
)

PROPOSE_EDIT_DESCRIPTION = (
    "Change an existing file by replacing a unique old_string with new_string. "
    "Use this to add, rewrite, or delete a snippet inside a file. Does not write "
    "until the user approves. Copy old_string verbatim from read_file."
)

PROPOSE_CREATE_DESCRIPTION = (
    "Create a new file at a relative path that does not exist yet. Provide the "
    "full file contents. Does not write until the user approves."
)

PROPOSE_DELETE_DESCRIPTION = (
    "Delete an entire existing file at a relative path. Does not delete until "
    "the user approves. Prefer propose_edit if you only want to remove some lines."
)

WRITE_TOOL_NAME = "propose_edit"
WRITE_KIND_BY_TOOL = {
    "propose_edit": "replace",
    "propose_create": "create",
    "propose_delete": "delete",
}
WRITE_TOOL_NAMES = frozenset(WRITE_KIND_BY_TOOL)


class SearchInput(BaseModel):
    query: str = Field(..., description="Exact text to find")
    max_results: int = Field(default=20, description="Max hits (1-50). Default 20.")


class CodebaseInput(BaseModel):
    query: str = Field(..., description="Natural-language or keyword query")
    max_results: int = Field(default=8, description="Max fused hits (1-20). Default 8.")


class GlobInput(BaseModel):
    pattern: str = Field(..., description="Glob such as **/*Agent*.java")
    max_results: int = Field(default=50, description="Max paths (1-100). Default 50.")


class ReadInput(BaseModel):
    path: str = Field(..., description="Relative path to read")
    start_line: int = Field(default=1, description="Start line")
    end_line: int = Field(default=None, description="End line")


class ProposeEditInput(BaseModel):
    path: str = Field(..., description="Existing file, relative to repo root")
    old_string: str = Field(..., description="Exact unique snippet to replace")
    new_string: str = Field(..., description="Replacement; empty string deletes the snippet")
    reason: str = Field(default="", description="Short reason")


class ProposeCreateInput(BaseModel):
    path: str = Field(..., description="New file path; must not already exist")
    content: str = Field(..., description="Full contents of the new file")
    reason: str = Field(default="", description="Short reason")


class ProposeDeleteInput(BaseModel):
    path: str = Field(..., description="Existing file to delete")
    reason: str = Field(default="", description="Short reason")


def _pending_note(reason: str, preview: str, kind: str) -> str:
    note = f"Reason: {reason}\n" if reason.strip() else ""
    return (
        f"{note}{preview}"
        f"Waiting for the user to approve or reject this {kind}. "
        "Do not claim the filesystem already changed."
    )


def make_repo_tools(repo: Path) -> list[BaseTool]:
    """Build search / read / write-proposal tools locked to ``repo``."""

    def _codebase(query: str, max_results: int = 8) -> str:
        try:
            ensure_index(repo)
            return hybrid_search(repo, query, max_results=max_results)
        except (RetrievalError, WorkspaceAccessError) as e:
            return f"Error in codebase_search: {e}"

    def _glob(pattern: str, max_results: int = 50) -> str:
        try:
            return glob_files(repo, pattern, max_results)
        except WorkspaceAccessError as e:
            return f"Error listing files: {e}"

    def _search(query: str, max_results: int = 20) -> str:
        try:
            return search_repo(repo, query, max_results)
        except WorkspaceAccessError as e:
            return f"Error searching repository: {e}"

    def _read(path: str, start_line: int = 1, end_line: int | None = None) -> str:
        try:
            return read_file(repo, path, start_line, end_line)
        except WorkspaceAccessError as e:
            return f"Error reading file: {e}"

    def _propose_edit(
        path: str, old_string: str, new_string: str, reason: str = ""
    ) -> str:
        try:
            preview = preview_replace(repo, path, old_string, new_string)
        except WorkspaceAccessError as e:
            return f"Error proposing edit: {e}"
        return _pending_note(reason, preview, "edit")

    def _propose_create(path: str, content: str, reason: str = "") -> str:
        try:
            preview = preview_create(repo, path, content)
        except WorkspaceAccessError as e:
            return f"Error proposing create: {e}"
        return _pending_note(reason, preview, "create")

    def _propose_delete(path: str, reason: str = "") -> str:
        try:
            preview = preview_delete(repo, path)
        except WorkspaceAccessError as e:
            return f"Error proposing delete: {e}"
        return _pending_note(reason, preview, "delete")

    return [
        StructuredTool.from_function(
            func=_codebase,
            name="codebase_search",
            description=CODEBASE_DESCRIPTION,
            args_schema=CodebaseInput,
        ),
        StructuredTool.from_function(
            func=_glob,
            name="glob_files",
            description=GLOB_DESCRIPTION,
            args_schema=GlobInput,
        ),
        StructuredTool.from_function(
            func=_search,
            name="search_code",
            description=SEARCH_DESCRIPTION,
            args_schema=SearchInput,
        ),
        StructuredTool.from_function(
            func=_read,
            name="read_file",
            description=READ_DESCRIPTION,
            args_schema=ReadInput,
        ),
        StructuredTool.from_function(
            func=_propose_edit,
            name="propose_edit",
            description=PROPOSE_EDIT_DESCRIPTION,
            args_schema=ProposeEditInput,
        ),
        StructuredTool.from_function(
            func=_propose_create,
            name="propose_create",
            description=PROPOSE_CREATE_DESCRIPTION,
            args_schema=ProposeCreateInput,
        ),
        StructuredTool.from_function(
            func=_propose_delete,
            name="propose_delete",
            description=PROPOSE_DELETE_DESCRIPTION,
            args_schema=ProposeDeleteInput,
        ),
    ]


def tool_names(tools: list[BaseTool] | None = None) -> list[str]:
    items = tools if tools is not None else make_repo_tools(Path("."))
    return [tool.name for tool in items]
