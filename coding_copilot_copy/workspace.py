"""
it is a sand box, allowing agent to access the local repositories, base on the repo and the path

"""

from __future__ import annotations

import difflib
import fnmatch
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

# static variables
MAX_READ_LINES = 200
DEFAULT_SEARCH_RESULTS = 20
MAX_SEARCH_RESULTS = 50
_MAX_QUERY_CHARS = 200
_MAX_MATCH_CHARS = 400
_SEARCH_TIMEOUT_SECONDS = 30
_MAX_REPLACE_CHARS = 8000
_MAX_FILE_CHARS = 100_000
DEFAULT_GLOB_RESULTS = 50
MAX_GLOB_RESULTS = 100
_SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        "node_modules",
        "target",
        ".venv",
        "venv",
        "dist",
        "build",
        "__pycache__",
        ".idea",
        ".mvn",
    }
)

# not allowed names
_RESTRICTED_NAMES = frozenset(
    {
        ".git",
        ".env",
        ".ssh",
        ".aws",
        ".gnupg",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "credentials.json",
    }
)
_RESTRICTED_SUFFIXES = (".pem", ".key")

class WorkspaceAccessError(Exception):
    """
    Exception raised when there is an error accessing the workspace.
    """
    pass

# read a file and return the content with the line numebr, 
def read_file(
    repo_path: str|Path,
    rel_path: str,
    start_line: int = 1,
    end_line: int | None = None,
) -> str:
    """Read a file and return the content with the line numebr, 
    the first line is the relative path+line numebr range, 
    and follow by the source codes with line numebr"""

    # get the repo, check if it is safe, if the rel_path is the relative path to the repo_root
    repo_root, file_path = _resolve_repo_file(repo_path, rel_path)
    #turn the absolute path to the relative path to the repo_root
    display_path = file_path.relative_to(repo_root).as_posix()

    if start_line < 1:
        raise WorkspaceAccessError(f"Start line must be at least 1, got {start_line}")
    if end_line is not None and end_line < start_line:
        raise WorkspaceAccessError(f"End line must be greater than or equal to start line, got {end_line} < {start_line}")
    
    # limit to at most 100 lines
    
    if end_line is None:
        end_line = start_line + MAX_READ_LINES - 1
    elif end_line - start_line +1 > MAX_READ_LINES:
        end_line = start_line + MAX_READ_LINES - 1
        

    # start to read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        raise WorkspaceAccessError(f"File {display_path} is empty")
    if start_line > len(lines):
        raise WorkspaceAccessError(f"Start line is out of range for file {display_path}, got {start_line} > {len(lines)}")
    selected_lines = lines[start_line-1:end_line]
    rows = [
        (start_line + i, line.rstrip("\r\n"))
        for i, line in enumerate(selected_lines)
    ]
    header = f"{display_path}:{rows[0][0]}-{rows[-1][0]}"
    body = "\n".join(f"{n}| {text}" for n, text in rows)
    return f"{header}\n{body}"



# search the keyword in the repo, get the result with the line numebr and the file path
def search_repo(
    repo_path: str|Path,
    query: str,
    max_results: int = DEFAULT_SEARCH_RESULTS,
) -> str:
    """Search the keyword in the repo, get the result with the line numebr and the file path 
    Search the repo with ripgrep and return ``path:line: text`` hits."""
    repo_root = _resolve_repo_root(repo_path)
    #check query
    if not query.strip():
        raise WorkspaceAccessError("Query cannot be empty")
    elif len(query) > _MAX_QUERY_CHARS:
        raise WorkspaceAccessError(f"Query is too long, got {len(query)} characters, max is {_MAX_QUERY_CHARS}")
    if not repo_root:
        raise WorkspaceAccessError(f"Repository path {repo_path} is not a valid repository")
    if not isinstance(max_results, int) or isinstance(max_results, bool) or max_results < 1:
        raise WorkspaceAccessError("max_results must be an integer >= 1")
    limited_to = min(max_results, MAX_SEARCH_RESULTS)
    command = [
        _rg_binary(),
        "--json",
        "-F",
        "--max-count",
        str(limited_to + 1),
        "--max-filesize",
        "1M",
        "--glob",
        "!.git/**",
        "--glob",
        "!**/.git/**",
        "--",
        query,
        ".",
    ]
    try:
        result = subprocess.run(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=_SEARCH_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as e:
        raise WorkspaceAccessError(f"Search timed out after {_SEARCH_TIMEOUT_SECONDS} seconds")
    except FileNotFoundError as e:
        raise WorkspaceAccessError(f"ripgrep binary not found: {e}") from e
    
    hits: list[str] = []
    truncated = False
    for line in result.stdout.splitlines():

        hit = _parse_search_hit(repo_root,line)
        if not hit:
            continue
        if len(hits) >= limited_to:
            truncated = True
            break
        hits.append(hit)
    
    if result.returncode not in (0, 1) and not hits:
        detail = result.stderr.strip() or f"ripgrep exited {result.returncode}"
        raise WorkspaceAccessError(f"search failed: {detail}")
    if not hits:
        return "No results found"
    header = f"Search results for '{query}'"
    body = "\n".join(hits)
    if truncated:
        return f"{body}\n(showing {len(hits)} matches; narrow the query for more)" 
    return f"{header}\n{body}"


def glob_files(
    repo_path: str | Path,
    pattern: str,
    max_results: int = DEFAULT_GLOB_RESULTS,
) -> str:
    """Return relative paths matching a glob (fnmatch on the posix relative path)."""
    repo_root = _resolve_repo_root(repo_path)
    if not isinstance(pattern, str) or not pattern.strip():
        raise WorkspaceAccessError("glob pattern is required")
    if not isinstance(max_results, int) or isinstance(max_results, bool) or max_results < 1:
        raise WorkspaceAccessError("max_results must be an integer >= 1")
    limited_to = min(max_results, MAX_GLOB_RESULTS)
    needle = pattern.strip().lstrip("./")
    hits: list[str] = []
    truncated = False
    for path in repo_root.rglob("*"):
        if _is_skipped_dir(path, repo_root):
            continue
        if not path.is_file():
            continue
        rel = path.relative_to(repo_root).as_posix()
        try:
            _reject_restricted(Path(rel), rel)
        except WorkspaceAccessError:
            continue
        if not (
            fnmatch.fnmatch(rel, needle)
            or fnmatch.fnmatch(path.name, needle)
            or fnmatch.fnmatch(rel, f"**/{needle}")
        ):
            continue
        hits.append(rel)
        if len(hits) >= limited_to:
            truncated = True
            break
    if not hits:
        return f"No files matching '{pattern}'"
    body = "\n".join(hits)
    if truncated:
        return f"{body}\n(showing {len(hits)} paths; narrow the glob for more)"
    return f"Files matching '{pattern}'\n{body}"


def _is_skipped_dir(path: Path, repo_root: Path) -> bool:
    try:
        relative = path.relative_to(repo_root)
    except ValueError:
        return True
    return any(part in _SKIP_DIR_NAMES or _is_restricted_name(part) for part in relative.parts)


def preview_replace(
    repo_path: str | Path,
    rel_path: str,
    old_string: str,
    new_string: str,
) -> str:
    """Return a unified diff for a unique substring replace. Does not write."""
    display_path, original, updated = _prepare_replace(
        repo_path, rel_path, old_string, new_string
    )
    return _format_replace_preview(display_path, original, updated)


def apply_replace(
    repo_path: str | Path,
    rel_path: str,
    old_string: str,
    new_string: str,
) -> str:
    """Apply a unique substring replace and return the same preview as confirmation."""
    display_path, original, updated = _prepare_replace(
        repo_path, rel_path, old_string, new_string
    )
    repo_root, file_path = _resolve_repo_file(repo_path, rel_path)
    file_path.write_text(updated, encoding="utf-8")
    return _format_file_diff(
        display_path, original, updated, f"Proposed edit for {display_path} (1 unique match):"
    )


def preview_create(repo_path: str | Path, rel_path: str, content: str) -> str:
    display_path, file_path, body = _prepare_create(repo_path, rel_path, content)
    return _format_file_diff(
        display_path, "", body, f"Proposed new file {display_path}:"
    )


def apply_create(repo_path: str | Path, rel_path: str, content: str) -> str:
    display_path, file_path, body = _prepare_create(repo_path, rel_path, content)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(body, encoding="utf-8")
    return _format_file_diff(
        display_path, "", body, f"Created file {display_path}:"
    )


def preview_delete(repo_path: str | Path, rel_path: str) -> str:
    display_path, file_path, original = _prepare_delete(repo_path, rel_path)
    return _format_file_diff(
        display_path, original, "", f"Proposed delete of {display_path}:"
    )


def apply_delete(repo_path: str | Path, rel_path: str) -> str:
    display_path, file_path, original = _prepare_delete(repo_path, rel_path)
    file_path.unlink()
    return _format_file_diff(
        display_path, original, "", f"Deleted file {display_path}:"
    )


def _prepare_create(
    repo_path: str | Path, rel_path: str, content: str
) -> tuple[str, Path, str]:
    if not isinstance(content, str):
        raise WorkspaceAccessError("content must be a string")
    if not content:
        raise WorkspaceAccessError("new file content cannot be empty")
    if len(content) > _MAX_FILE_CHARS:
        raise WorkspaceAccessError(
            f"file content is too long, got {len(content)} characters, max is {_MAX_FILE_CHARS}"
        )
    repo_root, file_path = _resolve_new_file(repo_path, rel_path)
    display_path = file_path.relative_to(repo_root).as_posix()
    return display_path, file_path, content


def _prepare_delete(repo_path: str | Path, rel_path: str) -> tuple[str, Path, str]:
    repo_root, file_path = _resolve_repo_file(repo_path, rel_path)
    display_path = file_path.relative_to(repo_root).as_posix()
    try:
        original = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise WorkspaceAccessError(
            f"File {display_path} is not valid UTF-8 text"
        ) from exc
    return display_path, file_path, original


def _resolve_new_file(repo_path: str | Path, rel_path: str) -> tuple[Path, Path]:
    repo_root = _resolve_repo_root(repo_path)
    relative = _parse_relative_path(rel_path)
    _reject_restricted(relative, rel_path)
    real_file = _realpath_inside(repo_root, relative, rel_path)
    _reject_restricted(real_file.relative_to(repo_root), rel_path)
    if real_file.exists():
        raise WorkspaceAccessError(f"File {rel_path} already exists")
    return repo_root, real_file


def _prepare_replace(
    repo_path: str | Path,
    rel_path: str,
    old_string: str,
    new_string: str,
) -> tuple[str, str, str]:
    if not isinstance(old_string, str) or not old_string:
        raise WorkspaceAccessError("old_string is required and cannot be empty")
    if not isinstance(new_string, str):
        raise WorkspaceAccessError("new_string must be a string")
    if old_string == new_string:
        raise WorkspaceAccessError("old_string and new_string are identical")
    if len(old_string) > _MAX_REPLACE_CHARS:
        raise WorkspaceAccessError(
            f"old_string is too long, got {len(old_string)} characters, max is {_MAX_REPLACE_CHARS}"
        )
    if len(new_string) > _MAX_REPLACE_CHARS:
        raise WorkspaceAccessError(
            f"new_string is too long, got {len(new_string)} characters, max is {_MAX_REPLACE_CHARS}"
        )

    repo_root, file_path = _resolve_repo_file(repo_path, rel_path)
    display_path = file_path.relative_to(repo_root).as_posix()
    try:
        original = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise WorkspaceAccessError(
            f"File {display_path} is not valid UTF-8 text"
        ) from exc

    matches = original.count(old_string)
    if matches == 0:
        raise WorkspaceAccessError(
            f"old_string was not found in {display_path}; read the file and copy an exact unique snippet"
        )
    if matches > 1:
        raise WorkspaceAccessError(
            f"old_string matched {matches} times in {display_path}; include more surrounding lines so it is unique"
        )
    updated = original.replace(old_string, new_string, 1)
    return display_path, original, updated


def _format_replace_preview(display_path: str, original: str, updated: str) -> str:
    return _format_file_diff(
        display_path, original, updated, f"Proposed edit for {display_path} (1 unique match):"
    )


def _format_file_diff(
    display_path: str, original: str, updated: str, headline: str
) -> str:
    diff_lines = list(
        difflib.unified_diff(
            original.splitlines(),
            updated.splitlines(),
            fromfile=f"a/{display_path}",
            tofile=f"b/{display_path}",
            lineterm="",
        )
    )
    diff = "\n".join(diff_lines)
    if diff:
        diff += "\n"
    return f"{headline}\n{diff}"


def _resolve_repo_root(repo_path: str|Path) -> Path:

    """check if the repo is there or not"""
    try:
        root = Path(repo_path).expanduser().resolve(strict=True)
    except FileNotFoundError as e:
        raise WorkspaceAccessError(f"Repository path {repo_path} not found") from e
    if not root.is_dir():
        raise WorkspaceAccessError(f"Repository path {repo_path} is not a directory")
    return root

def _resolve_repo_file(repo_path: str|Path, rel_path: str) -> tuple[Path, Path]:
    """Resolve the absolute path to the file in the repository, 
    check if the file is restricted or not"""
    repo_root = _resolve_repo_root(repo_path)
    # we first check if the repo is there or not
    relative = _parse_relative_path(rel_path)
    _reject_restricted(relative,rel_path)

    real_file = _realpath_inside(repo_root,relative,rel_path)
    _reject_restricted(real_file.relative_to(repo_root), rel_path)
    if not real_file.is_file():
        raise WorkspaceAccessError(f"File {rel_path} not found in repository {repo_path}")
    return repo_root, real_file

def _parse_relative_path(rel_path: str) -> Path:
    if not isinstance(rel_path, str) or not rel_path.strip():
        raise WorkspaceAccessError("relative file path is required")
    if rel_path.startswith("~/") or rel_path == "~":
        raise WorkspaceAccessError(f"absolute path is not allowed: {rel_path}")

    parsed = Path(rel_path)
    if parsed.is_absolute():
        raise WorkspaceAccessError(f"absolute path is not allowed: {rel_path}")
    return parsed

def _reject_restricted(relative: Path, rel_path: str) -> None:
    """ reject the restricted files and directories"""
    for name in relative.parts:
        if name in (".",".."):
            continue
        if _is_restricted_name(name):
            raise WorkspaceAccessError(f"File {rel_path} is restricted")

def _is_restricted_name(name: str) -> bool:
    """ check if the name is restricted, include the restricted names, .env, .pem, .key"""
    lower = name.lower()
    if lower in _RESTRICTED_NAMES:
        return True
    if lower.startswith(".env."):
        return True
    if lower.endswith(".pem") or lower.endswith(".key"):
        return True
    return False
    

def _realpath_inside(repo_root: Path, relative: Path, rel_path: str) -> Path:
    """ get the real path of the file inside the repository"""
    real_file = (repo_root / relative).resolve(strict=False)
    if not real_file.is_absolute():
        raise WorkspaceAccessError(f"path resolves outside the repository: {rel_path}")
    if ".." in real_file.parts:
        raise WorkspaceAccessError(f"path contains ..: {rel_path}")
    if not real_file.is_relative_to(repo_root):
        raise WorkspaceAccessError(f"path resolves outside the repository: {rel_path}")
    return real_file


# find the ripgrep binary, use the environment variable CODING_COPILOT_RG if it is set, otherwise use the system path
def _rg_binary() -> str:
    override = os.environ.get("CODING_COPILOT_RG")
    if override:
        if Path(override).is_file() and os.access(override, os.X_OK):
            return override
        raise WorkspaceAccessError(
            f"CODING_COPILOT_RG is not an executable ripgrep binary: {override}"
        )
    found = shutil.which("rg")
    if found:
        return found
    for candidate in _rg_fallback_paths():
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise WorkspaceAccessError(
        "ripgrep (rg) was not found. Install with: brew install ripgrep — "
        "or set CODING_COPILOT_RG to the full path of an rg binary"
    )


def _rg_fallback_paths() -> list[Path]:
    home = Path.home()
    paths = [
        Path("/opt/homebrew/bin/rg"),
        Path("/usr/local/bin/rg"),
        home / ".cargo" / "bin" / "rg",
    ]
    paths.extend(_editor_bundled_rg_paths())
    return paths


def _editor_bundled_rg_paths() -> list[Path]:
    """Ripgrep bundled with Cursor / VS Code when ``rg`` is not on PATH."""
    module_roots = [
        Path("/Applications/Cursor.app/Contents/Resources/app/node_modules"),
        Path("/Applications/Visual Studio Code.app/Contents/Resources/app/node_modules"),
        Path("/Applications/Code.app/Contents/Resources/app/node_modules"),
    ]
    rel_paths = [
        Path("@vscode/ripgrep/bin/rg"),
        Path("@cursor/ripgrep-darwin-arm64/bin/rg"),
        Path("@cursor/ripgrep-darwin-x64/bin/rg"),
    ]
    candidates: list[Path] = []
    for root in module_roots:
        if not root.is_dir():
            continue
        for rel in rel_paths:
            candidate = root / rel
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _parse_search_hit(repo_root: Path, line: str) -> str:
    """ parse the search hit found by rg, and check if it is a safe path
        JSON parse succeeded?  
    → is `type` `"match"`?  
    → take out `path`, `line_number`, and text  
    → strip leading `./`  
    → is the path safe via the sandbox check (`_search_display_path`)?  
    → safe: return `"src/A.java:3:   int pageSize;"`  
    → unsafe: return `None` (drop it)
    """
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return None
    if event["type"] != "match":
        return None
    data = event.get("data") or {};
    # get the rel path
    rel = _rg_path_text(data.get("path"))
    line_number = data.get("line_number")
    if not rel or not isinstance(line_number, int):
        return None
    
    # check if the beginning of the path is ./
    if rel.startswith("./"):
        rel = rel[2:]
    # check if the path is safe via the sandbox check (_search_display_path)
    try:
        root, file = _resolve_repo_file(repo_root, rel)
        display_path = file.relative_to(root).as_posix()
        if not display_path:
            return None
    except WorkspaceAccessError:
        return None
    
    lines = data.get("lines") or {}
    text = lines.get("text", "") if isinstance(lines, dict) else ""
    text = str(text).replace("\r", "").replace("\n", "")
    return f"{display_path}:{line_number}: {text}"

def _rg_path_text(path: object) -> str:
    if isinstance(path, dict):
        text = path.get("text","")
        return text if isinstance(text, str) else ""
    if isinstance(path, str):
        return path.strip()
    return ""



