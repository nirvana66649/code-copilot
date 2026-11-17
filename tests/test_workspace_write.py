from pathlib import Path

import pytest

from coding_copilot_copy.workspace import (
    WorkspaceAccessError,
    apply_create,
    apply_delete,
    apply_replace,
    preview_create,
    preview_replace,
)


def _repo(tmp_path: Path) -> Path:
    target = tmp_path / "repo"
    target.mkdir()
    (target / "sample.py").write_text("page_size = 10\nprint(page_size)\n", encoding="utf-8")
    return target


def test_preview_does_not_write(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    preview = preview_replace(repo, "sample.py", "page_size = 10", "page_size = 20")
    assert "page_size = 20" in preview
    assert (repo / "sample.py").read_text(encoding="utf-8") == "page_size = 10\nprint(page_size)\n"


def test_apply_writes_once(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    apply_replace(repo, "sample.py", "page_size = 10", "page_size = 20")
    assert (repo / "sample.py").read_text(encoding="utf-8") == "page_size = 20\nprint(page_size)\n"


def test_replace_must_be_unique(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    with pytest.raises(WorkspaceAccessError, match="matched 2 times"):
        preview_replace(repo, "sample.py", "page_size", "n")


def test_create_then_delete_file(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    preview = preview_create(repo, "extra.py", "x = 1\n")
    assert "Proposed new file extra.py" in preview
    assert not (repo / "extra.py").exists()
    apply_create(repo, "extra.py", "x = 1\n")
    assert (repo / "extra.py").read_text(encoding="utf-8") == "x = 1\n"
    apply_delete(repo, "extra.py")
    assert not (repo / "extra.py").exists()


def test_create_rejects_existing(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    with pytest.raises(WorkspaceAccessError, match="already exists"):
        apply_create(repo, "sample.py", "nope\n")
