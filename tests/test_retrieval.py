import hashlib
from pathlib import Path

from coding_copilot_copy.retrieval import (
    _iter_indexable_files,
    chunk_source,
    ensure_index,
    hybrid_search,
    _rrf,
)


def _fake_embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vectors.append([((digest[i % 32] / 255.0) - 0.5) for i in range(1536)])
    return vectors


def test_chunk_splits_long_files_with_overlap() -> None:
    text = "\n".join(f"line {i}" for i in range(1, 201))
    chunks = chunk_source("long.py", text)
    assert len(chunks) == 2
    assert chunks[0]["start_line"] == 1 and chunks[0]["end_line"] == 120
    assert chunks[1]["start_line"] == 101 and chunks[1]["end_line"] == 200


def test_index_size_limits_by_suffix(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    # 600 KiB source: allowed (2 MiB cap)
    (repo / "Big.java").write_bytes(b"x" * (600 * 1024))
    # 600 KiB json: skipped (512 KiB cap)
    (repo / "big.json").write_bytes(b"x" * (600 * 1024))
    # lockfile: never indexed
    (repo / "package-lock.json").write_text('{"name":"x"}\n', encoding="utf-8")
    paths = {rel for rel, _ in _iter_indexable_files(repo)}
    assert "Big.java" in paths
    assert "big.json" not in paths
    assert "package-lock.json" not in paths


def test_chunk_breaks_on_blank_line_when_possible() -> None:
    body = [f"x{i}" for i in range(1, 111)]
    body.append("")
    body.extend(f"y{i}" for i in range(1, 25))
    chunks = chunk_source("mix.txt", "\n".join(body))
    assert chunks[0]["end_line"] == 110
    assert chunks[0]["start_line"] == 1


def test_rrf_prefers_items_high_in_both_lists() -> None:
    a = ("a", {"path": "a"})
    b = ("b", {"path": "b"})
    fused = _rrf([[a, b], [a, b]])
    assert fused[0]["path"] == "a"


def test_hybrid_search_finds_lexical_token(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("page_size = 10\nprint(page_size)\n")
    ensure_index(repo, embed=_fake_embed)
    out = hybrid_search(repo, "page_size", embed=_fake_embed)
    assert "sample.py" in out
