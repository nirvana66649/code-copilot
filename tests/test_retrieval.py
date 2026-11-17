import hashlib
from pathlib import Path

from coding_copilot_copy.retrieval import chunk_source, ensure_index, hybrid_search, _rrf


def _fake_embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vectors.append([((digest[i % 32] / 255.0) - 0.5) for i in range(1536)])
    return vectors


def test_chunk_splits_on_class_header() -> None:
    text = "package demo;\n\npublic class A {}\n\npublic class B {}\n"
    chunks = chunk_source("Demo.java", text)
    assert len(chunks) >= 2
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
