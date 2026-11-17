from pathlib import Path

from coding_copilot_copy.workspace import glob_files


def test_glob_agent_filename(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "DailyNewsAgent.java").write_text("class DailyNewsAgent {}\n")
    (repo / "src" / "NewsController.java").write_text("class NewsController {}\n")
    out = glob_files(repo, "*Agent*")
    assert "DailyNewsAgent.java" in out
    assert "NewsController.java" not in out
