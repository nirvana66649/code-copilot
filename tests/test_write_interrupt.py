from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from coding_copilot_copy.agent import build_agent_graph
from coding_copilot_copy.tools import make_repo_tools


class FakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "propose_edit",
                        "args": {
                            "path": "sample.py",
                            "old_string": "page_size = 10",
                            "new_string": "page_size = 20",
                            "reason": "bump default",
                        },
                    }
                ],
            )
        return AIMessage(content="applied")


def test_interrupt_then_approve(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    sample = repo / "sample.py"
    sample.write_text("page_size = 10\n", encoding="utf-8")

    checkpointer = InMemorySaver()
    graph = build_agent_graph(
        FakeModel(),
        make_repo_tools(repo),
        repo,
        max_tool_calls=8,
        checkpointer=checkpointer,
    )
    config = {"configurable": {"thread_id": "t1"}}
    graph.invoke(
        {
            "messages": [
                SystemMessage(content="sys"),
                HumanMessage(content="change it"),
            ],
            "tool_calls_used": 0,
            "pending_edits": [],
        },
        config=config,
    )
    snap = graph.get_state(config)
    interrupts = [item.value for task in snap.tasks for item in task.interrupts]
    assert interrupts
    assert sample.read_text(encoding="utf-8") == "page_size = 10\n"

    graph.invoke(Command(resume={"action": "approve"}), config=config)
    assert sample.read_text(encoding="utf-8") == "page_size = 20\n"
