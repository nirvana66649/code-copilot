from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from coding_copilot_copy.agent import build_agent_graph
from coding_copilot_copy.tools import make_repo_tools


class FakeModel:
    def __init__(self) -> None:
        self.calls = 0
        self.bound_tools = False

    def bind_tools(self, tools):
        self.bound_tools = True
        return self

    def invoke(self, messages):
        self.calls += 1
        if self.calls < 3:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": f"call-{self.calls}",
                        "name": "glob_files",
                        "args": {"pattern": "*.py"},
                    }
                ],
            )
        return AIMessage(content="done")


def _start(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("x = 1\n", encoding="utf-8")
    model = FakeModel()
    checkpointer = InMemorySaver()
    graph = build_agent_graph(
        model,
        make_repo_tools(repo),
        repo,
        max_tool_calls=2,
        checkpointer=checkpointer,
    )
    config = {"configurable": {"thread_id": "t1"}}
    graph.invoke(
        {
            "messages": [
                SystemMessage(content="sys"),
                HumanMessage(content="look around"),
            ],
            "tool_calls_used": 0,
            "pending_edits": [],
        },
        config=config,
    )
    return graph, config, model


def test_tool_limit_continue_keeps_tools(tmp_path: Path) -> None:
    graph, config, model = _start(tmp_path)
    snap = graph.get_state(config)
    interrupts = [item.value for task in snap.tasks for item in task.interrupts]
    assert interrupts
    assert interrupts[0]["type"] == "tool_limit"
    assert model.bound_tools
    assert model.calls == 2

    graph.invoke(Command(resume={"action": "continue"}), config=config)
    snap = graph.get_state(config)
    interrupts = [item.value for task in snap.tasks for item in task.interrupts]
    assert not interrupts
    messages = snap.values["messages"]
    assert messages[-1].content == "done"
    assert snap.values["tool_calls_used"] == 0
    assert model.calls == 3


def test_tool_limit_stop_answers_once(tmp_path: Path) -> None:
    graph, config, model = _start(tmp_path)
    graph.invoke(Command(resume={"action": "stop"}), config=config)
    snap = graph.get_state(config)
    assert not [item.value for task in snap.tasks for item in task.interrupts]
    assert snap.values["messages"][-1].content == "done"
    assert snap.values["tool_calls_used"] >= 2
    assert model.calls == 3
