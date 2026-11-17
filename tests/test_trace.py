from langchain_core.messages import AIMessage, ToolMessage

from coding_copilot_copy.agent import emit_trace


def test_emit_trace_prints_tool_call_and_result() -> None:
    lines: list[str] = []
    emit_trace(
        {
            "agent": {
                "messages": [
                    AIMessage(
                        content="looking up",
                        tool_calls=[
                            {
                                "id": "1",
                                "name": "codebase_search",
                                "args": {"query": "optimize prompt"},
                            }
                        ],
                    )
                ]
            }
        },
        lines.append,
    )
    emit_trace(
        {
            "tools_read": {
                "messages": [
                    ToolMessage(content="DailyNewsAgent.java:1", tool_call_id="1")
                ]
            }
        },
        lines.append,
    )
    joined = "\n".join(lines)
    assert "→ agent" in joined
    assert "codebase_search" in joined
    assert "optimize prompt" in joined
    assert "→ tools_read" in joined
    assert "DailyNewsAgent.java" in joined
