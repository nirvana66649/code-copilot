from coding_copilot_copy.agent import AgentTurn
from coding_copilot_copy.server import serialize_turn


def test_serialize_completed_turn() -> None:
    turn = AgentTurn(text="hello", interrupted=False, thread_id="t1")
    out = serialize_turn(turn, ["→ agent"])
    assert out["status"] == "completed"
    assert out["message"]["content"] == "hello"
    assert out["trace"] == ["→ agent"]
    assert out["interrupt"] is None


def test_serialize_interrupt() -> None:
    payload = {"type": "edit_approval", "edits": []}
    turn = AgentTurn(text="paused", interrupted=True, payload=payload, thread_id="t1")
    out = serialize_turn(turn)
    assert out["status"] == "interrupted"
    assert out["interrupt"]["type"] == "edit_approval"
