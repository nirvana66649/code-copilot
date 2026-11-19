import { useEffect, useRef, useState } from "react";
import { activitySummary, lastSpoken, parseActivity } from "./activity.js";
import { createSession, getConfig, streamPost } from "./api.js";
import "./App.css";

export default function App() {
  const [repoInput, setRepoInput] = useState("");
  const [session, setSession] = useState(null);
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [interrupt, setInterrupt] = useState(null);
  const scroller = useRef(null);

  useEffect(() => {
    getConfig()
      .then((c) => {
        if (c.default_repo) setRepoInput(c.default_repo);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" });
  }, [messages, interrupt, busy]);

  async function startSession(e) {
    e?.preventDefault();
    setError("");
    try {
      const created = await createSession(repoInput.trim());
      setSession(created);
      setMessages([]);
      setInterrupt(null);
    } catch (err) {
      setError(String(err.message || err));
    }
  }

  async function send(content) {
    if (!session || busy || !content.trim()) return;
    setError("");
    setBusy(true);
    const traces = [];
    setMessages((prev) => [
      ...prev,
      { role: "user", content },
      { role: "assistant", content: "", traces: [], pending: true },
    ]);
    await streamPost(
      `/v1/sessions/${session.session_id}/messages/stream`,
      { content },
      {
        onTrace: (line) => {
          traces.push(line);
          patchLastAssistant(setMessages, traces);
        },
        onDone: (turn) => {
          setMessages((prev) => {
            const next = prev.slice(0, -1);
            return [
              ...next,
              {
                role: "assistant",
                content: turn.message?.content || "",
                traces: [...traces],
                interrupted: turn.status === "interrupted",
              },
            ];
          });
          setInterrupt(turn.status === "interrupted" ? turn.interrupt : null);
          setBusy(false);
        },
        onError: (detail) => {
          setError(String(detail));
          setMessages((prev) => prev.filter((m) => !m.pending));
          setBusy(false);
        },
      }
    );
  }

  async function resume(action) {
    if (!session || busy) return;
    setBusy(true);
    setError("");
    const traces = [];
    setMessages((prev) => [...prev, { role: "assistant", content: "", traces: [], pending: true }]);
    await streamPost(
      `/v1/sessions/${session.session_id}/resume/stream`,
      { action },
      {
        onTrace: (line) => {
          traces.push(line);
          patchLastAssistant(setMessages, traces);
        },
        onDone: (turn) => {
          setMessages((prev) => [
            ...prev.filter((m) => !m.pending),
            {
              role: "assistant",
              content: turn.message?.content || "",
              traces: [...traces],
              interrupted: turn.status === "interrupted",
            },
          ]);
          setInterrupt(turn.status === "interrupted" ? turn.interrupt : null);
          setBusy(false);
        },
        onError: (detail) => {
          setError(String(detail));
          setMessages((prev) => prev.filter((m) => !m.pending));
          setBusy(false);
        },
      }
    );
  }

  function onComposer(e) {
    e.preventDefault();
    const text = draft.trim();
    setDraft("");
    send(text);
  }

  return (
    <div className="shell">
      <aside className="rail">
        <div className="brand">
          <span className="mark" />
          Copilot
        </div>
        <form className="repo-form" onSubmit={startSession}>
          <label>Workspace</label>
          <input
            value={repoInput}
            onChange={(e) => setRepoInput(e.target.value)}
            placeholder="/path/to/repo"
            spellCheck={false}
          />
          <button type="submit">{session ? "New session" : "Connect"}</button>
        </form>
        {session && (
          <div className="meta">
            <div className="meta-k">repo</div>
            <div className="meta-v mono">{shortPath(session.repo_root)}</div>
            <div className="meta-k">thread</div>
            <div className="meta-v mono">{session.thread_id}</div>
          </div>
        )}
      </aside>

      <main className="main">
        <header className="top">
          <span>Agent</span>
          {busy && <span className="pulse">running</span>}
        </header>

        <div className="thread" ref={scroller}>
          {!session && (
            <div className="empty">
              Connect a local repo on the left, then ask about the code — search, glob, edits with
              approve/reject.
            </div>
          )}
          {session && messages.length === 0 && (
            <div className="empty">Ask a question about this repository.</div>
          )}
          {messages.map((m, i) => (
            <Message key={i} msg={m} />
          ))}
          {interrupt && (
            <InterruptCard interrupt={interrupt} disabled={busy} onAction={resume} />
          )}
        </div>

        {error && <div className="banner">{error}</div>}

        <form className="composer" onSubmit={onComposer}>
          <textarea
            rows={2}
            value={draft}
            disabled={!session || busy || Boolean(interrupt)}
            placeholder={
              interrupt
                ? "Resolve the pause above first"
                : session
                  ? "Ask about the repo…"
                  : "Connect a workspace first"
            }
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                onComposer(e);
              }
            }}
          />
          <button type="submit" disabled={!session || busy || !draft.trim() || Boolean(interrupt)}>
            Send
          </button>
        </form>
      </main>
    </div>
  );
}

function patchLastAssistant(setMessages, traces) {
  const steps = parseActivity(traces);
  const spoken = lastSpoken(steps);
  setMessages((prev) => {
    const next = [...prev];
    const last = next[next.length - 1];
    if (!last || last.role !== "assistant") return prev;
    next[next.length - 1] = {
      ...last,
      traces: [...traces],
      content: spoken || last.content || "",
      pending: true,
    };
    return next;
  });
}

function Message({ msg }) {
  if (msg.role === "user") {
    return (
      <article className="msg user">
        <div className="who">You</div>
        <div className="body">{msg.content}</div>
      </article>
    );
  }
  const steps = parseActivity(msg.traces);
  const live = msg.pending && !msg.content && steps.length === 0;
  return (
    <article className="msg assistant">
      <div className="who">Agent</div>
      {live && <div className="body muted">Working…</div>}
      {steps.length > 0 && <ActivityPanel steps={steps} pending={msg.pending} />}
      {msg.content && (
        <div className={`body ${msg.pending ? "streaming" : ""}`}>{msg.content}</div>
      )}
    </article>
  );
}

function ActivityPanel({ steps, pending }) {
  return (
    <details className="traces" open={pending || undefined}>
      <summary>
        {pending ? (
          <span className="live">
            <span className="dot" />
            Running · {activitySummary(steps)}
          </span>
        ) : (
          activitySummary(steps)
        )}
      </summary>
      <ol className="timeline">
        {steps.map((step, i) => (
          <li key={`${step.node}-${i}`} className="step">
            <div className="step-node mono">{labelNode(step.node)}</div>
            {step.events.map((event, j) => (
              <ActivityEvent key={j} event={event} />
            ))}
          </li>
        ))}
      </ol>
    </details>
  );
}

function ActivityEvent({ event }) {
  if (event.kind === "think") {
    return (
      <div className="ev think">
        <div className="ev-k">Thinking</div>
        <pre className="mono">{event.text}</pre>
      </div>
    );
  }
  if (event.kind === "tool") {
    return (
      <div className="ev tool">
        <div className="ev-k">
          Tool <span className="mono">{event.name}</span>
        </div>
        {event.args && <pre className="mono">{event.args}</pre>}
      </div>
    );
  }
  if (event.kind === "result") {
    return (
      <details className="ev result">
        <summary>Result</summary>
        <pre className="mono">{event.text}</pre>
      </details>
    );
  }
  if (event.kind === "say") return null;
  if (event.kind === "interrupt") {
    return <div className="ev interrupt-line">{event.text}</div>;
  }
  return (
    <div className="ev note">
      <pre className="mono">{event.text}</pre>
    </div>
  );
}

function labelNode(node) {
  if (node === "agent") return "model";
  if (node === "tools_read") return "tools";
  if (node === "write_review") return "write review";
  if (node === "tool_limit") return "limit";
  return node;
}

function InterruptCard({ interrupt, disabled, onAction }) {
  const kind = interrupt.type === "tool_limit" ? "tool_limit" : "edit_approval";
  return (
    <section className="interrupt">
      <div className="int-title">
        {kind === "tool_limit" ? "Tool limit" : "Review edits"}
      </div>
      {interrupt.prompt && <p>{interrupt.prompt}</p>}
      {(interrupt.edits || []).map((edit, i) => (
        <div key={i} className="diff-block">
          <div className="diff-path mono">{edit.path}</div>
          {edit.reason && <div className="muted">{edit.reason}</div>}
          <pre className="mono diff">{edit.preview}</pre>
        </div>
      ))}
      <div className="int-actions">
        {kind === "tool_limit" ? (
          <>
            <button disabled={disabled} onClick={() => onAction("continue")}>
              Continue
            </button>
            <button className="ghost" disabled={disabled} onClick={() => onAction("stop")}>
              Stop
            </button>
          </>
        ) : (
          <>
            <button className="ok" disabled={disabled} onClick={() => onAction("approve")}>
              Approve
            </button>
            <button className="ghost" disabled={disabled} onClick={() => onAction("reject")}>
              Reject
            </button>
          </>
        )}
      </div>
    </section>
  );
}

function shortPath(path) {
  if (!path) return "";
  const parts = path.split("/").filter(Boolean);
  if (parts.length <= 3) return path;
  return "…/" + parts.slice(-3).join("/");
}
