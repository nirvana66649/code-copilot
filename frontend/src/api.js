export async function getConfig() {
  const res = await fetch("/v1/config");
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function createSession(repoPath) {
  const res = await fetch("/v1/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo_path: repoPath || null }),
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function streamPost(url, body, { onTrace, onDone, onError }) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    onError(await readError(res));
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";
    for (const part of parts) {
      const line = part
        .split("\n")
        .filter((l) => l.startsWith("data:"))
        .map((l) => l.slice(5).trim())
        .join("");
      if (!line) continue;
      let event;
      try {
        event = JSON.parse(line);
      } catch {
        continue;
      }
      if (event.event === "trace") onTrace(event.line);
      else if (event.event === "done") onDone(event.turn);
      else if (event.event === "error") onError(event.detail || "request failed");
    }
  }
}

async function readError(res) {
  try {
    const data = await res.json();
    return data.detail || res.statusText;
  } catch {
    return res.statusText;
  }
}
