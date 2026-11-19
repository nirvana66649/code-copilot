export function parseActivity(traces) {
  const steps = [];
  let current = null;
  for (const raw of traces || []) {
    const line = String(raw);
    if (line.startsWith("→ ")) {
      current = { node: line.slice(2).trim(), events: [] };
      steps.push(current);
      continue;
    }
    if (line.startsWith("⏸")) {
      steps.push({
        node: "interrupt",
        events: [{ kind: "interrupt", text: line.replace(/^⏸\s*/, "") }],
      });
      current = null;
      continue;
    }
    const trimmed = line.trim();
    if (!current) continue;
    if (trimmed.startsWith("think:")) {
      current.events.push({ kind: "think", text: trimmed.slice(6).trim() });
    } else if (trimmed.startsWith("say:")) {
      current.events.push({ kind: "say", text: trimmed.slice(4).trim() });
    } else if (trimmed.startsWith("note:")) {
      current.events.push({ kind: "note", text: trimmed.slice(5).trim() });
    } else if (trimmed.startsWith("tool ")) {
      const match = trimmed.match(/^tool\s+(\S+)\((.*)\)\s*$/s);
      current.events.push({
        kind: "tool",
        name: match ? match[1] : "tool",
        args: match ? match[2] : trimmed.slice(5),
      });
    } else if (trimmed.startsWith("←")) {
      current.events.push({ kind: "result", text: trimmed.slice(1).trim() });
    } else if (trimmed.startsWith("pending edits")) {
      current.events.push({ kind: "note", text: trimmed });
    }
  }
  return steps;
}

export function lastSpoken(steps) {
  let text = "";
  for (const step of steps) {
    for (const event of step.events) {
      if (event.kind === "say") text = event.text;
    }
  }
  return text;
}

export function activitySummary(steps) {
  let tools = 0;
  let thinks = 0;
  for (const step of steps) {
    for (const event of step.events) {
      if (event.kind === "tool") tools += 1;
      if (event.kind === "think") thinks += 1;
    }
  }
  const parts = [];
  if (tools) parts.push(`${tools} tool${tools === 1 ? "" : "s"}`);
  if (thinks) parts.push("thinking");
  return parts.join(" · ") || "Activity";
}
