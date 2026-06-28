import type { ChatResponse, Message } from "./types";

export type StreamEvent =
  | { event: "status"; stage: string }
  | { event: "token"; delta: string }
  | { event: "final"; data: ChatResponse }
  | { event: "error"; detail: string };

let sessionPromise: Promise<void> | null = null;

export async function bootstrapSession(): Promise<void> {
  const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || "";
  await ensureSession(apiBaseUrl);
}

async function ensureSession(apiBaseUrl: string): Promise<void> {
  sessionPromise ??= fetch(`${apiBaseUrl}/api/session`, {
    method: "GET",
    credentials: "include",
  }).then(async (response) => {
    if (!response.ok) {
      const body = (await response.json().catch(() => ({}))) as { detail?: string };
      throw new Error(body.detail || `Session request failed (${response.status})`);
    }
  });
  return sessionPromise;
}

export async function streamChat(
  messages: Message[],
  clauseType: string,
  limit: number,
  onEvent: (event: StreamEvent) => void,
): Promise<void> {
  const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || "";
  await ensureSession(apiBaseUrl);
  const response = await fetch(`${apiBaseUrl}/chat/stream`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: messages.slice(-8).map(({ role, content }) => ({ role, content })),
      clause_type: clauseType || null,
      limit,
      rerank_mode: "off",
    }),
  });
  if (!response.ok || !response.body) {
    const retryAfter = response.headers.get("Retry-After");
    const body = (await response.json().catch(() => ({}))) as { detail?: string };
    const suffix = retryAfter ? ` Try again in ${retryAfter} seconds.` : "";
    throw new Error(`${body.detail || `Request failed (${response.status})`}${suffix}`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.trim()) continue;
      onEvent(JSON.parse(line) as StreamEvent);
    }
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer) as StreamEvent);
}
