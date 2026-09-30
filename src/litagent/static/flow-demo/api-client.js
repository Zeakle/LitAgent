function normalizeError(status, payload) {
  const detail = payload?.detail;
  if (detail && typeof detail === "object") {
    return { status, code: detail.code || `http_${status}`, message: detail.message || detail.code, details: detail };
  }
  return { status, code: `http_${status}`, message: String(detail || payload?.message || status), details: payload || null };
}

export function createApiClient(getOperationToken) {
  let controller = new AbortController();

  return {
    abort() {
      controller.abort();
      controller = new AbortController();
    },
    async request(path, { method = "GET", body, mutate = false, signal } = {}) {
      const headers = {};
      if (body !== undefined) headers["Content-Type"] = "application/json";
      if (mutate) headers["X-LitAgent-Operation-Token"] = getOperationToken() || "";
      const response = await fetch(path, {
        method,
        headers,
        signal: signal || controller.signal,
        body: body === undefined ? undefined : JSON.stringify(body),
      });
      if (!response.ok) {
        let payload = null;
        try { payload = await response.json(); } catch { payload = null; }
        throw normalizeError(response.status, payload);
      }
      return response;
    },
    async json(path, options) {
      return (await this.request(path, options)).json();
    },
  };
}

export function openEventStream(runId, handlers = {}) {
  const source = new EventSource(`/survey/${encodeURIComponent(runId)}/events`);
  source.addEventListener("trace", (event) => handlers.onTrace?.(JSON.parse(event.data)));
  const settle = (event) => {
    handlers.onSettle?.(event.type);
    source.close();
  };
  source.addEventListener("state", settle);
  source.addEventListener("done", settle);
  source.onerror = () => {
    handlers.onError?.();
    source.close();
  };
  return source;
}
