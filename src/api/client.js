const BASE = import.meta.env.VITE_API_URL || "/api/v1";

async function request(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...options.headers };

  const res = await fetch(`${BASE}${path}`, {
    headers,
    credentials: "include",
    ...options,
  });
  if (!res.ok) {
    if (res.status === 401) {
      if (window.location.pathname !== "/") {
        window.location.href = "/?expired=1";
      }
      throw new Error("Unauthorized");
    }
    const err = await res.text();
    throw new Error(err || res.statusText);
  }
  if (res.status === 204) return null;
  return res.json();
}

export const api = {
  login: async (username, password) => {
    const formData = new URLSearchParams();
    formData.append("username", username);
    formData.append("password", password);
    const res = await fetch(`${BASE}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: formData,
      credentials: "include",
    });
    if (!res.ok) throw new Error("Login failed");
    return res.json();
  },
  logout: async () => {
    const res = await fetch(`${BASE}/auth/logout`, {
      method: "POST",
      credentials: "include",
    });
    if (!res.ok) throw new Error("Logout failed");
    return res.json();
  },
  me: () => request("/auth/me"),
  // Direct download URL: a top-level <a href> navigation sends the
  // SameSite=Lax auth cookie, so attachment downloads work without JS blobs.
  exportUrl: (conversationId, format = "json") =>
    `${BASE}/conversations/${conversationId}/export?format=${format}`,
  health: () => request("/health"),
  dashboard: () => request("/analytics/dashboard"),
  trends: (params) => request(`/analytics/trends?${new URLSearchParams(params)}`),
  topKeywords: (window = "7d") => request(`/analytics/top-keywords?window=${window}`),
  conversations: (limit = 50) => request(`/conversations?limit=${limit}`),
  conversation: (id) => request(`/conversations/${id}`),
  reassignSpeaker: (conversationId, segmentId, newSpeaker) =>
    request(`/conversations/${conversationId}/segments/${segmentId}/reassign`, {
      method: "POST",
      body: JSON.stringify({ new_speaker: newSpeaker }),
    }),
  analyze: (body) =>
    request("/conversations/analyze", { method: "POST", body: JSON.stringify(body) }),
  analyzeText: (text, sector = "omni") =>
    request("/conversations/analyze-text", {
      method: "POST",
      body: JSON.stringify({ text, sector }),
    }),
  uploadAudio: async (file, sector = "omni") => {
    const formData = new window.FormData();
    formData.append("file", file);
    const res = await fetch(`${BASE}/conversations/upload?sector=${sector}`, {
      method: "POST",
      body: formData,
      credentials: "include",
    });
    if (!res.ok) throw new Error((await res.text()) || "Upload failed");
    return res.json();
  },
  keywordRules: () => request("/keyword-rules"),
  createRule: (body) => request("/keyword-rules", { method: "POST", body: JSON.stringify(body) }),
  updateRule: (id, body) =>
    request(`/keyword-rules/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
  deleteRule: (id) => request(`/keyword-rules/${id}`, { method: "DELETE" }),
  testRule: (body) =>
    request("/keyword-rules/test", { method: "POST", body: JSON.stringify(body) }),
  topics: () => request("/topics"),
  alerts: (acknowledged) =>
    request(`/alerts${acknowledged !== undefined ? `?acknowledged=${acknowledged}` : ""}`),
  alertRules: () => request("/alerts/rules"),
  createAlertRule: (body) =>
    request("/alerts/rules", { method: "POST", body: JSON.stringify(body) }),
  acknowledgeAlert: (id) => request(`/alerts/${id}/acknowledge`, { method: "PATCH" }),
  evaluateAlerts: () => request("/alerts/evaluate", { method: "POST" }),
  auditLogs: (params = {}) => {
    const query = new URLSearchParams(
      Object.fromEntries(Object.entries(params).filter(([, v]) => v !== undefined && v !== ""))
    );
    return request(`/audit-logs${query.toString() ? `?${query}` : ""}`);
  },
};

export function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(1).padStart(4, "0")}`;
}

export function severityColor(severity) {
  if (severity === "critical") return "var(--danger)";
  if (severity === "warning") return "var(--warning)";
  return "var(--info)";
}
