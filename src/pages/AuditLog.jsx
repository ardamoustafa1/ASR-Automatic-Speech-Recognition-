import { useState, useEffect } from "react";
import { ShieldCheck, Filter } from "lucide-react";
import { api } from "../api/client";

export default function AuditLogPage() {
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({ username: "", action: "" });

  const load = async (activeFilters) => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.auditLogs(activeFilters);
      setEntries(data);
    } catch {
      setError("Denetim kayıtları yüklenemedi. Bu sayfa yalnızca admin ve auditor rollerine açıktır.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load(filters);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="page-content">
      <header className="page-header">
        <h1>Denetim Kaydı (Audit Log)</h1>
        <p>Kim, neyi, ne zaman görüntüledi veya değiştirdi — uyumluluk ve güvenlik denetimi için.</p>
      </header>

      <div className="glass-panel" style={{ marginBottom: "1.5rem" }}>
        <div className="panel-header">
          <span>Filtre</span>
          <h2>
            <Filter size={16} style={{ marginRight: "0.4rem", verticalAlign: "middle" }} />
            Kayıtları Daralt
          </h2>
        </div>
        <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
          <input
            placeholder="Kullanıcı adı"
            value={filters.username}
            onChange={(e) => setFilters({ ...filters, username: e.target.value })}
            style={{ padding: "8px 10px" }}
          />
          <input
            placeholder="Aksiyon (GET, POST, VIEW, EXPORT...)"
            value={filters.action}
            onChange={(e) => setFilters({ ...filters, action: e.target.value })}
            style={{ padding: "8px 10px" }}
          />
          <button className="btn-primary" onClick={() => load(filters)}>
            Uygula
          </button>
        </div>
      </div>

      <div className="glass-panel">
        <div className="panel-header">
          <span>Kayıtlar</span>
          <h2>{entries.length} kayıt</h2>
        </div>

        {loading ? (
          <div className="page-loading">Yükleniyor...</div>
        ) : error ? (
          <div className="empty-state compact">
            <ShieldCheck size={32} className="icon" />
            <p>{error}</p>
          </div>
        ) : entries.length === 0 ? (
          <div className="empty-state compact">
            <ShieldCheck size={32} className="icon" />
            <p>Kayıt bulunamadı.</p>
          </div>
        ) : (
          <table className="data-table" style={{ width: "100%" }}>
            <thead>
              <tr>
                <th>Zaman</th>
                <th>Kullanıcı</th>
                <th>Aksiyon</th>
                <th>Kaynak</th>
                <th>IP</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((entry) => (
                <tr key={entry.id}>
                  <td>{new Date(entry.timestamp).toLocaleString("tr-TR")}</td>
                  <td>{entry.username || "—"}</td>
                  <td>{entry.action}</td>
                  <td style={{ fontSize: "0.8rem", color: "var(--asr-muted)" }}>
                    {entry.target_resource || "—"}
                  </td>
                  <td>{entry.ip_address || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
