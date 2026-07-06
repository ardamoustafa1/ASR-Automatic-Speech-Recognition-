import { useState, useEffect } from "react";
import { Bell, CheckCircle2, RefreshCw } from "lucide-react";
import { api } from "../api/client";
import { useAppStore } from "../store/useAppStore";

export default function AlertsPage() {
  const isAdmin = useAppStore((state) => state.user?.role === "admin");
  const [events, setEvents] = useState([]);
  const [rules, setRules] = useState([]);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    const [e, r] = await Promise.all([api.alerts(false), api.alertRules()]);
    setEvents(e);
    setRules(r);
    setLoading(false);
  };

  useEffect(() => {
    load();
  }, []);

  const ack = async (id) => {
    await api.acknowledgeAlert(id);
    load();
  };

  const evaluate = async () => {
    await api.evaluateAlerts();
    load();
  };

  if (loading) return <div className="page-loading">Yükleniyor...</div>;

  return (
    <div className="page-content">
      <header className="page-header row">
        <div>
          <h1>Uyarı Merkezi</h1>
          <p>Trend eşiklerine göre otomatik raporlar</p>
        </div>
        {isAdmin && (
          <button className="btn-secondary" onClick={evaluate}>
            <RefreshCw size={16} /> Uyarıları Değerlendir
          </button>
        )}
      </header>

      <div className="alerts-grid">
        <div className="glass-panel">
          <div className="panel-header">
            <span>Aktif Uyarılar</span>
            <h2>{events.length} okunmamış</h2>
          </div>
          {events.length === 0 ? (
            <div className="empty-state compact">
              <Bell size={32} className="icon" />
              <p>Henüz tetiklenen uyarı yok</p>
            </div>
          ) : (
            <ul className="alert-list">
              {events.map((e) => (
                <li className={`alert-item ${e.severity}`} key={e.id}>
                  <div className="alert-head">
                    <strong>{e.title}</strong>
                    <span className="alert-date">
                      {new Date(e.created_at).toLocaleString("tr-TR")}
                    </span>
                  </div>
                  <p>{e.summary}</p>
                  {e.payload?.pct_change != null && (
                    <span className="trend-badge danger">+{e.payload.pct_change}%</span>
                  )}
                  {isAdmin && (
                    <button className="btn-secondary sm" onClick={() => ack(e.id)}>
                      <CheckCircle2 size={14} /> Okundu
                    </button>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="glass-panel">
          <div className="panel-header">
            <span>Uyarı Kuralları</span>
            <h2>Yapılandırma</h2>
          </div>
          {rules.map((r) => (
            <div className="alert-rule-card" key={r.id}>
              <strong>{r.name}</strong>
              <p>
                Eşik: %{r.condition?.threshold ?? "—"} artış, pencere: {r.condition?.window ?? "7d"}
              </p>
              <span className={r.is_active ? "active" : "inactive"}>
                {r.is_active ? "Aktif" : "Pasif"}
              </span>
            </div>
          ))}
          {rules.length === 0 && (
            <p className="text-muted">Varsayılan kurallar ilk analizde oluşturulur.</p>
          )}
        </div>
      </div>
    </div>
  );
}
