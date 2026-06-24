import { useState, useEffect } from "react";
import { Activity, Bell, Hash, Phone } from "lucide-react";
import { api } from "../api/client";
import { TrendChart, Sparkline } from "../components/TrendChart";

export default function DashboardPage() {
  const [summary, setSummary] = useState(null);
  const [trend, setTrend] = useState(null);
  const [topKeywords, setTopKeywords] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const [dash, top] = await Promise.all([api.dashboard(), api.topKeywords("7d")]);
        setSummary(dash);
        setTopKeywords(top);
        if (dash.top_rising_keyword?.keyword) {
          const t = await api.trends({ keyword: dash.top_rising_keyword.keyword, window: "7d" });
          setTrend(t);
        } else if (top[0]?.keyword) {
          const t = await api.trends({ keyword: top[0].keyword, window: "7d" });
          setTrend(t);
        }
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) return <div className="page-loading">Yükleniyor...</div>;

  const pct = trend?.pct_change ?? 0;
  const pctClass = pct >= 30 ? "danger" : pct >= 15 ? "warning" : "neutral";

  return (
    <div className="page-content">
      <header className="page-header">
        <h1>Anahtar Kelime & Konu Analitiği</h1>
        <p>Görüşmelerde otomatik kelime tespiti ve trend raporlama</p>
      </header>

      <div className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-label">
            <Hash size={16} /> Bugünkü Eşleşmeler
          </div>
          <div className="kpi-value">{summary?.hits_today ?? 0}</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">
            <Phone size={16} /> Toplam Görüşme
          </div>
          <div className="kpi-value">{summary?.conversations_total ?? 0}</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">
            <Bell size={16} /> Aktif Uyarılar
          </div>
          <div className="kpi-value">{summary?.active_alerts ?? 0}</div>
        </div>
        <div className="kpi-card highlight">
          <div className="kpi-label">
            <Activity size={16} /> En Çok Artan
          </div>
          <div className="kpi-value">{(trend?.keyword || "—").toUpperCase()}</div>
          <div className={`kpi-foot ${pctClass}`}>
            {pct > 0 ? "▲" : pct < 0 ? "▼" : "—"} %{Math.abs(pct).toFixed(1)}
          </div>
          {trend?.daily_series && <Sparkline data={trend.daily_series} />}
        </div>
      </div>

      <div className="dashboard-grid">
        <div className="glass-panel">
          <div className="panel-header">
            <span>Trend Analizi</span>
            <h2>{trend?.keyword ? `"${trend.keyword}" — Son 7 Gün` : "Trend Verisi"}</h2>
          </div>
          {trend && (
            <div className="trend-summary">
              <div className="trend-stat">
                <span className="trend-num">{trend.current_count}</span>
                <span className="trend-label">eşleşme</span>
              </div>
              <div className="trend-stat">
                <span className="trend-num">{trend.current_conversations}</span>
                <span className="trend-label">görüşme</span>
              </div>
              <div className={`trend-badge ${pctClass}`}>
                {pct >= 0 ? "+" : ""}
                {pct}% önceki döneme göre
              </div>
            </div>
          )}
          <TrendChart data={trend?.daily_series} />
        </div>

        <div className="glass-panel">
          <div className="panel-header">
            <span>En Sık Kelimeler</span>
            <h2>Son 7 Gün</h2>
          </div>
          <ul className="keyword-list">
            {topKeywords.map((kw, i) => (
              <li key={kw.keyword}>
                <span className="kw-rank">{i + 1}</span>
                <span className="kw-name">{kw.keyword}</span>
                <span className="kw-count">{kw.hit_count} eşleşme</span>
              </li>
            ))}
            {!topKeywords.length && <li className="empty">Henüz veri yok</li>}
          </ul>
        </div>
      </div>
    </div>
  );
}
