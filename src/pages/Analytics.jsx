import { useState, useEffect } from "react";
import { api } from "../api/client";
import { TrendChart } from "../components/TrendChart";
import { SkeletonList } from "../components/common/Skeleton";

const KEYWORDS = ["zam", "fatura", "iade", "iptal", "şikayet"];

export default function AnalyticsPage() {
  const [keyword, setKeyword] = useState("zam");
  const [window, setWindow] = useState("7d");
  const [trend, setTrend] = useState(null);
  const [compareKeyword, setCompareKeyword] = useState("");
  const [compareTrend, setCompareTrend] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const t = await api.trends({ keyword, window });
        setTrend(t);
        if (compareKeyword) {
          const c = await api.trends({ keyword: compareKeyword, window });
          setCompareTrend(c);
        } else {
          setCompareTrend(null);
        }
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [keyword, window, compareKeyword]);

  const pct = trend?.pct_change ?? 0;
  const pctClass = pct >= 30 ? "danger" : pct >= 15 ? "warning" : "neutral";

  const exportToCSV = () => {
    if (!trend?.daily_series) return;
    const header = "Tarih,Eşleşme Sayısı\n";
    const csv = trend.daily_series.map((d) => `${d.date},${d.hit_count}`).join("\n");
    const blob = new Blob([header + csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `analytics_${keyword}_${window}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="page-content">
      <header
        className="page-header"
        style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}
      >
        <div>
          <h1>Trend Analitiği</h1>
          <p>Zaman penceresi bazında kelime frekans değişimleri</p>
        </div>
        <button className="btn-secondary" onClick={exportToCSV} disabled={!trend?.daily_series}>
          CSV Olarak İndir
        </button>
      </header>

      <div className="filter-bar">
        <div className="control-group inline">
          <label>Kelime</label>
          <select value={keyword} onChange={(e) => setKeyword(e.target.value)}>
            {KEYWORDS.map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
        </div>
        <div className="control-group inline">
          <label>Pencere</label>
          <select value={window} onChange={(e) => setWindow(e.target.value)}>
            <option value="24h">24 Saat</option>
            <option value="7d">7 Gün</option>
            <option value="30d">30 Gün</option>
          </select>
        </div>
        <div className="control-group inline">
          <label>Karşılaştır</label>
          <select value={compareKeyword} onChange={(e) => setCompareKeyword(e.target.value)}>
            <option value="">—</option>
            {KEYWORDS.filter((k) => k !== keyword).map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
        </div>
      </div>

      {loading ? (
        <SkeletonList count={3} height="150px" />
      ) : (
        <div className="analytics-hero glass-panel">
          <div className="analytics-title">
            <h2>{keyword.toUpperCase()}</h2>
            <div className={`trend-badge large ${pctClass}`}>
              {pct >= 0 ? "▲" : "▼"} %{Math.abs(pct).toFixed(1)}
            </div>
          </div>
          <div className="trend-summary">
            <div className="trend-stat">
              <span className="trend-num">{trend?.current_count ?? 0}</span>
              <span className="trend-label">bu dönem</span>
            </div>
            <div className="trend-stat">
              <span className="trend-num">{trend?.previous_count ?? 0}</span>
              <span className="trend-label">önceki dönem</span>
            </div>
            <div className="trend-stat">
              <span className="trend-num">{trend?.current_conversations ?? 0}</span>
              <span className="trend-label">görüşme</span>
            </div>
          </div>
          {trend?.anomaly && (
            <div className="anomaly-banner">
              Anomali tespit edildi: %{pct.toFixed(0)} artış — raporlanabilir eşik aşıldı.
            </div>
          )}
          <TrendChart
            data={trend?.daily_series}
            color={pct >= 30 ? "var(--danger)" : "var(--accent)"}
          />
        </div>
      )}

      {compareTrend && (
        <div className="glass-panel" style={{ marginTop: "1.5rem" }}>
          <div className="panel-header">
            <span>Karşılaştırma</span>
            <h2>{compareKeyword.toUpperCase()}</h2>
          </div>
          <div className="trend-summary">
            <div className="trend-stat">
              <span className="trend-num">{compareTrend.current_count}</span>
              <span className="trend-label">eşleşme</span>
            </div>
            <div className={`trend-badge ${compareTrend.pct_change >= 30 ? "danger" : "neutral"}`}>
              {compareTrend.pct_change >= 0 ? "+" : ""}
              {compareTrend.pct_change}%
            </div>
          </div>
          <TrendChart data={compareTrend.daily_series} color="var(--info)" />
        </div>
      )}
    </div>
  );
}
