import { useState, useEffect } from "react";
import { ChevronRight } from "lucide-react";
import { api, formatTime, severityColor } from "../api/client";

function highlightText(text, hits) {
  if (!hits?.length) return text;
  let result = text;
  const sorted = [...hits].sort((a, b) => b.matched_text.length - a.matched_text.length);
  for (const hit of sorted) {
    const re = new RegExp(`(${hit.matched_text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "gi");
    result = result.replace(re, '<mark class="kw-highlight">$1</mark>');
  }
  return result;
}

export default function ConversationsPage() {
  const [list, setList] = useState([]);
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .conversations()
      .then(setList)
      .finally(() => setLoading(false));
  }, []);

  const openDetail = async (id) => {
    const detail = await api.conversation(id);
    setSelected(detail);
  };

  if (loading) return <div className="page-loading">Yükleniyor...</div>;

  return (
    <div className="page-content">
      <header className="page-header">
        <h1>Görüşmeler</h1>
        <p>Anahtar kelime eşleşmeleriyle birlikte deşifre kayıtları</p>
      </header>

      <div className="conv-layout">
        <div className="conv-list glass-panel">
          {list.length === 0 ? (
            <div className="empty-state compact">
              <p>Henüz kayıtlı görüşme yok. Streamlit veya API ile analiz yapın.</p>
            </div>
          ) : (
            list.map((c) => (
              <div
                className={`conv-item ${selected?.id === c.id ? "active" : ""}`}
                key={c.id}
                onClick={() => openDetail(c.id)}
              >
                <div className="conv-item-head">
                  <span className="sector-badge">{c.sector}</span>
                  <span className="hit-badge">{c.hit_count} hit</span>
                </div>
                <p className="conv-preview">{c.full_transcript.slice(0, 80)}...</p>
                <span className="conv-date">{new Date(c.created_at).toLocaleString("tr-TR")}</span>
                <ChevronRight size={16} className="conv-arrow" />
              </div>
            ))
          )}
        </div>

        <div className="conv-detail glass-panel">
          {!selected ? (
            <div className="empty-state compact">
              <p>Detay görmek için bir görüşme seçin</p>
            </div>
          ) : (
            <>
              <div className="panel-header">
                <span>{selected.sector}</span>
                <h2>Görüşme Detayı</h2>
                <p>{selected.hit_count} anahtar kelime eşleşmesi</p>
              </div>

              {selected.topics?.length > 0 && (
                <div className="chip-row" style={{ marginBottom: "1rem" }}>
                  {selected.topics.map((t) => (
                    <span className="chip topic" key={t.slug}>
                      {t.label_tr}
                    </span>
                  ))}
                </div>
              )}

              {selected.hits?.length > 0 && (
                <div className="hits-timeline">
                  {selected.hits.map((h) => (
                    <div
                      className="hit-pill"
                      key={h.id}
                      style={{ borderColor: severityColor(h.severity) }}
                    >
                      <span className="hit-time">{formatTime(h.timestamp_sec)}</span>
                      <strong>{h.matched_text}</strong>
                      <span className="hit-type">{h.match_type}</span>
                    </div>
                  ))}
                </div>
              )}

              <div className="transcript-box">
                {selected.segments.map((seg, _idx) => {
                  const segHits = selected.hits.filter(
                    (h) => Math.abs(h.timestamp_sec - seg.start) < 0.5
                  );
                  return (
                    <div className="transcript-row" key={seg.id}>
                      <div className="t-time">{formatTime(seg.start)}</div>
                      <div className="t-content">
                        {seg.speaker && <strong>{seg.speaker}</strong>}
                        <p dangerouslySetInnerHTML={{ __html: highlightText(seg.text, segHits) }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
