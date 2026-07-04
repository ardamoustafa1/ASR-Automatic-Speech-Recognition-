import { Fragment, useState, useEffect } from "react";
import { ChevronRight, Upload, FileAudio, AlertTriangle, CheckCircle2, Award } from "lucide-react";
import { api, formatTime, severityColor } from "../api/client";
import { SkeletonList } from "../components/common/Skeleton";

function highlightedRanges(text, hits) {
  if (!text || !hits?.length) return [];

  const lowerText = text.toLocaleLowerCase("tr-TR");
  const ranges = [];
  const sorted = [...hits]
    .map((hit) => hit.matched_text || "")
    .filter(Boolean)
    .sort((a, b) => b.length - a.length);

  for (const term of sorted) {
    const needle = term.toLocaleLowerCase("tr-TR");
    let fromIndex = 0;

    while (needle && fromIndex < lowerText.length) {
      const start = lowerText.indexOf(needle, fromIndex);
      if (start === -1) break;

      const end = start + needle.length;
      const overlaps = ranges.some((range) => start < range.end && end > range.start);
      if (!overlaps) ranges.push({ start, end });
      fromIndex = end;
    }
  }

  return ranges.sort((a, b) => a.start - b.start);
}

function renderHighlightedText(text, hits) {
  const ranges = highlightedRanges(text, hits);
  if (!ranges.length) return text;

  const parts = [];
  let cursor = 0;
  ranges.forEach((range, index) => {
    if (range.start > cursor) {
      parts.push(text.slice(cursor, range.start));
    }
    parts.push(
      <mark className="kw-highlight" key={`${range.start}-${range.end}-${index}`}>
        {text.slice(range.start, range.end)}
      </mark>
    );
    cursor = range.end;
  });
  if (cursor < text.length) parts.push(text.slice(cursor));

  return parts.map((part, index) => <Fragment key={index}>{part}</Fragment>);
}

export default function ConversationsPage() {
  const [list, setList] = useState([]);
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState("");

  const loadList = () => {
    setLoading(true);
    api
      .conversations()
      .then(setList)
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadList();
  }, []);

  const openDetail = async (id) => {
    const detail = await api.conversation(id);
    setSelected(detail);
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadMsg("Ses dosyası yükleniyor ve yapay zeka analizine alınıyor...");
    try {
      await api.uploadAudio(file);
      setUploadMsg("Başarıyla arka plana alındı! Deşifre tamamlandığında listede görünecek.");
      setTimeout(() => {
        loadList();
        setUploadMsg("");
      }, 3500);
    } catch (err) {
      setUploadMsg(`Yükleme hatası: ${err.message}`);
    } finally {
      setUploading(false);
    }
  };

  if (loading) {
    return (
      <div className="page-content">
        <header className="page-header">
          <h1>Görüşmeler</h1>
          <p>Anahtar kelime eşleşmeleriyle birlikte deşifre kayıtları</p>
        </header>
        <SkeletonList count={5} height="80px" />
      </div>
    );
  }

  return (
    <div className="page-content">
      <header
        className="page-header"
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "1rem",
        }}
      >
        <div>
          <h1>Görüşmeler</h1>
          <p>Anahtar kelime eşleşmeleriyle birlikte deşifre kayıtları</p>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
          <label
            className="btn btn-primary"
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "0.5rem",
              cursor: "pointer",
              background: "linear-gradient(135deg, #3B82F6, #6366F1)",
              border: "none",
              padding: "0.6rem 1.2rem",
              borderRadius: "8px",
              fontWeight: 600,
              color: "#fff",
              transition: "all 0.2s",
            }}
          >
            <Upload size={18} />
            <span>{uploading ? "Analiz Ediliyor..." : "Ses Dosyası Yükle & Analiz Et"}</span>
            <input
              type="file"
              accept="audio/*"
              onChange={handleFileUpload}
              disabled={uploading}
              style={{ display: "none" }}
            />
          </label>
        </div>
      </header>

      {uploadMsg && (
        <div
          className="glass-panel"
          style={{
            padding: "0.8rem 1.2rem",
            marginBottom: "1rem",
            borderRadius: "8px",
            border: "1px solid rgba(99,102,241,0.3)",
            background: "rgba(99,102,241,0.1)",
            color: "#E0E7FF",
            display: "flex",
            alignItems: "center",
            gap: "0.6rem",
            fontSize: "0.9rem",
          }}
        >
          <FileAudio size={18} color="#818CF8" />
          <span>{uploadMsg}</span>
        </div>
      )}

      <div className="conv-layout">
        <div className="conv-list glass-panel">
          {list.length === 0 ? (
            <div className="empty-state compact">
              <p>
                Henüz kayıtlı görüşme yok. Yukarıdaki butondan ses dosyası yükleyebilir veya API ile
                analiz yapabilirsiniz.
              </p>
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

              {selected.metadata_json && (
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
                    gap: "1rem",
                    margin: "1.2rem 0",
                  }}
                >
                  <div
                    className="glass-panel"
                    style={{
                      padding: "1rem",
                      background: "rgba(255,255,255,0.03)",
                      border: "1px solid rgba(255,255,255,0.08)",
                      borderRadius: "10px",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "0.5rem",
                        fontSize: "0.75rem",
                        color: "var(--text-muted)",
                        textTransform: "uppercase",
                        letterSpacing: "0.5px",
                        marginBottom: "0.4rem",
                      }}
                    >
                      <Award size={14} color="#10B981" />
                      <span>Empati & Soft Skill Skoru</span>
                    </div>
                    <div
                      style={{
                        fontSize: "1.6rem",
                        fontWeight: 700,
                        color:
                          selected.metadata_json.empathy_score >= 70
                            ? "#10B981"
                            : selected.metadata_json.empathy_score >= 50
                              ? "#F59E0B"
                              : "#EF4444",
                      }}
                    >
                      {selected.metadata_json.empathy_score !== undefined
                        ? `${selected.metadata_json.empathy_score} / 100`
                        : "Hesaplanmadı"}
                    </div>
                    {selected.metadata_json.empathy_summary && (
                      <div
                        style={{
                          fontSize: "0.8rem",
                          color: "var(--text-secondary)",
                          marginTop: "0.4rem",
                          lineHeight: 1.4,
                        }}
                      >
                        {selected.metadata_json.empathy_summary}
                      </div>
                    )}
                  </div>

                  <div
                    className="glass-panel"
                    style={{
                      padding: "1rem",
                      background: "rgba(255,255,255,0.03)",
                      border: "1px solid rgba(255,255,255,0.08)",
                      borderRadius: "10px",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "0.5rem",
                        fontSize: "0.75rem",
                        color: "var(--text-muted)",
                        textTransform: "uppercase",
                        letterSpacing: "0.5px",
                        marginBottom: "0.4rem",
                      }}
                    >
                      {selected.metadata_json.is_high_risk ? (
                        <AlertTriangle size={14} color="#EF4444" />
                      ) : (
                        <CheckCircle2 size={14} color="#10B981" />
                      )}
                      <span>Müşteri Churn (Kayıp) Riski</span>
                    </div>
                    <div
                      style={{
                        fontSize: "1.6rem",
                        fontWeight: 700,
                        color: selected.metadata_json.is_high_risk ? "#EF4444" : "#10B981",
                      }}
                    >
                      {selected.metadata_json.is_high_risk
                        ? "🚨 Yüksek Risk"
                        : selected.metadata_json.churn_risk !== undefined
                          ? `✅ Düşük Risk (${selected.metadata_json.churn_risk})`
                          : "Stabil"}
                    </div>
                    <div
                      style={{
                        fontSize: "0.8rem",
                        color: "var(--text-secondary)",
                        marginTop: "0.4rem",
                        lineHeight: 1.4,
                      }}
                    >
                      {selected.metadata_json.is_high_risk
                        ? "Acil müşteri geri kazanımı ve arama önerilir."
                        : "Müşteri memnuniyet ve bağlılık seviyesi stabil."}
                    </div>
                  </div>
                </div>
              )}

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
                  const meta = selected.metadata_json || {};
                  const isAgent =
                    seg.speaker &&
                    (seg.speaker === meta.agent_speaker_id || seg.speaker === "SPEAKER_00");
                  const isCustomer =
                    seg.speaker &&
                    (seg.speaker === meta.customer_speaker_id || seg.speaker === "SPEAKER_01");

                  return (
                    <div className="transcript-row" key={seg.id}>
                      <div className="t-time">{formatTime(seg.start)}</div>
                      <div className="t-content">
                        {seg.speaker && (
                          <div style={{ marginBottom: "0.3rem" }}>
                            {isAgent ? (
                              <span
                                style={{
                                  display: "inline-block",
                                  background: "rgba(16, 185, 129, 0.15)",
                                  color: "#10B981",
                                  border: "1px solid rgba(16,185,129,0.3)",
                                  padding: "0.15rem 0.55rem",
                                  borderRadius: "6px",
                                  fontSize: "0.75rem",
                                  fontWeight: 600,
                                }}
                              >
                                👤 Temsilci ({seg.speaker})
                              </span>
                            ) : isCustomer ? (
                              <span
                                style={{
                                  display: "inline-block",
                                  background: "rgba(59, 130, 246, 0.15)",
                                  color: "#60A5FA",
                                  border: "1px solid rgba(59,130,246,0.3)",
                                  padding: "0.15rem 0.55rem",
                                  borderRadius: "6px",
                                  fontSize: "0.75rem",
                                  fontWeight: 600,
                                }}
                              >
                                🎧 Müşteri ({seg.speaker})
                              </span>
                            ) : (
                              <strong style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>
                                {seg.speaker}
                              </strong>
                            )}
                          </div>
                        )}
                        <p>{renderHighlightedText(seg.text, segHits)}</p>
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
