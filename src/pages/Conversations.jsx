import { Fragment, useState, useEffect } from "react";
import { ChevronRight } from "lucide-react";
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
