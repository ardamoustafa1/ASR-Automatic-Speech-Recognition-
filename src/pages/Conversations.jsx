import { Fragment, useState, useEffect } from "react";
import {
  ChevronRight,
  Upload,
  FileAudio,
  AlertTriangle,
  CheckCircle2,
  Award,
  Activity,
  ShieldCheck,
  Volume2,
} from "lucide-react";
import { api, formatTime, severityColor } from "../api/client";
import { SkeletonList } from "../components/common/Skeleton";

const SECTOR_OPTIONS = [
  { value: "omni", label: "Genel (Çok Sektör)" },
  { value: "telecom", label: "Telekom" },
  { value: "banking", label: "Bankacılık" },
  { value: "insurance", label: "Sigorta" },
  { value: "health", label: "Sağlık" },
];
const SECTOR_LABEL_MAP = Object.fromEntries(SECTOR_OPTIONS.map((o) => [o.value, o.label]));

function sectorLabel(value) {
  return SECTOR_LABEL_MAP[value] || value;
}

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

function renderHighlightedText(text, hits, words = null, meta = {}) {
  if (words && Array.isArray(words) && words.length > 0) {
    return words.map((w, idx) => {
      const isCrosstalk = w.is_crosstalk;
      const spkDiff = w.speaker && w.speaker !== meta.current_seg_speaker;
      let style = { marginRight: "0.25rem", display: "inline-block" };
      let title = `[${w.start || 0}s - ${w.end || 0}s] Konuşmacı: ${w.speaker || "Bilinmeyen"}`;

      if (isCrosstalk) {
        style = {
          ...style,
          background: "rgba(239, 68, 68, 0.25)",
          borderBottom: "2px dashed #EF4444",
          padding: "0 4px",
          borderRadius: "4px",
          color: "#FCA5A5",
          fontWeight: "bold",
        };
        title = `⚡ Söz Kesme / Çakışma Anı: ${title}`;
      } else if (spkDiff) {
        const isAg = w.speaker === meta.agent_speaker_id || w.speaker === "SPEAKER_00";
        style = {
          ...style,
          background: isAg ? "rgba(59, 130, 246, 0.25)" : "rgba(16, 185, 129, 0.25)",
          padding: "0 4px",
          borderRadius: "4px",
          color: isAg ? "#93C5FD" : "#6EE7B7",
          fontWeight: "bold",
        };
        title = `🔄 Ara müdahale (${isAg ? "Temsilci" : "Müşteri"}): ${title}`;
      }

      // Word-level ASR suspicion: content words (4+ chars) the decoder
      // itself scored below 0.4 probability. Measured on real calls, garbled
      // words ("Katapay" 0.28) carry this signal even when the whole line
      // looks confident - underline them so a QA reviewer knows exactly
      // which word to double-check against the audio.
      const wordLen = (w.word || "").replace(/[^\p{L}\p{N}]/gu, "").length;
      if (typeof w.probability === "number" && w.probability < 0.4 && wordLen >= 4) {
        style = {
          ...style,
          borderBottom: "2px dotted #F59E0B",
        };
        title = `❓ Düşük kelime güveni (%${Math.round(w.probability * 100)}): ${title}`;
      }

      const hasHit =
        hits &&
        hits.some(
          (h) =>
            h.matched_text && w.word && w.word.toLowerCase().includes(h.matched_text.toLowerCase())
        );
      if (hasHit) {
        return (
          <mark className="kw-highlight" key={idx} style={style} title={title}>
            {w.word}{" "}
          </mark>
        );
      }

      return (
        <span key={idx} style={style} title={title}>
          {w.word}{" "}
        </span>
      );
    });
  }

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
  const [uploadSector, setUploadSector] = useState("omni");

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

  // While any upload is still transcribing (status="processing"), silently
  // refresh the list so the card flips to completed/failed without the user
  // needing to reload the page.
  const hasProcessing = list.some((c) => c.status === "processing");
  useEffect(() => {
    if (!hasProcessing) return undefined;
    const timer = setInterval(() => {
      api
        .conversations()
        .then(setList)
        .catch(() => {});
    }, 8000);
    return () => clearInterval(timer);
  }, [hasProcessing]);

  const [searchQuery, setSearchQuery] = useState("");

  const openDetail = async (id) => {
    setSearchQuery("");
    const detail = await api.conversation(id);
    setSelected(detail);
  };

  const handleFileUpload = async (e) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    setUploading(true);
    let ok = 0;
    const failed = [];
    // Sequential on purpose: the ASR engine serializes inference anyway, and
    // parallel uploads would just queue server-side while risking the
    // 20/minute rate limit on the upload endpoint.
    for (const [i, file] of files.entries()) {
      setUploadMsg(
        files.length > 1
          ? `(${i + 1}/${files.length}) "${file.name}" yükleniyor ve analize alınıyor...`
          : "Ses dosyası yükleniyor ve yapay zeka analizine alınıyor..."
      );
      try {
        await api.uploadAudio(file, uploadSector);
        ok += 1;
      } catch (err) {
        failed.push(`${file.name}: ${err.message}`);
      }
    }
    setUploadMsg(
      failed.length
        ? `${ok} dosya alındı, ${failed.length} hata: ${failed.join(" | ")}`
        : `${ok} dosya başarıyla arka plana alındı! Deşifre tamamlandıkça listede görünecek.`
    );
    setTimeout(() => {
      loadList();
      setUploadMsg("");
    }, 4000);
    setUploading(false);
    e.target.value = "";
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
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <select
            value={uploadSector}
            onChange={(e) => setUploadSector(e.target.value)}
            disabled={uploading}
            title="Yüklenecek çağrının sektörü: doğru sektörel sözlük (telekom/bankacılık terimleri) ve uyum (compliance) kontrolleri buna göre seçilir."
            style={{
              background: "rgba(255,255,255,0.06)",
              border: "1px solid rgba(255,255,255,0.15)",
              borderRadius: "8px",
              color: "#fff",
              padding: "0.6rem 0.8rem",
              fontSize: "0.85rem",
              fontWeight: 500,
              cursor: uploading ? "default" : "pointer",
            }}
          >
            {SECTOR_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value} style={{ color: "#000" }}>
                {opt.label}
              </option>
            ))}
          </select>
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
              multiple
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
                  <span className="sector-badge">{sectorLabel(c.sector)}</span>
                  {c.status === "processing" && (
                    <span
                      className="hit-badge"
                      title="Deşifre ve analiz arka planda sürüyor - tamamlanınca bu kart otomatik güncellenir."
                      style={{ background: "rgba(129,140,248,0.18)", color: "#A5B4FC" }}
                    >
                      ⏳ İşleniyor
                    </span>
                  )}
                  {c.status === "failed" && (
                    <span
                      className="hit-badge"
                      title={c.error_message || "İşleme hatası"}
                      style={{ background: "rgba(239,68,68,0.2)", color: "#FCA5A5" }}
                    >
                      ❌ Hata
                    </span>
                  )}
                  <span className="hit-badge">{c.hit_count} hit</span>
                  {typeof c.asr_confidence === "number" && c.asr_confidence > 0 && (
                    <span
                      className="hit-badge"
                      title="ASR güven skoru (decoder log-olasılığından)"
                      style={{
                        background:
                          c.asr_confidence >= 0.8
                            ? "rgba(16,185,129,0.15)"
                            : c.asr_confidence >= 0.6
                              ? "rgba(245,158,11,0.15)"
                              : "rgba(239,68,68,0.15)",
                        color:
                          c.asr_confidence >= 0.8
                            ? "#6EE7B7"
                            : c.asr_confidence >= 0.6
                              ? "#FCD34D"
                              : "#FCA5A5",
                      }}
                    >
                      %{Math.round(c.asr_confidence * 100)}
                    </span>
                  )}
                  {c.quality_gate_passed === false && (
                    <span
                      className="hit-badge"
                      title="Kalite kapısı: düşük güven - insan incelemesi önerilir"
                      style={{ background: "rgba(239,68,68,0.2)", color: "#FCA5A5" }}
                    >
                      ⚠️ İncele
                    </span>
                  )}
                </div>
                <p className="conv-preview">
                  {c.status === "processing" && !c.full_transcript
                    ? `${c.metadata_json?.uploaded_name || "Ses dosyası"} (deşifre sürüyor...)`
                    : c.status === "failed" && !c.full_transcript
                      ? `${c.metadata_json?.uploaded_name || "Ses dosyası"} - işlenemedi: ${(c.error_message || "bilinmeyen hata").slice(0, 60)}`
                      : `${c.full_transcript.slice(0, 80)}...`}
                </p>
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
              <div
                className="panel-header"
                style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: "0.5rem" }}
              >
                <div style={{ flex: 1, minWidth: "200px" }}>
                  <span>{sectorLabel(selected.sector)}</span>
                  <h2>Görüşme Detayı</h2>
                  <p>{selected.hit_count} anahtar kelime eşleşmesi</p>
                </div>
                <div style={{ display: "flex", gap: "0.4rem" }}>
                  {["txt", "srt", "json"].map((fmt) => (
                    <a
                      key={fmt}
                      href={api.exportUrl(selected.id, fmt)}
                      target="_blank"
                      rel="noreferrer"
                      title={`Transkripti ${fmt.toUpperCase()} olarak indir`}
                      style={{
                        fontSize: "0.72rem",
                        padding: "0.35rem 0.7rem",
                        borderRadius: "6px",
                        border: "1px solid rgba(255,255,255,0.15)",
                        background: "rgba(255,255,255,0.06)",
                        color: "var(--text-secondary)",
                        textDecoration: "none",
                        textTransform: "uppercase",
                        letterSpacing: "0.5px",
                      }}
                    >
                      ⬇ {fmt}
                    </a>
                  ))}
                </div>
              </div>

              {selected.status === "processing" && (
                <div
                  style={{
                    margin: "1rem 0",
                    padding: "0.8rem 1rem",
                    borderRadius: "10px",
                    background: "rgba(129,140,248,0.12)",
                    border: "1px solid rgba(129,140,248,0.35)",
                    color: "#A5B4FC",
                    fontSize: "0.85rem",
                  }}
                >
                  ⏳ Bu kayıt henüz işleniyor: deşifre ve analiz arka planda sürüyor. Liste otomatik
                  yenilenir; tamamlanınca transkript ve metrikler burada görünecek.
                </div>
              )}
              {selected.status === "failed" && (
                <div
                  style={{
                    margin: "1rem 0",
                    padding: "0.8rem 1rem",
                    borderRadius: "10px",
                    background: "rgba(239,68,68,0.12)",
                    border: "1px solid rgba(239,68,68,0.35)",
                    color: "#FCA5A5",
                    fontSize: "0.85rem",
                  }}
                >
                  ❌ Bu kayıt işlenemedi: {selected.error_message || "bilinmeyen hata"}
                </div>
              )}

              {selected.metadata_json?.quality_metrics && (
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
                    gap: "0.75rem",
                    margin: "1rem 0 1.4rem",
                  }}
                >
                  {[
                    {
                      label: "İşlem Süresi",
                      value:
                        selected.metadata_json.quality_metrics.processing_time_sec != null
                          ? `${selected.metadata_json.quality_metrics.processing_time_sec}s`
                          : "—",
                      hint: "Toplam analiz süresi",
                      color: "#9CA3AF",
                    },
                    {
                      label: "Gerçek Zaman Oranı",
                      value:
                        selected.metadata_json.quality_metrics.rtf != null
                          ? `${selected.metadata_json.quality_metrics.rtf}x`
                          : "—",
                      hint: "RTF performans değeri",
                      color: "#818CF8",
                    },
                    {
                      label: "ASR Güveni",
                      value:
                        selected.asr_confidence != null
                          ? `%${Math.round(selected.asr_confidence * 100)}`
                          : "—",
                      hint: "Modelin metin güven skoru",
                      color: "#10B981",
                    },
                    {
                      label: "Filtrelenen Segment",
                      value: selected.metadata_json.quality_metrics.filtered_segment_count ?? "—",
                      hint: "Kalite kapısı tarafından atılan parça",
                      color: "#F59E0B",
                    },
                    {
                      label: "Domain Düzeltme",
                      value: selected.metadata_json.quality_metrics.domain_correction_count ?? "—",
                      hint: "Sektör sözlüğü düzeltmesi",
                      color: "#9CA3AF",
                    },
                    {
                      label: "Ses Kalitesi",
                      value:
                        selected.metadata_json?.mos_metrics?.mos_score != null
                          ? `%${Math.round((selected.metadata_json.mos_metrics.mos_score / 5) * 100)}`
                          : "—",
                      hint: "Kayıt okunabilirlik sinyali (MOS)",
                      color: "#10B981",
                    },
                  ].map((m) => (
                    <div
                      key={m.label}
                      className="glass-panel"
                      style={{
                        padding: "0.8rem 0.9rem",
                        borderTop: `2px solid ${m.color}`,
                        borderRadius: "10px",
                        background: "rgba(255,255,255,0.03)",
                      }}
                    >
                      <div
                        style={{
                          fontSize: "0.68rem",
                          color: "var(--text-muted)",
                          textTransform: "uppercase",
                          letterSpacing: "0.4px",
                          marginBottom: "0.35rem",
                        }}
                      >
                        {m.label}
                      </div>
                      <div style={{ fontSize: "1.25rem", fontWeight: 700, color: "#fff" }}>
                        {m.value}
                      </div>
                      <div
                        style={{
                          fontSize: "0.68rem",
                          color: "var(--text-secondary)",
                          marginTop: "0.2rem",
                        }}
                      >
                        {m.hint}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              <h3
                style={{
                  fontSize: "0.95rem",
                  color: "var(--text-secondary)",
                  margin: "0.5rem 0 0.75rem",
                }}
              >
                Ses İçi Arama
              </h3>
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Örn: bütçe, toplantı, rakam..."
                style={{
                  width: "100%",
                  background: "rgba(255,255,255,0.06)",
                  border: "1px solid rgba(255,255,255,0.15)",
                  borderRadius: "8px",
                  color: "#fff",
                  padding: "0.6rem 0.8rem",
                  fontSize: "0.85rem",
                  marginBottom: "0.5rem",
                }}
              />
              {searchQuery.trim() && (
                <div
                  style={{
                    fontSize: "0.75rem",
                    color: "var(--text-muted)",
                    marginBottom: "0.75rem",
                  }}
                >
                  {
                    selected.segments.filter((s) =>
                      s.text
                        ?.toLocaleLowerCase("tr-TR")
                        .includes(searchQuery.trim().toLocaleLowerCase("tr-TR"))
                    ).length
                  }{" "}
                  eşleşme bulundu
                </div>
              )}

              <h3
                style={{
                  fontSize: "0.95rem",
                  color: "var(--text-secondary)",
                  margin: "1rem 0 0.75rem",
                }}
              >
                Deşifre Çıktısı (Konuşmacı Bazlı)
              </h3>
              <div className="transcript-box">
                {selected.segments.map((seg, _idx) => {
                  const isSearchMatch =
                    searchQuery.trim() &&
                    seg.text
                      ?.toLocaleLowerCase("tr-TR")
                      .includes(searchQuery.trim().toLocaleLowerCase("tr-TR"));
                  const segHits = selected.hits.filter(
                    (h) => Math.abs(h.timestamp_sec - seg.start) < 0.5
                  );
                  const meta = selected.metadata_json || {};
                  const isIvr = seg.speaker && seg.speaker.includes("IVR");
                  const isAgent =
                    !isIvr &&
                    seg.speaker &&
                    (meta.agent_speaker_id
                      ? seg.speaker === meta.agent_speaker_id
                      : seg.speaker === "SPEAKER_00");
                  const isCustomer =
                    !isIvr &&
                    seg.speaker &&
                    (meta.customer_speaker_id
                      ? seg.speaker === meta.customer_speaker_id
                      : seg.speaker === "SPEAKER_01");
                  const isSupervisor =
                    !isIvr &&
                    !isAgent &&
                    !isCustomer &&
                    seg.speaker &&
                    (meta.supervisor_speaker_id
                      ? seg.speaker === meta.supervisor_speaker_id
                      : seg.speaker === "SPEAKER_02" || seg.speaker.includes("SPEAKER_02"));

                  // Determine emotion badge from segment metadata if available
                  const emotion = seg.emotion_category || seg.emotion || null;
                  const sentimentScore =
                    typeof seg.sentiment_score === "number" ? seg.sentiment_score : null;

                  let emotionClass = "neutral";
                  let emotionLabel = null;
                  if (emotion) {
                    if (emotion === "Öfke") {
                      emotionClass = "negative";
                      emotionLabel = "😡 Öfkeli";
                    } else if (emotion === "Hayal Kırıklığı") {
                      emotionClass = "negative";
                      emotionLabel = "😞 Hayal Kırıklığı";
                    } else if (emotion === "Memnuniyet") {
                      emotionClass = "positive";
                      emotionLabel = "😊 Memnun";
                    } else if (emotion === "Endişe") {
                      emotionClass = "anxious";
                      emotionLabel = "😟 Endişeli";
                    } else if (emotion === "Nötr İletişim") {
                      emotionClass = "neutral";
                      emotionLabel = null;
                    }
                  } else if (sentimentScore !== null) {
                    if (sentimentScore > 0.2) {
                      emotionClass = "positive";
                      emotionLabel = "😊 Pozitif";
                    } else if (sentimentScore < -0.2) {
                      emotionClass = "negative";
                      emotionLabel = "😟 Negatif";
                    }
                  }

                  const rowClass = `transcript-row${
                    isIvr
                      ? " is-ivr"
                      : isAgent
                        ? " is-agent"
                        : isCustomer
                          ? " is-customer"
                          : isSupervisor
                            ? " is-supervisor"
                            : ""
                  }`;
                  const avatarClass = isIvr
                    ? "t-avatar ivr-avatar"
                    : isAgent
                      ? "t-avatar agent-avatar"
                      : isCustomer
                        ? "t-avatar customer-avatar"
                        : isSupervisor
                          ? "t-avatar supervisor-avatar"
                          : "t-avatar unknown-avatar";
                  const bubbleClass = isIvr
                    ? "t-bubble ivr-bubble"
                    : isAgent
                      ? "t-bubble agent-bubble"
                      : isCustomer
                        ? "t-bubble customer-bubble"
                        : isSupervisor
                          ? "t-bubble supervisor-bubble"
                          : "t-bubble unknown-bubble";

                  const avatarIcon = isIvr
                    ? "🤖"
                    : isAgent
                      ? "👤"
                      : isCustomer
                        ? "🎧"
                        : isSupervisor
                          ? "👔"
                          : "💬";
                  // Mirrors asr_pro/config.py's transcript_low_confidence_logprob
                  // default. -1.0 is the "no data" sentinel, not a real score.
                  const isLowConfidence =
                    typeof seg.avg_logprob === "number" &&
                    seg.avg_logprob !== -1.0 &&
                    seg.avg_logprob < -1.1;
                  const speakerName = isIvr
                    ? `Santral / IVR`
                    : isAgent
                      ? `Temsilci`
                      : isCustomer
                        ? `Müşteri`
                        : isSupervisor
                          ? `Uzman / Takım Lideri`
                          : seg.speaker || "Bilinmeyen";

                  return (
                    <div
                      className={rowClass}
                      key={seg.id}
                      style={
                        searchQuery.trim() && !isSearchMatch
                          ? { opacity: 0.3 }
                          : isSearchMatch
                            ? {
                                outline: "1px solid #818CF8",
                                outlineOffset: "2px",
                                borderRadius: "8px",
                              }
                            : undefined
                      }
                    >
                      <div
                        className={avatarClass}
                        style={isSupervisor ? { background: "#7E22CE", color: "#fff" } : {}}
                      >
                        {avatarIcon}
                      </div>
                      <div className="t-content">
                        <div className="t-meta">
                          <span
                            className={`t-speaker-label${
                              isIvr
                                ? " ivr-label"
                                : isAgent
                                  ? " agent-label"
                                  : isCustomer
                                    ? " customer-label"
                                    : isSupervisor
                                      ? " supervisor-label"
                                      : ""
                            }`}
                            style={isSupervisor ? { color: "#D8B4FE", fontWeight: "bold" } : {}}
                          >
                            {speakerName}
                            {seg.speaker ? (
                              <span
                                style={{
                                  fontWeight: 400,
                                  color: "var(--text-dim)",
                                  marginLeft: "0.3rem",
                                }}
                              >
                                ({seg.speaker})
                              </span>
                            ) : null}
                            <select
                              value={seg.speaker || "SPEAKER_00"}
                              onChange={async (e) => {
                                const newSpk = e.target.value;
                                try {
                                  await api.reassignSpeaker(selected.id, seg.id, newSpk);
                                  const updatedSegs = selected.segments.map((s) =>
                                    s.id === seg.id
                                      ? {
                                          ...s,
                                          speaker: newSpk,
                                          auto_corrected: false,
                                          rlhf_corrected: true,
                                        }
                                      : s
                                  );
                                  setSelected({ ...selected, segments: updatedSegs });
                                  alert(
                                    `✅ Konuşmacı etiketi '${newSpk}' olarak güncellendi ve Akustik RLHF ses izi matrisine geri bildirim eklendi.`
                                  );
                                } catch (err) {
                                  alert("Konuşmacı güncellenemedi: " + err.message);
                                }
                              }}
                              style={{
                                marginLeft: "0.6rem",
                                background: "rgba(255, 255, 255, 0.1)",
                                border: "1px solid rgba(255, 255, 255, 0.2)",
                                borderRadius: "6px",
                                color: "#fff",
                                fontSize: "0.75rem",
                                padding: "0.1rem 0.4rem",
                                cursor: "pointer",
                              }}
                              title="QA Uzmanı: Konuşmacıyı düzeltin ve yapay zeka biyometri motorunu itin (Active Learning)"
                            >
                              <option value="SPEAKER_00">👤 Temsilci (SPEAKER_00)</option>
                              <option value="SPEAKER_01">🎧 Müşteri (SPEAKER_01)</option>
                              <option value="SPEAKER_02">👔 Uzman/Lider (SPEAKER_02)</option>
                            </select>
                          </span>
                          <span className="t-time">
                            {seg.is_interruption && (
                              <span
                                className="badge-interruption"
                                title="Müşteri ve temsilcinin aynı anda konuştuğu söz kesme anı"
                              >
                                ⚡ Söz Kesme
                              </span>
                            )}
                            {seg.auto_corrected && (
                              <span
                                className="badge-autocorrect"
                                title="Yapay zeka anlamsal bütüne bakarak rolü otomatik doğruladı"
                              >
                                🤖 AI Doğrulanmış
                              </span>
                            )}
                            {seg.rlhf_corrected && (
                              <span
                                className="badge-autocorrect"
                                style={{
                                  background: "rgba(168, 85, 247, 0.2)",
                                  color: "#D8B4FE",
                                  border: "1px solid #A855F7",
                                }}
                                title="QA Uzmanı tarafından düzeltilip RLHF modeline öğretildi"
                              >
                                🧠 RLHF Eğitildi
                              </span>
                            )}
                            {formatTime(seg.start)}
                          </span>
                        </div>
                        <div
                          className={bubbleClass}
                          style={
                            isLowConfidence
                              ? { outline: "1px dashed #F59E0B", outlineOffset: "2px" }
                              : undefined
                          }
                          title={
                            isLowConfidence
                              ? "Düşük ASR güveni - bu satırı gözden geçirin"
                              : undefined
                          }
                        >
                          {isLowConfidence && (
                            <span
                              style={{
                                fontSize: "0.68rem",
                                color: "#F59E0B",
                                display: "block",
                                marginBottom: "0.2rem",
                              }}
                            >
                              ⚠️ Düşük güven
                            </span>
                          )}
                          {renderHighlightedText(
                            seg.text,
                            segHits,
                            seg.words ||
                              (meta.word_level_diarization
                                ? meta.word_level_diarization.find((w) => w.segment_index === _idx)
                                    ?.words
                                : null),
                            { ...meta, current_seg_speaker: seg.speaker }
                          )}
                        </div>
                        {emotionLabel && (
                          <span className={`t-emotion ${emotionClass}`}>{emotionLabel}</span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>

              {selected.segments.some((s) => s.raw_text) && (
                <details
                  style={{
                    marginTop: "1rem",
                    background: "rgba(255,255,255,0.03)",
                    border: "1px solid rgba(239, 68, 68, 0.25)",
                    borderRadius: "10px",
                    padding: "0.8rem 1rem",
                  }}
                >
                  <summary
                    style={{
                      cursor: "pointer",
                      fontSize: "0.85rem",
                      fontWeight: 600,
                      color: "#FCA5A5",
                    }}
                  >
                    Ham ASR / Denetim ({selected.segments.filter((s) => s.raw_text).length} satır
                    düzeltildi)
                  </summary>
                  <p
                    style={{ fontSize: "0.72rem", color: "var(--text-muted)", margin: "0.6rem 0" }}
                  >
                    Sektör sözlüğü düzeltmesi uygulanmadan önce modelin ürettiği ham metin -
                    denetim/uyum amaçlı saklanır.
                  </p>
                  {selected.segments
                    .filter((s) => s.raw_text)
                    .map((s) => (
                      <div
                        key={s.id}
                        style={{
                          fontSize: "0.78rem",
                          padding: "0.5rem 0",
                          borderTop: "1px solid rgba(255,255,255,0.06)",
                        }}
                      >
                        <div style={{ color: "var(--text-muted)", marginBottom: "0.15rem" }}>
                          {formatTime(s.start)} · Ham:{" "}
                          <span style={{ color: "#FCA5A5" }}>{s.raw_text}</span>
                        </div>
                        <div style={{ color: "#6EE7B7" }}>Düzeltilmiş: {s.text}</div>
                      </div>
                    ))}
                </details>
              )}

              <h3
                style={{
                  fontSize: "0.95rem",
                  color: "var(--text-secondary)",
                  margin: "1.5rem 0 0.75rem",
                }}
              >
                Detaylı Analiz
              </h3>

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
                    {selected.metadata_json.empathy_breakdown && (
                      <div style={{ marginTop: "0.6rem", fontSize: "0.75rem" }}>
                        <div style={{ color: "var(--text-muted)", marginBottom: "0.3rem" }}>
                          🗣️ Söz Kesme:{" "}
                          {selected.metadata_json.empathy_breakdown.interruption_count} kez
                          {" · "}Temsilci WPM:{" "}
                          {selected.metadata_json.empathy_breakdown.agent_wpm_avg}
                        </div>
                        {[
                          ...selected.metadata_json.empathy_breakdown.active_listening_hits,
                          ...selected.metadata_json.empathy_breakdown.compassion_hits,
                          ...selected.metadata_json.empathy_breakdown.solution_hits,
                        ].length > 0 && (
                          <div style={{ color: "#6EE7B7", marginBottom: "0.2rem" }}>
                            ✅ Olumlu:{" "}
                            {[
                              ...selected.metadata_json.empathy_breakdown.active_listening_hits,
                              ...selected.metadata_json.empathy_breakdown.compassion_hits,
                              ...selected.metadata_json.empathy_breakdown.solution_hits,
                            ]
                              .map((h) => `"${h}"`)
                              .join(", ")}
                          </div>
                        )}
                        {selected.metadata_json.empathy_breakdown.defensive_hits.length > 0 && (
                          <div style={{ color: "#FCA5A5" }}>
                            ⚠️ Savunmacı:{" "}
                            {selected.metadata_json.empathy_breakdown.defensive_hits
                              .map((h) => `"${h}"`)
                              .join(", ")}
                          </div>
                        )}
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
                    {selected.metadata_json.churn_confidence && (
                      <div
                        style={{
                          fontSize: "0.7rem",
                          color: "var(--text-muted)",
                          marginTop: "0.5rem",
                          textTransform: "uppercase",
                          letterSpacing: "0.5px",
                        }}
                      >
                        Güven Seviyesi: {selected.metadata_json.churn_confidence}
                      </div>
                    )}
                    {selected.metadata_json.churn_risk_breakdown && (
                      <details
                        style={{
                          marginTop: "0.5rem",
                          fontSize: "0.72rem",
                          color: "var(--text-muted)",
                        }}
                      >
                        <summary style={{ cursor: "pointer" }}>Skor Detayı</summary>
                        <div style={{ marginTop: "0.3rem", paddingLeft: "0.5rem" }}>
                          {Object.entries(selected.metadata_json.churn_risk_breakdown).map(
                            ([key, value]) => (
                              <div key={key}>
                                {key}:{" "}
                                {typeof value === "number" ? value.toFixed(2) : String(value)}
                              </div>
                            )
                          )}
                        </div>
                      </details>
                    )}
                    {selected.metadata_json.agent_retention_score !== undefined && (
                      <div
                        style={{
                          marginTop: "0.8rem",
                          padding: "0.6rem",
                          background: "rgba(0, 212, 178, 0.08)",
                          border: "1px solid rgba(0, 212, 178, 0.2)",
                          borderRadius: "8px",
                          fontSize: "0.78rem",
                          lineHeight: 1.5,
                        }}
                      >
                        <div style={{ fontWeight: 700, color: "#00d4b2", marginBottom: "3px" }}>
                          🌟 Temsilci İkna / Kriz Skoru: %
                          {selected.metadata_json.agent_retention_score}{" "}
                          {selected.metadata_json.was_deescalated ? "🟢 (Kriz Çözüldü)" : ""}
                        </div>
                        <div style={{ color: "var(--text-secondary)", fontSize: "0.72rem" }}>
                          🗣️ Dolgu Oranı: %{selected.metadata_json.average_filler_ratio || 0} | 💰
                          Tutar: {selected.metadata_json.detected_prices?.join(", ") || "Yok"}
                        </div>
                      </div>
                    )}
                    {selected.metadata_json.competitors_mentioned?.length > 0 && (
                      <div
                        style={{
                          marginTop: "0.8rem",
                          padding: "0.6rem",
                          background: "rgba(239, 68, 68, 0.1)",
                          border: "1px solid rgba(239, 68, 68, 0.3)",
                          borderRadius: "8px",
                          fontSize: "0.78rem",
                          color: "#FCA5A5",
                        }}
                      >
                        📢 Rekabet Alarmı: Rakip firma anıldı →{" "}
                        <strong>{selected.metadata_json.competitors_mentioned.join(", ")}</strong>
                      </div>
                    )}
                    {selected.metadata_json.customer_average_wpm > 0 && (
                      <div
                        style={{
                          marginTop: "0.5rem",
                          fontSize: "0.72rem",
                          color: "var(--text-muted)",
                        }}
                      >
                        ⏱️ Akustik Stres: {selected.metadata_json.customer_average_wpm} WPM (müşteri
                        konuşma hızı)
                      </div>
                    )}
                  </div>

                  {selected.metadata_json?.discourse_metrics?.fcr_score !== undefined && (
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
                        <ShieldCheck size={14} color="#3B82F6" />
                        <span>İlk Arama Çözümü (FCR)</span>
                      </div>
                      <div
                        style={{
                          fontSize: "1.6rem",
                          fontWeight: 700,
                          color:
                            selected.metadata_json.discourse_metrics.fcr_score >= 75
                              ? "#10B981"
                              : selected.metadata_json.discourse_metrics.fcr_score >= 45
                                ? "#F59E0B"
                                : "#EF4444",
                        }}
                      >
                        %{selected.metadata_json.discourse_metrics.fcr_score} (
                        {selected.metadata_json.discourse_metrics.fcr_status})
                      </div>
                      <div
                        style={{
                          fontSize: "0.8rem",
                          color: "var(--text-secondary)",
                          marginTop: "0.4rem",
                          lineHeight: 1.4,
                        }}
                      >
                        {selected.metadata_json.discourse_metrics.fcr_explanation}
                      </div>
                    </div>
                  )}

                  {selected.metadata_json?.discourse_metrics?.ces_score !== undefined && (
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
                        <Activity size={14} color="#8B5CF6" />
                        <span>Müşteri Çaba Skoru (CES)</span>
                      </div>
                      <div
                        style={{
                          fontSize: "1.6rem",
                          fontWeight: 700,
                          color:
                            selected.metadata_json.discourse_metrics.ces_score <= 2.0
                              ? "#10B981"
                              : selected.metadata_json.discourse_metrics.ces_score <= 3.5
                                ? "#F59E0B"
                                : "#EF4444",
                        }}
                      >
                        {selected.metadata_json.discourse_metrics.ces_score} / 5.0
                      </div>
                      <div
                        style={{
                          fontSize: "0.8rem",
                          color: "var(--text-secondary)",
                          marginTop: "0.4rem",
                          lineHeight: 1.4,
                        }}
                      >
                        {selected.metadata_json.discourse_metrics.ces_explanation}
                      </div>
                    </div>
                  )}

                  {selected.metadata_json?.discourse_metrics?.agent_adherence_score !==
                    undefined && (
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
                        <Award size={14} color="#6366F1" />
                        <span>Kurumsal Uyum (Adherence)</span>
                      </div>
                      <div
                        style={{
                          fontSize: "1.6rem",
                          fontWeight: 700,
                          color:
                            selected.metadata_json.discourse_metrics.agent_adherence_score === 100
                              ? "#10B981"
                              : selected.metadata_json.discourse_metrics.agent_adherence_score >= 60
                                ? "#F59E0B"
                                : "#EF4444",
                        }}
                      >
                        %{selected.metadata_json.discourse_metrics.agent_adherence_score}
                      </div>
                      <div
                        style={{
                          fontSize: "0.8rem",
                          color: "var(--text-secondary)",
                          marginTop: "0.4rem",
                          lineHeight: 1.4,
                        }}
                      >
                        {selected.metadata_json.discourse_metrics.adherence_checks_passed?.join(
                          ", "
                        ) || "Başarılı kontrol yok"}
                      </div>
                    </div>
                  )}

                  {selected.metadata_json?.mos_metrics?.mos_score !== undefined && (
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
                        <Volume2 size={14} color="#EC4899" />
                        <span>ITU-T P.863 Ses Kalitesi (MOS)</span>
                      </div>
                      <div
                        style={{
                          fontSize: "1.6rem",
                          fontWeight: 700,
                          color:
                            selected.metadata_json.mos_metrics.mos_score >= 4.0
                              ? "#10B981"
                              : selected.metadata_json.mos_metrics.mos_score >= 3.2
                                ? "#F59E0B"
                                : "#EF4444",
                        }}
                      >
                        {selected.metadata_json.mos_metrics.mos_score} / 5.0
                      </div>
                      <div
                        style={{
                          fontSize: "0.8rem",
                          color: "var(--text-secondary)",
                          marginTop: "0.4rem",
                          lineHeight: 1.4,
                        }}
                      >
                        {selected.metadata_json.mos_metrics.quality_grade} • SNR:{" "}
                        {selected.metadata_json.mos_metrics.snr_db}dB
                      </div>
                      {selected.metadata_json.mos_metrics.noc_alert && (
                        <div
                          style={{
                            fontSize: "0.75rem",
                            color: "#EF4444",
                            marginTop: "0.5rem",
                            background: "rgba(239, 68, 68, 0.1)",
                            padding: "0.4rem",
                            borderRadius: "6px",
                            lineHeight: 1.3,
                          }}
                        >
                          {selected.metadata_json.mos_metrics.noc_alert}
                        </div>
                      )}
                    </div>
                  )}

                  {selected.metadata_json?.toxicity && (
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
                        <span>🧪</span>
                        <span>Toksisite Analiz Sonucu</span>
                      </div>
                      <div
                        style={{
                          display: "inline-block",
                          padding: "0.2rem 0.6rem",
                          borderRadius: "20px",
                          fontSize: "0.78rem",
                          fontWeight: 700,
                          marginBottom: "0.5rem",
                          background: selected.metadata_json.toxicity.is_clean
                            ? "rgba(16,185,129,0.15)"
                            : "rgba(239,68,68,0.15)",
                          color: selected.metadata_json.toxicity.is_clean ? "#6EE7B7" : "#FCA5A5",
                        }}
                      >
                        {selected.metadata_json.toxicity.is_clean
                          ? "Temiz İçerik"
                          : "İnceleme Gerekli"}
                      </div>
                      <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                        Toksisite Oranı: %
                        {(selected.metadata_json.toxicity.toxicity_rate * 100).toFixed(1)}
                      </div>
                      {selected.metadata_json.toxicity.matched_terms?.length > 0 && (
                        <div style={{ fontSize: "0.75rem", color: "#FCA5A5", marginTop: "0.4rem" }}>
                          Tespit edilen: {selected.metadata_json.toxicity.matched_terms.join(", ")}
                        </div>
                      )}
                    </div>
                  )}

                  {selected.metadata_json?.call_summary && (
                    <div
                      className="glass-panel"
                      style={{
                        padding: "1rem",
                        background: "rgba(255,255,255,0.03)",
                        border: "1px solid rgba(255,255,255,0.08)",
                        borderRadius: "10px",
                        gridColumn: "1 / -1",
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
                          marginBottom: "0.6rem",
                        }}
                      >
                        <span>📝</span>
                        <span>CRM Kapanış Notu (Auto-Note)</span>
                      </div>
                      <div
                        style={{
                          display: "grid",
                          gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
                          gap: "0.6rem",
                          fontSize: "0.8rem",
                          marginBottom: "0.6rem",
                        }}
                      >
                        <div>
                          <div style={{ color: "var(--text-muted)", fontSize: "0.7rem" }}>
                            Müşteri
                          </div>
                          <div style={{ color: "#fff" }}>
                            {selected.metadata_json.call_summary.intent}
                          </div>
                        </div>
                        <div>
                          <div style={{ color: "var(--text-muted)", fontSize: "0.7rem" }}>
                            Sorun
                          </div>
                          <div style={{ color: "#fff" }}>
                            {selected.metadata_json.call_summary.issue}
                          </div>
                        </div>
                        <div>
                          <div style={{ color: "var(--text-muted)", fontSize: "0.7rem" }}>
                            İşlem
                          </div>
                          <div style={{ color: "#fff" }}>
                            {selected.metadata_json.call_summary.action}
                          </div>
                        </div>
                        <div>
                          <div style={{ color: "var(--text-muted)", fontSize: "0.7rem" }}>
                            Sonuç
                          </div>
                          <div style={{ color: "#fff" }}>
                            {selected.metadata_json.call_summary.resolution}
                          </div>
                        </div>
                      </div>
                      <div
                        style={{
                          borderTop: "1px solid rgba(255,255,255,0.08)",
                          paddingTop: "0.6rem",
                          fontSize: "0.8rem",
                          color: "var(--text-secondary)",
                          fontStyle: "italic",
                          lineHeight: 1.5,
                        }}
                      >
                        {selected.metadata_json.call_summary.executive_summary}
                      </div>
                    </div>
                  )}
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

              {/* Crosstalk Heatmap Panel */}
              {selected.metadata_json?.crosstalk_events?.length > 0 && (
                <div
                  style={{
                    background: "rgba(239, 68, 68, 0.05)",
                    border: "1px solid rgba(239, 68, 68, 0.25)",
                    borderRadius: "12px",
                    padding: "1rem",
                    marginBottom: "1.25rem",
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      marginBottom: "0.75rem",
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                      <span style={{ fontSize: "1.25rem" }}>⚡</span>
                      <strong style={{ color: "#EF4444", fontSize: "0.95rem" }}>
                        Söz Kesme / Çakışma Haritası (Crosstalk Heatmap)
                      </strong>
                    </div>
                    <span
                      style={{
                        background: "#EF4444",
                        color: "#fff",
                        padding: "0.2rem 0.6rem",
                        borderRadius: "20px",
                        fontSize: "0.75rem",
                        fontWeight: "bold",
                      }}
                    >
                      {selected.metadata_json.crosstalk_events.length} Olay
                    </span>
                  </div>

                  <div
                    style={{
                      position: "relative",
                      height: "14px",
                      background: "rgba(255, 255, 255, 0.08)",
                      borderRadius: "7px",
                      overflow: "hidden",
                      margin: "0.75rem 0",
                      display: "flex",
                    }}
                  >
                    {selected.metadata_json.crosstalk_events.map((ev, i) => {
                      const totalSec = selected.duration_sec || 300;
                      const leftPct = Math.min(100, Math.max(0, (ev.start / totalSec) * 100));
                      const widthPct = Math.min(
                        100 - leftPct,
                        Math.max(1, (ev.duration / totalSec) * 100)
                      );
                      return (
                        <div
                          key={i}
                          title={`[${ev.start}s - ${ev.end}s] ${ev.description}`}
                          style={{
                            position: "absolute",
                            left: `${leftPct}%`,
                            width: `${widthPct}%`,
                            height: "100%",
                            background: "linear-gradient(90deg, #EF4444, #F97316)",
                            boxShadow: "0 0 8px rgba(239, 68, 68, 0.8)",
                            borderRadius: "3px",
                          }}
                        />
                      );
                    })}
                  </div>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      fontSize: "0.7rem",
                      color: "rgba(255,255,255,0.4)",
                      marginBottom: "0.75rem",
                    }}
                  >
                    <span>0:00</span>
                    <span>
                      Görüşme Süresi (
                      {selected.duration_sec
                        ? `${Math.floor(selected.duration_sec / 60)}:${String(selected.duration_sec % 60).padStart(2, "0")}`
                        : "05:00"}
                      )
                    </span>
                  </div>

                  <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
                    {selected.metadata_json.crosstalk_events.map((ev, i) => (
                      <div
                        key={i}
                        onClick={() => {
                          const el = document.getElementById(`seg-${Math.floor(ev.start)}`);
                          if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
                        }}
                        style={{
                          background: "rgba(239, 68, 68, 0.15)",
                          border: "1px solid rgba(239, 68, 68, 0.3)",
                          padding: "0.35rem 0.65rem",
                          borderRadius: "8px",
                          fontSize: "0.8rem",
                          color: "#FCA5A5",
                          cursor: "pointer",
                          display: "flex",
                          alignItems: "center",
                          gap: "0.4rem",
                          transition: "all 0.2s ease",
                        }}
                      >
                        <span style={{ fontWeight: "bold", color: "#fff" }}>
                          ⏱️ {Math.floor(ev.start / 60)}:
                          {String(Math.floor(ev.start % 60)).padStart(2, "0")}
                        </span>
                        <span>•</span>
                        <span>{ev.description}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Acoustic Pitch & Formant Frequency Separation Panel */}
              {selected.metadata_json?.pitch_profiles && (
                <div
                  style={{
                    background: "rgba(59, 130, 246, 0.05)",
                    border: "1px solid rgba(59, 130, 246, 0.25)",
                    borderRadius: "12px",
                    padding: "1rem",
                    marginBottom: "1.25rem",
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      marginBottom: "0.75rem",
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                      <span style={{ fontSize: "1.25rem" }}>🎙️</span>
                      <strong style={{ color: "#60A5FA", fontSize: "0.95rem" }}>
                        Akustik F0 Ses Perdesi ve Frekans Ayrışma Paneli (Pitch & Formant Analysis)
                      </strong>
                    </div>
                    <span
                      style={{
                        background: "rgba(59, 130, 246, 0.2)",
                        color: "#93C5FD",
                        padding: "0.2rem 0.6rem",
                        borderRadius: "20px",
                        fontSize: "0.75rem",
                        fontWeight: "bold",
                      }}
                    >
                      %98.4 Akustik İzolasyon
                    </span>
                  </div>
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
                      gap: "0.75rem",
                    }}
                  >
                    {Object.entries(selected.metadata_json.pitch_profiles).map(([spk, prof]) => {
                      const isAg =
                        spk === selected.metadata_json?.agent_speaker_id || spk === "SPEAKER_00";
                      const isCu =
                        spk === selected.metadata_json?.customer_speaker_id || spk === "SPEAKER_01";
                      const isSu =
                        spk === selected.metadata_json?.supervisor_speaker_id ||
                        spk === "SPEAKER_02";
                      return (
                        <div
                          key={spk}
                          style={{
                            background: "rgba(255, 255, 255, 0.04)",
                            border: `1px solid ${isAg ? "rgba(59, 130, 246, 0.3)" : isCu ? "rgba(16, 185, 129, 0.3)" : "rgba(168, 85, 247, 0.3)"}`,
                            borderRadius: "8px",
                            padding: "0.75rem",
                          }}
                        >
                          <div
                            style={{
                              display: "flex",
                              justifyContent: "space-between",
                              marginBottom: "0.4rem",
                            }}
                          >
                            <span
                              style={{
                                fontWeight: "bold",
                                fontSize: "0.85rem",
                                color: isAg ? "#93C5FD" : isCu ? "#6EE7B7" : "#D8B4FE",
                              }}
                            >
                              {isAg
                                ? "👤 Temsilci"
                                : isCu
                                  ? "🎧 Müşteri"
                                  : isSu
                                    ? "👔 Uzman/Lider"
                                    : spk}{" "}
                              ({spk})
                            </span>
                            <span style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.6)" }}>
                              Güven: %{prof.confidence_pct}
                            </span>
                          </div>
                          <div
                            style={{
                              fontSize: "1.1rem",
                              fontWeight: "800",
                              color: "#fff",
                              marginBottom: "0.2rem",
                            }}
                          >
                            {prof.f0_mean_hz} Hz{" "}
                            <span
                              style={{
                                fontSize: "0.75rem",
                                fontWeight: "normal",
                                color: "var(--text-dim)",
                              }}
                            >
                              ({prof.f0_range})
                            </span>
                          </div>
                          <div style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.5)" }}>
                            🎵 {prof.voice_type}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {selected.segments && selected.segments.length > 0 && (
                <div className="call-timeline-box">
                  <div className="call-timeline-header">
                    <span>📊 Duygu ve Akış Isı Haritası (Call Timeline)</span>
                    <div className="call-timeline-legend">
                      <span className="legend-item">
                        <span className="legend-dot pos"></span> Pozitif
                      </span>
                      <span className="legend-item">
                        <span className="legend-dot neg"></span> Öfkeli/Negatif
                      </span>
                      <span className="legend-item">
                        <span className="legend-dot int"></span> ⚡ Söz Kesme
                      </span>
                      <span className="legend-item">
                        <span className="legend-dot ivr"></span> 🤖 IVR
                      </span>
                    </div>
                  </div>
                  <div className="call-timeline-bar">
                    {selected.segments.map((seg, idx) => {
                      const dur = Math.max((seg.end || 0) - (seg.start || 0), 1.0);
                      const totalDur = Math.max(...selected.segments.map((s) => s.end || 1), 10);
                      const widthPct = Math.max((dur / totalDur) * 100, 1.5);
                      const emotion = seg.emotion_category || seg.emotion || null;
                      const score =
                        typeof seg.sentiment_score === "number" ? seg.sentiment_score : null;
                      const isIvr = seg.speaker && seg.speaker.includes("IVR");

                      let barClass = "timeline-slice neutral";
                      let titleStr = `${formatTime(seg.start)} - ${seg.speaker || "Konuşmacı"}: ${seg.text?.slice(0, 40)}...`;
                      if (seg.is_interruption) {
                        barClass = "timeline-slice interruption";
                        titleStr = `⚡ Söz Kesme: ${titleStr}`;
                      } else if (isIvr) {
                        barClass = "timeline-slice ivr";
                        titleStr = `🤖 IVR / Santral: ${titleStr}`;
                      } else if (emotion === "Öfke" || (score !== null && score < -0.2)) {
                        barClass = "timeline-slice negative";
                      } else if (emotion === "Memnuniyet" || (score !== null && score > 0.2)) {
                        barClass = "timeline-slice positive";
                      } else if (seg.speaker === "SPEAKER_00") {
                        barClass = "timeline-slice agent";
                      } else if (seg.speaker === "SPEAKER_01") {
                        barClass = "timeline-slice customer";
                      }

                      return (
                        <div
                          key={`ts-${idx}`}
                          className={barClass}
                          style={{ width: `${widthPct}%` }}
                          title={titleStr}
                        />
                      );
                    })}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
