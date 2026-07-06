import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  Mic,
  MicOff,
  Activity,
  CheckCircle,
  AlertTriangle,
  Brain,
  ShieldCheck,
  Copy,
  Trash2,
} from "lucide-react";
import { api } from "../api/client";

const WS_STATUS = {
  DISCONNECTED: "disconnected",
  CONNECTING: "connecting",
  CONNECTED: "connected",
};

function ScoreBadge({ label, value, icon: Icon, color }) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "0.25rem",
        background: "rgba(0,0,0,0.25)",
        borderRadius: "10px",
        padding: "0.75rem 1rem",
        border: `1px solid ${color}33`,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "0.4rem",
          color: "var(--asr-muted)",
          fontSize: "0.72rem",
          fontWeight: 600,
          textTransform: "uppercase",
          letterSpacing: "0.05em",
        }}
      >
        <Icon size={12} /> {label}
      </div>
      <div style={{ fontSize: "1.5rem", fontWeight: 800, color }}>{value}</div>
    </div>
  );
}

export default function LiveASRPage() {
  const [isRecording, setIsRecording] = useState(false);
  // committedText: server-confirmed (final) transcript, persisted to DB.
  // partialText: tentative, not-yet-committed tail — re-transcribed and replaced on every update.
  const [committedText, setCommittedText] = useState("");
  const [partialText, setPartialText] = useState("");
  const [wsStatus, setWsStatus] = useState(WS_STATUS.DISCONNECTED);
  const [latencyMs, setLatencyMs] = useState(null);
  const [coachingAlert, setCoachingAlert] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [isMockMode, setIsMockMode] = useState(false);
  const [wsError, setWsError] = useState(null);
  const mockIntervalRef = useRef(null);

  const mediaRecorderRef = useRef(null);
  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const committedTextRef = useRef("");

  const transcript = (committedText + (partialText ? " " + partialText : "")).trim();

  const stopMockMode = () => {
    if (mockIntervalRef.current) clearInterval(mockIntervalRef.current);
    setIsMockMode(false);
    setIsRecording(false);
    setWsStatus(WS_STATUS.DISCONNECTED);
  };

  const stopRecording = useCallback(() => {
    if (isMockMode) {
      stopMockMode();
      return;
    }
    mediaRecorderRef.current?.state === "recording" && mediaRecorderRef.current.stop();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    if (wsRef.current?.readyState === WebSocket.OPEN) wsRef.current.close();
    setIsRecording(false);
    setWsStatus(WS_STATUS.DISCONNECTED);
  }, [isMockMode]);

  // Resolve WebSocket base URL
  const wsBase = (import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1")
    .replace("http://", "ws://")
    .replace("https://", "wss://")
    .replace("/api/v1", "");

  useEffect(() => () => stopRecording(), [stopRecording]);

  // ── Debounced analysis trigger ──────────────────────────────────────────────
  const analyzeTranscript = useCallback(async (text) => {
    if (!text || text.trim().length < 10) return;
    setAnalyzing(true);
    try {
      const res = await api.analyzeText(text, "omni");
      setAnalysis(res);
    } catch {
      // silent — analysis is best-effort
    } finally {
      setAnalyzing(false);
    }
  }, []);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      setWsStatus(WS_STATUS.CONNECTING);
      setWsError(null);

      const ws = new WebSocket(`${wsBase}/ws/live-asr`);
      wsRef.current = ws;
      ws.binaryType = "arraybuffer";

      ws.onmessage = async (event) => {
        const msg = JSON.parse(event.data);

        if (msg.coaching_alert) {
          setCoachingAlert(msg.coaching_alert);
        }

        // ── Auth challenge-response (secure: token never in URL) ──────────────
        if (msg.type === "auth_required") {
          const token = localStorage.getItem("asr_token");
          ws.send(JSON.stringify({ type: "auth", token }));
          return;
        }

        if (msg.type === "auth_ok") {
          setWsStatus(WS_STATUS.CONNECTED);
          setIsRecording(true);
          setCommittedText("");
          setPartialText("");
          setAnalysis(null);
          setCoachingAlert(null);
          setWsError(null);
          committedTextRef.current = "";
          startMediaRecorder(stream, ws);
          return;
        }

        if (msg.type === "partial") {
          setPartialText(msg.text ?? "");
          setLatencyMs(msg.latency_ms ?? null);
          return;
        }

        if (msg.type === "final") {
          const fullText = msg.transcript_so_far ?? committedTextRef.current;
          committedTextRef.current = fullText;
          setCommittedText(fullText);
          setPartialText("");
          setLatencyMs(msg.latency_ms ?? null);
          // NLP analysis only runs on committed (final) text to avoid redundant
          // load and flicker while a partial hypothesis is still in flux.
          analyzeTranscript(fullText);
          return;
        }

        if (msg.type === "error") {
          setWsError(msg.message || "Ses akışı işlenirken bir hata oluştu.");
          return;
        }
      };

      ws.onclose = () => {
        setWsStatus(WS_STATUS.DISCONNECTED);
        setIsRecording(false);
      };

      ws.onerror = () => {
        setWsStatus(WS_STATUS.DISCONNECTED);
        setIsRecording(false);
        setWsError("Bağlantı hatası oluştu.");
      };
    } catch (err) {
      console.error("Microphone error:", err);
      const useMock = window.confirm("Mikrofon erişimi reddedildi veya bulunamadı. Demo (Mock) modu başlatılsın mı?");
      if (useMock) {
        setIsMockMode(true);
        startMockMode();
      } else {
        setWsStatus(WS_STATUS.DISCONNECTED);
      }
    }
  };

  const startMockMode = () => {
    setWsStatus(WS_STATUS.CONNECTED);
    setIsRecording(true);
    setCommittedText("");
    setPartialText("");
    setAnalysis(null);
    setWsError(null);
    committedTextRef.current = "";

    const mockPhrases = [
      "Merhaba, kolay gelsin. ",
      "Kredi kartı ekstremdeki bir işleme itiraz etmek istiyorum. ",
      "Gecikme faizi yansıtılmış ama ben ödemeyi yapmıştım. ",
      "Bunu hemen iptal eder misiniz? ",
      "Aksi takdirde kartımı kapatacağım."
    ];
    let currentIndex = 0;

    mockIntervalRef.current = setInterval(() => {
      if (currentIndex < mockPhrases.length) {
        committedTextRef.current += mockPhrases[currentIndex];
        setCommittedText(committedTextRef.current);
        setLatencyMs(Math.floor(Math.random() * 200) + 100);
        analyzeTranscript(committedTextRef.current);
        currentIndex++;
      } else {
        stopMockMode();
      }
    }, 2500);
  };

  const startMediaRecorder = (stream, ws) => {
    const mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
    mediaRecorderRef.current = mediaRecorder;
    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0 && ws.readyState === WebSocket.OPEN) {
        ws.send(event.data);
      }
    };
    mediaRecorder.start(2000); // 2-second chunks
  };

  // ── Derived analysis values ───────────────────────────────────────────────
  const hitCount = analysis?.hit_count ?? 0;
  const topHit = analysis?.hits?.[0];
  const statusColor =
    wsStatus === WS_STATUS.CONNECTED
      ? "var(--asr-success)"
      : wsStatus === WS_STATUS.CONNECTING
        ? "var(--asr-warning)"
        : "var(--asr-muted)";

  return (
    <div className="page-content">
      <header className="page-header">
        <h1>Canlı Analiz (Live ASR)</h1>
        <p>Mikrofon sesini gerçek zamanlı metne çevirin ve anlık NLP analizi yapın.</p>
      </header>

      {wsError && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.5rem",
            background: "rgba(239,68,68,0.1)",
            border: "1px solid rgba(239,68,68,0.3)",
            borderRadius: "8px",
            padding: "0.6rem 0.9rem",
            marginBottom: "1rem",
            color: "var(--asr-danger)",
            fontSize: "0.85rem",
          }}
        >
          <AlertTriangle size={16} />
          <span>{wsError}</span>
        </div>
      )}

      <div className="main-grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
        {/* ── Left: Controls ─────────────────────────────────────────────── */}
        <div
          className="glass-panel"
          style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}
        >
          <div className="panel-header">
            <span>Bağlantı</span>
            <h2>Mikrofon Kontrolü</h2>
          </div>

          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              padding: "1.5rem 0",
            }}
          >
            <button
              id="live-asr-toggle"
              onClick={isRecording ? stopRecording : startRecording}
              disabled={wsStatus === WS_STATUS.CONNECTING}
              style={{
                width: "120px",
                height: "120px",
                borderRadius: "50%",
                border: `2px solid ${isRecording ? "var(--asr-danger)" : "var(--asr-accent)"}`,
                background: isRecording ? "rgba(239,68,68,0.1)" : "rgba(99,102,241,0.1)",
                color: isRecording ? "var(--asr-danger)" : "var(--asr-accent)",
                cursor: wsStatus === WS_STATUS.CONNECTING ? "wait" : "pointer",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                gap: "0.5rem",
                fontWeight: 600,
                boxShadow: isRecording ? "0 0 24px rgba(239,68,68,0.4)" : "none",
                animation: isRecording ? "pulse 1.5s infinite" : "none",
                transition: "all 0.2s ease",
              }}
            >
              {isRecording ? <MicOff size={32} /> : <Mic size={32} />}
              <span style={{ fontSize: "0.8rem" }}>
                {wsStatus === WS_STATUS.CONNECTING
                  ? "Bağlanıyor..."
                  : isRecording
                    ? isMockMode ? "Mock Durdur" : "Durdur"
                    : "Dinle"}
              </span>
            </button>
          </div>

          {/* Status KPIs */}
          <div className="kpi-grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
            <div className="kpi-card">
              <span className="kpi-label">WS Durumu</span>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "0.5rem",
                  marginTop: "0.5rem",
                }}
              >
                {wsStatus === WS_STATUS.CONNECTED ? (
                  <CheckCircle size={16} color="var(--asr-success)" />
                ) : (
                  <AlertTriangle size={16} color={statusColor} />
                )}
                <span style={{ fontWeight: 700, color: statusColor, fontSize: "0.9rem" }}>
                  {wsStatus === WS_STATUS.CONNECTED
                    ? "Bağlı"
                    : wsStatus === WS_STATUS.CONNECTING
                      ? "Bağlanıyor"
                      : "Kopuk"}
                </span>
              </div>
            </div>
            <div className="kpi-card">
              <span className="kpi-label">Gecikme</span>
              <span className="kpi-value" style={{ fontSize: "1.1rem" }}>
                {latencyMs !== null ? `${latencyMs}ms` : "—"}
              </span>
            </div>
          </div>

          {/* Real-Time AI Coaching Banner */}
          {coachingAlert && (
            <div
              style={{
                marginBottom: "1.25rem",
                padding: "1rem 1.15rem",
                borderRadius: "10px",
                background:
                  coachingAlert.severity === "high"
                    ? "rgba(239, 68, 68, 0.15)"
                    : coachingAlert.severity === "warning"
                      ? "rgba(245, 158, 11, 0.15)"
                      : "rgba(59, 130, 246, 0.15)",
                borderLeft: `5px solid ${
                  coachingAlert.severity === "high"
                    ? "var(--asr-danger)"
                    : coachingAlert.severity === "warning"
                      ? "var(--asr-warning)"
                      : "var(--asr-accent)"
                }`,
                color: "var(--asr-fg)",
                display: "flex",
                flexDirection: "column",
                gap: "0.35rem",
                boxShadow: "0 4px 15px rgba(0,0,0,0.15)",
                animation: "fadeIn 0.3s ease",
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "0.5rem",
                  fontWeight: 700,
                  fontSize: "0.95rem",
                  color:
                    coachingAlert.severity === "high"
                      ? "var(--asr-danger)"
                      : coachingAlert.severity === "warning"
                        ? "var(--asr-warning)"
                        : "var(--asr-accent)",
                }}
              >
                <Activity size={18} />
                <span>{coachingAlert.title}</span>
                <span style={{ marginLeft: "auto", fontSize: "0.75rem", opacity: 0.8 }}>
                  {coachingAlert.timestamp_sec}s
                </span>
              </div>
              <div style={{ fontSize: "0.88rem", lineHeight: 1.4, opacity: 0.95 }}>
                {coachingAlert.message}
              </div>
            </div>
          )}

          {/* Real-time analysis scores */}
          {(analysis || analyzing) && (
            <div>
              <div
                style={{
                  fontSize: "0.72rem",
                  fontWeight: 600,
                  color: "var(--asr-muted)",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                  marginBottom: "0.75rem",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  {analyzing ? <Activity size={14} style={{ animation: "pulse 1.5s infinite" }} /> : <Activity size={14} />}
                  {analyzing ? "Analiz ediliyor..." : "Anlık Analiz"}
                </div>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.6rem" }}>
                <ScoreBadge
                  label="Kelime Eşleşmesi"
                  value={analyzing ? "…" : hitCount}
                  icon={ShieldCheck}
                  color={
                    hitCount > 3
                      ? "var(--asr-danger)"
                      : hitCount > 0
                        ? "var(--asr-warning)"
                        : "var(--asr-success)"
                  }
                />
                <ScoreBadge
                  label="Tespit"
                  value={analyzing ? "…" : topHit ? topHit.rule_name?.slice(0, 10) : "Temiz"}
                  icon={Brain}
                  color={topHit ? "var(--asr-warning)" : "var(--asr-success)"}
                />
              </div>
              {topHit && (
                <div
                  style={{
                    marginTop: "0.75rem",
                    padding: "0.6rem 0.8rem",
                    background: "rgba(239,68,68,0.1)",
                    border: "1px solid rgba(239,68,68,0.3)",
                    borderRadius: "8px",
                    fontSize: "0.78rem",
                    color: "var(--asr-danger)",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                    <AlertTriangle size={14} />
                    <strong>Eşleşme:</strong> "{topHit.matched_text}" — {topHit.rule_name}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── Right: Transcript ───────────────────────────────────────────── */}
        <div className="glass-panel" style={{ display: "flex", flexDirection: "column" }}>
          <div className="panel-header">
            <span>Canlı Sonuç</span>
            <h2>Deşifre Edilen Metin</h2>
          </div>

          <div
            style={{
              flex: 1,
              background: "rgba(0,0,0,0.3)",
              borderRadius: "8px",
              padding: "1.5rem",
              minHeight: "300px",
              maxHeight: "500px",
              overflowY: "auto",
              border: "1px solid var(--asr-stroke)",
            }}
          >
            {transcript ? (
              <p
                style={{
                  fontSize: "1.05rem",
                  lineHeight: "1.9",
                  color: "#e4e4e7",
                  whiteSpace: "pre-wrap",
                }}
              >
                {committedText}
                {partialText && (
                  <span style={{ color: "var(--asr-muted)", fontStyle: "italic" }}>
                    {committedText ? " " : ""}
                    {partialText}
                  </span>
                )}
              </p>
            ) : (
              <div className="empty-state">
                {isRecording ? (
                  <>
                    <Activity
                      size={32}
                      color="var(--asr-accent)"
                      style={{ animation: "pulse 1.5s infinite", marginBottom: "1rem" }}
                    />
                    <p>Dinliyorum, lütfen konuşun...</p>
                  </>
                ) : (
                  <>
                    <Mic size={32} color="var(--asr-muted)" style={{ marginBottom: "1rem" }} />
                    <p>Kayıt başladığında metin burada görünecek.</p>
                  </>
                )}
              </div>
            )}
          </div>

          {transcript && (
            <div style={{ marginTop: "0.75rem", display: "flex", gap: "0.5rem" }}>
              <button
                className="btn-secondary"
                onClick={() => navigator.clipboard.writeText(transcript)}
                style={{ fontSize: "0.8rem", padding: "0.4rem 0.8rem", display: "flex", alignItems: "center", gap: "0.5rem" }}
              >
                <Copy size={14} /> Kopyala
              </button>
              <button
                className="btn-secondary"
                onClick={() => {
                  setCommittedText("");
                  setPartialText("");
                  setAnalysis(null);
                  committedTextRef.current = "";
                }}
                style={{ fontSize: "0.8rem", padding: "0.4rem 0.8rem", display: "flex", alignItems: "center", gap: "0.5rem" }}
              >
                <Trash2 size={14} /> Temizle
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
