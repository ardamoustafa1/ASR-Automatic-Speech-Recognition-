import { useEffect } from "react";
import {
  Activity,
  AudioLines,
  BarChart3,
  Bell,
  Hash,
  MessageSquareText,
  PanelLeft,
  UploadCloud,
  Mic,
} from "lucide-react";
import { Routes, Route, NavLink, useNavigate } from "react-router-dom";
import { Toaster, toast } from "react-hot-toast";

import DashboardPage from "./pages/Dashboard";
import KeywordsPage from "./pages/Keywords";
import AnalyticsPage from "./pages/Analytics";
import ConversationsPage from "./pages/Conversations";
import AlertsPage from "./pages/Alerts";
import LiveASRPage from "./pages/LiveASR";
import LoginPage from "./pages/Login";
import ErrorBoundary from "./components/ErrorBoundary";
import { api } from "./api/client";
import { useAppStore } from "./store/useAppStore";
import "./styles.css";

// Move AnalyzePage inside pages later if needed, kept here for brevity based on existing code
import { Play, FileAudio } from "lucide-react";
import { useState } from "react";
import { Skeleton, SkeletonList } from "./components/common/Skeleton";

function AnalyzePage() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("idle");
  const [result, setResult] = useState(null);
  const [textInput, setTextInput] = useState(
    "Fatura itirazı için aramıştım. Geçen ayki zam yanlış yansımış, iade talep ediyorum."
  );

  const analyzeText = async () => {
    setStatus("analyzing");
    try {
      const res = await api.analyzeText(textInput);
      const segments = [{ start: 0, end: 0, text: textInput }];
      const saved = await api.analyze({
        segments,
        full_transcript: textInput,
        sector: "omni",
      });
      setResult({ ...saved, preview: textInput });
      toast.success("Analiz başarıyla tamamlandı!");
      setStatus("done");
    } catch (e) {
      console.error(e);
      toast.error("Analiz sırasında bir hata oluştu.");
      setStatus("idle");
    }
  };

  return (
    <div className="page-content">
      <header className="page-header">
        <h1>Hızlı Analiz</h1>
        <p>Metin girerek anahtar kelime tespiti test edin</p>
      </header>
      <div className="main-grid">
        <div className="glass-panel">
          <div className="panel-header">
            <span>Metin Girişi</span>
            <h2>Örnek Görüşme</h2>
          </div>
          <textarea
            className="analyze-textarea"
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            rows={6}
          />
          <button className="btn-primary" disabled={status === "analyzing"} onClick={analyzeText}>
            {status === "analyzing" ? (
              "Analiz ediliyor..."
            ) : (
              <>
                <Play size={16} /> Analizi Başlat
              </>
            )}
          </button>
        </div>
        <div className="glass-panel">
          <div className="panel-header">
            <span>Sonuç</span>
            <h2>Tespit Edilen Kelimeler</h2>
          </div>
          {status === "analyzing" ? (
            <SkeletonList count={4} height="30px" />
          ) : status === "done" && result ? (
            <>
              <div className="chip-row">
                {result.topics?.map((t) => (
                  <span className="chip topic" key={t.topic_id}>
                    {t.label_tr}
                  </span>
                ))}
              </div>
              <div className="hits-timeline">
                {result.hits?.map((h, i) => (
                  <div className="hit-pill" key={i}>
                    <strong>{h.matched_text}</strong>
                    <span>{h.rule_name}</span>
                    <span className="hit-type">{h.match_type}</span>
                  </div>
                ))}
              </div>
              {result.hit_count === 0 && <p className="text-muted">Eşleşme bulunamadı</p>}
            </>
          ) : (
            <div className="empty-state compact">
              <FileAudio size={40} className="icon" />
              <p>Analiz sonuçları burada görünecek</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const NAV = [
  { to: "/", icon: Activity, label: "Dashboard" },
  { to: "/live", icon: Mic, label: "Canlı ASR (WebSockets)" },
  { to: "/analyze", icon: UploadCloud, label: "Toplu Analiz" },
  { to: "/conversations", icon: MessageSquareText, label: "Görüşmeler" },
  { to: "/keywords", icon: Hash, label: "Kelime Kuralları" },
  { to: "/analytics", icon: BarChart3, label: "Trend Analitiği" },
  { to: "/alerts", icon: Bell, label: "Uyarılar" },
];

function App() {
  const token = useAppStore((state) => state.token);
  const setToken = useAppStore((state) => state.setToken);
  const sidebarOpen = useAppStore((state) => state.sidebarOpen);
  const setSidebarOpen = useAppStore((state) => state.setSidebarOpen);
  const alertCount = useAppStore((state) => state.alertCount);
  const setAlertCount = useAppStore((state) => state.setAlertCount);

  useEffect(() => {
    if (token) {
      api
        .alerts(false)
        .then((a) => setAlertCount(a.length))
        .catch(() => {
          toast.error("Uyarılar yüklenemedi");
        });
    }
  }, [token, setAlertCount]);

  const handleLogout = () => {
    setToken(null);
    toast.success("Çıkış yapıldı");
  };

  if (!token) {
    return (
      <>
        <Toaster position="top-right" />
        <LoginPage
          onLogin={(newToken) => {
            setToken(newToken);
            toast.success("Giriş başarılı");
          }}
        />
      </>
    );
  }

  return (
    <div className="app-container">
      <Toaster position="top-right" />
      <aside className={`sidebar ${sidebarOpen ? "" : "collapsed"}`}>
        <div className="brand-block">
          <div className="brand-icon">
            <AudioLines size={24} />
          </div>
          <div className="brand-text">
            <h1>ASR Command</h1>
            <span>Kelime & Konu Tespiti</span>
          </div>
        </div>

        <div className="section-title">Modüller</div>
        <nav className="nav-list">
          {NAV.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}
            >
              <Icon size={18} />
              <span>{label}</span>
              {label === "Uyarılar" && alertCount > 0 && (
                <span className="nav-badge">{alertCount}</span>
              )}
            </NavLink>
          ))}

          <button
            className="nav-item"
            onClick={handleLogout}
            style={{
              border: "none",
              background: "none",
              width: "100%",
              textAlign: "left",
              cursor: "pointer",
              color: "var(--asr-danger)",
              marginTop: "auto",
            }}
          >
            <span>Çıkış Yap</span>
          </button>
        </nav>
      </aside>

      <main className="workspace full">
        <button className="sidebar-toggle" onClick={() => setSidebarOpen(!sidebarOpen)}>
          <PanelLeft size={18} />
        </button>
        <ErrorBoundary>
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/live" element={<LiveASRPage />} />
            <Route path="/analyze" element={<AnalyzePage />} />
            <Route path="/conversations" element={<ConversationsPage />} />
            <Route path="/keywords" element={<KeywordsPage />} />
            <Route path="/analytics" element={<AnalyticsPage />} />
            <Route path="/alerts" element={<AlertsPage />} />
          </Routes>
        </ErrorBoundary>
      </main>
    </div>
  );
}

export default App;
