import React, { useState } from "react";
import { Lock, User } from "lucide-react";
import { api } from "../api/client";

export default function LoginPage({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  
  React.useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("expired")) {
      setError("Oturum süreniz doldu veya token geçersiz. Lütfen tekrar giriş yapın.");
      window.history.replaceState({}, document.title, window.location.pathname);
    }
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const res = await api.login(username, password);
      localStorage.setItem("asr_token", res.access_token);
      onLogin(res.access_token);
    } catch {
      setError("Giriş başarısız. Lütfen bilgilerinizi kontrol edin.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        display: "flex",
        height: "100vh",
        alignItems: "center",
        justifyContent: "center",
        background: "var(--asr-bg)",
      }}
    >
      <div
        className="glass-panel"
        style={{
          width: "400px",
          padding: "3rem",
          display: "flex",
          flexDirection: "column",
          gap: "1.5rem",
        }}
      >
        <div style={{ textAlign: "center", marginBottom: "1rem" }}>
          <Lock size={48} color="var(--asr-accent)" style={{ marginBottom: "1rem" }} />
          <h2>Kurumsal Giriş</h2>
          <p className="panel-caption">Lütfen yetkili hesabınızla giriş yapın.</p>
        </div>

        {error && (
          <div className="chip warn" style={{ textAlign: "center", marginBottom: "1rem" }}>
            {error}
          </div>
        )}
        
        <div style={{ background: "rgba(66, 153, 225, 0.1)", padding: "1rem", borderRadius: "8px", fontSize: "0.85rem", color: "var(--asr-accent)", border: "1px solid rgba(66, 153, 225, 0.2)" }}>
          <strong>İlk Kurulum (Admin Onboarding):</strong><br/>
          Sisteme ilk kez giriş yapıyorsanız lütfen <code>.env</code> dosyasındaki <code>ASR_ADMIN_PASSWORD</code> bilgisini kullanın. (Varsayılan kullanıcı adı: <code>admin</code>)
        </div>

        <form
          onSubmit={handleSubmit}
          style={{ display: "flex", flexDirection: "column", gap: "1rem" }}
        >
          <div>
            <label
              style={{
                display: "block",
                marginBottom: "0.5rem",
                fontSize: "0.85rem",
                color: "var(--asr-muted)",
              }}
            >
              Kullanıcı Adı
            </label>
            <div style={{ position: "relative" }}>
              <User
                size={18}
                style={{
                  position: "absolute",
                  left: "12px",
                  top: "12px",
                  color: "var(--asr-muted)",
                }}
              />
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                style={{ width: "100%", padding: "10px 10px 10px 40px", boxSizing: "border-box" }}
                placeholder="örn: admin"
                required
              />
            </div>
          </div>

          <div>
            <label
              style={{
                display: "block",
                marginBottom: "0.5rem",
                fontSize: "0.85rem",
                color: "var(--asr-muted)",
              }}
            >
              Şifre
            </label>
            <div style={{ position: "relative" }}>
              <Lock
                size={18}
                style={{
                  position: "absolute",
                  left: "12px",
                  top: "12px",
                  color: "var(--asr-muted)",
                }}
              />
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                style={{ width: "100%", padding: "10px 10px 10px 40px", boxSizing: "border-box" }}
                placeholder="••••••••"
                required
              />
            </div>
          </div>

          <button
            type="submit"
            className="btn-primary"
            style={{ marginTop: "1rem" }}
            disabled={loading}
          >
            {loading ? "Giriş Yapılıyor..." : "Giriş Yap"}
          </button>
        </form>
      </div>
    </div>
  );
}
