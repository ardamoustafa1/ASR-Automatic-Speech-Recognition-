import React from "react";
import { FolderOpen, AlertCircle } from "lucide-react";

export function EmptyState({ title = "Kayıt Bulunamadı", description = "Henüz veri oluşturulmamış.", icon: Icon = FolderOpen }) {
  return (
    <div style={{ padding: "3rem", textAlign: "center", color: "var(--asr-muted)" }}>
      <Icon size={48} style={{ margin: "0 auto 1rem", opacity: 0.5 }} />
      <h3 style={{ margin: "0 0 0.5rem", color: "var(--asr-text)" }}>{title}</h3>
      <p style={{ margin: 0 }}>{description}</p>
    </div>
  );
}

export function ErrorState({ title = "Bir Hata Oluştu", error = "Veri yüklenemedi.", retry }) {
  return (
    <div style={{ padding: "3rem", textAlign: "center", color: "var(--asr-danger)", background: "rgba(239, 68, 68, 0.05)", borderRadius: "var(--asr-radius)" }}>
      <AlertCircle size={48} style={{ margin: "0 auto 1rem" }} />
      <h3 style={{ margin: "0 0 0.5rem" }}>{title}</h3>
      <p style={{ margin: "0 0 1rem", opacity: 0.8 }}>{error?.message || String(error)}</p>
      {retry && (
        <button onClick={retry} className="primary-btn" style={{ padding: "0.5rem 1rem", border: "none", borderRadius: "4px", background: "var(--asr-accent)", color: "#fff", cursor: "pointer" }}>
          Tekrar Dene
        </button>
      )}
    </div>
  );
}
