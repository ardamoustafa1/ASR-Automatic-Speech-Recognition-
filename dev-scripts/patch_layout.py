
file_path = "ASR.py"
with open(file_path, encoding="utf-8") as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if line.startswith("def render_command_header("):
        start_idx = i
    if line.startswith("def render_empty_state("):
        # Find the end of render_empty_state
        for j in range(i, len(lines)):
            if lines[j].strip() == 'unsafe_allow_html=True,':
                # plus two lines to cover the closing parenthesis
                end_idx = j + 2
                break

if start_idx != -1 and end_idx != -1:
    new_code = """def render_command_header(mode, model_size, profile_key, domain_key, runtime_device):
    profile = ASR_PROFILES.get(profile_key, ASR_PROFILES["smart"])
    domain = get_domain_profile(domain_key)
    st.markdown(
        f'''
        <style>
        .command-header {{ display: flex; flex-direction: column; gap: 1.5rem; padding: 2rem !important; margin-bottom: 2rem !important; }}
        .header-row {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; }}
        .eyebrow {{ font-weight: 800; color: var(--asr-accent) !important; letter-spacing: 1px; margin-bottom: 0.5rem; text-transform: uppercase; font-size: 0.8rem; }}
        .command-header h1 {{ margin-bottom: 0.75rem !important; font-size: 1.8rem !important; font-weight: 700; line-height: 1.3; color: white !important; }}
        .subtitle {{ font-size: 1rem; color: var(--asr-muted) !important; max-width: 85%; line-height: 1.6; }}
        .data-strip {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.5rem; margin-top: 1rem; border-top: 1px solid var(--asr-border); padding-top: 1.5rem; }}
        .data-item {{ display: flex; flex-direction: column; gap: 0.4rem; padding: 0.5rem; }}
        .data-item strong {{ font-size: 1.3rem; color: #fff !important; font-weight: 800; }}
        .data-item span {{ font-size: 0.8rem; color: var(--asr-muted) !important; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 700; }}
        .status-line {{ display: flex; gap: 0.5rem; flex-wrap: wrap; justify-content: flex-end; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.25rem; margin-bottom: 1.5rem; }}
        .kpi-card {{ display: flex; flex-direction: column; gap: 0.5rem; padding: 1.25rem !important; }}
        .kpi-label {{ font-size: 0.75rem; text-transform: uppercase; font-weight: 800; color: var(--asr-muted) !important; }}
        .kpi-value {{ font-size: 1.75rem !important; font-weight: 800; color: white !important; }}
        .kpi-foot {{ font-size: 0.8rem; color: var(--asr-accent) !important; font-weight: 600; }}
        .panel {{ padding: 1.5rem !important; display: flex; flex-direction: column; gap: 0.5rem; }}
        .panel-title {{ display: flex; justify-content: space-between; align-items: center; font-size: 1.15rem; font-weight: 800; color: white !important; margin-bottom: 0.25rem; }}
        .panel-caption {{ font-size: 0.9rem; color: var(--asr-muted) !important; line-height: 1.5; margin-bottom: 1rem; }}
        .feature-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; margin-top: 0.5rem; }}
        .feature-pill {{ padding: 0.6rem 1rem !important; text-align: center; font-size: 0.85rem; font-weight: 600; border-radius: 8px !important; }}
        .empty-state {{ padding: 3rem 2rem !important; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; gap: 0.5rem; }}
        .empty-state strong {{ font-size: 1.1rem; color: white !important; }}
        .empty-state span {{ font-size: 0.9rem; color: var(--asr-muted) !important; }}
        </style>
        <div class="command-header">
            <div class="header-row">
                <div>
                    <div class="eyebrow">Enterprise Speech Intelligence</div>
                    <h1>Kurumsal ses kayıtlarını denetlenebilir metne dönüştür.</h1>
                    <div class="subtitle">
                        Adaptif kötü ses kurtarma, sektör sözlüğü ve kalite kapısı tek akışta. Hedef: Kelime doğruluğu %95+ ve WER %5 altı.
                    </div>
                </div>
                <div class="status-line">
                    <span class="chip good">Canlı Sistem</span>
                    <span class="chip info">{safe_html(runtime_device.upper())}</span>
                    <span class="chip warn">{safe_html(mode)}</span>
                </div>
            </div>
            <div class="data-strip">
                <div class="data-item"><strong>{safe_html(model_size.upper())}</strong><span>Model Sınıfı</span></div>
                <div class="data-item"><strong>{safe_html(profile.label)}</strong><span>ASR Profili</span></div>
                <div class="data-item"><strong>{safe_html(domain.label)}</strong><span>Sektör Sözlüğü</span></div>
                <div class="data-item"><strong>%{QUALITY_GATE_ACCURACY:.0f}</strong><span>Doğruluk Hedefi</span></div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

def render_kpi_grid(cpu_percent, ram_usage, target_latency_s, runtime_device):
    ram_percent = int(ram_usage.percent)
    active_calls = 4703
    daily_calls = 357
    agents = 94
    avg_duration = "2 dk 59 sn"
    st.markdown(
        f'''
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Temsilci Sayısı</div>
                <div class="kpi-value">{agents}</div>
                <div class="kpi-foot">Bugün aktif</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Çağrı Sayısı</div>
                <div class="kpi-value">{daily_calls}</div>
                <div class="kpi-foot">Günlük işlenen</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Aktif Çağrılar</div>
                <div class="kpi-value">{active_calls}</div>
                <div class="kpi-foot">Kuyrukta bekleyen</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Ortalama Süre</div>
                <div class="kpi-value">{avg_duration}</div>
                <div class="kpi-foot">SLA {target_latency_s} sn analiz</div>
            </div>
        </div>
        <div class="panel">
            <div class="panel-title">
                <span>Operasyon Özeti</span>
                <span class="chip info">CPU %{cpu_percent:.0f} · RAM %{ram_percent} · {safe_html(runtime_device.upper())}</span>
            </div>
            <div class="feature-grid">
                <div class="feature-pill">Zaman damgalı deşifre</div>
                <div class="feature-pill">Toksisite & risk skoru</div>
                <div class="feature-pill">Sektör sözlüğü</div>
                <div class="feature-pill">WER kalite kapısı</div>
                <div class="feature-pill">TXT, SRT, PDF çıktı</div>
                <div class="feature-pill">Toplu işleme</div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

def render_panel(title, caption="", right_label=""):
    right_html = f'<span class="chip info">{safe_html(right_label)}</span>' if right_label else ""
    st.markdown(
        f'''
        <div class="panel">
            <div class="panel-title"><span>{safe_html(title)}</span>{right_html}</div>
            <div class="panel-caption">{safe_html(caption)}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

def render_empty_state(title, subtitle):
    st.markdown(
        f'''
        <div class="empty-state">
            <div class="blinking-cursor"></div>
            <strong>{safe_html(title)}</strong>
            <span>{safe_html(subtitle)}</span>
        </div>
        ''',
        unsafe_allow_html=True,
    )
"""
    new_lines = lines[:start_idx] + [new_code + "\n"] + lines[end_idx:]
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print("UI layout rendering patched successfully!")
else:
    print(f"Failed to find block. Start: {start_idx}, End: {end_idx}")
