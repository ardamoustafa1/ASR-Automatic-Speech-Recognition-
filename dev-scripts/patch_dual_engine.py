import os

FILE = "ASR/ASR.py"

with open(FILE, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Insert MLXWhisperWrapper before load_whisper_model
wrapper_code = '''
class MLXWhisperWrapper:
    """Mac cihazlar için donanımsal hızlandırıcı MLX kullanan adaptör sınıf."""
    def __init__(self, actual_model: str):
        self.repo_name = f"mlx-community/whisper-{actual_model}"
        import streamlit as st
        st.toast(f"🍏 Apple Silicon MLX motoru devrede! ({self.repo_name})", icon="🚀")
        
    def transcribe(self, audio, **kwargs):
        import mlx_whisper
        from collections import namedtuple
        
        mlx_kwargs = {}
        for key in ["temperature", "compression_ratio_threshold", "logprob_threshold", 
                    "no_speech_threshold", "condition_on_previous_text", "initial_prompt", 
                    "word_timestamps", "language"]:
            if key in kwargs:
                mlx_kwargs[key] = kwargs[key]
                
        # Optional: beam_size isn't natively exposed via standard transcribe kwargs in the same way,
        # but we can pass it to decode_options if we want, or just let MLX use its fast greedy search.
                
        res = mlx_whisper.transcribe(audio, path_or_hf_repo=self.repo_name, **mlx_kwargs)
        
        Segment = namedtuple("Segment", ["start", "end", "text", "avg_logprob", "no_speech_prob", "compression_ratio", "words"])
        segments = []
        for s in res.get("segments", []):
            segments.append(Segment(
                start=float(s.get("start", 0)),
                end=float(s.get("end", 0)),
                text=str(s.get("text", "")),
                avg_logprob=float(s.get("avg_logprob", 0)),
                no_speech_prob=float(s.get("no_speech_prob", 0)),
                compression_ratio=float(s.get("compression_ratio", 1.0)),
                words=s.get("words", [])
            ))
            
        class Info:
            language = mlx_kwargs.get("language", "tr")
            language_probability = 1.0
            
        return segments, Info()

@st.cache_resource
def load_whisper_model(model_size: str, engine_type: str = "Windows"):
'''

content = content.replace("@st.cache_resource\ndef load_whisper_model(model_size: str):", wrapper_code)

# 2. Modify the inside of load_whisper_model
old_load = '''def load_whisper_model(model_size: str, engine_type: str = "Windows"):
    """Faster-Whisper modelini yükler (yerel cache - hızlı yükleme)."""
    try:
        WhisperModel = get_whisper_model_class()
        actual_model = resolve_model_name(model_size)
        cpu_threads = max(4, os.cpu_count() or 4)'''

new_load = '''def load_whisper_model(model_size: str, engine_type: str = "Windows"):
    """Faster-Whisper veya MLX modelini yükler (donanıma göre)."""
    try:
        actual_model = resolve_model_name(model_size)
        if "Mac" in engine_type:
            return MLXWhisperWrapper(actual_model)
            
        WhisperModel = get_whisper_model_class()
        cpu_threads = max(4, os.cpu_count() or 4)'''
        
content = content.replace(old_load, new_load)

# 3. Inject hardware_engine radio button in Sidebar
old_ui = '''    st.markdown('<div class="sidebar-separator"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">Model stratejisi</div>', unsafe_allow_html=True)'''

new_ui = '''    st.markdown('<div class="sidebar-separator"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">Donanım Motoru (Hardware)</div>', unsafe_allow_html=True)
    
    hardware_engine = st.radio(
        "Altyapı",
        ["🖥️ Windows (Nvidia CUDA / Standart)", "🍏 Mac (Apple Silicon MLX - Çok Hızlı)"],
        index=0,
        help="Sistemi çalıştırdığınız bilgisayara göre motor seçin. Apple M serisi kullanıyorsanız Mac seçeneği 10 kat daha hızlı sonuç verir.",
        label_visibility="collapsed"
    )
    st.session_state["hardware_engine"] = hardware_engine

    st.markdown('<div class="sidebar-separator"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">Model stratejisi</div>', unsafe_allow_html=True)'''

content = content.replace(old_ui, new_ui)

# 4. Modify load_whisper_model call in UI to pass hardware_engine
old_call1 = '''    model, classifier = load_models()'''

new_call1 = '''    model, classifier = load_models(st.session_state.get("hardware_engine", "Windows"))'''

content = content.replace(old_call1, new_call1)

# We must update `load_models` definition to accept and pass the arg
old_load_models = '''def load_models():
    """Modelleri yükler ve spinner gösterir."""
    with st.spinner("Modeller yükleniyor (Sadece ilk açılışta sürer)..."):
        model = load_whisper_model(st.session_state.get("model_size", DEFAULT_MODEL_SIZE))
        classifier = load_toxicity_classifier()
        return model, classifier'''

new_load_models = '''def load_models(engine_type="Windows"):
    """Modelleri yükler ve spinner gösterir."""
    with st.spinner("Modeller yükleniyor (Sadece ilk açılışta sürer)..."):
        model = load_whisper_model(st.session_state.get("model_size", DEFAULT_MODEL_SIZE), engine_type)
        classifier = load_toxicity_classifier()
        return model, classifier'''

content = content.replace(old_load_models, new_load_models)


with open(FILE, "w", encoding="utf-8") as f:
    f.write(content)

print("Dual Engine Patch Applied.")
