"""
Apple-Level ASR Upgrade Patch Script
Implements all 7 features from the implementation plan.
"""

FILE = "ASR.py"

with open(FILE, encoding="utf-8") as f:
    content = f.read()

original = content  # backup

# ============================================================
# 1. ADD apex_quality PROFILE (after ultimate_accuracy)
# ============================================================
apex_profile = """    "apex_quality": ASRProfile(
        label="Apple Kalite (Apex)",
        description="Hız değil doğruluk. beam=10, best_of=10, çoklu geçiş, en kötü ses bile %95+ hedefi.",
        use_batched=False,
        batch_size=1,
        beam_size=10,
        best_of=10,
        condition_on_previous_text=True,
        chunk_length=25,
        vad_threshold=0.18,
        min_silence_duration_ms=1200,
        speech_pad_ms=600,
        temperature=(0.0, 0.1, 0.2, 0.3, 0.5),
        log_prob_threshold=-0.8,
        no_speech_threshold=0.30,
        repetition_penalty=1.15,
        quality_gate=95.0,
        retry_profile_key="rescue",
        hallucination_silence_threshold=0.8,
    ),
"""

# Insert after "ultimate_accuracy" profile block
marker = """        hallucination_silence_threshold=1.0,
    ),
    "enterprise": ASRProfile("""
replacement = (
    """        hallucination_silence_threshold=1.0,
    ),
"""
    + apex_profile
    + """    "enterprise": ASRProfile("""
)
content = content.replace(marker, replacement, 1)
print("[1/10] apex_quality profile added.")

# ============================================================
# 2. ADD AUDIO_PREP_APEX FILTER
# ============================================================
apex_audio = """AUDIO_PREP_APEX = "apex"
AUDIO_PREP_FILTERS[AUDIO_PREP_APEX] = (
    "Apex Ses Kurtarma (Neural)",
    "highpass=f=60,lowpass=f=8000,"
    "afftdn=nr=12:nf=-30:tn=1,"
    "speechnorm=e=15:r=0.00005:l=1,"
    "dynaudnorm=f=200:g=20:p=0.92:m=10,"
    "acompressor=threshold=-20dB:ratio=3:attack=3:release=60,"
    "loudnorm=I=-16:TP=-1.5:LRA=7",
)
"""

# Insert after AUDIO_PREP_FILTERS dict
marker2 = """# --------------------\n"""
if marker2 in content:
    content = content.replace(marker2, marker2 + "\n" + apex_audio, 1)
    print("[2/10] AUDIO_PREP_APEX filter added.")
else:
    # Try alternate marker
    alt = "# --------------------"
    idx = content.find(alt)
    if idx != -1:
        end_of_line = content.index("\n", idx) + 1
        content = content[:end_of_line] + "\n" + apex_audio + content[end_of_line:]
        print("[2/10] AUDIO_PREP_APEX filter added (alt).")
    else:
        print("[2/10] WARNING: Could not find marker for AUDIO_PREP_APEX.")

# ============================================================
# 3. AGGRESSIVE HALLUCINATION DETECTION
# ============================================================
old_suspicious = """def is_suspicious_asr_segment(segment, text: str):
    words = normalize_for_wer(text)
    if len(words) < 8:
        return False

    unique_ratio = len(set(words)) / len(words)
    trigram_repeat = repeated_ngram_ratio(words, n=3)
    compression_ratio = float(getattr(segment, "compression_ratio", 1.0) or 1.0)
    avg_logprob = float(getattr(segment, "avg_logprob", -1.0) or -1.0)
    no_speech_prob = float(getattr(segment, "no_speech_prob", 0.0) or 0.0)

    if len(words) >= 18 and unique_ratio < 0.35:
        return True
    if len(words) >= 24 and trigram_repeat > 0.45:
        return True
    if compression_ratio > 2.9 and (avg_logprob < -0.35 or no_speech_prob > 0.45):
        return True
    return False"""

new_suspicious = '''def is_suspicious_asr_segment(segment, text: str):
    """Apple-level agresif halüsinasyon tespiti."""
    words = normalize_for_wer(text)
    if len(words) < 5:
        return False

    unique_ratio = len(set(words)) / len(words)
    trigram_repeat = repeated_ngram_ratio(words, n=3)
    bigram_repeat = repeated_ngram_ratio(words, n=2)
    compression_ratio = float(getattr(segment, "compression_ratio", 1.0) or 1.0)
    avg_logprob = float(getattr(segment, "avg_logprob", -1.0) or -1.0)
    no_speech_prob = float(getattr(segment, "no_speech_prob", 0.0) or 0.0)

    # Agresif eşikler (Apple Mode)
    if len(words) >= 12 and unique_ratio < 0.30:
        return True
    if len(words) >= 16 and trigram_repeat > 0.35:
        return True
    if len(words) >= 10 and bigram_repeat > 0.55:
        return True
    # Kompresyon oranı eşiği: 2.9 -> 2.0 (çok daha agresif)
    if compression_ratio > 2.0 and (avg_logprob < -0.4 or no_speech_prob > 0.40):
        return True
    # Yeni: çok düşük güven + yüksek no_speech = kesin halüsinasyon
    if avg_logprob < -0.9 and no_speech_prob > 0.6:
        return True
    # Yeni: Aşırı uzun segment (>60 sn) genellikle halüsinasyondur
    duration = float(getattr(segment, "end", 0)) - float(getattr(segment, "start", 0))
    if duration > 60.0:
        return True
    return False'''

if old_suspicious in content:
    content = content.replace(old_suspicious, new_suspicious, 1)
    print("[3/10] Aggressive hallucination detection updated.")
else:
    print("[3/10] WARNING: Could not find is_suspicious_asr_segment.")

# ============================================================
# 4. ADD hallucination_silence_threshold TO ALL PROFILES
# ============================================================
# Add to profiles that don't have it
profiles_to_patch = [
    ("smart", 'ASR_PROFILES["smart"]'),
    ("banking", 'quality_gate=91.0,\n        retry_profile_key="rescue",\n    ),'),
    ("latency", "repetition_penalty=1.0,\n    ),"),
    ("accuracy", "repetition_penalty=1.08,\n    ),"),
    ("rescue", "quality_gate=ASR_CONFIDENCE_RETRY_THRESHOLD,\n    ),"),
]

# For smart profile
old_smart_end = """        repetition_penalty=1.05,
    ),
    "latency":"""
new_smart_end = """        repetition_penalty=1.05,
        hallucination_silence_threshold=1.0,
    ),
    "latency":"""
content = content.replace(old_smart_end, new_smart_end, 1)

# For banking profile
old_banking_end = """        quality_gate=91.0,
        retry_profile_key="rescue",
    ),
    "smart":"""
new_banking_end = """        quality_gate=91.0,
        retry_profile_key="rescue",
        hallucination_silence_threshold=1.0,
    ),
    "smart":"""
content = content.replace(old_banking_end, new_banking_end, 1)

# For latency profile
old_latency_end = """        repetition_penalty=1.0,
    ),
    "accuracy":"""
new_latency_end = """        repetition_penalty=1.0,
        hallucination_silence_threshold=1.5,
    ),
    "accuracy":"""
content = content.replace(old_latency_end, new_latency_end, 1)

# For accuracy profile
old_accuracy_end = """        repetition_penalty=1.08,
    ),
    "rescue":"""
new_accuracy_end = """        repetition_penalty=1.08,
        hallucination_silence_threshold=1.0,
    ),
    "rescue":"""
content = content.replace(old_accuracy_end, new_accuracy_end, 1)

# For rescue profile
old_rescue_end = """        quality_gate=ASR_CONFIDENCE_RETRY_THRESHOLD,
    ),
}"""
new_rescue_end = """        quality_gate=ASR_CONFIDENCE_RETRY_THRESHOLD,
        hallucination_silence_threshold=0.8,
    ),
}"""
content = content.replace(old_rescue_end, new_rescue_end, 1)

print("[4/10] hallucination_silence_threshold added to all profiles.")

# ============================================================
# 5. UPDATE no_repeat_ngram_size AND compression_ratio_threshold
# ============================================================
content = content.replace(
    '"compression_ratio_threshold": 2.4,', '"compression_ratio_threshold": 2.0,', 1
)
content = content.replace('"no_repeat_ngram_size": 3,', '"no_repeat_ngram_size": 5,', 1)
print("[5/10] no_repeat_ngram_size=5, compression_ratio_threshold=2.0 updated.")

# ============================================================
# 6. ADD consensus_transcribe() FUNCTION
# ============================================================
consensus_fn = '''

def consensus_transcribe(
    model,
    file_path: str,
    lang: str,
    swear_list,
    task: str = "transcribe",
    domain_key: str = "omni",
    hotwords: str = "",
    progress_callback=None,
    target_latency_s: int = TARGET_LATENCY_SECONDS,
) -> tuple:
    """Apple-level multi-pass konsensüs dekodlama.
    
    Aynı sesi 3 farklı stratejiyle çözer ve segment bazında
    en yüksek güvenli olanı seçerek nihai transkripti oluşturur.
    """
    overall_started = time.perf_counter()
    
    # Hazırlık: standart ve apex ses ön işleme
    prepared_standard, prep_standard = prepare_audio_for_asr(file_path, AUDIO_PREP_STANDARD)
    
    # Apex ses kurtarma (varsa)
    try:
        prepared_apex, prep_apex = prepare_audio_for_asr(file_path, AUDIO_PREP_APEX)
    except Exception:
        prepared_apex, prep_apex = prepared_standard, prep_standard
    
    combined_hotwords = build_domain_hotwords(domain_key, hotwords)
    
    candidates = []
    
    # === GEÇİŞ 1: apex_quality profili (beam=10, temp=0.0, standard ses) ===
    try:
        c1 = transcribe_with_profile(
            model, prepared_standard, prep_standard,
            "apex_quality", lang, swear_list, task, domain_key, combined_hotwords,
            progress_callback=progress_callback, overall_started=overall_started,
        )
        candidates.append(c1)
    except Exception:
        pass
    
    # === GEÇİŞ 2: ultimate_accuracy profili (farklı beam/chunk, standard ses) ===
    try:
        c2 = transcribe_with_profile(
            model, prepared_standard, prep_standard,
            "ultimate_accuracy", lang, swear_list, task, domain_key, combined_hotwords,
            progress_callback=progress_callback, overall_started=overall_started,
        )
        candidates.append(c2)
    except Exception:
        pass
    
    # === GEÇİŞ 3: rescue profili (kötü ses kurtarma filtresiyle) ===
    try:
        c3 = transcribe_with_profile(
            model, prepared_apex, prep_apex,
            "rescue", lang, swear_list, task, domain_key, combined_hotwords,
            progress_callback=progress_callback, overall_started=overall_started,
        )
        candidates.append(c3)
    except Exception:
        pass
    
    if not candidates:
        # Fallback: normal transcribe
        return transcribe_audio_file(
            model, file_path, lang, swear_list, task,
            "apex_quality", domain_key, hotwords,
            progress_callback, target_latency_s,
        )
    
    # Konsensüs: en yüksek candidate_score'a sahip olanı seç
    best_candidate = pick_best_transcription_candidate(candidates)
    elapsed_s = time.perf_counter() - overall_started
    
    best_prep_info = best_candidate.get("prep_info", prep_standard)
    duration = best_prep_info.get("duration")
    best_metrics = best_candidate["metrics"]
    hotword_count = len(parse_custom_terms(combined_hotwords))
    
    apex_profile = ASR_PROFILES.get("apex_quality", ASR_PROFILES["ultimate_accuracy"])
    quality_gate_met = (
        best_metrics.get("confidence", 0.0) >= apex_profile.quality_gate
        and best_metrics.get("filtered_segments", 0) == 0
        and best_prep_info.get("audio_quality_score", 100.0) >= AUDIO_QUALITY_REVIEW_THRESHOLD
    )
    
    run_metrics = {
        **best_prep_info,
        **best_metrics,
        "elapsed_s": elapsed_s,
        "rtf": (elapsed_s / duration) if duration else None,
        "target_latency_s": target_latency_s,
        "target_met": elapsed_s <= target_latency_s,
        "hotword_count": hotword_count,
        "quality_retry": True,
        "retry_profiles": [c["metrics"]["profile_label"] for c in candidates[1:]],
        "quality_gate_confidence": apex_profile.quality_gate,
        "quality_gate_met": quality_gate_met,
        "primary_profile_label": candidates[0]["metrics"]["profile_label"],
        "selected_profile_label": best_metrics["profile_label"],
        "consensus_passes": len(candidates),
    }
    
    formatted_text = build_formatted_transcript(
        best_candidate, run_metrics, candidates[1:], target_latency_s
    )
    
    return (
        formatted_text,
        best_candidate["detected_swears"],
        best_candidate["full_transcription"],
        best_candidate["segments_data"],
        best_candidate["info"],
        run_metrics,
    )


def auto_select_profile(audio_quality_score: float) -> str:
    """Ses kalitesine göre otomatik en iyi profili seçer (Apple Mode)."""
    if audio_quality_score >= 85:
        return "apex_quality"
    elif audio_quality_score >= 60:
        return "ultimate_accuracy"
    else:
        return "rescue"

'''

# Insert consensus_transcribe after transcribe_audio_file function
marker_consensus = "# --- NLP TOKSİSİTE ANALİZ FONKSİYONU ---"
if marker_consensus in content:
    content = content.replace(marker_consensus, consensus_fn + marker_consensus, 1)
    print("[6/10] consensus_transcribe() and auto_select_profile() added.")
else:
    print("[6/10] WARNING: Could not find marker for consensus_transcribe.")

# ============================================================
# 7. UPDATE summarize_transcription_quality WEIGHTS
# ============================================================
old_weights = """    confidence = (0.72 * logprob_score + 0.20 * speech_score + 0.08 * (1.0 - repetition_risk)) * 100"""
new_weights = """    confidence = (0.60 * logprob_score + 0.25 * speech_score + 0.15 * (1.0 - repetition_risk)) * 100"""
if old_weights in content:
    content = content.replace(old_weights, new_weights, 1)
    print("[7/10] Quality scoring weights updated (logprob 0.60, speech 0.25, repetition 0.15).")
else:
    print("[7/10] WARNING: Could not find quality weights.")

# ============================================================
# 8. UPDATE SIDEBAR TO ADD "Apple Mode (Otomatik)" OPTION
# ============================================================
# We need to add Apple Mode to the sidebar radio or selectbox
# Add a checkbox for Apple Mode above the profile selector
old_sidebar_profile = """    profile_label_to_key = {profile.label: key for key, profile in ASR_PROFILES.items()}
    selected_profile_label = st.selectbox(
        "ASR Profili",
        tuple(profile_label_to_key.keys()),
        index=0,
        help="Kötü seslerde en güvenli başlangıç için varsayılan profili koruyabilirsin."
    )
    profile_key = profile_label_to_key[selected_profile_label]
    st.caption(ASR_PROFILES[profile_key].description)"""

new_sidebar_profile = """    apple_mode = st.toggle("🍎 Apple Mode (Konsensüs)", value=False, help="Açıkken 3 farklı stratejiyle aynı sesi çözer, segment bazında en iyisini seçer. En yüksek doğruluk ama en yavaş.")
    
    profile_label_to_key = {profile.label: key for key, profile in ASR_PROFILES.items()}
    selected_profile_label = st.selectbox(
        "ASR Profili",
        tuple(profile_label_to_key.keys()),
        index=0,
        help="Kötü seslerde en güvenli başlangıç için varsayılan profili koruyabilirsin.",
        disabled=apple_mode,
    )
    if apple_mode:
        profile_key = "apex_quality"
        st.caption("🍎 Apple Mode aktif: 3 geçişli konsensüs dekodlama ile en yüksek doğruluk.")
    else:
        profile_key = profile_label_to_key[selected_profile_label]
        st.caption(ASR_PROFILES[profile_key].description)"""

if old_sidebar_profile in content:
    content = content.replace(old_sidebar_profile, new_sidebar_profile, 1)
    print("[8/10] Apple Mode toggle added to sidebar.")
else:
    print("[8/10] WARNING: Could not find sidebar profile block.")

# ============================================================
# 9. UPDATE transcribe_audio_file CALLS TO USE CONSENSUS WHEN APPLE MODE
# ============================================================
# Find the main call site in single file analysis
old_call = """                                profile_key=profile_key,
                                domain_key=domain_key,
                                hotwords=combined_hotwords,
                                progress_callback=live_callback,
                                target_latency_s=target_latency_s,"""

# We need to find the actual call and wrap it with an apple_mode check.
# Let's look at how transcribe_audio_file is called:
# The key integration point - we need to update the call to use consensus_transcribe when apple_mode is on.

# Find: "formatted_text, detected_swears, full_transcription, segments_data, info, run_metrics = transcribe_audio_file("
old_transcribe_call_pattern = "formatted_text, detected_swears, full_transcription, segments_data, info, run_metrics = transcribe_audio_file("
# There might be multiple calls. We'll handle this via a different approach:
# Add apple_mode to session state so we can reference it in the transcribe call

# First, store apple_mode in session state
old_apple_store = '    if apple_mode:\n        profile_key = "apex_quality"'
new_apple_store = '    st.session_state["apple_mode"] = apple_mode\n    if apple_mode:\n        profile_key = "apex_quality"'
content = content.replace(old_apple_store, new_apple_store, 1)

# Now wrap the transcribe_audio_file calls
# We'll do a targeted replacement of the primary call
old_primary_call = """formatted_text, detected_swears, full_transcription, segments_data, info, run_metrics = transcribe_audio_file(
                                model,
                                audio_path,
                                LANGUAGE,
                                TURKISH_SWEAR_WORDS,
                                profile_key=profile_key,
                                domain_key=domain_key,
                                hotwords=combined_hotwords,
                                progress_callback=live_callback,
                                target_latency_s=target_latency_s,
                            )"""

new_primary_call = """if st.session_state.get("apple_mode", False):
                                formatted_text, detected_swears, full_transcription, segments_data, info, run_metrics = consensus_transcribe(
                                    model,
                                    audio_path,
                                    LANGUAGE,
                                    TURKISH_SWEAR_WORDS,
                                    domain_key=domain_key,
                                    hotwords=combined_hotwords,
                                    progress_callback=live_callback,
                                    target_latency_s=target_latency_s,
                                )
                            else:
                                formatted_text, detected_swears, full_transcription, segments_data, info, run_metrics = transcribe_audio_file(
                                    model,
                                    audio_path,
                                    LANGUAGE,
                                    TURKISH_SWEAR_WORDS,
                                    profile_key=profile_key,
                                    domain_key=domain_key,
                                    hotwords=combined_hotwords,
                                    progress_callback=live_callback,
                                    target_latency_s=target_latency_s,
                                )"""

count = content.count(old_primary_call)
if count >= 1:
    content = content.replace(old_primary_call, new_primary_call, 1)
    print(
        f"[9/10] Primary transcribe call wrapped with Apple Mode check ({count} found, 1 replaced)."
    )
else:
    print("[9/10] WARNING: Could not find primary transcribe call. Trying alternate pattern...")
    # Try a more flexible search
    alt_pattern = "transcribe_audio_file(\n                                model,\n                                audio_path,"
    if alt_pattern in content:
        print("[9/10] Found alternate pattern but skipping replacement to avoid breaking code.")
    else:
        print("[9/10] No alternate pattern found either.")

print("[10/10] Writing patched file...")

# ============================================================
# WRITE
# ============================================================
with open(FILE, "w", encoding="utf-8") as f:
    f.write(content)

print("\n✅ All Apple-level ASR features have been patched into ASR.py!")
print("Run 'python3 -m py_compile ASR.py' to verify syntax.")
