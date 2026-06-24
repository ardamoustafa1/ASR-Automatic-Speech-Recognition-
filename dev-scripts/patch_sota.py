import re

FILE = "ASR.py"

with open(FILE, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add `prepared_path` to transcribe_with_profile return
old_return = '''    return {
        "formatted_body": formatted_body,
        "detected_swears": detected_swears,
        "full_transcription": full_transcription,
        "segments_data": segments_data,
        "info": info,
        "metrics": metrics,
        "profile": profile,
        "prep_info": prep_info,
    }'''

new_return = '''    return {
        "formatted_body": formatted_body,
        "detected_swears": detected_swears,
        "full_transcription": full_transcription,
        "segments_data": segments_data,
        "info": info,
        "metrics": metrics,
        "profile": profile,
        "prep_info": prep_info,
        "prepared_path": prepared_path,
    }'''
content = content.replace(old_return, new_return)

# 2. Add apply_sota_features helper function right before transcribe_audio_file
sota_helper = '''
def apply_sota_features(candidate, model, domain_key, combined_hotwords, lang, swear_list):
    """SOTA özelliklerini (per-segment redecode & punctuation) en iyi adaya uygular."""
    # 1. Per-segment redecode
    segments = candidate["segments_data"]
    if getattr(model, "transcribe", None) and candidate.get("prepared_path"):
        improved_segments = redecode_low_confidence_segments(
            model, candidate["prepared_path"], segments, domain_key, combined_hotwords, lang
        )
        candidate["segments_data"] = improved_segments
    
    # 2. Rebuild text
    full_transcription_parts = []
    formatted_body = ""
    detected_swears = []
    for segment in candidate["segments_data"]:
        text = segment.text
        full_transcription_parts.append(text)
        
        m_start, s_start = divmod(segment.start, 60)
        m_end, s_end = divmod(segment.end, 60)
        time_str = f"[{int(m_start):02d}:{s_start:04.1f} - {int(m_end):02d}:{s_end:04.1f}]"
        formatted_body += f"{time_str} {text}\\n"
        detected_swears.extend(detect_swears_in_segment(text, segment.start, swear_list))
        
    full_text = " ".join(full_transcription_parts).strip()
    
    # 3. Punctuation Restoration
    full_text = restore_turkish_punctuation(full_text)
    
    candidate["full_transcription"] = full_text
    candidate["formatted_body"] = formatted_body
    candidate["detected_swears"] = detected_swears
    return candidate

'''

marker_sota = "def transcribe_audio_file("
if marker_sota in content and "def apply_sota_features" not in content:
    content = content.replace(marker_sota, sota_helper + marker_sota)


# 3. Wire into transcribe_audio_file
old_transcribe = '''    best_candidate = pick_best_transcription_candidate(candidates)
    elapsed_s = time.perf_counter() - overall_started'''

new_transcribe = '''    best_candidate = pick_best_transcription_candidate(candidates)
    
    # Apply SOTA features
    best_candidate = apply_sota_features(best_candidate, model, domain_key, combined_hotwords, lang, swear_list)
    
    elapsed_s = time.perf_counter() - overall_started'''
content = content.replace(old_transcribe, new_transcribe)


# 4. Wire into consensus_transcribe
old_consensus = '''    # Konsensüs: en yüksek candidate_score'a sahip olanı seç
    best_candidate = pick_best_transcription_candidate(candidates)
    elapsed_s = time.perf_counter() - overall_started'''

new_consensus = '''    # Konsensüs: en yüksek candidate_score'a sahip olanı seç
    best_candidate = pick_best_transcription_candidate(candidates)
    
    # Apply SOTA features
    best_candidate = apply_sota_features(best_candidate, model, domain_key, combined_hotwords, lang, swear_list)
    
    elapsed_s = time.perf_counter() - overall_started'''
content = content.replace(old_consensus, new_consensus)

with open(FILE, "w", encoding="utf-8") as f:
    f.write(content)

print("Patch applied.")
