# Speech recognition service integrating Faster-Whisper and MLX acceleration.
"""Thread-safe Singleton ASR Service using Faster-Whisper."""

import contextvars
import os
import platform
import re
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from loguru import logger

from asr_pro.config import settings
from asr_pro.observability.metrics import asr_transcribe_duration_seconds, time_block

# Per-request counter of segments dropped by the quality gate (no_speech_prob
# too high, avg_logprob too low + garbled compression ratio) or the known-
# hallucination blocklist. Exposed as a QA metric to callers - "how much did
# we throw away for this call". A ContextVar rather than an instance
# attribute: ASRService is a shared singleton, so concurrent upload requests
# on different threads must not see each other's counts.
_filtered_segment_count: contextvars.ContextVar[int] = contextvars.ContextVar(
    "asr_filtered_segment_count", default=0
)


def reset_filtered_segment_counter() -> None:
    _filtered_segment_count.set(0)


def get_filtered_segment_count() -> int:
    return _filtered_segment_count.get()


def _count_filtered(n: int = 1) -> None:
    _filtered_segment_count.set(_filtered_segment_count.get() + n)


try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None  # type: ignore[assignment]

warnings.simplefilter("ignore")


@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    # Whisper decoder confidence: average log-probability of the segment's
    # tokens. -1.0 = unknown/not provided. Carried end-to-end so the API and
    # UI can surface real per-line confidence instead of a fabricated constant.
    avg_logprob: float = -1.0
    no_speech_prob: float = 0.0
    # Word-level timestamps ([{word, start, end, probability}, ...]) when the
    # engine produced them; None otherwise.
    words: Any = None
    # Text before DomainAdaptationService's phonetic correction, for
    # compliance/QA audit ("what did the model actually hear"). Empty when
    # correction made no change to this segment.
    raw_text: str = ""


from asr_pro.utils.lang_utils import lock_language


class ASRService:
    """Thread-safe Singleton for Faster-Whisper model management."""

    _instance: Optional["ASRService"] = None
    _lock = threading.Lock()
    _model = None
    _model_size = "large-v3"

    @classmethod
    def get_instance(cls) -> "ASRService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._device = self._choose_device()
        self._compute_type = self._choose_compute_type()
        # Serializes calls into the underlying model: CTranslate2/mlx are not
        # guaranteed safe for concurrent inference from multiple threads on a
        # single model instance (relevant once live streaming sessions and
        # batch uploads can call transcribe() at the same time).
        self._inference_lock = threading.Lock()
        # MLX binds GPU streams to the thread that created them: weights
        # loaded on thread A then decoded on thread B raise "There is no
        # Stream(gpu, N) in current thread" (observed live when startup
        # preload + the background job executor used different threads). Pin
        # every MLX operation - load and decode alike - to this ONE
        # persistent thread. Single worker also matches _inference_lock
        # semantics, so it costs no parallelism.
        self._mlx_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-inference")

    def _mlx_call(self, fn, /, *args, **kwargs):
        """Run fn on the dedicated MLX thread and return its result."""
        return self._mlx_executor.submit(fn, *args, **kwargs).result()

    def _choose_device(self) -> str:
        if platform.system() == "Darwin" and platform.machine() in ["arm64", "aarch64"]:
            # Apple Silicon: faster-whisper falls back to CPU (use mlx-whisper for native MPS)
            return "cpu"
        return "cuda" if self._check_cuda() else "cpu"

    def _check_cuda(self) -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    @staticmethod
    def _gpu_compute_capability() -> tuple[int, int]:
        """Return (major, minor) CUDA compute capability of device 0, or (0, 0) if unavailable."""
        try:
            import torch

            return torch.cuda.get_device_capability(0)
        except Exception:
            return (0, 0)

    def _choose_compute_type(self) -> str:
        if self._device == "cuda":
            return self._choose_cuda_compute_type()
        if self._device == "mps":
            return "float16"
        return "int8"

    def _choose_cuda_compute_type(self) -> str:
        """Select the initial CTranslate2 compute type for the detected GPU generation.

        Blackwell (compute capability 10.x on B100/B200/GB200, 12.x on
        RTX 50-series) has no CTranslate2 quantization kernel confirmed as
        supported by this project as of this writing - CTranslate2 only ships
        prebuilt kernels for the CUDA compute capabilities its own release
        targeted, and whether a given deployment's installed ctranslate2
        build has Blackwell kernels depends entirely on which version is
        installed. Hardcoding an unverified "Blackwell fast path" here would
        be a false claim, not an engineering decision.

        Instead: detect and log the compute capability for observability,
        return the best generally-supported type, and let load_model()'s
        fallback chain (see _compute_type_fallback_chain) actually discover
        at load time whether a faster type (e.g. int8_float16) works on this
        specific GPU + installed CTranslate2 build - degrading automatically
        instead of crashing if it doesn't.
        """
        major, minor = self._gpu_compute_capability()
        if major >= 10:
            logger.info(
                f"ASRService: GPU compute capability {major}.{minor} detected "
                "(Blackwell-class or newer). No Blackwell-specific CTranslate2 quantization "
                "kernel is confirmed available in this deployment; will attempt float16 first "
                "and probe faster compute types via automatic fallback in load_model()."
            )
        return "float16"

    def _compute_type_fallback_chain(self) -> list[str]:
        """Ordered compute types to attempt when loading the model on CUDA.

        Tries the selected type first, then progressively more conservative
        alternatives - so an unsupported compute_type/architecture
        combination in the installed CTranslate2 build (e.g. a new GPU
        generation released after that build) degrades to a working
        configuration instead of raising.
        """
        if self._device != "cuda":
            return [self._compute_type]
        chain = [self._compute_type]
        for candidate in ("float16", "int8_float16", "int8"):
            if candidate not in chain:
                chain.append(candidate)
        return chain

    def load_model(self, model_size: str | None = None) -> "WhisperModel | None":
        model_size = model_size or getattr(settings, "asr_model_size", None) or "large-v3"
        if self._model is not None and self._model_size == model_size:
            return self._model
        if getattr(self, "_is_mlx", False) and self._model_size == model_size:
            return None

        # Apple Silicon MLX Check
        if platform.system() == "Darwin" and platform.machine() in ["arm64", "aarch64"]:
            try:
                import mlx_whisper  # noqa: F401

                self._device = "mps"
                self._compute_type = "float16"
                self._model_size = model_size
                self._is_mlx = True
                logger.info(
                    f"Initialized ASR model '{model_size}' using hardware-accelerated Apple MLX engine."
                )
                return None  # MLX handles loading during transcribe
            except ImportError:
                logger.warning(
                    "mlx-whisper module not detected. Falling back to CPU int8 execution. For optimal Apple Silicon performance, install mlx-whisper."
                )
                self._is_mlx = False
        else:
            self._is_mlx = False

        if WhisperModel is None:
            raise RuntimeError("faster-whisper is not installed. Run: pip install faster-whisper")

        cpu_threads = max(4, os.cpu_count() or 4)
        num_workers = 1 if self._device == "cuda" else min(4, max(1, cpu_threads // 2))

        last_exc: Exception | None = None
        for compute_type in self._compute_type_fallback_chain():
            logger.info(f"Loading ASR model '{model_size}' on {self._device} ({compute_type})")
            try:
                self._model = WhisperModel(
                    model_size,
                    device=self._device,
                    compute_type=compute_type,
                    cpu_threads=cpu_threads,
                    num_workers=num_workers,
                )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    f"ASRService: compute_type={compute_type!r} failed to load on "
                    f"{self._device} ({exc}); trying next fallback compute type."
                )
                continue
            self._compute_type = compute_type
            self._model_size = model_size
            logger.info(
                f"ASR model '{model_size}' loaded successfully with compute_type={compute_type}."
            )
            return self._model

        raise RuntimeError(
            f"Could not load ASR model '{model_size}' on {self._device} with any compute type "
            f"({self._compute_type_fallback_chain()}). Last error: {last_exc}"
        ) from last_exc

    def ensure_model_loaded(self) -> None:
        """Pull model weights into memory now, so the first customer upload
        doesn't pay the ~10-20s large-v3 load on top of decode time.

        On the faster-whisper path load_model() already does the work; on the
        MLX path load_model() only sets flags (mlx_whisper lazy-loads inside
        transcribe()), so we warm mlx_whisper's ModelHolder cache directly
        with the exact repo transcribe() will request.
        """
        self.load_model()
        if getattr(self, "_is_mlx", False):
            import mlx.core as mx
            from mlx_whisper.transcribe import ModelHolder

            mlx_repo_overrides = {"large-v3": "large-v3-mlx"}
            repo_size = mlx_repo_overrides.get(self._model_size, self._model_size)
            # Load on the SAME dedicated thread that will decode - MLX GPU
            # streams are thread-bound (see _mlx_call).
            self._mlx_call(ModelHolder.get_model, f"mlx-community/whisper-{repo_size}", mx.float16)

    @staticmethod
    def _is_stereo_file(file_path: Any) -> bool:
        if not file_path or not isinstance(file_path, str) or not os.path.exists(file_path):
            return False
        try:
            import av

            with av.open(file_path, mode="r", metadata_errors="ignore") as container:
                if container.streams.audio:
                    return getattr(container.streams.audio[0].codec_context, "channels", 1) >= 2
        except Exception:
            pass
        try:
            import wave

            with wave.open(file_path, "rb") as wf:
                return wf.getnchannels() >= 2
        except Exception:
            pass
        return False

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """Sanitize Whisper hallucinations and phrase repetition loops."""
        return ASRService._sanitize_text_with_raw(text)[0]

    @staticmethod
    def _sanitize_text_with_raw(text: str) -> tuple[str, str]:
        """Like _sanitize_text, but also returns the text as it stood right
        before DomainAdaptationService's phonetic correction ran - i.e. the
        decoder's own output after hallucination/repetition cleanup, but
        before any "VoLTE"/"IBAN"-style spelling fixes. Returns ("", "") for
        empty input; returns (final, "") when correction made no change, so
        callers can skip storing a redundant audit-trail copy.
        """
        if not text or not text.strip():
            return text, ""
        text = re.sub(r"(\s*\.\s*){3,}", "... ", text)
        text = re.sub(r"\.{4,}", "... ", text)
        text = re.sub(r"(?:\.\s*){3,}$", ".", text)
        # Short token+punctuation loops glued with NO whitespace between
        # repeats (e.g. "7.7.7.7.7...7.") - observed as a real hallucination
        # on noisy telephony audio with the large-v3-turbo model. The n-gram
        # collapse loop below requires whitespace between repeats and never
        # matches this punctuation-glued form, so it needs its own pass.
        text = re.sub(r"\b(\w{1,4}[.,;:!?]\s*)\1{2,}", r"\1", text)
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        if len(parts) > 1:
            cleaned_parts = []
            for p in parts:
                p_clean = p.strip()
                if not p_clean:
                    continue
                p_norm = re.sub(r"[^\w\s]", "", p_clean.lower()).strip()
                if cleaned_parts:
                    prev_norm = re.sub(r"[^\w\s]", "", cleaned_parts[-1].lower()).strip()
                    if p_norm and prev_norm == p_norm:
                        continue
                cleaned_parts.append(p_clean)
            text = " ".join(cleaned_parts)
        for n in range(5, 0, -1):
            min_repeat = 2 if n >= 3 else 3
            pattern = (
                r"\b((?:\w+)(?:[,.!?]?\s+\w+){"
                + str(n - 1)
                + r"})(?:[,.!?]?\s+(?:\1)){"
                + str(min_repeat - 1)
                + r",}\b"
            )
            text = re.sub(pattern, r"\1", text, flags=re.IGNORECASE)
        pre_correction = text.strip()
        from asr_pro.services.domain_adaptation import DomainAdaptationService

        corrected = DomainAdaptationService.correct_terms(pre_correction).strip()
        return corrected, (pre_correction if pre_correction != corrected else "")

    # Known Whisper Turkish hallucination artifacts: the model was trained on
    # YouTube-style data, so on ring tones / hold music / channel-start noise
    # it emits subtitle-outro phrases that never occur in real call-center
    # speech. Observed live: "İzlediğiniz için teşekkür ederim." decoded from
    # a ring tone at t=0.2s of a production call recording.
    _HALLUCINATION_PATTERNS = [
        re.compile(p, re.IGNORECASE)
        for p in (
            r"izledi[ğg]iniz için teşekkür",
            r"altyazı\s*m\.?\s*k\.?",
            r"abone olmayı unutmayın",
            r"videoyu be[ğg]en",
            r"kanal[ıi]m[ıi]za abone",
            r"video için teşekkür",
            r"dinledi[ğg]iniz için teşekkür",
            r"bu dizinin betimlemesi",
            r"altyazılar.*sağlanmıştır",
        )
    ]

    @classmethod
    def _is_known_hallucination(cls, text: str) -> bool:
        return any(p.search(text) for p in cls._HALLUCINATION_PATTERNS)

    # Turkish function words excluded from prompt-echo matching: they appear
    # in the conditioning prompt's natural-language preamble AND in nearly
    # every real utterance, so counting them would let short grammatical
    # sentences built around 2-3 domain terms cross the echo threshold.
    _ECHO_STOPWORDS = frozenset(
        {
            "bu",
            "bir",
            "ve",
            "için",
            "de",
            "da",
            "ile",
            "mi",
            "mı",
            "mu",
            "mü",
            "ne",
            "o",
            "şu",
            "çok",
            "en",
            "gibi",
            "ama",
            "ki",
            "ya",
            "hem",
            "olan",
            "olarak",
            "daha",
            "her",
            "size",
            "sizin",
            "ben",
            "biz",
            "siz",
        }
    )

    @classmethod
    def _prompt_token_set(cls, prompt: str) -> tuple[str, ...]:
        """Ordered, stopword-free token sequence of the conditioning prompt.

        Ordered (not a set) because the strongest echo signal is ORDER: a
        mutated echo still reproduces long runs of prompt tokens in the
        prompt's own sequence, which real speech never does.
        """
        return tuple(
            t for t in re.findall(r"\w+", (prompt or "").lower()) if t not in cls._ECHO_STOPWORDS
        )

    @classmethod
    def _is_prompt_echo(cls, text: str, prompt_tokens: tuple[str, ...]) -> bool:
        """Detect Whisper regurgitating the initial_prompt as a "transcription".

        On long silence/hold stretches Whisper sometimes emits the
        conditioning prompt itself - observed live on a real call: the
        domain-vocabulary list ("Vodafone Türkiye, Türk Telekom, Turkcell,
        VoLTE, eSIM, ...") appeared as a 16-second agent segment. For a
        telecom client, competitor names surfacing as if the agent said them
        is an unshippable defect, so this is checked on every segment.

        Echoes MUTATE nondeterministically: three decodes of the same hold
        stretch produced three different tails ("tayfan, borç sorgulama" /
        "fiyat altyapı, fiyat altyapı" / "tayinli bir hizmet, tarif ve
        bilgi"), so a bag-of-words overlap ratio alone is fragile - the third
        mutation diluted it to 0.58. Two complementary detectors:

        1. Ordered-run: ≥5 consecutive prompt tokens appearing in the
           segment in the prompt's own order. Every observed mutation starts
           with the intact prompt prefix (vodafone, türkiye, türk, telekom,
           turkcell, volte, esim - a 7-token run); real speech never
           reproduces 5+ prompt tokens in exact prompt order ("Vodafone,
           Türk Telekom, Turkcell karşılaştırması" has only a 3-run because
           the customer doesn't say "Türkiye").
        2. Overlap ratio ≥0.6 with ≥5 matched content tokens, as a net for
           echoes from deeper inside the prompt where the prefix is absent.
        """
        if not prompt_tokens:
            return False
        tokens = [t for t in re.findall(r"\w+", text.lower()) if t not in cls._ECHO_STOPWORDS]
        if len(tokens) < 5:
            return False

        # Detector 1: longest run of consecutive prompt tokens, in order.
        best_run = 0
        for i in range(len(tokens)):
            for j in range(len(prompt_tokens)):
                if tokens[i] != prompt_tokens[j]:
                    continue
                k = 0
                while (
                    i + k < len(tokens)
                    and j + k < len(prompt_tokens)
                    and tokens[i + k] == prompt_tokens[j + k]
                ):
                    k += 1
                best_run = max(best_run, k)
            if best_run >= 5:
                return True

        # Detector 2: bag-of-words overlap.
        prompt_set = set(prompt_tokens)
        matched = sum(1 for t in tokens if t in prompt_set)
        return matched >= 5 and (matched / len(tokens)) >= 0.6

    # Short Turkish backchannel acknowledgements that customers/agents genuinely
    # repeat many times per call ("evet ... evet ... evet") - these must never
    # be treated as hallucination loops unless they repeat back-to-back.
    _BACKCHANNEL_NORMS = frozenset(
        {
            "evet",
            "tamam",
            "tamamdır",
            "peki",
            "olur",
            "hı hı",
            "hıhı",
            "anladım",
            "tabii",
            "tabi",
            "teşekkürler",
            "teşekkür ederim",
            "iyi günler",
            "aynen",
            "doğru",
            "yok",
            "hayır",
        }
    )

    @staticmethod
    def _deduplicate_segments(segments: list[Any]) -> list[Any]:
        """Drop hallucinated segment loops without deleting genuinely repeated speech.

        Whisper hallucination loops repeat the SAME text back-to-back (gap of at
        most a couple of seconds, usually 0). Real conversations also repeat short
        phrases ("Evet.", "Tamam.") legitimately across the call, so wide time
        windows delete real speech - a previous 25s window was observed dropping
        real customer confirmations on production call-center audio.
        """
        if not segments:
            return []
        cleaned = []
        recent_by_speaker: dict[str, list[tuple[float, float, str]]] = {}
        repeat_count_by_speaker: dict[str, int] = {}
        for seg in segments:
            spk = getattr(seg, "speaker", "default") or "default"
            text = getattr(seg, "text", "").strip()
            if not text:
                continue
            norm = re.sub(r"[^\w\s]", "", text.lower()).strip()
            if not norm:
                continue
            words = norm.split()
            start = float(getattr(seg, "start", 0))
            end = float(getattr(seg, "end", 0))
            is_backchannel = norm in ASRService._BACKCHANNEL_NORMS or len(words) <= 1

            is_dup = False
            history = recent_by_speaker.get(spk, [])
            if history:
                _prev_start, prev_end, prev_norm = history[-1]
                gap = start - prev_end
                if norm == prev_norm:
                    streak = repeat_count_by_speaker.get(spk, 0) + 1
                    repeat_count_by_speaker[spk] = streak
                    if is_backchannel:
                        # Backchannels only count as loops when they repeat
                        # essentially immediately, or keep repeating 3+ times.
                        is_dup = gap < 1.2 or (streak >= 3 and gap < 6.0)
                    else:
                        # Longer phrases repeating consecutively within a few
                        # seconds are decoder loops; a second+ repetition is
                        # dropped even with a slightly larger gap.
                        is_dup = gap < 3.0 or (streak >= 2 and gap < 10.0)
                else:
                    repeat_count_by_speaker[spk] = 0
                    if (
                        not is_backchannel
                        and len(words) <= 4
                        and (norm in prev_norm or prev_norm in norm)
                        and gap < 0.8
                    ):
                        # Partial overlap of a short phrase right at a VAD
                        # boundary (same words re-decoded across a window split).
                        is_dup = True
            if is_dup:
                logger.debug(
                    f"ASR Deduplication: Dropping repeated hallucination [{spk}] '{text}' at {start:.1f}s"
                )
                _count_filtered()
                continue
            recent_by_speaker.setdefault(spk, []).append((start, end, norm))
            cleaned.append(seg)
        return cleaned

    @staticmethod
    def _split_into_sentences(segments: list[Any]) -> list[Any]:
        """Apple Silicon Top-Tier Sentence & Timestamp Splitter.
        Breaks long multi-sentence Whisper chunks into individual sentences with proportional
        timestamps so stereo & mono dialogues are chronologically ordered.
        """
        if not segments:
            return []
        refined = []
        for seg in segments:
            is_dict = isinstance(seg, dict)
            start = float(seg["start"] if is_dict else getattr(seg, "start", 0.0))
            end = float(seg["end"] if is_dict else getattr(seg, "end", 0.0))
            text = str(seg["text"] if is_dict else getattr(seg, "text", "")).strip()
            speaker = (
                seg["speaker"] if (is_dict and "speaker" in seg) else getattr(seg, "speaker", None)
            )

            if not text:
                continue

            sentences = [s.strip() for s in re.split(r"(?<=[.?!;])\s+", text) if s.strip()]

            if len(sentences) <= 1 or (end - start) <= 2.5:
                refined.append(seg)
                continue

            total_chars = sum(max(len(s), 1) for s in sentences)
            duration = max(end - start, 0.1)

            cur_start = start
            for idx, sent in enumerate(sentences):
                sent_len = max(len(sent), 1)
                if idx == len(sentences) - 1:
                    sent_end = end
                else:
                    sent_dur = duration * (sent_len / total_chars)
                    sent_end = round(cur_start + sent_dur, 2)

                if is_dict:
                    new_seg = dict(seg)
                    new_seg.update({"start": round(cur_start, 2), "end": sent_end, "text": sent})
                elif hasattr(seg, "_replace"):
                    replace_kwargs = {"start": round(cur_start, 2), "end": sent_end, "text": sent}
                    if hasattr(seg, "_fields") and "words" in seg._fields:
                        replace_kwargs["words"] = getattr(seg, "words", None) or []
                    try:
                        new_seg = seg._replace(**replace_kwargs)
                    except Exception:
                        replace_kwargs["words"] = None
                        new_seg = seg._replace(**replace_kwargs)
                elif isinstance(seg, TranscriptionSegment):
                    sent_words = None
                    seg_words = getattr(seg, "words", None)
                    if seg_words:
                        # Keep only the words whose midpoint falls inside this
                        # sentence's estimated time window.
                        sent_words = [
                            w
                            for w in seg_words
                            if isinstance(w, dict)
                            and cur_start
                            <= (float(w.get("start", 0)) + float(w.get("end", 0))) / 2
                            < sent_end + 0.01
                        ] or None
                    new_seg = TranscriptionSegment(
                        start=round(cur_start, 2),
                        end=sent_end,
                        text=sent,
                        speaker=speaker,
                        avg_logprob=getattr(seg, "avg_logprob", -1.0),
                        no_speech_prob=getattr(seg, "no_speech_prob", 0.0),
                        words=sent_words,
                        # The pre-correction text was captured for the whole
                        # original (pre-split) segment, so it's carried as-is
                        # to each sentence split from it - same tradeoff as
                        # avg_logprob/no_speech_prob above.
                        raw_text=getattr(seg, "raw_text", ""),
                    )
                else:
                    import copy

                    new_seg = copy.copy(seg)
                    if hasattr(new_seg, "start"):
                        new_seg.start = round(cur_start, 2)
                    if hasattr(new_seg, "end"):
                        new_seg.end = sent_end
                    if hasattr(new_seg, "text"):
                        new_seg.text = sent
                refined.append(new_seg)
                cur_start = sent_end
        return refined

    @staticmethod
    def _safe_float_attr(obj, attr, default=0.0):
        val = getattr(obj, attr, default) if not isinstance(obj, dict) else obj.get(attr, default)
        if val is None or "mock" in str(type(val)).lower():
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _vad_speech_regions(
        pcm: np.ndarray, sample_rate: int = 16000
    ) -> Optional[list[tuple[int, int]]]:
        """Merge Silero VAD speech timestamps into padded decode regions.

        Returns None when the VAD model isn't available or its verdict can't be
        trusted (caller then decodes the full buffer), or a possibly-empty list
        of (start_sample, end_sample) regions covering detected speech.

        Decoding only VAD-confirmed speech is the primary defense against
        Whisper hallucinating text on hold music, IVR tones, and silence - and
        it removes the pathological slow decode loops those windows cause.
        """
        try:
            from asr_pro.services.vad_service import VADService

            vad = VADService.get_instance()
            if not vad.loaded:
                return None
            ts = vad.filter_speech_timestamps(pcm, sampling_rate=sample_rate)
        except Exception as exc:
            logger.debug(f"ASR VAD gating unavailable ({exc}); decoding full buffer.")
            return None
        if not ts:
            # Empty could mean "true silence" OR "VAD internally errored" (the
            # service returns [] for both). Only trust the silence verdict when
            # the buffer really is acoustically quiet - otherwise fall back to
            # a full decode rather than dropping potentially real speech.
            rms = float(np.sqrt(np.mean(np.square(pcm)))) if pcm.size else 0.0
            return [] if rms < 0.006 else None
        pad = int(0.30 * sample_rate)
        # Wider merging = fewer decode regions = fewer repetitions of the
        # fixed per-region cost (mel spectrogram setup + the ~60-token domain
        # prompt run through the decoder for every region). A real bad-line
        # call produced 20 regions per channel at 1.2s; typical dialog turn
        # gaps are 1-4s, so 3s merging roughly halves region count. The
        # bridged silence is bounded (≤3s) and the hallucination guards
        # (no-speech filter, echo detector, glued-loop sanitizer) cover the
        # silence-hallucination risk that VAD gating exists for.
        merge_gap = int(settings.asr_vad_region_merge_gap_sec * sample_rate)
        regions: list[tuple[int, int]] = []
        for t in ts:
            s = max(0, int(t["start"]) - pad)
            e = min(len(pcm), int(t["end"]) + pad)
            if regions and s - regions[-1][1] <= merge_gap:
                regions[-1] = (regions[-1][0], e)
            else:
                regions.append((s, e))
        return regions

    @staticmethod
    def _offset_words(words: Any, offset: float) -> Any:
        """Shift MLX/faster-whisper word timestamps by a region offset, as plain dicts."""
        if not words:
            return None
        shifted = []
        for w in words:
            if isinstance(w, dict):
                word_txt = w.get("word", "")
                w_start, w_end = w.get("start", 0.0), w.get("end", 0.0)
                prob = w.get("probability", 0.0)
            else:
                word_txt = getattr(w, "word", "")
                w_start, w_end = getattr(w, "start", 0.0), getattr(w, "end", 0.0)
                prob = getattr(w, "probability", 0.0)
            try:
                shifted.append(
                    {
                        "word": str(word_txt),
                        "start": round(float(w_start) + offset, 2),
                        "end": round(float(w_end) + offset, 2),
                        "probability": round(float(prob), 3),
                    }
                )
            except (TypeError, ValueError):
                continue
        return shifted or None

    def _decode_clip(self, clip: np.ndarray, language: str, prompt: str) -> tuple[str, float, Any]:
        """Decode a short audio clip and return (text, duration-weighted avg_logprob, words).

        Used by second-pass rescue; routes to whichever engine (MLX or
        faster-whisper) this service instance runs on.
        """
        raw: list[Any] = []
        if self._is_mlx:
            import mlx_whisper

            mlx_repo_overrides = {"large-v3": "large-v3-mlx"}
            repo_size = mlx_repo_overrides.get(self._model_size, self._model_size)
            with self._inference_lock:
                res = self._mlx_call(
                    mlx_whisper.transcribe,
                    clip,
                    path_or_hf_repo=f"mlx-community/whisper-{repo_size}",
                    language=language,
                    condition_on_previous_text=False,
                    initial_prompt=prompt,
                    compression_ratio_threshold=2.0,
                    no_speech_threshold=settings.asr_no_speech_threshold,
                    word_timestamps=settings.asr_word_timestamps,
                    # Greedy-only for the rescue clip: the whole point is a
                    # cleaner second opinion, not a high-temperature gamble.
                    temperature=(0.0,),
                )
            raw = list(res.get("segments", []))
        else:
            with self._inference_lock:
                gen, _info = self._model.transcribe(
                    clip,
                    language=language,
                    beam_size=max(8, settings.asr_beam_size),
                    initial_prompt=prompt,
                    word_timestamps=settings.asr_word_timestamps,
                    condition_on_previous_text=False,
                    repetition_penalty=1.20,
                    no_repeat_ngram_size=3,
                    log_prob_threshold=settings.asr_log_prob_threshold,
                    no_speech_threshold=settings.asr_no_speech_threshold,
                    temperature=(0.0,),
                )
                raw = list(gen)

        texts, weighted, total_w, words = [], 0.0, 0.0, []
        for s in raw:
            no_speech = self._safe_float_attr(s, "no_speech_prob", 0.0)
            if no_speech > 0.6:
                continue
            text = (s["text"] if isinstance(s, dict) else s.text).strip()
            if not text:
                continue
            lp = self._safe_float_attr(s, "avg_logprob", -1.0)
            s_start = self._safe_float_attr(s, "start", 0.0)
            s_end = self._safe_float_attr(s, "end", 0.0)
            w = max(s_end - s_start, 0.1)
            weighted += lp * w
            total_w += w
            texts.append(text)
            seg_words = s.get("words") if isinstance(s, dict) else getattr(s, "words", None)
            if seg_words:
                words.extend(seg_words)
        if not texts or total_w == 0.0:
            return "", -10.0, None
        return " ".join(texts), weighted / total_w, words or None

    def _second_pass_rescue(
        self,
        audio: np.ndarray,
        segments: list[TranscriptionSegment],
        language: str,
        domain_prompt: str,
        sample_rate: int = 16000,
    ) -> list[TranscriptionSegment]:
        """Re-decode low-confidence segments in isolation and keep the better hypothesis.

        A fresh acoustic window (no surrounding audio) plus the preceding
        confident text as context prompt often fixes misrecognitions caused by
        Whisper's 30s window boundaries. The rewrite is only accepted when the
        decoder itself scores it meaningfully higher, so this pass can improve
        but never degrade a transcript.
        """
        if not settings.asr_second_pass_enabled or not segments:
            return segments
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            return segments
        threshold = settings.asr_second_pass_logprob_threshold
        margin = settings.asr_second_pass_margin
        pad = int(0.30 * sample_rate)
        rescued = 0
        attempts = 0
        for i, seg in enumerate(segments):
            if attempts >= settings.asr_second_pass_max_attempts:
                logger.debug("ASR second pass: attempt budget exhausted for this channel.")
                break
            lp = seg.avg_logprob
            dur = seg.end - seg.start
            # Two independent triggers:
            #  (a) whole-segment confidence below threshold, or
            #  (b) the segment LOOKS confident overall but contains ≥2
            #      suspect content words (word-level probability < 0.4).
            #      Measured on real persisted calls, garbled content words
            #      ("Katapay" 0.28, "retle" 0.30) carry low word probability
            #      even when the segment's avg_logprob sits in the healthy
            #      -0.2..-0.3 band, so (a) alone never catches them. Short
            #      words are excluded - backchannels ("Bu", "Tamam") score
            #      low at boundaries without being wrong.
            low_conf_trigger = lp != -1.0 and lp < threshold
            suspect_words = 0
            if not low_conf_trigger and seg.words:
                suspect_words = sum(
                    1
                    for w in seg.words
                    if isinstance(w, dict)
                    and len(re.sub(r"\W", "", str(w.get("word", "")))) >= 4
                    and float(w.get("probability", 1.0)) < 0.4
                )
            if not low_conf_trigger and suspect_words < 2:
                continue
            if dur < 0.3 or dur > settings.asr_second_pass_max_segment_sec:
                continue
            attempts += 1
            clip_start = max(0, int(seg.start * sample_rate) - pad)
            clip_end = min(len(audio), int(seg.end * sample_rate) + pad)
            clip = audio[clip_start:clip_end]
            if len(clip) < 1600:
                continue
            # Preceding confident text primes the decoder with real
            # conversational context instead of a cold start.
            context = next(
                (
                    segments[j].text
                    for j in range(i - 1, -1, -1)
                    if segments[j].avg_logprob >= threshold and segments[j].text
                ),
                "",
            )
            prompt = f"{domain_prompt} {context}".strip()[-400:]
            try:
                new_text, new_lp, new_words = self._decode_clip(clip, language, prompt)
            except Exception as exc:
                logger.debug(f"ASR second pass decode failed at {seg.start:.1f}s: {exc}")
                continue
            new_text, new_raw_text = (
                self._sanitize_text_with_raw(new_text) if new_text else ("", "")
            )
            if (
                not new_text
                or self._is_known_hallucination(new_text)
                or self._is_prompt_echo(new_text, self._prompt_token_set(domain_prompt))
            ):
                continue
            offset_new_words = self._offset_words(new_words, clip_start / sample_rate)
            if low_conf_trigger:
                # Whole-segment low confidence: accept only a meaningfully
                # higher-likelihood hypothesis.
                accept = new_lp >= lp + margin
            else:
                # Suspect-word trigger: the original logprob is already
                # healthy, so a margin over it is an unreachable bar. Accept
                # when the decoder likes the rewrite at least as much AND the
                # word-level suspicion actually went down - both conditions
                # so a same-score different-text gamble is never taken.
                new_suspect = sum(
                    1
                    for w in offset_new_words or []
                    if isinstance(w, dict)
                    and len(re.sub(r"\W", "", str(w.get("word", "")))) >= 4
                    and float(w.get("probability", 1.0)) < 0.4
                )
                accept = new_lp >= lp and new_suspect < suspect_words
            if not accept:
                continue
            logger.debug(
                f"ASR second pass rescued [{seg.start:.1f}s] lp {lp:.2f}->{new_lp:.2f}: "
                f"'{seg.text}' -> '{new_text}'"
            )
            seg.text = new_text
            seg.avg_logprob = round(new_lp, 3)
            seg.words = offset_new_words
            seg.raw_text = new_raw_text
            rescued += 1
        if rescued:
            logger.info(f"ASR second pass: rescued {rescued} low-confidence segment(s).")
        return segments

    def _transcribe_single_channel(
        self, audio_input: Any, language: str = "tr", sector: str = "telecom"
    ) -> tuple[list[TranscriptionSegment], float]:
        """Transcribe a mono file path or 1D audio numpy array."""
        language = lock_language(language)
        from asr_pro.services.audio_conditioning import condition_telephony_audio
        from asr_pro.services.domain_adaptation import DomainAdaptationService

        audio_input = condition_telephony_audio(audio_input)
        domain_prompt = (
            f"{settings.asr_initial_prompt or ''} "
            f"{DomainAdaptationService.get_initial_prompt(sector)}"
        ).strip()
        prompt_tokens = self._prompt_token_set(domain_prompt)
        if self._is_mlx:
            import mlx_whisper

            # mlx-community's repo naming isn't a uniform "whisper-{size}"
            # pattern (e.g. large-v3 is hosted as "large-v3-mlx", not
            # "large-v3") - translate known exceptions, otherwise mirror
            # size unchanged (works for "turbo", "small", etc).
            mlx_repo_overrides = {"large-v3": "large-v3-mlx"}
            repo_size = mlx_repo_overrides.get(self._model_size, self._model_size)
            repo = f"mlx-community/whisper-{repo_size}"

            duration = float(len(audio_input)) / 16000.0 if len(audio_input) else 0.0
            temperature_schedule_mlx = tuple(
                float(t.strip()) for t in settings.asr_temperature.split(",") if t.strip()
            )
            regions = self._vad_speech_regions(audio_input)
            if regions is not None and not regions:
                logger.debug("ASR MLX: VAD found no speech in buffer; skipping decode.")
                return [], duration
            if regions is None:
                regions = [(0, len(audio_input))]
            logger.debug(f"Transcribing via MLX ({repo}) across {len(regions)} speech region(s)...")
            raw_segments: list[tuple[float, dict]] = []
            with self._inference_lock:
                for start_sample, end_sample in regions:
                    clip = audio_input[start_sample:end_sample]
                    if len(clip) < 160:  # <10ms, nothing decodable
                        continue
                    offset = start_sample / 16000.0
                    res = self._mlx_call(
                        mlx_whisper.transcribe,
                        clip,
                        path_or_hf_repo=repo,
                        language=language,
                        condition_on_previous_text=False,
                        initial_prompt=domain_prompt,
                        compression_ratio_threshold=2.0,
                        no_speech_threshold=settings.asr_no_speech_threshold,
                        word_timestamps=settings.asr_word_timestamps,
                        # Without an explicit schedule mlx_whisper retries
                        # failing segments at up to temperature 1.0 (six
                        # steps) - on noisy telephony audio those high-temp
                        # retries produced literal word salad ("Habi çöp,
                        # oğuz, içim, hındır" on a real call) and most of the
                        # decode time. Use the same bounded schedule as the
                        # faster-whisper path (default 0.0,0.2,0.4).
                        temperature=temperature_schedule_mlx,
                    )
                    for s in res.get("segments", []):
                        raw_segments.append((offset, s))
            segments_gen = []
            for offset, s in raw_segments:
                no_speech = self._safe_float_attr(s, "no_speech_prob", 0.0)
                logprob = self._safe_float_attr(s, "avg_logprob", 0.0)
                comp_ratio = self._safe_float_attr(s, "compression_ratio", 1.0)
                if no_speech > 0.6 or (logprob < -0.8 and comp_ratio > 1.8):
                    _count_filtered()
                    continue
                cleaned_text, raw_text = self._sanitize_text_with_raw(s["text"].strip())
                if cleaned_text and self._is_known_hallucination(cleaned_text):
                    logger.debug(f"ASR: Dropping known hallucination artifact: '{cleaned_text}'")
                    _count_filtered()
                    continue
                if cleaned_text and self._is_prompt_echo(cleaned_text, prompt_tokens):
                    logger.warning(
                        f"ASR: Dropping initial-prompt echo hallucination: '{cleaned_text}'"
                    )
                    _count_filtered()
                    continue
                if cleaned_text:
                    segments_gen.append(
                        TranscriptionSegment(
                            start=round(offset + float(s["start"]), 2),
                            end=round(offset + float(s["end"]), 2),
                            text=cleaned_text,
                            avg_logprob=logprob,
                            no_speech_prob=no_speech,
                            words=self._offset_words(s.get("words"), offset),
                            raw_text=raw_text,
                        )
                    )
            segments_gen = self._second_pass_rescue(
                audio_input, segments_gen, language, domain_prompt
            )
            return self._split_into_sentences(self._deduplicate_segments(segments_gen)), duration

        temperature_schedule = tuple(
            float(t.strip()) for t in settings.asr_temperature.split(",") if t.strip()
        )

        with self._inference_lock:
            segments_gen_fw, info = self._model.transcribe(
                audio_input,
                language=language,
                beam_size=settings.asr_beam_size,
                initial_prompt=domain_prompt,
                word_timestamps=settings.asr_word_timestamps,
                condition_on_previous_text=False,
                repetition_penalty=1.20,
                no_repeat_ngram_size=3,
                compression_ratio_threshold=2.0,
                log_prob_threshold=settings.asr_log_prob_threshold,
                no_speech_threshold=settings.asr_no_speech_threshold,
                temperature=temperature_schedule,
                vad_filter=True,
                vad_parameters={
                    "threshold": settings.asr_vad_threshold,
                    "min_speech_duration_ms": settings.asr_vad_min_speech_ms,
                    "min_silence_duration_ms": settings.asr_vad_min_silence_ms,
                    "speech_pad_ms": settings.asr_vad_speech_pad_ms,
                },
            )
            # Materialize the generator while still holding the lock: faster-whisper
            # decodes lazily as the generator is consumed, so a concurrent caller
            # could otherwise interleave inference with this one.
            segments_gen_fw = list(segments_gen_fw)

        segments = []
        for s in segments_gen_fw:
            no_speech = self._safe_float_attr(s, "no_speech_prob", 0.0)
            logprob = self._safe_float_attr(s, "avg_logprob", 0.0)
            comp_ratio = self._safe_float_attr(s, "compression_ratio", 1.0)
            if no_speech > 0.6 or (logprob < -0.8 and comp_ratio > 1.8):
                _count_filtered()
                continue
            cleaned_text, raw_text = self._sanitize_text_with_raw(s.text.strip())
            if cleaned_text and self._is_known_hallucination(cleaned_text):
                logger.debug(f"ASR: Dropping known hallucination artifact: '{cleaned_text}'")
                _count_filtered()
                continue
            if cleaned_text and self._is_prompt_echo(cleaned_text, prompt_tokens):
                logger.warning(f"ASR: Dropping initial-prompt echo hallucination: '{cleaned_text}'")
                _count_filtered()
                continue
            if cleaned_text:
                segments.append(
                    TranscriptionSegment(
                        start=s.start,
                        end=s.end,
                        text=cleaned_text,
                        avg_logprob=logprob,
                        no_speech_prob=no_speech,
                        words=self._offset_words(getattr(s, "words", None), 0.0),
                        raw_text=raw_text,
                    )
                )
        if isinstance(audio_input, np.ndarray):
            segments = self._second_pass_rescue(audio_input, segments, language, domain_prompt)
        return self._split_into_sentences(self._deduplicate_segments(segments)), info.duration

    def _transcribe_stereo(
        self, audio_path: str, language: str = "tr", sector: str = "telecom"
    ) -> tuple[list[TranscriptionSegment], float]:
        """Transcribe stereo audio by isolating left/right channels with post-transcription RMS bleed filtering.

        Left channel is assigned to SPEAKER_00 and right channel to SPEAKER_01.
        An acoustic RMS energy comparison filters out bleed-through after transcription so each party's words
        are only retained on their actual physical channel.
        """
        from faster_whisper import decode_audio

        logger.info(
            f"ASR: Stereo audio detected for {audio_path}. Isolating channels with post-transcription RMS bleed filtering."
        )
        left_ch, right_ch = decode_audio(audio_path, sampling_rate=16000, split_stereo=True)
        from asr_pro.services.audio_conditioning import is_ivr_segment

        # NOTE: channels are passed RAW into _transcribe_single_channel, which
        # applies condition_telephony_audio exactly once itself. Conditioning
        # here too would (a) double-filter the signal and (b) normalize each
        # channel's RMS independently to the same target level, which destroys
        # the L/R energy comparison the bleed filter below depends on (a quiet
        # bleed-through channel would get boosted ~10x to match the loud one).

        with time_block(asr_transcribe_duration_seconds, mode="batch"):
            segs_left, dur_left = self._transcribe_single_channel(
                left_ch, language=language, sector=sector
            )
            segs_right, dur_right = self._transcribe_single_channel(
                right_ch, language=language, sector=sector
            )

        def _get_rms(channel_data: np.ndarray, start_sec: float, end_sec: float) -> float:
            start_idx = int(start_sec * 16000)
            end_idx = int(end_sec * 16000)
            slice_data = channel_data[start_idx:end_idx]
            if len(slice_data) == 0:
                return 0.0
            return float(np.sqrt(np.mean(slice_data**2)))

        clean_left = []
        for s in segs_left:
            rms_l = _get_rms(left_ch, s.start, s.end)
            rms_r = _get_rms(right_ch, s.start, s.end)
            if rms_r > rms_l * 4.0 and rms_l < 0.015:
                continue
            s.speaker = (
                "[🤖 IVR / Sistem Mesajı]" if is_ivr_segment(s.text, s.start) else "SPEAKER_00"
            )
            clean_left.append(s)

        clean_right = []
        for s in segs_right:
            rms_r = _get_rms(right_ch, s.start, s.end)
            rms_l = _get_rms(left_ch, s.start, s.end)
            if rms_l > rms_r * 4.0 and rms_r < 0.015:
                continue
            s.speaker = (
                "[🤖 IVR / Sistem Mesajı]" if is_ivr_segment(s.text, s.start) else "SPEAKER_01"
            )
            clean_right.append(s)

        merged_segments = sorted(clean_left + clean_right, key=lambda x: (x.start, x.end))
        duration = max(dur_left, dur_right)
        if duration == 0.0 and len(merged_segments) > 0:
            duration = max(s.end for s in merged_segments)

        return self._split_into_sentences(self._deduplicate_segments(merged_segments)), duration

    @staticmethod
    def compute_confidence(segments: list[TranscriptionSegment]) -> float:
        """Aggregate per-segment decoder log-probabilities into a 0-1 confidence.

        exp(avg_logprob) approximates the mean per-token probability of a
        segment; the duration-weighted mean over all segments gives an honest
        file-level confidence (replacing the former hardcoded constant).
        Segments without confidence data (avg_logprob == -1.0 sentinel from
        engines that don't report it) are excluded.
        """
        import math

        weighted, total_w = 0.0, 0.0
        for s in segments:
            lp = getattr(s, "avg_logprob", -1.0)
            if lp is None or lp <= -1.0 or lp > 0.0:
                continue
            w = max(float(getattr(s, "end", 0.0)) - float(getattr(s, "start", 0.0)), 0.1)
            weighted += math.exp(lp) * w
            total_w += w
        if total_w == 0.0:
            return 0.0
        return round(min(max(weighted / total_w, 0.0), 1.0), 3)

    def transcribe(
        self, audio_path: Any, language: str = "tr", sector: str = "telecom"
    ) -> tuple[list[TranscriptionSegment], float]:
        """Transcribes an audio file or numpy array using Faster-Whisper / MLX.

        Stereo files are transcribed by isolating left/right channels with DSP crosstalk suppression.
        """
        language = lock_language(language)
        if self._model is None and not getattr(self, "_is_mlx", False):
            self.load_model()

        if isinstance(audio_path, str) and self._is_stereo_file(audio_path):
            try:
                return self._transcribe_stereo(audio_path, language=language, sector=sector)
            except Exception as exc:
                logger.warning(
                    f"Stereo channel isolation transcription failed ({exc}), falling back to mono downmix..."
                )
                try:
                    from faster_whisper import decode_audio

                    mono_audio = decode_audio(audio_path, sampling_rate=16000)
                    with time_block(asr_transcribe_duration_seconds, mode="batch"):
                        return self._transcribe_single_channel(
                            mono_audio, language=language, sector=sector
                        )
                except Exception as exc2:
                    logger.warning(
                        f"Mono downmix fallback also failed ({exc2}), falling back to raw file..."
                    )

        with time_block(asr_transcribe_duration_seconds, mode="batch"):
            return self._transcribe_single_channel(audio_path, language=language, sector=sector)

    def transcribe_array(
        self, pcm: np.ndarray, language: str = "tr", sector: str = "telecom"
    ) -> tuple[list[TranscriptionSegment], float]:
        """Transcribe an in-memory mono float32 PCM array (16kHz) for live streaming sessions.

        Bypasses stereo-file detection and disk I/O entirely — the caller (a
        StreamingASRSession) is responsible for supplying a bounded, already-decoded window.
        """
        language = lock_language(language)
        if self._model is None and not getattr(self, "_is_mlx", False):
            self.load_model()
        if pcm.size == 0:
            return [], 0.0
        with time_block(asr_transcribe_duration_seconds, mode="streaming"):
            return self._transcribe_single_channel(pcm, language=language, sector=sector)
