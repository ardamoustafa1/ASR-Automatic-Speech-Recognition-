import argparse
import time
from pathlib import Path


def benchmark_mlx(audio_file: str, model_name: str):
    import mlx.core as mx
    import mlx_whisper
    from mlx_whisper.transcribe import ModelHolder

    repo = model_name if model_name.startswith("mlx-community/") else f"mlx-community/whisper-{model_name}"

    load_started = time.perf_counter()
    ModelHolder.get_model(repo, mx.float16)
    load_s = time.perf_counter() - load_started

    transcribe_started = time.perf_counter()
    result = mlx_whisper.transcribe(
        audio_file,
        path_or_hf_repo=repo,
        language="tr",
        task="transcribe",
        temperature=(0.0,),
        condition_on_previous_text=False,
        word_timestamps=False,
        fp16=True,
    )
    transcribe_s = time.perf_counter() - transcribe_started

    return {
        "engine": "mlx",
        "model": repo,
        "load_s": load_s,
        "transcribe_s": transcribe_s,
        "text": result.get("text", "").strip(),
        "segments": len(result.get("segments", [])),
    }


def benchmark_faster_whisper(audio_file: str, model_name: str):
    import ctranslate2
    from faster_whisper import WhisperModel

    cpu_threads = 10
    supported = ctranslate2.get_supported_compute_types("cpu")
    compute_type = "int8_float32" if "int8_float32" in supported else "int8"

    load_started = time.perf_counter()
    model = WhisperModel(
        model_name,
        device="cpu",
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=4,
    )
    load_s = time.perf_counter() - load_started

    transcribe_started = time.perf_counter()
    segments, _ = model.transcribe(
        audio_file,
        language="tr",
        task="transcribe",
        beam_size=2,
        best_of=2,
        temperature=(0.0,),
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters={"threshold": 0.42, "min_silence_duration_ms": 320, "speech_pad_ms": 160},
        word_timestamps=False,
    )
    segments = list(segments)
    transcribe_s = time.perf_counter() - transcribe_started

    return {
        "engine": "faster-whisper-cpu",
        "model": model_name,
        "load_s": load_s,
        "transcribe_s": transcribe_s,
        "text": " ".join(s.text.strip() for s in segments).strip(),
        "segments": len(segments),
    }


def main():
    parser = argparse.ArgumentParser(description="ASR speed benchmark with load/transcribe split.")
    parser.add_argument("--audio", default="ses2.wav", help="Audio file path")
    parser.add_argument("--model", default="large-v3-turbo", help="Model name")
    parser.add_argument("--engine", choices=("mlx", "faster-whisper"), default="mlx")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    if args.engine == "mlx":
        result = benchmark_mlx(str(audio_path), args.model)
    else:
        result = benchmark_faster_whisper(str(audio_path), args.model)

    print(f"Engine: {result['engine']}")
    print(f"Model: {result['model']}")
    print(f"Model load/cache: {result['load_s']:.2f}s")
    print(f"Transcribe only: {result['transcribe_s']:.2f}s")
    print(f"Segments: {result['segments']}")
    print("Text:")
    print(result["text"])


if __name__ == "__main__":
    main()
