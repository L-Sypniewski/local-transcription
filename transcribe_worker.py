#!/usr/bin/env python3
"""Standalone faster-whisper transcription (runs in its own process).

Keeping transcription in a separate process means ctranslate2's CUDA memory is
released back to the driver when this process EXITS, so the diarization subprocess
that follows gets the full GPU (ctranslate2 otherwise keeps ~3.9GB reserved even
after del+gc, leaving only ~4GB for diarization on an 8GB card).

Reads /work/audio.wav, writes /work/segments.json = {"language": ..., "segments": [...]}.

Env vars:
  MODEL              faster-whisper model (default large-v3)
  TRANSCRIBE_DEVICE  cuda|cpu|auto  (auto = use GPU if present)
  COMPUTE_TYPE       float16|int8_float16|int8  (default: float16 on cuda, int8 on cpu)
  LANGUAGE           language code or auto
  BEAM_SIZE          decoding beam size (default 5)
"""
import gc
import json
import os
import time
from pathlib import Path

WAV = "/work/audio.wav"
OUT = Path("/work/segments.json")
WHISPER_MODELS = Path("/models/whisper")

MODEL = os.environ.get("MODEL", "large-v3")
DEVICE = os.environ.get("TRANSCRIBE_DEVICE") or os.environ.get("DEVICE", "auto")
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE")
LANGUAGE = os.environ.get("LANGUAGE", "auto")
BEAM_SIZE = int(os.environ.get("BEAM_SIZE", "5"))


def resolve_device(envval):
    if envval != "auto":
        return envval
    try:
        import ctranslate2
        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


def main():
    device = resolve_device(DEVICE)
    compute = COMPUTE_TYPE or ("float16" if device == "cuda" else "int8")
    WHISPER_MODELS.mkdir(parents=True, exist_ok=True)
    print(f"[whisper] model={MODEL} device={device} compute_type={compute} language={LANGUAGE}", flush=True)
    from faster_whisper import WhisperModel
    t0 = time.time()
    model = WhisperModel(MODEL, device=device, compute_type=compute, download_root=str(WHISPER_MODELS))
    kw = {"vad_filter": True, "beam_size": BEAM_SIZE}
    if LANGUAGE and LANGUAGE != "auto":
        kw["language"] = LANGUAGE
    segments, info = model.transcribe(str(WAV), **kw)
    segs = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]
    print(f"[whisper] {len(segs)} segments in {time.time()-t0:.1f}s "
          f"(language={info.language} p={info.language_probability:.2f})", flush=True)
    del model
    gc.collect()
    OUT.write_text(json.dumps({"language": info.language, "segments": segs}))


if __name__ == "__main__":
    main()
