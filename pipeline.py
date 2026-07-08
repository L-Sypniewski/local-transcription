#!/usr/bin/env python3
"""
Whisper transcription + speaker diarization pipeline (Docker entrypoint).

Transcription and diarization run on INDEPENDENT devices and in SEPARATE processes
(ctranslate2 and onnxruntime each own a CUDA context and corrupt each other if run
in one process, so diarization is spawned as a subprocess).

Env vars:
  INPUT              path under /input to transcribe (default: first file in /input)
  MODEL              faster-whisper model: tiny|base|small|medium|large-v3|large-v3-turbo  (default large-v3)
  TRANSCRIBE_DEVICE  cuda|cpu|auto  (default auto = use GPU if present)
  DIARIZE_DEVICE     cuda|cpu|auto  (default auto = use GPU if present)
  DEVICE             fallback for both of the above if they are unset.
  COMPUTE_TYPE       float16|int8_float16|int8  (default: float16 on cuda, int8 on cpu)
  LANGUAGE           language code, e.g. pl|en, or auto  (default auto)
  NUM_SPEAKERS       number of speakers for clustering, or 0 = auto  (default 3)
  DIARIZE            1|0  run speaker diarization  (default 1)
  SPEAKER_NAMES      optional comma list mapping speaker index -> name, e.g. "Alice,Bob,Carol"
  BEAM_SIZE          decoding beam size  (default 5)
"""
import gc
import json
import os
import subprocess
import sys
import time
from glob import glob
from pathlib import Path

INPUT_DIR = Path("/input")
OUTPUT_DIR = Path("/output")
WHISPER_MODELS = Path("/models/whisper")

MODEL = os.environ.get("MODEL", "large-v3")
TRANSCRIBE_DEVICE = os.environ.get("TRANSCRIBE_DEVICE") or os.environ.get("DEVICE", "auto")
DIARIZE_DEVICE = os.environ.get("DIARIZE_DEVICE") or os.environ.get("DEVICE", "auto")
LANGUAGE = os.environ.get("LANGUAGE", "auto")
NUM_SPEAKERS = int(os.environ.get("NUM_SPEAKERS", "3"))
DIARIZE = os.environ.get("DIARIZE", "1") == "1"
BEAM_SIZE = int(os.environ.get("BEAM_SIZE", "5"))
SPEAKER_NAMES = [s for s in os.environ.get("SPEAKER_NAMES", "").split(",") if s.strip()]


def resolve_device(envval):
    if envval != "auto":
        return envval
    try:
        import ctranslate2
        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


def log(msg):
    print(msg, flush=True)


def find_input():
    if os.environ.get("INPUT"):
        p = Path(os.environ["INPUT"])
        if not p.is_absolute():
            p = INPUT_DIR / p
        return p
    exts = ("*.mp4", "*.mkv", "*.mov", "*.mp3", "*.wav", "*.m4a", "*.webm", "*.ogg",
            "*.oga", "*.flac", "*.aac")
    for e in exts:
        hits = sorted(glob(str(INPUT_DIR / e)))
        if hits:
            return Path(hits[0])
    raise SystemExit(f"No input file found in {INPUT_DIR}. Mount one or set INPUT.")


def extract_audio(src):
    wav = Path("/work/audio.wav")
    wav.parent.mkdir(parents=True, exist_ok=True)
    log(f"[ffmpeg] extracting 16kHz mono WAV from {src.name} ...")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
        check=True,
    )
    return wav


def transcribe(wav, device):
    from faster_whisper import WhisperModel
    compute = os.environ.get("COMPUTE_TYPE") or ("float16" if device == "cuda" else "int8")
    WHISPER_MODELS.mkdir(parents=True, exist_ok=True)
    log(f"[whisper] model={MODEL} device={device} compute_type={compute} language={LANGUAGE}")
    t0 = time.time()
    model = WhisperModel(MODEL, device=device, compute_type=compute, download_root=str(WHISPER_MODELS))
    kw = {"vad_filter": True, "beam_size": BEAM_SIZE}
    if LANGUAGE and LANGUAGE != "auto":
        kw["language"] = LANGUAGE
    segments, info = model.transcribe(str(wav), **kw)
    segs = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]
    log(f"[whisper] {len(segs)} segments in {time.time()-t0:.1f}s "
        f"(language={info.language} p={info.language_probability:.2f})")
    del model  # free CUDA memory before the diarization subprocess uses the GPU
    gc.collect()
    return segs, info.language


def diarize(wav, device):
    """Run sherpa-onnx diarization in a SEPARATE process (CUDA-context isolation)."""
    n = NUM_SPEAKERS if NUM_SPEAKERS > 0 else -1
    provider = "cuda" if device == "cuda" else "cpu"
    log(f"[sherpa] diarizing (provider={provider}, num_speakers={n}) in subprocess ...")
    env = {**os.environ, "DIARIZE_DEVICE": device, "NUM_SPEAKERS": str(NUM_SPEAKERS)}
    t0 = time.time()
    subprocess.run([sys.executable, "/app/diarize_worker.py"], env=env, check=True)
    turns = json.loads(Path("/work/turns.json").read_text())
    log(f"[sherpa] {len(turns)} turns in {time.time()-t0:.1f}s")
    return [(t[0], t[1], t[2]) for t in turns]


def merge(segs, turns):
    def overlap(a_s, a_e, b_s, b_e):
        return max(0.0, min(a_e, b_e) - max(a_s, b_s))
    for s in segs:
        best_sp, best_ov, near_sp, near_d = None, -1.0, None, 1e18
        for ds, de, sp in turns:
            ov = overlap(s["start"], s["end"], ds, de)
            if ov > best_ov:
                best_ov, best_sp = ov, sp
            mid = (s["start"] + s["end"]) / 2
            d = min(abs(mid - ds), abs(mid - de))
            if d < near_d:
                near_d, near_sp = d, sp
        s["speaker"] = best_sp if best_ov > 0 else near_sp
    return segs


def speaker_label(idx):
    if idx < len(SPEAKER_NAMES):
        return SPEAKER_NAMES[idx]
    return f"SPEAKER_{idx:02d}"


def ts(x):
    h = int(x // 3600); m = int((x % 3600) // 60); s = x % 60
    ms = int((s - int(s)) * 1000); s = int(s)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_outputs(segs, stem, language, turns):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUTPUT_DIR / stem
    for s in segs:
        if "speaker" in s:
            s["speaker_name"] = speaker_label(s["speaker"])
    txt = "".join(s["text"] + " " for s in segs).strip()
    (base.with_suffix(".txt")).write_text(txt + "\n", encoding="utf-8")
    spk_time = {}
    for ds, de, sp in turns:
        spk_time[sp] = spk_time.get(sp, 0) + (de - ds)
    with open(base.with_suffix(".srt"), "w", encoding="utf-8") as f:
        for i, s in enumerate(segs, 1):
            sp = f"[{s['speaker_name']}] " if s.get("speaker_name") else ""
            f.write(f"{i}\n{sp}{ts(s['start'])} --> {ts(s['end'])}\n{s['text']}\n\n")
    with open(base.with_suffix(".vtt"), "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        prev = None
        for s in segs:
            if s.get("speaker_name") and s["speaker_name"] != prev:
                f.write(f"<v {s['speaker_name']}>\n"); prev = s["speaker_name"]
            f.write(f"{ts(s['start']).replace(',', '.')} --> {ts(s['end']).replace(',', '.')}\n{s['text']}\n\n")
    with open(base.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump({"language": language, "speaking_time_seconds": spk_time, "segments": segs},
                  f, ensure_ascii=False, indent=2)
    log(f"[done] -> {base}.txt | .srt | .vtt | .json")
    if spk_time:
        log("speaking time: " + ", ".join(f"{speaker_label(k)}={v:.0f}s" for k, v in sorted(spk_time.items())))


def main():
    src = find_input()
    log(f"input: {src}")
    tdev = resolve_device(TRANSCRIBE_DEVICE)
    ddev = resolve_device(DIARIZE_DEVICE)
    log(f"transcribe device: {tdev} | diarize device: {ddev}")
    wav = extract_audio(src)
    segs, language = transcribe(wav, tdev)
    turns = []
    if DIARIZE:
        turns = diarize(wav, ddev)
        segs = merge(segs, turns)
    else:
        for s in segs:
            s["speaker_name"] = None
    write_outputs(segs, src.stem, language, turns)


if __name__ == "__main__":
    main()
