#!/usr/bin/env python3
"""
Whisper transcription + speaker diarization pipeline (Docker entrypoint).

The orchestrator does NO GPU work itself: it extracts audio (ffmpeg), then spawns
TWO subprocesses — transcription then diarization — and merges their JSON results.
Each CUDA engine (ctranslate2 / onnxruntime / torch) runs alone in its own process;
crucially, the transcription process EXITS before diarization starts, so ctranslate2's
VRAM is fully released and diarization gets the whole GPU (ctranslate2 otherwise keeps
~3.9GB reserved even after del+gc, which would starve an 8GB card).

Env vars:
  INPUT              path under /input to transcribe (default: first file in /input)
  MODEL              faster-whisper model: tiny|base|small|medium|large-v3|large-v3-turbo  (default large-v3)
  TRANSCRIBE_DEVICE  cuda|cpu|auto  (default auto = use GPU if present)
  DIARIZE_DEVICE     cuda|cpu|auto  (default auto = use GPU if present)
  DEVICE             fallback for both of the above if they're unset.
  COMPUTE_TYPE       float16|int8_float16|int8  (default: float16 on cuda, int8 on cpu)
  LANGUAGE           language code, e.g. pl|en, or auto  (default auto)
  NUM_SPEAKERS       number of speakers for clustering, or 0 = auto  (default 2)
  DIARIZE            1|0  run speaker diarization  (default 1)
  DIARIZER           pyannote|sherpa  diarization backend  (default pyannote)
  HF_TOKEN           required for pyannote (gated models); ignored by sherpa
  SPEAKER_NAMES      optional comma list mapping speaker index -> name, e.g. "Alice,Bob,Carol"
  BEAM_SIZE          decoding beam size  (default 5)
"""
import json
import os
import subprocess
import sys
import time
from glob import glob
from pathlib import Path

INPUT_DIR = Path("/input")
OUTPUT_DIR = Path("/output")

TRANSCRIBE_DEVICE = os.environ.get("TRANSCRIBE_DEVICE") or os.environ.get("DEVICE", "auto")
DIARIZE_DEVICE = os.environ.get("DIARIZE_DEVICE") or os.environ.get("DEVICE", "auto")
NUM_SPEAKERS = int(os.environ.get("NUM_SPEAKERS", "2"))
DIARIZE = os.environ.get("DIARIZE", "1") == "1"
DIARIZER = os.environ.get("DIARIZER", "pyannote").lower()
SPEAKER_NAMES = [s for s in os.environ.get("SPEAKER_NAMES", "").split(",") if s.strip()]


def resolve_device(envval):
    """Detect GPU presence WITHOUT importing any CUDA library, so the orchestrator
    never reserves VRAM. (The workers resolve 'auto' themselves too, but this keeps
    the startup log line accurate.)"""
    if envval != "auto":
        return envval
    try:
        subprocess.run(["nvidia-smi"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return "cuda"
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


def transcribe(device):
    """Run transcription in a SEPARATE process so its VRAM is freed on exit."""
    log(f"[whisper] transcribing (device={device}) in subprocess ...")
    env = {**os.environ, "TRANSCRIBE_DEVICE": device}
    t0 = time.time()
    subprocess.run([sys.executable, "/app/transcribe_worker.py"], env=env, check=True)
    data = json.loads(Path("/work/segments.json").read_text())
    log(f"[whisper] done in {time.time()-t0:.1f}s (language={data['language']})")
    return data["segments"], data["language"]


def diarize(wav, device):
    """Run diarization in a SEPARATE process (CUDA-context isolation).
    pyannote 3.x/4.x live in separate venvs (/opt/p3, /opt/p4) because their torch/
    huggingface_hub deps conflict; pick the matching interpreter per backend."""
    has_token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    if DIARIZER == "sherpa":
        py, worker = sys.executable, "/app/diarize_worker.py"
        n = NUM_SPEAKERS if NUM_SPEAKERS > 0 else -1
        provider = "cuda" if device == "cuda" else "cpu"
        log(f"[sherpa] diarizing (provider={provider}, num_speakers={n}) in subprocess ...")
    elif DIARIZER == "pyannote4":
        if not has_token:
            raise SystemExit("DIARIZER=pyannote4 but HF_TOKEN is not set (community-1 is gated).")
        # community-1's reconstruction needs ~9.5GB → OOMs on <12GB GPUs (issue #1963); force CPU.
        device = "cpu"
        py, worker = "/opt/p4/bin/python", "/app/diarize_pyannote4.py"
        spk = NUM_SPEAKERS if NUM_SPEAKERS > 0 else "auto"
        log(f"[pyannote4] diarizing with community-1 on CPU (slow; num_speakers={spk}) ...")
    else:  # pyannote (3.x)
        if not has_token:
            raise SystemExit(
                "DIARIZER=pyannote but HF_TOKEN is not set. Either set HF_TOKEN (free, "
                "see diarize_pyannote.py) or fall back with -e DIARIZER=sherpa")
        py, worker = "/opt/p3/bin/python", "/app/diarize_pyannote.py"
        spk = NUM_SPEAKERS if NUM_SPEAKERS > 0 else "auto"
        log(f"[pyannote] diarizing (device={device}, num_speakers={spk}) in subprocess ...")
    env = {**os.environ, "DIARIZE_DEVICE": device, "NUM_SPEAKERS": str(NUM_SPEAKERS)}
    t0 = time.time()
    subprocess.run([py, worker], env=env, check=True)
    turns = json.loads(Path("/work/turns.json").read_text())
    log(f"[{DIARIZER}] {len(turns)} turns in {time.time()-t0:.1f}s")
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
    segs, language = transcribe(tdev)
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
