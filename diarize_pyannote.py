"""Standalone pyannote.audio 3.1 diarization (runs in its own process for CUDA isolation).
Reads /work/audio.wav, writes /work/turns.json (list of [start, end, speaker]).

pyannote models are gated — HF_TOKEN must be set (free: accept terms at
  https://hf.co/pyannote/speaker-diarization-3.1
  https://hf.co/pyannote/segmentation-3.0
then create a token at https://huggingface.co/settings/tokens).
"""
import json
import os
import time
from pathlib import Path

WAV = "/work/audio.wav"
OUT = Path("/work/turns.json")
TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
NUM_SPEAKERS = int(os.environ.get("NUM_SPEAKERS", "0"))
DIARIZE_DEVICE = os.environ.get("DIARIZE_DEVICE", "auto")

if not TOKEN:
    raise SystemExit(
        "HF_TOKEN not set. pyannote models are gated: create a free token at "
        "https://huggingface.co/settings/tokens after accepting the terms at "
        "https://hf.co/pyannote/speaker-diarization-3.1 and "
        "https://hf.co/pyannote/segmentation-3.0, then run with -e HF_TOKEN=hf_xxx"
    )

import torch
from pyannote.audio import Pipeline

use_cuda = (DIARIZE_DEVICE == "cuda") or (DIARIZE_DEVICE == "auto" and torch.cuda.is_available())
device = torch.device("cuda" if use_cuda else "cpu")

print(f"[pyannote] loading speaker-diarization-3.1 (device={device.type}) ...", flush=True)
t0 = time.time()
# Auth via HF_TOKEN env var (auto-read by huggingface_hub) — version-agnostic,
# so no use_auth_token kwarg that broke between pyannote 3.x and 4.x.
pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
pipe.to(device)
print(f"[pyannote] loaded in {time.time()-t0:.1f}s", flush=True)

kw = {}
if NUM_SPEAKERS > 0:
    kw["num_speakers"] = NUM_SPEAKERS
    spk = str(NUM_SPEAKERS)
else:
    spk = "auto"
# Pre-load audio as a {waveform, sample_rate} dict. pyannote 4.x's default audio
# decoder (torchcodec) needs ffmpeg shared libs that aren't in this image; feeding
# a pre-decoded tensor via soundfile bypasses torchcodec entirely.
import soundfile as sf
data, sr = sf.read(WAV, dtype="float32")
waveform = torch.from_numpy(data).unsqueeze(0)  # (channel=1, time)
print(f"[pyannote] diarizing {len(data)/sr:.0f}s (num_speakers={spk}) ...", flush=True)
t1 = time.time()
diarization = pipe({"waveform": waveform, "sample_rate": sr}, **kw)

label_to_idx = {}
turns = []
for segment, _track, label in diarization.itertracks(yield_label=True):
    if label not in label_to_idx:
        label_to_idx[label] = len(label_to_idx)
    turns.append([round(segment.start, 3), round(segment.end, 3), label_to_idx[label]])
turns.sort()
print(f"[pyannote] {len(turns)} turns ({len(label_to_idx)} speakers) in {time.time()-t1:.1f}s", flush=True)
OUT.write_text(json.dumps(turns))
