"""Standalone pyannote.audio 4.x (community-1) diarization — runs in its own process
under the /opt/p4 venv, on CPU. Reads /work/audio.wav, writes /work/turns.json.

Why CPU: community-1's reconstruction step allocates ~9.5GB (pyannote issue #1963,
unfixed in 4.0.7), which OOMs on <12GB GPUs. On CPU it uses system RAM instead.
Quality is device-independent, so you get community-1's better speaker counting/
assignment at the cost of speed (~0.5-1x realtime on a multi-core CPU).

Auth for the gated model comes from the HF_TOKEN env var (auto-read by huggingface_hub).
Audio is pre-decoded via soundfile into a {waveform, sample_rate} dict to bypass
torchcodec (4.x's default decoder, which needs ffmpeg shared libs not present here).
"""
import json
import os
import time
from pathlib import Path

WAV = "/work/audio.wav"
OUT = Path("/work/turns.json")
TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
NUM_SPEAKERS = int(os.environ.get("NUM_SPEAKERS", "0"))

if not TOKEN:
    raise SystemExit(
        "HF_TOKEN not set. community-1 is gated: create a free token at "
        "https://huggingface.co/settings/tokens after accepting the terms at "
        "https://hf.co/pyannote/speaker-diarization-community-1 (and segmentation-3.0), "
        "then run with -e HF_TOKEN=hf_xxx"
    )

import torch
from pyannote.audio import Pipeline

print("[pyannote4] loading speaker-diarization-community-1 (cpu) ...", flush=True)
t0 = time.time()
pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
pipe.to(torch.device("cpu"))
print(f"[pyannote4] loaded in {time.time()-t0:.1f}s", flush=True)

# pre-decode audio via soundfile (bypasses torchcodec)
import soundfile as sf
data, sr = sf.read(WAV, dtype="float32")
waveform = torch.from_numpy(data).unsqueeze(0)  # (channel=1, time)

kw = {}
if NUM_SPEAKERS > 0:
    kw["num_speakers"] = NUM_SPEAKERS  # community-1's VBx clustering needs a speaker count
spk = str(NUM_SPEAKERS) if NUM_SPEAKERS > 0 else "auto"
print(f"[pyannote4] diarizing {len(data)/sr:.0f}s on CPU (num_speakers={spk}) — this is slow ...", flush=True)
t1 = time.time()
output = pipe({"waveform": waveform, "sample_rate": sr}, **kw)

# 4.x returns a DiarizeOutput; use .speaker_diarization (an Annotation) for the turns
ann = output.speaker_diarization
label_to_idx = {}
turns = []
for segment, _track, label in ann.itertracks(yield_label=True):
    if label not in label_to_idx:
        label_to_idx[label] = len(label_to_idx)
    turns.append([round(segment.start, 3), round(segment.end, 3), label_to_idx[label]])
turns.sort()
print(f"[pyannote4] {len(turns)} turns ({len(label_to_idx)} speakers) in {time.time()-t1:.1f}s", flush=True)
OUT.write_text(json.dumps(turns))
