"""Standalone sherpa-onnx diarization (runs in its own process for CUDA isolation).
Reads /work/audio.wav, writes /work/turns.json (list of [start, end, speaker])."""
import io
import json
import os
import tarfile
import time
import urllib.request
from pathlib import Path

import sherpa_onnx
import soundfile as sf

SHERPA_MODELS = Path("/models/sherpa")
WAV = "/work/audio.wav"
OUT = Path("/work/turns.json")
DEVICE = os.environ.get("DIARIZE_DEVICE", "cpu")
N = int(os.environ.get("NUM_SPEAKERS", "3"))
N = N if N > 0 else -1

SEG_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
EMB_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"


def ensure_models():
    SHERPA_MODELS.mkdir(parents=True, exist_ok=True)
    seg_dir = SHERPA_MODELS / "sherpa-onnx-pyannote-segmentation-3-0"
    seg_onnx = seg_dir / "model.onnx"
    emb_onnx = SHERPA_MODELS / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
    if not seg_onnx.exists():
        print("[sherpa] downloading segmentation model ...", flush=True)
        buf = urllib.request.urlopen(SEG_URL).read()
        with tarfile.open(fileobj=io.BytesIO(buf), mode="r:bz2") as tf:
            tf.extractall(SHERPA_MODELS)
    if not emb_onnx.exists():
        print("[sherpa] downloading speaker embedding model ...", flush=True)
        urllib.request.urlretrieve(EMB_URL, emb_onnx)
    return seg_onnx, emb_onnx


seg_onnx, emb_onnx = ensure_models()
provider = "cuda" if DEVICE == "cuda" else "cpu"
cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
    segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
        pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(seg_onnx)),
        provider=provider,
    ),
    embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb_onnx), provider=provider),
    clustering=sherpa_onnx.FastClusteringConfig(num_clusters=N, threshold=0.5),
    min_duration_on=0.3,
    min_duration_off=0.5,
)
if not cfg.validate():
    raise SystemExit("sherpa config invalid")
sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
audio, sr = sf.read(WAV, dtype="float32")
if sr != sd.sample_rate:
    import librosa
    audio = librosa.resample(audio, orig_sr=sr, target_sr=sd.sample_rate)
print(f"[sherpa] diarizing {len(audio)/sd.sample_rate:.0f}s (provider={provider}, num_speakers={N})", flush=True)
t0 = time.time()
turns = [[t.start, t.end, t.speaker] for t in sd.process(audio).sort_by_start_time()]
print(f"[sherpa] {len(turns)} turns in {time.time()-t0:.1f}s", flush=True)
OUT.write_text(json.dumps(turns))
