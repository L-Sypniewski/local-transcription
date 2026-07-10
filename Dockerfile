# Transcription (faster-whisper/CTranslate2) + diarization (pyannote.audio 3.x OR 4.x
# OR sherpa-onnx). Single image; each stage independently switchable via
# TRANSCRIBE_DEVICE / DIARIZE_DEVICE, and the diarizer via DIARIZER:
#   pyannote  = pyannote.audio 3.x, speaker-diarization-3.1  (GPU, fast, default)
#   pyannote4 = pyannote.audio 4.x, speaker-diarization-community-1  (CPU, best quality)
#   sherpa    = sherpa-onnx  (GPU/CPU, token-free fallback)
#
# pyannote 3.x and 4.x CANNOT share a Python env (same package; 3.x needs torch<2.6 +
# huggingface_hub<1.0, 4.x needs torch>=2.8 + huggingface_hub>=1.x), so they live in two
# virtual environments and the orchestrator spawns the matching one:
#   /opt/p3  pyannote.audio 3.x + torch 2.5.1+cu124   (GPU)
#   /opt/p4  pyannote.audio 4.x + CPU torch           (CPU — see below)
#
# Transcription and diarization run in SEPARATE processes (ctranslate2/onnxruntime/torch
# each own a CUDA context and corrupt each other if run in one process). Transcription's
# process EXITS before diarization starts, freeing all VRAM for diarization.
#
# pyannote 4.x (community-1) runs on CPU: its reconstruction step needs a ~9.5GB
# allocation that OOMs on <12GB GPUs (pyannote issue #1963, unfixed in 4.0.7). On CPU it
# uses system RAM. pyannote 3.x runs on GPU (sliding-window model, ~2-3GB peak, fits 8GB).
# All pyannote models are gated — a free HF token (HF_TOKEN) is required once.

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HOME=/models/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv ffmpeg ca-certificates libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Base: orchestrator + transcription (faster-whisper) + sherpa diarization.
# No torch here — keeps the base lean; the pyannote venvs each bring their own torch.
RUN pip install \
        faster-whisper soundfile librosa \
        "sherpa-onnx==1.13.4+cuda12.cudnn9" -f https://k2-fsa.github.io/sherpa/onnx/cuda.html

# /opt/p3 — pyannote.audio 3.x on GPU. torch 2.5.1 (v2.6 broke 3.x checkpoint loading via
# weights_only=True); huggingface_hub<1.0 (v1 removed `use_auth_token` that 3.x uses).
RUN python3 -m venv /opt/p3 \
 && /opt/p3/bin/pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchaudio==2.5.1 \
 && /opt/p3/bin/pip install "pyannote.audio<4" "huggingface_hub<1.0" soundfile

# /opt/p4 — pyannote.audio 4.x (community-1) on CPU. CPU-only torch (~200MB, not the ~3GB
# CUDA13 wheel — it's dead weight since 4.x is forced to CPU). 4.x needs torch>=2.8 for
# the version; the CPU build satisfies it. Uninstall torchcodec (a 4.x dep that prints 6
# noisy CUDA-loader tracebacks on CPU); pyannote has a clean fallback and the worker feeds
# a pre-decoded {waveform, sample_rate} dict anyway, so torchcodec is never actually used.
RUN python3 -m venv /opt/p4 \
 && /opt/p4/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio \
 && /opt/p4/bin/pip install pyannote.audio soundfile \
 && /opt/p4/bin/pip uninstall -y torchcodec

WORKDIR /app
COPY pipeline.py transcribe_worker.py diarize_worker.py diarize_pyannote.py diarize_pyannote4.py /app/

ENTRYPOINT ["python3", "/app/pipeline.py"]
