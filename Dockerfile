# Transcription (faster-whisper/CTranslate2) + diarization (pyannote.audio OR sherpa-onnx).
# Single image serves BOTH GPU and CPU; each stage is independently switchable
# via TRANSCRIBE_DEVICE / DIARIZE_DEVICE (see .env.example). The diarization backend
# is chosen via DIARIZER=pyannote (default) or DIARIZER=sherpa.
#
# IMPORTANT: transcription and diarization run in SEPARATE processes (pipeline.py
# spawns the diarize worker). ctranslate2, onnxruntime and torch each own a CUDA
# context; running any pair in one process corrupts memory (std::length_error).
# Process isolation lets transcription use the GPU safely. The diarize subprocess
# runs ONE backend per invocation, so the engines never coexist in a process.
#
# pyannote.audio 3.1 runs on GPU: it processes audio in sliding windows, so GPU
# memory is bounded by window size (not total duration) and a long meeting fits on
# an 8GB card. Its speaker embeddings are language-agnostic (good for Polish).
# pyannote's models are gated — a free HF token (HF_TOKEN) is required once.
# sherpa-onnx (CUDA) is kept as a token-free fallback (DIARIZER=sherpa).
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HOME=/models/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg ca-certificates libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install \
        faster-whisper soundfile librosa \
        "sherpa-onnx==1.13.4+cuda12.cudnn9" -f https://k2-fsa.github.io/sherpa/onnx/cuda.html

# pyannote.audio diarization on GPU. Install CUDA torch first (bundled CUDA 12.4
# runtime + cuDNN run fine on the 12.8 base image) so pyannote.audio reuses it.
# Pin pyannote.audio<4: v4 needs torch>=2.8 (CUDA 13 build, needs a newer driver),
# while v3 keeps CUDA-12 torch and matches the speaker-diarization-3.1 model.
RUN pip install --index-url https://download.pytorch.org/whl/cu124 torch torchaudio \
 && pip install "pyannote.audio<4"

WORKDIR /app
COPY pipeline.py diarize_worker.py diarize_pyannote.py /app/

ENTRYPOINT ["python3", "/app/pipeline.py"]
