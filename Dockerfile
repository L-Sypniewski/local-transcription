# Transcription (faster-whisper/CTranslate2) + diarization (sherpa-onnx).
# Single image serves BOTH GPU and CPU; each stage is independently switchable
# via TRANSCRIBE_DEVICE / DIARIZE_DEVICE (see .env.example).
#
# IMPORTANT: transcription and diarization run in SEPARATE processes (pipeline.py
# spawns diarize_worker.py). ctranslate2 and onnxruntime each own a CUDA context;
# running both in one process corrupts memory (std::length_error). Process
# isolation lets both use the GPU safely.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install \
        faster-whisper soundfile librosa \
        "sherpa-onnx==1.13.4+cuda12.cudnn9" -f https://k2-fsa.github.io/sherpa/onnx/cuda.html

WORKDIR /app
COPY pipeline.py diarize_worker.py /app/

ENTRYPOINT ["python3", "/app/pipeline.py"]
