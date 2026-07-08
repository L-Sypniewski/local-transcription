# whisper-diarize

Transcribe audio/video and label speakers, fully offline, in Docker. Powered by
**faster-whisper** (Whisper large-v3 on GPU) for transcription and **sherpa-onnx**
(pyannote segmentation + 3D-Speaker embeddings) for diarization.

Everything runs locally — no API keys, no cloud. Pick any Whisper model, run on
GPU or CPU (independently per stage), and keep models cached in a volume so
nothing re-downloads.

## Quick start

```bash
# 1. drop your file into input/
cp /path/to/meeting.mp4 input/

# 2. run (CPU-only — works on any machine)
docker compose run --rm transcribe

# 2b. or run on GPU (needs nvidia-container-toolkit, see below)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm transcribe
```

Results land in `output/`:
- `<name>.txt`  — plain transcript (with speaker labels inline)
- `<name>.srt`  — subtitles, `[Speaker]` + timestamps
- `<name>.vtt`  — web subtitles (`<v Name>` voice tags)
- `<name>.json` — segments + speaker + speaking-time stats

## Prerequisites

- **Docker** + **Docker Compose v2**
- **For GPU only:** an NVIDIA GPU with a working driver, plus the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
  Verify GPU access from inside a container with:
  ```bash
  docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
  ```

CUDA toolkit, Python, and ffmpeg are bundled in the image — nothing extra to install.

## Configuration

All options are environment variables. Set them inline (`-e KEY=VAL`) or copy
`.env.example` to `.env` and edit. Defaults are sensible.

| Variable | Default | Description |
|---|---|---|
| `INPUT` | first file in `input/` | filename inside `input/` (or absolute path under `/input`) |
| `MODEL` | `large-v3` | `tiny` `base` `small` `medium` `large-v3` `large-v3-turbo` |
| `TRANSCRIBE_DEVICE` | `auto` | `cuda` `cpu` `auto` (auto = use GPU if present) |
| `DIARIZE_DEVICE` | `auto` | `cuda` `cpu` `auto` — **independent** of transcription |
| `DEVICE` | — | fallback for both of the above if they're unset |
| `COMPUTE_TYPE` | auto | `float16` `int8_float16` `int8` (auto: float16 on cuda, int8 on cpu) |
| `LANGUAGE` | `auto` | language code (`pl`, `en`, …) or `auto` to detect |
| `NUM_SPEAKERS` | `3` | known speaker count, or `0` to auto-cluster |
| `DIARIZE` | `1` | `1` run diarization, `0` transcription only |
| `SPEAKER_NAMES` | — | comma list mapping speaker index → name, e.g. `Alice,Bob,Carol` |
| `BEAM_SIZE` | `5` | decoding beam size |

### Per-stage device combinations

Transcription and diarization run on **separate, independently chosen** devices:

```bash
# full-GPU (fastest)
TRANSCRIBE_DEVICE=cuda DIARIZE_DEVICE=cuda \
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm transcribe

# transcribe on GPU + diarize on CPU (most robust; avoids any GPU-diarization edge cases)
TRANSCRIBE_DEVICE=cuda DIARIZE_DEVICE=cpu \
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm transcribe

# CPU-only
TRANSCRIBE_DEVICE=cpu DIARIZE_DEVICE=cpu docker compose run --rm transcribe
```

### Label speakers by name

Speaker IDs (`SPEAKER_00`, `SPEAKER_01`, …) are arbitrary. After a first run,
identify who is who from `output/<name>.txt`, then re-run with names:

```bash
docker compose ... run --rm -e SPEAKER_NAMES="Alice,Bob,Carol" transcribe
```

### Examples

```bash
# English, 2 speakers, faster model, named
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm \
  -e MODEL=large-v3-turbo -e LANGUAGE=en -e NUM_SPEAKERS=2 \
  -e SPEAKER_NAMES="Alice,Bob" transcribe

# transcription only, skip diarization
docker compose run --rm -e DIARIZE=0 transcribe
```

## Storage layout

| Host path | Container | Purpose |
|---|---|---|
| `./input/` | `/input` (read-only) | your audio/video file |
| `./output/` | `/output` | transcription results |
| `models` (named volume) | `/models` | cached Whisper + sherpa models |

The named `models` volume persists across runs, so models download once. To reset
it: `docker compose down -v`.

## Performance (RTX 3060 Ti, 52-min meeting)

| Stage | GPU (cuda) | CPU |
|---|---|---|
| Transcription (large-v3) | ~5 min | ~60+ min |
| Diarization (3 speakers) | ~9 min | ~16 min |

GPU transcription is a large win. GPU diarization is faster but clusters slightly
differently than CPU (fp differences); CPU diarization tends to keep similar
voices more separated. If diarization quality matters more than speed, use
`DIARIZE_DEVICE=cpu`.

## How it works / notes

- **Two engines, two processes.** faster-whisper (CTranslate2) and sherpa-onnx
  (ONNX Runtime) each own a CUDA context. Running both in one Python process
  corrupts memory (`std::length_error`). `pipeline.py` therefore spawns
  `diarize_worker.py` as a separate process for diarization, so both can safely
  use the GPU. Whisper GPU memory is freed before the diarization subprocess runs.
- **Base image.** `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04` provides CUDA
  12.8 + cuDNN matching the sherpa-onnx `cuda12.cudnn9` wheel. The same image
  works for CPU runs (just don't attach a GPU).
- **Pipeline.** ffmpeg extracts 16 kHz mono WAV → faster-whisper transcribes →
  sherpa-onnx segments + embeds + clusters speakers → speaker turns are mapped
  onto transcript segments by time overlap → outputs written.

## Files

```
Dockerfile              image definition
docker-compose.yml      base service (CPU-capable) + volumes + env
docker-compose.gpu.yml  GPU override (attaches --gpus all)
.env.example            documented defaults
pipeline.py             orchestrator / entrypoint
diarize_worker.py       standalone diarization (spawned as subprocess)
input/                  mount your media here
output/                 results appear here
```
