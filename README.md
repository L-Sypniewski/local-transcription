# whisper-diarize

Transcribe audio/video and label speakers, fully offline, in Docker. Powered by
**faster-whisper** (Whisper large-v3 on GPU) for transcription and one of three
diarization backends:

- **`pyannote4`** (default) — pyannote.audio **4.x** `speaker-diarization-community-1` on **CPU**. Best
  quality (better speaker counting/assignment) but slow (~0.5–1× realtime). Runs on CPU
  because its reconstruction step needs ~9.5 GB (OOMs on <12 GB GPUs — pyannote issue #1963).
- **`pyannote`** — pyannote.audio **3.x** `speaker-diarization-3.1` on GPU. Fast, fits an 8 GB
  card (sliding-window model). Good everyday choice when speed matters.
- **`sherpa`** — sherpa-onnx. Token-free fallback.

Everything runs locally — no cloud. Pick any Whisper model, run on GPU or CPU
(independently per stage), and keep models cached in a volume so nothing re-downloads.

> **Gated models.** Both pyannote backends need a free `HF_TOKEN` (inference stays 100%
> local). `pyannote4` additionally requires accepting terms at
> `pyannote/speaker-diarization-community-1`. `sherpa` needs no token.

## Quick start

```bash
# 1. drop your file into input/
cp /path/to/meeting.mp4 input/

# 2. run (CPU-only — works on any machine)
docker compose run --rm transcribe

# 2b. or run on GPU (needs nvidia-container-toolkit, see below)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm transcribe
```

For the default pyannote4 (or pyannote), first put a free HuggingFace token in `.env`:
```bash
# accept terms, then create a read token:
#   https://hf.co/pyannote/speaker-diarization-3.1
#   https://hf.co/pyannote/segmentation-3.0
#   https://hf.co/pyannote/speaker-diarization-community-1   (needed for the pyannote4 default)
#   https://huggingface.co/settings/tokens
echo 'HF_TOKEN=hf_xxx' >> .env
```
Switch diarizer with `DIARIZER`: `-e DIARIZER=pyannote` (3.x, GPU, fast) or
`-e DIARIZER=sherpa` (no token).

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
| `DIARIZER` | `pyannote4` | `pyannote4` (4.x community-1, CPU, best quality) · `pyannote` (3.x, GPU, fast) · `sherpa` (token-free) |
| `HF_TOKEN` | — | free HuggingFace token; required for pyannote & pyannote4, ignored by sherpa |
| `SPEAKER_NAMES` | — | comma list mapping speaker index → name, e.g. `Alice,Bob,Carol` |
| `BEAM_SIZE` | `5` | decoding beam size |

> Keep comments in `.env` on their **own lines** — Docker Compose does not strip
> inline comments after an empty value, so `KEY=  # comment` breaks parsing.

### Per-stage device combinations

Transcription and diarization run on **separate, independently chosen** devices:

```bash
# full-GPU (fastest)
TRANSCRIBE_DEVICE=cuda DIARIZE_DEVICE=cuda \
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm transcribe

# transcribe on GPU + diarize on CPU
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

## Performance (RTX 3060 Ti 8GB, 25-min meeting)

| Stage | GPU (cuda) | CPU |
|---|---|---|
| Transcription (large-v3) | ~3 min | ~60+ min |
| Diarization — pyannote 3.x (3 speakers) | ~45s | ~6 min |
| Diarization — pyannote4 community-1 (3 speakers) | n/a (OOMs <12GB) | ~10–25 min |
| Diarization — sherpa (3 speakers) | ~4 min | ~7 min |

GPU transcription is a large win. The default `pyannote4` (community-1) gives the best
diarization quality but runs on CPU (its reconstruction step needs ~9.5 GB, OOMing on
<12 GB GPUs — pyannote issue #1963), so it's ~0.5–1× realtime. `pyannote` (3.x) on GPU is
the fast alternative (~45 s); `sherpa` is the token-free fallback.

## Diarization backend comparison

All three backends were compared on the same 25-min, 3-speaker Polish meeting. Speakers
are matched between engines by maximum co-speech time, then agreement is the fraction of
co-speech time assigned to the matched speaker (a DER-complement). Reproduce with
`cmp/compare.py`.

**Pairwise agreement** (who's-speaking agreement, after optimal speaker matching):

| | sherpa ↔ pyannote 3 | sherpa ↔ pyannote4 | pyannote 3 ↔ pyannote4 |
|---|---|---|---|
| 25-min meeting | 94.9% | 86.1% | 87.0% |
| 3.5-min clip | 90.8% | 89.2% | 98.5% |

**Speaking time per speaker** (seconds, each engine ranked most→least talkative), 25-min meeting:

| Rank | sherpa | pyannote 3 | pyannote 4 |
|---|---|---|---|
| #1 (most) | 767 | 741 | 729 |
| #2 | 519 | 522 | **412** |
| #3 (least) | **21** | **17** | **139** |

**What this shows:**

- **sherpa ≈ pyannote 3.x** — ~95% agreement and near-identical speaker-time splits; they
  agree even down to the speaker indices (0↔0, 1↔1, 2↔2). pyannote 3.x is marginally more
  granular. Either is a fine fast option; pyannote 3.x is faster on GPU.
- **pyannote4 (community-1) is the quality outlier.** It identifies a substantial 3rd speaker
  (412 s, rank #2) that sherpa and pyannote 3.x almost entirely drop (~17–21 s). In other
  words, the older engines merged/mis-assigned a real participant; community-1's improved
  speaker counting/assignment recovered them. On the short clip, where the 3rd speaker is
  genuinely marginal, all three agree (~98%).
- **Recommendation:** use `pyannote4` when quality matters (it's the default). Use `pyannote`
  for quick everyday runs. Disagreements concentrate on short backchannels and overlapped
  speech — exactly where diarization is hard — so a brief listen around disputed regions is
  the final arbiter.

## How it works / notes

- **Three processes, three engines.** The orchestrator (`pipeline.py`) does no GPU work:
  it extracts audio with ffmpeg, then spawns **two subprocesses** — transcription then
  diarization — and merges their JSON results. faster-whisper (CTranslate2), pyannote
  (torch) and sherpa-onnx (ONNX Runtime) each own a CUDA context and corrupt each other
  if run in one process, so each runs alone.
- **Why transcription is its own process.** CTranslate2 keeps ~3.9 GB of VRAM reserved
  even after the model is deleted and `gc.collect()` — it only hands memory back when the
  process exits. Running transcription in a subprocess means its VRAM is fully released
  on exit, so the diarization subprocess that follows gets the **entire GPU**.
- **Two pyannote venvs.** pyannote.audio 3.x and 4.x can't share a Python environment
  (same package; 3.x needs `torch<2.6` + `huggingface_hub<1.0`, 4.x needs `torch>=2.8` +
  `huggingface_hub>=1.x`). They live in `/opt/p3` (3.x, GPU torch) and `/opt/p4` (4.x, CPU
  torch); the orchestrator spawns the matching interpreter. 4.x runs on CPU because its
  reconstruction step allocates ~9.5 GB (issue #1963, unfixed in 4.0.7).
- **Base image.** `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04`. `/opt/p3` uses torch
  2.5.1 + CUDA 12.4 (torch 2.6 broke 3.x checkpoint loading via `weights_only=True`).
  `/opt/p4` uses CPU-only torch (4.x is forced to CPU, so the CUDA wheel would be dead
  weight). CUDA 12.4 torch runs fine on the 12.8 base + your host driver.
- **Pipeline.** ffmpeg extracts 16 kHz mono WAV → faster-whisper transcribes → pyannote
  segments + embeds + clusters speakers → speaker turns are mapped onto transcript
  segments by time overlap → outputs written.
- **Auth.** pyannote models are gated; pass a free `HF_TOKEN` (read from `.env`/env).
  sherpa needs no token.

## Files

```
Dockerfile              image definition (base + /opt/p3 + /opt/p4 venvs)
docker-compose.yml      base service (CPU-capable) + volumes + env
docker-compose.gpu.yml  GPU override (attaches --gpus all)
.env.example            documented defaults
pipeline.py             orchestrator / entrypoint (no GPU work)
transcribe_worker.py    faster-whisper transcription (subprocess)
diarize_pyannote.py     pyannote.audio 3.x diarization (subprocess, /opt/p3, GPU, default)
diarize_pyannote4.py    pyannote.audio 4.x community-1 diarization (subprocess, /opt/p4, CPU)
diarize_worker.py       sherpa-onnx diarization (subprocess, fallback)
cmp/compare.py          3-way diarization comparison utility (see "backend comparison")
input/                  mount your media here
output/                 results appear here
```
