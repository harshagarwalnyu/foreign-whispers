# Foreign Whispers -- Aegean AI Course Specs

All specification documents from [aegean.ai](https://aegean.ai) for the Foreign Whispers pipeline project.

**Course:** Introduction to AI, Spring 2026
**Weight:** 20% of final grade
**Instructor:** Pantelis Monogioudis, Ph.D

---

## Table of Contents

1. [Pipeline End-to-End](#1-pipeline-end-to-end)
2. [Download Integration](#2-download-integration)
3. [Transcription Integration](#3-transcription-integration)
4. [Translation Integration](#4-translation-integration)
5. [TTS Integration](#5-tts-integration)
6. [Stitch Integration](#6-stitch-integration)
7. [Diarization Integration](#7-diarization-integration)
8. [Alignment Integration](#8-alignment-integration)

---

## 1. Pipeline End-to-End

**Source:** <https://aegean.ai/aiml-common/projects/nlp/foreign-whispers/pipeline_end_to_end/pipeline_end_to_end.md>
**Colab:** [Open In Colab](https://colab.research.google.com/github/aegean-ai/foreign-whispers/blob/main/notebooks/pipeline_end_to_end/pipeline_end_to_end.ipynb)

> Complete foreign whispers dubbing pipeline

Commercial dubbing services like [ElevenLabs](https://elevenlabs.io) can take a video, transcribe it, translate it, clone the speaker's voice, and return a dubbed video in the target language.

You are going to build the same thing from open-source components. No API keys to a proprietary service. No per-minute billing. The entire pipeline runs on your own GPU server.

**Where this matters:**

* **Media localization** -- dub documentaries, lectures, or interviews into multiple languages at scale
* **Accessibility** -- make video content available to non-English-speaking audiences without manual voiceover
* **Research** -- experiment with duration-aware translation, prosody alignment, and speaker-aware TTS in a controllable pipeline
* **Education** -- learn how production ML systems compose ASR, MT, TTS, and audio engineering into a single product

### Architecture

| Layer | What it is | Where it runs |
|-------|-----------|---------------|
| **GPU services** | Whisper STT (port 8000), Chatterbox TTS (port 8020) | Dedicated GPU containers |
| **API** | FastAPI orchestrator (port 8080) -- proxies to GPU services | CPU container |
| **`foreign_whispers` library** | Alignment logic, metrics, evaluation | Pure Python -- no GPU needed |

```
+--------------------+
|  API (CPU :8080)    |  orchestrates the pipeline
+--+---------+-------+
   | HTTP    | HTTP
   v         v
+--------+ +--------+
| STT    | | TTS    |   GPU containers
| :8000  | | :8020  |
+--------+ +--------+
```

### Pipeline Flow

```
YouTube URL -> Download -> Transcribe -> Translate -> TTS (+ alignment) -> Stitch -> Dubbed Video
```

### Production Tools

* **[FastAPI](https://fastapi.tiangolo.com/tutorial/first-steps/)** -- layered backend with Pydantic schemas, dependency injection, and async request handling
* **[Logfire](https://www.youtube.com/watch?v=on5RKukQzIg)** -- every pipeline step emits structured traces to Pydantic's observability platform
* **Docker Compose** -- four coordinated containers with GPU passthrough

### Per-Stage Integration Notebooks

| Notebook | Stage |
|----------|-------|
| `download_integration/` | YouTube download + caption fetching |
| `transcription_integration/` | Whisper vs YouTube captions |
| `diarization_integration/` | Speaker diarization (student assignment) |
| `translation_integration/` | argostranslate + duration-aware re-ranking |
| `alignment_integration/` | Temporal alignment: metrics, policies, global optimizer |
| `tts_integration/` | Chatterbox TTS + voice cloning |
| `stitch_integration/` | Final assembly + captions |

### Requirements

```bash
docker compose --profile nvidia up -d   # start GPU services + API
uv sync                                 # install library deps locally
```

### Pipeline Stages (SDK)

**P1 -- Download**
```python
VIDEO_URL = "https://www.youtube.com/watch?v=GYQ5yGV_-Oc"
with logfire.span("P1.download"):
    dl = fw.download(VIDEO_URL)
```

**P2 -- Transcribe**
```python
with logfire.span("P2.transcribe", video_id=video_id):
    transcript = fw.transcribe(video_id)
```

**P3 -- Translate**
```python
with logfire.span("P3.translate", video_id=video_id):
    translation = fw.translate(video_id)
```

**P4 -- TTS**
```python
with logfire.span("P4.tts", video_id=video_id):
    tts_result = fw.tts(video_id, alignment=True)
```

**P5 -- Stitch**
```python
with logfire.span("P5.stitch", video_id=video_id):
    stitch_result = fw.stitch(video_id)
```

### Pipeline Artifacts

| Step | Tool | Output |
|------|------|--------|
| P1 -- Download | `yt-dlp` via API | `videos/*.mp4`, `youtube_captions/*.txt` |
| P2 -- Transcribe | Whisper STT (GPU) | `transcriptions/whisper/*.json` |
| P3 -- Translate | `argostranslate` | `translations/argos/*.json` |
| P4 -- TTS | Chatterbox (GPU) | `tts_audio/chatterbox/{config}/*.wav` |
| P5 -- Stitch | `ffmpeg` | `dubbed_videos/{config}/*.mp4`, `dubbed_captions/*.vtt` |

All artifacts are cached in `pipeline_data/api/`. Re-running skips completed steps.

---

## 2. Download Integration

**Source:** <https://aegean.ai/aiml-common/projects/nlp/foreign-whispers/download_integration/download_integration.md>
**Colab:** [Open In Colab](https://colab.research.google.com/github/aegean-ai/foreign-whispers/blob/main/notebooks/download_integration/download_integration.ipynb)

> Downloading source videos and closed captions from YouTube

This notebook demonstrates the **Download** stage of the Foreign Whispers dubbing pipeline. It downloads a YouTube video and its closed captions via `yt-dlp` through the FastAPI backend.

**Prerequisites:**

* The Docker stack must be running (`docker compose --profile nvidia up -d`).
* The API should be accessible at `http://localhost:8080`.

### Setup

```python
from foreign_whispers.client import FWClient
fw = FWClient("http://localhost:8080")
fw.healthz()
```

### Download Video and Captions

The API wraps `yt-dlp` to download the video MP4 and extract any available closed captions. The `fw.download()` call returns a dict with `video_id`, `title`, and `caption_segments`.

```python
VIDEO_URL = "https://www.youtube.com/watch?v=GYQ5yGV_-Oc"
with logfire.span("download", video_url=VIDEO_URL):
    dl = fw.download(VIDEO_URL)

print(f"Video ID:       {dl['video_id']}")
print(f"Title:          {dl['title']}")
print(f"Caption count:  {len(dl['caption_segments'])}")
```

### Artifacts

* `pipeline_data/api/videos/` -- source MP4 files
* `pipeline_data/api/youtube_captions/` -- extracted caption JSON files

---

## 3. Transcription Integration

**Source:** <https://aegean.ai/aiml-common/projects/nlp/foreign-whispers/transcription_integration/transcription_integration.md>
**Colab:** [Open In Colab](https://colab.research.google.com/github/aegean-ai/foreign-whispers/blob/main/notebooks/transcription_integration/transcription_integration.ipynb)

> Speech-to-text transcription using Whisper

The transcription endpoint supports a `use_youtube_captions` flag:

- **`use_youtube_captions=True`** (default): YouTube captions are preferred when available, skipping Whisper entirely. Faster since no GPU inference is needed. Falls back to Whisper if YouTube captions are not available.
- **`use_youtube_captions=False`**: Whisper STT always runs on the video's audio track, regardless of whether YouTube captions exist. Produces more accurate timestamps and can provide word-level detail.

The transcription result is cached as JSON in `pipeline_data/api/transcriptions/whisper/`. Subsequent calls with `use_youtube_captions=True` return the cached result with `skipped=True`.

### When to Force Whisper Over YouTube Captions

- **More accurate timestamps**: Whisper uses acoustic features to precisely locate speech boundaries, while YouTube captions often have coarser timing.
- **Word-level detail**: Whisper can produce word-level timestamps, useful for precise alignment in the TTS stage.
- **Consistency**: YouTube captions vary in quality across videos; Whisper provides a uniform baseline.

### Segment JSON Format

```json
{
  "id": 0,
  "start": 0.0,
  "end": 3.5,
  "text": "Hello world"
}
```

### Artifacts

* `pipeline_data/api/transcriptions/whisper/*.json` -- Whisper-format segment JSON

---

## 4. Translation Integration

**Source:** <https://aegean.ai/aiml-common/projects/nlp/foreign-whispers/translation_integration/translation_integration.md>
**Colab:** [Open In Colab](https://colab.research.google.com/github/aegean-ai/foreign-whispers/blob/main/notebooks/translation_integration/translation_integration.ipynb)

> Translating transcribed text to target language

Translate transcription segments from source to target language using **argostranslate** (offline, OpenNMT-based).

Key limitation: argostranslate has **no duration budget** -- the translation length is unconstrained. Romance languages (Spanish, French, Italian) typically produce longer text than the English source, which creates timing challenges for the downstream TTS stage.

### Student Assignment: Duration-Aware Re-ranking (P8)

The function `get_shorter_translations()` in `foreign_whispers/reranking.py` is a **stub** that currently returns an empty list. Students implement it to produce shorter target-language translations that fit within a TTS duration budget.

**Specification:**

* Input/output contract defined in docstring
* Duration heuristic: ~15 chars/second for Romance languages
* Suggested approaches: rule-based, multi-backend, LLM, hybrid

```python
from foreign_whispers import get_shorter_translations, TranslationCandidate

candidates = get_shorter_translations(
    source_text=source_text,
    baseline_es=baseline_es,
    target_duration_s=target_duration_s,
)
```

### Artifacts

* `pipeline_data/api/translations/argos/*.json` -- translated segment JSON (same format as transcription, with `text` replaced by translated text)

---

## 5. TTS Integration

**Source:** <https://aegean.ai/aiml-common/projects/nlp/foreign-whispers/tts_integration/tts_integration.md>
**Colab:** [Open In Colab](https://colab.research.google.com/github/aegean-ai/foreign-whispers/blob/main/notebooks/tts_integration/tts_integration.ipynb)

> Text-to-speech synthesis using Chatterbox TTS

The Chatterbox container supports voice cloning -- it accepts a reference audio file via the `/v1/audio/speech/upload` endpoint for voice matching. However, the Foreign Whispers pipeline currently uses `default.wav` and never exposes speaker selection through the API.

### Current State

| File | Current state |
|------|---------------|
| `tts.py` -> `ChatterboxClient` | Supports `speaker_wav` kwarg per call |
| `CHATTERBOX_SPEAKER_WAV` env var | Defaults to empty string |
| `api/src/routers/tts.py` | No speaker parameter |
| `api/src/services/tts_service.py` | No speaker passthrough |
| `pipeline_data/speakers/{lang}/` | Reference WAVs exist but unused |
| Docker volume mount | `./pipeline_data/speakers:/app/voices` mounted |

### Task 2: Voice Resolution Function (TDD)

**File to create:** `foreign_whispers/voice_resolution.py`

**Resolution order:**

1. If speaker-specific WAV exists: `speakers/{lang}/{speaker_id}.wav`
2. If language default exists: `speakers/{lang}/default.wav`
3. Fall back to global: `speakers/default.wav`

Return value is a **relative path** (e.g. `"es/SPEAKER_00.wav"`) -- this is what the Chatterbox container expects relative to `/app/voices/`.

### Task 3: API Speaker Parameter

**Files to modify:**

- `api/src/core/config.py` -- add `speakers_dir` property
- `api/src/routers/tts.py` -- add `speaker_wav` query parameter
- `api/src/services/tts_service.py` -- pass `speaker_wav` through to `tts.py`

```python
@property
def speakers_dir(self) -> Path:
    return self.data_dir / "speakers"
```

### Task 4: Per-Speaker Voice Assignment

When transcription segments have `speaker` labels (from diarization), automatically assign different reference voices to different speakers.

**Approach:** Pass `voice_map` as a dict to `text_file_to_speech`. Inside the function, for each segment, look up `voice_map[segment["speaker"]]` and pass it as `speaker_wav` to `tts_to_file()`.

### Evaluation Criteria

| # | Criterion | How to verify |
|---|-----------|---------------|
| 1 | Tests pass | Re-run voice resolution tests -- all 5 green |
| 2 | API accepts `speaker_wav` | `POST /api/tts/{video_id}?speaker_wav=es/default.wav` works |
| 3 | Auto-resolution works | Omitting `speaker_wav` selects language default automatically |
| 4 | Per-speaker mapping | With diarized segments, different speakers get different voices |
| 5 | Fallback chain | Unknown speaker/language falls back to `default.wav` |
| 6 | Code quality | Follows existing patterns (query params, service layer, config properties) |

---

## 6. Stitch Integration

**Source:** <https://aegean.ai/aiml-common/projects/nlp/foreign-whispers/stitch_integration/stitch_integration.md>
**Colab:** [Open In Colab](https://colab.research.google.com/github/aegean-ai/foreign-whispers/blob/main/notebooks/stitch_integration/stitch_integration.ipynb)

> Stitching dubbed audio segments back into the video

This notebook demonstrates the **Stitch** stage of the Foreign Whispers dubbing pipeline. It performs final video assembly: combining the original video with dubbed TTS audio and rolling two-line translated captions via ffmpeg. The stitch uses audio-only remux (no re-encoding), preserving original video quality.

**Prerequisites:**

* Prior pipeline stages (download, transcribe, translate, TTS) must have completed for the target video.

### Stitch Video

```python
video_id = "GYQ5yGV_-Oc"
with logfire.span("stitch", video_id=video_id):
    result = fw.stitch(video_id)

print(f"Video ID:   {result['video_id']}")
print(f"Video path: {result['video_path']}")
print(f"Config:     {result['config']}")
```

### Output Artifacts

* `pipeline_data/api/dubbed_videos/{config}/` -- final dubbed MP4 files
* `pipeline_data/api/dubbed_captions/` -- target-language VTT caption files

### Captions

The stitch stage generates VTT captions in a rolling two-line format: the current translated line appears on top, and the previous line is shown on the bottom, giving viewers context continuity.

### Playback

1. **Frontend:** Open `http://localhost:8501` and select the video from the list. The UI will load the dubbed MP4 with captions overlay.
2. **Direct file:** Play the MP4 directly from `pipeline_data/api/dubbed_videos/{config}/{video_id}.mp4` using any media player (e.g., VLC, mpv). Load the corresponding VTT file from `pipeline_data/api/dubbed_captions/` as an external subtitle track.

---

## 7. Diarization Integration

**Source:** <https://aegean.ai/aiml-common/projects/nlp/foreign-whispers/diarization_integration/diarization_integration.md>
**Colab:** [Open In Colab](https://colab.research.google.com/github/aegean-ai/foreign-whispers/blob/main/notebooks/diarization_integration/diarization_integration.ipynb)

> Speaker diarization for multi-speaker video dubbing

### Pipeline Flow

```
CURRENT:  Download -> Transcribe -> Translate -> TTS -> Stitch
TARGET:   Download -> Transcribe -> Diarize -> Translate -> TTS (per-speaker) -> Stitch
                                    ^
                              YOUR WORK HERE
```

### Prerequisites

* `FW_HF_TOKEN` is set in your `.env` or environment
* You have accepted the [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) model license on HuggingFace
* A multi-speaker test video is in `video_registry.yml` and downloaded

### Provided Code

| File | What it does |
|------|-------------|
| `foreign_whispers/diarization.py` | `diarize_audio(audio_path, hf_token)` -- calls pyannote, returns `[{start_s, end_s, speaker}]` |
| `api/src/services/alignment_service.py` | `AlignmentService.diarize()` -- service wrapper |
| `api/src/core/config.py` | `Settings.hf_token` -- reads `FW_HF_TOKEN` env var |
| `pipeline_data/speakers/` | Per-language directories with reference WAV files for TTS voice cloning |

### Task 1: Segment-Speaker Merge Function

**Goal:** Write a pure function that assigns a speaker label to each transcription segment based on diarization output.

**Algorithm:** For each transcription segment, find which diarization speaker has the most temporal overlap with that segment's `[start, end]` range.

**File to modify:** `foreign_whispers/diarization.py`

**Hints:**

1. Create a **copy** of each segment dict (don't mutate the input)
2. For each segment, compute overlap with every diarization interval:
   ```python
   overlap = max(0, min(seg_end, diar_end) - max(seg_start, diar_start))
   ```
3. Pick the diarization interval with the **largest overlap**
4. If no overlap or diarization is empty, default to `"SPEAKER_00"`

**Tests (TDD):**

```python
def run_tests():
    from foreign_whispers.diarization import assign_speakers

    # Test 1: Single speaker
    # Test 2: Two speakers
    # Test 3: Empty diarization defaults to SPEAKER_00
    # Test 4: Does not mutate input
```

### Task 2: Diarize API Endpoint

**Goal:** Create `POST /api/diarize/{video_id}` that extracts audio, runs diarization, and returns speaker segments.

**Files to create:**

* `api/src/schemas/diarize.py`
* `api/src/routers/diarize.py`

**Files to modify:**

* `api/src/main.py` (register the router)
* `api/src/core/config.py` (add `diarizations_dir` property)

**Schema:**

```python
class DiarizeSpeakerSegment(BaseModel):
    start_s: float
    end_s: float
    speaker: str

class DiarizeResponse(BaseModel):
    video_id: str
    speakers: list[str]
    segments: list[DiarizeSpeakerSegment]
    skipped: bool = False
```

**Endpoint steps:**

1. Extract audio from video via ffmpeg: `ffmpeg -i <video_path> -vn -acodec pcm_s16le -ar 16000 -y <audio_path>`
2. Run diarization: `_alignment_service.diarize(str(audio_path))`
3. Extract unique speakers: `sorted(set(s["speaker"] for s in diar_segments))`
4. Cache result as JSON
5. Return `DiarizeResponse`

### Task 3: Merge Speaker Labels Into Transcription

**Goal:** After diarization runs, update the transcription JSON so each segment has a `speaker` field.

**File to modify:** `api/src/routers/diarize.py`

```python
from foreign_whispers.diarization import assign_speakers

transcript_path = settings.transcriptions_dir / f"{title}.json"
if transcript_path.exists():
    transcript = json.loads(transcript_path.read_text())
    labeled_segments = assign_speakers(transcript.get("segments", []), diar_segments)
    transcript["segments"] = labeled_segments
    transcript_path.write_text(json.dumps(transcript))
```

### Task 4: Frontend Pipeline Integration

**Files to modify:**

* `frontend/src/lib/api.ts` -- add `diarizeVideo` function
* `frontend/src/lib/types.ts` -- add `"diarize"` to `PipelineStage`
* `frontend/src/hooks/use-pipeline.ts` -- call diarize between transcribe and translate
* `frontend/src/components/pipeline-table.tsx` -- add diarize row
* `frontend/src/components/pipeline-status-bar.tsx` -- add status message

### Task 5: Per-Speaker TTS Voice Selection

**Goal:** When speaker labels exist in the translated segments, use a different Chatterbox reference voice per speaker.

**Files to modify:**

* `api/src/routers/tts.py`
* `api/src/services/tts_service.py`

**Approach:**

1. Read the translated JSON -- each segment now has a `speaker` field
2. Map each unique speaker to a reference WAV from `pipeline_data/speakers/{lang}/`
3. Pass the speaker-to-voice mapping to the TTS engine so it switches voices per segment

### Evaluation Criteria

| # | Criterion | How to verify |
|---|-----------|---------------|
| 1 | Tests pass | Re-run the test cell in Task 1.4 -- all 4 green |
| 2 | API works | `POST /api/diarize/{video_id}` returns speaker segments |
| 3 | Merge works | Transcription JSON has `speaker` fields after diarization |
| 4 | Frontend works | Diarize stage appears in pipeline table when enabled |
| 5 | Caching works | Second API call returns `skipped: true` |
| 6 | Code quality | Follows existing patterns (file-exists cache, service layer, Pydantic schemas) |

---

## 8. Alignment Integration

**Source:** <https://aegean.ai/aiml-common/projects/nlp/foreign-whispers/alignment_integration/alignment_integration.md>
**Colab:** [Open In Colab](https://colab.research.google.com/github/aegean-ai/foreign-whispers/blob/main/notebooks/alignment_integration/alignment_integration.ipynb)

> Temporal alignment between source speech and target-language TTS audio

Covers segment metrics, fallback policy, and global timeline optimization. Loads from `pipeline_data/` (no API call needed).

### Fallback Policy

| Stretch Factor | Action | Description |
|---------------|--------|-------------|
| <= 1.1 | ACCEPT | Fits naturally, no change needed |
| 1.1 - 1.4 | TIME_STRETCH | Apply pyrubberband time-stretch |
| 1.4 - 1.8 | SHIFT_INTO_GAP | Borrow from adjacent silence gap |
| 1.8 - 2.5 | REQUEST_SHORTER | Request a shorter translation |
| > 2.5 | FAIL | Unfixable, fall back to silence |

### Task 1: Duration Prediction

**Goal:** Replace the ~15 chars/second heuristic with a better duration predictor.

**Approach:**

- Collect ground-truth durations by running TTS on a sample of segments and measuring actual WAV duration
- Compare predictors: character count, syllable count (use a Spanish syllabifier), and a simple regression model trained on (text features -> actual TTS duration)
- Plug your predictor into `compute_segment_metrics` by modifying the `predicted_tts_duration_s` calculation in `foreign_whispers/alignment.py`

**Evaluation:**

- Mean absolute duration error (predicted vs actual TTS output)
- Calibration: does the predictor work equally well for short and long utterances?
- Downstream: does the improved predictor change the action distribution?

### Task 2: Translation Re-ranking

**Goal:** For segments that exceed the timing budget, generate shorter translation candidates and pick the one that best fits the source window while preserving meaning.

**Approach:**

- Filter for segments where `decide_action(m)` returns `REQUEST_SHORTER`
- For each, generate 2-3 shorter Spanish alternatives
- Score candidates by: `(predicted_duration - target_duration)^2 + lambda * semantic_distance`
- Implement in `foreign_whispers/reranking.py` -- the `get_shorter_translations()` stub

**Evaluation:**

- How many REQUEST_SHORTER segments can you bring down to ACCEPT or TIME_STRETCH?
- Semantic preservation: compare original and shortened translations using embedding cosine similarity

### Task 3: Global Optimizer

**Goal:** Implement a better global optimizer and compare it against the greedy baseline.

The greedy left-to-right `global_align()` makes locally optimal decisions but can't look ahead.

**Approach (pick one):**

- **Dynamic programming:** Minimize cumulative drift over all segments with gap-borrowing constraints
- **Integer linear programming:** Formulate as an optimization problem with timing and non-overlap constraints
- **Beam search:** Explore multiple alignment paths and pick the best

**Evaluation:**

- Total cumulative drift (lower is better)
- Number of segments requiring severe stretch (>1.4x)
- Number of segments that overlap in the scheduled timeline

### Task 4: Quality Scorecard

**Goal:** Design and implement a richer evaluation framework that scores clips across multiple dimensions.

**Dimensions to consider:**

- **Timing accuracy:** mean absolute duration error, percentage of severe stretches, cumulative drift
- **Intelligibility:** TTS the Spanish, then STT it back -- compare against the translation. Word error rate of the round-trip measures intelligibility.
- **Semantic fidelity:** compare source English and back-translated English using embedding cosine similarity
- **Naturalness:** speaking rate variance across segments -- is it consistent or does it jump between fast and slow?

**Implementation:** `dubbing_scorecard(metrics, aligned_segments, align_report)` in `foreign_whispers/evaluation.py`

**Evaluation:**

- Does your scorecard distinguish between good and bad clips?
- Do the dimensions correlate with each other, or do they capture independent quality aspects?
- Run on multiple videos from `video_registry.yml` and compare

### Summary Table

| # | What you build | File to modify | Evaluation |
|---|---------------|----------------|------------|
| 1 | Duration Prediction | `alignment.py` | Mean absolute error vs ground truth |
| 2 | Translation Re-ranking | `reranking.py` -- `get_shorter_translations` | Segments moved from REQUEST_SHORTER to ACCEPT |
| 3 | Global Optimizer | `alignment.py` -- new `global_align_dp` | Total drift, severe stretch count |
| 4 | Quality Scorecard | `evaluation.py` -- new `dubbing_scorecard` | Dimension independence, cross-clip consistency |

---

## Quick Reference: All Links

| Stage | Spec URL |
|-------|----------|
| Pipeline E2E | <https://aegean.ai/aiml-common/projects/nlp/foreign-whispers/pipeline_end_to_end/pipeline_end_to_end.md> |
| Download | <https://aegean.ai/aiml-common/projects/nlp/foreign-whispers/download_integration/download_integration.md> |
| Transcription | <https://aegean.ai/aiml-common/projects/nlp/foreign-whispers/transcription_integration/transcription_integration.md> |
| Translation | <https://aegean.ai/aiml-common/projects/nlp/foreign-whispers/translation_integration/translation_integration.md> |
| TTS | <https://aegean.ai/aiml-common/projects/nlp/foreign-whispers/tts_integration/tts_integration.md> |
| Stitch | <https://aegean.ai/aiml-common/projects/nlp/foreign-whispers/stitch_integration/stitch_integration.md> |
| Diarization | <https://aegean.ai/aiml-common/projects/nlp/foreign-whispers/diarization_integration/diarization_integration.md> |
| Alignment | <https://aegean.ai/aiml-common/projects/nlp/foreign-whispers/alignment_integration/alignment_integration.md> |
