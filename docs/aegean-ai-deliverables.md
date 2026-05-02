# Aegean AI Agent Context

Last updated: 2026-03-25

Canonical website source:
- `https://aegean.ai/sitemap.xml`

Canonical in-repo website inventory:
- `docs/aegean-ai-site-links.txt`

This file is the durable briefing for agents working in this repository. It is
meant to answer two questions quickly:

1. What aegean.ai surface area matters to this repo?
2. What is already delivered here versus what is still left in the backlog?

## Scope

This repository is the implementation home for the Foreign Whispers pipeline:

- download a source video and captions
- transcribe speech
- translate segments
- synthesize dubbed audio
- stitch and serve outputs
- expose the workflow through a web UI and API

The aegean.ai site already contains Foreign Whispers project pages, broader AI
course material, media, products, and research pages. The site inventory is
captured in `docs/aegean-ai-site-links.txt` so future agents can align repo work
to the public website without re-crawling the site every session.

## Website Inventory Snapshot

Snapshot date:
- 2026-03-25

Snapshot size:
- 265 URLs from `https://aegean.ai/sitemap.xml`

Largest sitemap groups:
- `aiml-common/lectures`: 127 URLs
- `products/applications`: 39 URLs
- `aiml-common/assignments`: 11 URLs
- `products/tech-demonstrators`: 8 URLs
- `aiml-common/resources`: 8 URLs
- `aiml-common/projects`: 7 URLs
- `courses/ai`: 6 URLs
- `courses/cv`: 6 URLs
- `courses/robotics`: 6 URLs

Highest-signal aegean.ai pages for this repo:
- `https://aegean.ai/about`
- `https://aegean.ai/products`
- `https://aegean.ai/courses`
- `https://aegean.ai/aiml-common/projects/nlp/foreign-whispers/download_integration/download_integration`
- `https://aegean.ai/aiml-common/projects/nlp/foreign-whispers/transcription_integration/transcription_integration`
- `https://aegean.ai/aiml-common/projects/nlp/foreign-whispers/translation_integration/translation_integration`
- `https://aegean.ai/aiml-common/projects/nlp/foreign-whispers/tts_integration/tts_integration`
- `https://aegean.ai/aiml-common/projects/nlp/foreign-whispers/diarization_integration/diarization_integration`
- `https://aegean.ai/aiml-common/projects/nlp/foreign-whispers/pipeline_end_to_end/pipeline_end_to_end`
- `https://aegean.ai/aiml-common/lectures/speech/text-to-speech-and-voice-cloning`

Refresh command:

```powershell
$xml = [xml](Invoke-WebRequest -Uri 'https://aegean.ai/sitemap.xml' -UseBasicParsing).Content
@($xml.urlset.url | ForEach-Object { $_.loc }) | Sort-Object
```

## Deliverables For This Repo

The deliverables below are the repo-level expectations that future agents
should preserve unless the user explicitly changes scope.

### 1. End-to-end dubbing pipeline

Expected outcome:
- a working pipeline that downloads, transcribes, translates, synthesizes, and
  stitches dubbed output

Primary repo surfaces:
- `api/src/`
- `foreign_whispers/`
- `notebooks/`
- `pipeline_data/`

### 2. API surface for orchestration and artifact serving

Expected outcome:
- stable HTTP endpoints for pipeline execution, health checks, video/audio
  serving, captions, and evaluation support

Primary repo surfaces:
- `api/src/main.py`
- `api/src/routers/`
- `api/src/services/`
- `api/src/schemas/`

### 3. User-facing studio frontend

Expected outcome:
- a frontend that can enumerate videos, drive the pipeline, and present outputs

Primary repo surfaces:
- `frontend/src/app/`
- `frontend/src/components/`
- `frontend/src/hooks/use-pipeline.ts`
- `frontend/src/lib/api.ts`

### 4. Local and containerized execution

Expected outcome:
- reproducible local setup and Docker Compose workflows for CPU and GPU use

Primary repo surfaces:
- `README.md`
- `docker-compose.yml`
- `Makefile`
- `pyproject.toml`

### 5. Evidence, evaluation, and maintainability

Expected outcome:
- tests, notebooks, and docs that make the implementation explainable and
  verifiable

Primary repo surfaces:
- `tests/`
- `docs/`
- `notebooks/`

### 6. Agent continuity

Expected outcome:
- all future agents can recover context quickly without inventing new trackers

Persistent sources of truth:
- this file for scope and deliverables
- `docs/aegean-ai-site-links.txt` for canonical site links
- `bd` for active work tracking

## Already Present In The Codebase

These are not future tasks. They are already implemented or scaffolded in the
current repository state and should be treated as existing deliverables.

- FastAPI app factory, lifespan hooks, `/healthz`, and `/api/videos` are
  present in `api/src/main.py`.
- Router modules already exist for download, transcribe, translate, TTS,
  stitch, diarization, and evaluation under `api/src/routers/`.
- A Next.js frontend already exists with `frontend/src/app/page.tsx` as the
  landing page and a studio-style component set under `frontend/src/components/`.
- Docker Compose already exists in `docker-compose.yml` with `nvidia` and `cpu`
  profiles plus API and frontend services.
- The Python package already exists in `foreign_whispers/` with alignment,
  diarization, evaluation, client, reranking, backends, VAD, and voice
  resolution modules.
- Notebook-based project evidence already exists under `notebooks/`, including
  integration folders that mirror the Foreign Whispers project pages on
  aegean.ai.
- Automated tests already exist under `tests/` and cover API routers, services,
  alignment, inference, evaluation, diarization, and path portability.

## Work Tracking Policy

Authoritative tracking lives in `bd`, not in markdown.

Use:
- `bd ready --json`
- `bd show <id>`
- `bd update <id> --claim`
- `bd close <id> --reason "Done"`

Do not create markdown TODO lists in this repo. If new work is discovered,
create or update a beads issue instead.

## Backlog Snapshot

Snapshot date:
- 2026-03-25

Tracked backup status from `.beads/backup/issues.jsonl`:
- 49 total issues
- 40 closed
- 2 in progress
- 7 open

Current in-progress work:
- `fw-tov`: Fix subtitle positioning and TTS temporal alignment in dubbed video
- `fw-lf7`: Implement `foreign_whispers` library

Current open work:
- `fw-b9s`: Add Logfire tracing spans to pipeline stages
- `fw-20b`: Evaluate LiveKit for real-time audio/video in the Foreign Whispers pipeline
- `fw-3h2`: Add voices section to sidebar with speaker preview audio player
- `fw-69b`: Add ElevenLabs-style transcript viewer with word-level source/target alignment
- `fw-9pw`: Add source/target language selectors to studio sidebar
- `fw-lua`: Wire pyannote diarization into the pipeline API and notebook
- `fw-z9e`: Evaluate whether local/remote backend abstraction in `config.py` is necessary

Representative work already done:
- `fw-r7s`: FastAPI app skeleton and project layout closed
- `fw-iy7`: Docker Compose for FastAPI deployment closed
- `fw-ttl`: ImageMagick dependency and startup handling closed
- `fw-2it`: translated WebVTT sidecar captions closed
- `fw-42s`: Streamlit UI replaced with Next.js + shadcn/ui closed
- `fw-4jc`: studio layout and dubbing method selector closed
- `fw-jhg`: Hugging Face Space deployment closed
- `fw-29a`: force-STT flag and bypass status closed

## Recommended Execution Order

If the user asks for roadmap guidance and does not override priorities, use this
order:

1. Finish the two active implementation threads:
   `fw-tov`, `fw-lf7`.
2. Complete observability and UI/product gaps that affect daily use:
   `fw-b9s`, `fw-3h2`, `fw-69b`, `fw-9pw`.
3. Complete speaker-aware pipeline work:
   `fw-lua`.
4. Resolve architecture and platform decisions before larger rewrites:
   `fw-z9e`, `fw-20b`.
5. After the above, refresh this file and the site-link snapshot if aegean.ai
   scope or repo deliverables have changed.

## Agent Rules

- Read this file before proposing scope or deliverables.
- Use `docs/aegean-ai-site-links.txt` when the user asks what aegean.ai pages
  exist.
- Treat this file as context, not as a task tracker.
- Put new work into `bd`, not into markdown checklists.
- Update this file only when aegean.ai scope, deliverables, or backlog framing
  materially changes.
