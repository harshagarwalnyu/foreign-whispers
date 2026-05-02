"""Foreign Whispers FastAPI application."""

import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.src.core.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lazy model loading — models are loaded on first use, not at startup.

    This avoids blocking startup in Docker Compose where Whisper and TTS
    inference may be handled by separate containers (speaches, Chatterbox).
    """
    # Ensure pipeline data directories exist
    for d in (
        settings.data_dir,
        settings.videos_dir,
        settings.youtube_captions_dir,
        settings.transcriptions_dir,
        settings.translations_dir,
        settings.tts_audio_dir,
        settings.dubbed_videos_dir,
        settings.dubbed_captions_dir,
        settings.diarizations_dir,
        settings.speakers_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)

    # Expose public state attributes for routers/tests while keeping lazy loads.
    app.state.whisper_model = SimpleNamespace()
    app.state.tts_model = SimpleNamespace()
    app.state._whisper_model = None
    app.state._tts_model = None
    logger.info("Application ready (models will load on first use).")

    # Configure Logfire if a write token is available
    if settings.logfire_write_token:
        try:
            import logfire
            logfire.configure(
                write_token=settings.logfire_write_token,
                service_name="foreign-whispers",
            )
            logfire.instrument_fastapi(app)
            logger.info("Logfire tracing enabled.")
        except ImportError:
            logger.info("Logfire not installed — tracing disabled.")

    yield

    # Cleanup
    if hasattr(app.state, "whisper_model"):
        del app.state.whisper_model
    if hasattr(app.state, "tts_model"):
        del app.state.tts_model
    if hasattr(app.state, "_whisper_model"):
        del app.state._whisper_model
    if hasattr(app.state, "_tts_model"):
        del app.state._tts_model
    logger.info("Models unloaded.")


def get_whisper_model(app):
    """Lazy-load Whisper model on first use."""
    if app.state._whisper_model is not None:
        return app.state._whisper_model

    if vars(app.state.whisper_model):
        app.state._whisper_model = app.state.whisper_model
        return app.state.whisper_model

    if app.state.whisper_model is None:
        app.state.whisper_model = SimpleNamespace()

    if not vars(app.state.whisper_model):
        logger.info("Loading Whisper model (%s)...", settings.whisper_model)
        import whisper
        model = whisper.load_model(settings.whisper_model)
        app.state.whisper_model = model
        app.state._whisper_model = model
        logger.info("Whisper model loaded.")
    return app.state.whisper_model


def get_tts_model(app):
    """Lazy-load TTS model on first use."""
    if app.state._tts_model is not None:
        return app.state._tts_model

    if vars(app.state.tts_model):
        app.state._tts_model = app.state.tts_model
        return app.state.tts_model

    if app.state.tts_model is None:
        app.state.tts_model = SimpleNamespace()

    if not vars(app.state.tts_model):
        logger.info("Loading TTS model (%s)...", settings.tts_model_name)
        from TTS.api import TTS
        model = TTS(model_name=settings.tts_model_name, progress_bar=False)
        app.state.tts_model = model
        app.state._tts_model = model
        logger.info("TTS model loaded.")
    return app.state.tts_model


def create_app() -> FastAPI:
    """Application factory — creates and configures the FastAPI instance."""
    app = FastAPI(
        title=settings.app_title,
        lifespan=lifespan,
    )

    if settings.cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    from api.src.routers.download import router as download_router
    from api.src.routers.transcribe import router as transcribe_router
    from api.src.routers.translate import router as translate_router
    from api.src.routers.tts import router as tts_router
    from api.src.routers.stitch import router as stitch_router

    app.include_router(download_router)
    app.include_router(transcribe_router)
    app.include_router(translate_router)
    app.include_router(tts_router)
    app.include_router(stitch_router)
    from api.src.routers.eval import router as eval_router
    app.include_router(eval_router)
    from api.src.routers.diarize import router as diarize_router
    app.include_router(diarize_router)

    @app.get("/healthz")
    async def healthz():
        """Health check endpoint."""
        return {"status": "ok"}

    @app.get("/api/videos")
    async def list_videos():
        """Return the video catalog from video_registry.yml."""
        from api.src.core.video_registry import get_all_videos
        return [
            {"id": v.id, "title": v.title, "url": v.url}
            for v in get_all_videos()
        ]

    return app


app = create_app()
