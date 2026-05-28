"""
Entry point: FastAPI app + Celery beat schedule.
Run API:    uvicorn main:app --host 0.0.0.0 --port 8000
Run worker: celery -A main.celery_app worker --loglevel=info
Run beat:   celery -A main.celery_app beat --loglevel=info
"""
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.settings import settings
from database.db import init_db
from database.models import MusicStyle, VideoStatus
from scheduler.tasks import app as celery_app, build_pipeline
from scheduler.pipeline import scheduler
from analytics.analyzer import analytics_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(settings.LOGS_DIR / "bot.log")),
    ],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("Database initialized")
    yield


app = FastAPI(
    title="CyberpunkYouTubeBot",
    description="Automated AI music video generation and YouTube publishing system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─── Celery Beat Schedule (cron jobs) ────────────────────────────────────────

celery_app.conf.beat_schedule = {
    "daily-video-batch": {
        "task": "tasks.daily_batch",
        "schedule": 86400,  # every 24 hours
        "options": {"expires": 3600},
    },
    "update-analytics": {
        "task": "tasks.update_analytics",
        "schedule": 21600,  # every 6 hours
        "options": {"expires": 1800},
    },
    "daily-snapshot": {
        "task": "tasks.daily_snapshot",
        "schedule": 86400,
        "options": {"expires": 3600},
    },
}


@celery_app.task(name="tasks.daily_batch")
def daily_batch_task():
    task_ids = scheduler.schedule_daily_batch()
    logger.info(f"Daily batch started: {len(task_ids)} jobs")
    return task_ids


@celery_app.task(name="tasks.update_analytics")
def update_analytics_task():
    updated = analytics_service.update_all_video_analytics()
    logger.info(f"Analytics updated for {updated} videos")
    return updated


@celery_app.task(name="tasks.daily_snapshot")
def daily_snapshot_task():
    analytics_service.save_daily_snapshot()
    logger.info("Daily channel snapshot saved")


# ─── API Routes ───────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    style: str
    duration_seconds: int = 3600
    publish_at: str | None = None


@app.post("/generate")
async def trigger_generate(req: GenerateRequest):
    """Manually trigger a video generation job."""
    try:
        style = MusicStyle(req.style)
    except ValueError:
        raise HTTPException(400, f"Invalid style. Choose from: {[s.value for s in MusicStyle]}")

    if req.duration_seconds not in [600, 1800, 3600, 7200]:
        raise HTTPException(400, "duration_seconds must be 600, 1800, 3600, or 7200")

    task_id = build_pipeline(style.value, req.duration_seconds, req.publish_at)
    return {"task_id": task_id, "style": style.value, "duration": req.duration_seconds}


@app.get("/jobs")
async def list_jobs(limit: int = 20, status: str | None = None):
    from database.db import get_db
    from database.models import VideoJob
    with get_db() as db:
        q = db.query(VideoJob).order_by(VideoJob.created_at.desc())
        if status:
            q = q.filter(VideoJob.status == VideoStatus(status))
        jobs = q.limit(limit).all()
        return [
            {
                "id": j.id,
                "uuid": j.uuid,
                "status": j.status.value,
                "style": j.style.value,
                "duration_seconds": j.duration_seconds,
                "created_at": j.created_at.isoformat() if j.created_at else None,
                "published_at": j.published_at.isoformat() if j.published_at else None,
            }
            for j in jobs
        ]


@app.get("/analytics/report")
async def get_report():
    return analytics_service.generate_performance_report()


@app.post("/analytics/update")
async def trigger_analytics_update(background_tasks: BackgroundTasks):
    background_tasks.add_task(analytics_service.update_all_video_analytics)
    return {"status": "analytics update queued"}


@app.post("/batch/now")
async def trigger_daily_batch():
    task_ids = scheduler.schedule_daily_batch()
    return {"jobs_started": len(task_ids), "task_ids": task_ids}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": settings.ENVIRONMENT,
    }
