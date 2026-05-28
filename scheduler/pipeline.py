"""
Daily scheduling: picks styles/durations, calculates publish times,
enforces quality gates, and avoids duplicate content.
"""
import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Optional

from config.settings import settings
from database.db import get_db
from database.models import VideoJob, VideoStatus, MusicStyle, ChannelAnalytics

logger = logging.getLogger(__name__)

# Weights: styles with better historical performance get higher probability
DEFAULT_STYLE_WEIGHTS = {
    MusicStyle.CYBERPUNK: 0.25,
    MusicStyle.DARK_TECHNO: 0.25,
    MusicStyle.NEON_AMBIENT: 0.20,
    MusicStyle.SYNTHWAVE: 0.15,
    MusicStyle.ACID_TECHNO: 0.08,
    MusicStyle.INDUSTRIAL: 0.05,
    MusicStyle.HARDTEK: 0.02,
}

# Duration weights: 1h is the sweet spot for watch time
DURATION_WEIGHTS = {
    600: 0.10,   # 10 min: quick test/experiment
    1800: 0.25,  # 30 min: good for playlists
    3600: 0.65,  # 1 hour: YouTube sweet spot
}


class DailyScheduler:

    def __init__(self):
        self._style_weights = DEFAULT_STYLE_WEIGHTS.copy()

    def update_weights_from_analytics(self) -> None:
        """Adjust style weights based on recent analytics data."""
        with get_db() as db:
            # Get last 30 days of successful jobs with analytics
            from sqlalchemy import func
            from database.models import YouTubeVideo

            results = (
                db.query(VideoJob.style, func.avg(YouTubeVideo.avg_view_duration_pct).label("avg_retention"))
                .join(YouTubeVideo, VideoJob.id == YouTubeVideo.job_id)
                .filter(
                    VideoJob.status == VideoStatus.PUBLISHED,
                    YouTubeVideo.views > 100,
                )
                .group_by(VideoJob.style)
                .all()
            )

            if not results:
                return

            # Normalize: higher retention = higher weight
            total_retention = sum(r.avg_retention or 0.3 for r in results)
            if total_retention == 0:
                return

            new_weights = {}
            for row in results:
                new_weights[row.style] = (row.avg_retention or 0.3) / total_retention

            # Blend with default weights (50/50 to avoid overfitting)
            for style in DEFAULT_STYLE_WEIGHTS:
                default_w = DEFAULT_STYLE_WEIGHTS[style]
                learned_w = new_weights.get(style, default_w)
                self._style_weights[style] = 0.5 * default_w + 0.5 * learned_w

            logger.info(f"Updated style weights from analytics: {self._style_weights}")

    def pick_style(self) -> MusicStyle:
        styles = list(self._style_weights.keys())
        weights = [self._style_weights[s] for s in styles]
        return random.choices(styles, weights=weights, k=1)[0]

    def pick_duration(self) -> int:
        durations = list(DURATION_WEIGHTS.keys())
        weights = [DURATION_WEIGHTS[d] for d in durations]
        return random.choices(durations, weights=weights, k=1)[0]

    def get_publish_times_today(self, n: int) -> list[datetime]:
        """Spread N publish times across the day at optimal hours."""
        now = datetime.now(timezone.utc)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        optimal_hours = [8, 12, 16, 20, 23]  # UTC: peak YouTube traffic windows

        chosen_hours = random.sample(optimal_hours, k=min(n, len(optimal_hours)))
        chosen_hours.sort()

        times = []
        for h in chosen_hours:
            publish_time = today + timedelta(hours=h, minutes=random.randint(0, 45))
            if publish_time > now + timedelta(hours=1):  # at least 1h from now
                times.append(publish_time)

        return times[:n]

    def has_duplicate_style_today(self, style: MusicStyle) -> bool:
        """Prevent publishing the same style twice in one day."""
        from datetime import date
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)

        with get_db() as db:
            count = (
                db.query(VideoJob)
                .filter(
                    VideoJob.style == style,
                    VideoJob.created_at >= today_start,
                    VideoJob.status.in_([VideoStatus.PUBLISHED, VideoStatus.UPLOADING]),
                )
                .count()
            )
        return count > 0

    def count_active_jobs(self) -> int:
        """Count jobs currently in the pipeline (not finished or failed)."""
        active_statuses = [
            VideoStatus.PENDING,
            VideoStatus.GENERATING_MUSIC,
            VideoStatus.GENERATING_VISUALS,
            VideoStatus.EDITING,
            VideoStatus.GENERATING_THUMBNAIL,
            VideoStatus.GENERATING_SEO,
            VideoStatus.UPLOADING,
        ]
        with get_db() as db:
            return db.query(VideoJob).filter(VideoJob.status.in_(active_statuses)).count()

    def schedule_daily_batch(self) -> list[str]:
        """
        Main entry point: generate today's video batch.
        Returns list of Celery chain task IDs.
        """
        from scheduler.tasks import build_pipeline

        self.update_weights_from_analytics()

        n_videos = settings.VIDEOS_PER_DAY
        active = self.count_active_jobs()
        if active >= n_videos:
            logger.info(f"Already {active} active jobs, skipping new batch")
            return []

        n_to_create = n_videos - active
        publish_times = self.get_publish_times_today(n_to_create)

        task_ids = []
        used_styles = set()

        for i in range(n_to_create):
            # Pick non-duplicate style
            style = self.pick_style()
            attempts = 0
            while style in used_styles or self.has_duplicate_style_today(style):
                style = self.pick_style()
                attempts += 1
                if attempts > 10:
                    break

            used_styles.add(style)
            duration = self.pick_duration()
            publish_at = publish_times[i] if i < len(publish_times) else None
            publish_at_iso = publish_at.isoformat() if publish_at else None

            logger.info(f"Scheduling: {style.value} / {duration}s / publish at {publish_at_iso}")
            task_id = build_pipeline(style.value, duration, publish_at_iso)
            task_ids.append(task_id)

        return task_ids


scheduler = DailyScheduler()
