"""
Pulls analytics from YouTube Analytics API and stores them in the DB.
Identifies what's working and feeds data back to the scheduler.
"""
import logging
from datetime import datetime, timedelta, timezone, date

import google.oauth2.credentials
from googleapiclient.discovery import build

from config.settings import settings
from database.db import get_db
from database.models import YouTubeVideo, VideoJob, ChannelAnalytics

logger = logging.getLogger(__name__)


class AnalyticsService:

    def __init__(self):
        self._analytics_service = None
        self._data_service = None

    def _get_services(self):
        if self._analytics_service is None:
            from youtube_uploader.uploader import YouTubeUploader
            uploader = YouTubeUploader()
            creds = uploader._get_credentials()
            self._analytics_service = build("youtubeAnalytics", "v2", credentials=creds)
            self._data_service = build("youtube", "v3", credentials=creds)

    def fetch_video_analytics(self, video_id: str, days_back: int = 30) -> dict:
        """Fetch per-video analytics for the last N days."""
        self._get_services()

        end_date = date.today().isoformat()
        start_date = (date.today() - timedelta(days=days_back)).isoformat()

        try:
            response = self._analytics_service.reports().query(
                ids=f"channel==MINE",
                startDate=start_date,
                endDate=end_date,
                metrics="views,estimatedMinutesWatched,averageViewDuration,averageViewPercentage,likes,comments,subscribersGained,impressions,impressionClickThroughRate,estimatedRevenue",
                dimensions="video",
                filters=f"video=={video_id}",
            ).execute()

            rows = response.get("rows", [])
            if not rows:
                return {}

            row = rows[0]
            headers = [h["name"] for h in response["columnHeaders"]]
            data = dict(zip(headers, row[1:]))  # skip video dimension

            return {
                "views": int(data.get("views", 0)),
                "watch_time_hours": float(data.get("estimatedMinutesWatched", 0)) / 60,
                "avg_view_duration_pct": float(data.get("averageViewPercentage", 0)) / 100,
                "likes": int(data.get("likes", 0)),
                "comments": int(data.get("comments", 0)),
                "impressions": int(data.get("impressions", 0)),
                "ctr": float(data.get("impressionClickThroughRate", 0)) / 100,
                "revenue_usd": float(data.get("estimatedRevenue", 0)),
            }
        except Exception as e:
            logger.error(f"Analytics fetch failed for {video_id}: {e}")
            return {}

    def update_all_video_analytics(self) -> int:
        """Update analytics for all published videos. Returns count updated."""
        with get_db() as db:
            videos = (
                db.query(YouTubeVideo)
                .filter(YouTubeVideo.youtube_id.isnot(None))
                .all()
            )

        updated = 0
        for video in videos:
            data = self.fetch_video_analytics(video.youtube_id)
            if not data:
                continue

            with get_db() as db:
                v = db.query(YouTubeVideo).filter(YouTubeVideo.id == video.id).first()
                v.views = data.get("views", v.views)
                v.watch_time_hours = data.get("watch_time_hours", v.watch_time_hours)
                v.avg_view_duration_pct = data.get("avg_view_duration_pct", v.avg_view_duration_pct)
                v.likes = data.get("likes", v.likes)
                v.comments = data.get("comments", v.comments)
                v.impressions = data.get("impressions", v.impressions)
                v.ctr = data.get("ctr", v.ctr)
                v.revenue_usd = data.get("revenue_usd", v.revenue_usd)
                v.analytics_updated_at = datetime.now(timezone.utc)

            updated += 1

        logger.info(f"Updated analytics for {updated} videos")
        return updated

    def fetch_channel_snapshot(self) -> dict:
        """Daily channel summary."""
        self._get_services()
        end_date = date.today().isoformat()
        start_date = end_date

        try:
            response = self._analytics_service.reports().query(
                ids="channel==MINE",
                startDate=start_date,
                endDate=end_date,
                metrics="views,estimatedMinutesWatched,subscribersGained,subscribersLost,estimatedRevenue",
            ).execute()

            rows = response.get("rows", [])
            if not rows:
                return {}

            row = rows[0]
            return {
                "total_views": int(row[0]),
                "total_watch_time_hours": float(row[1]) / 60,
                "subscriber_gain": int(row[2]),
                "subscriber_loss": int(row[3]),
                "total_revenue_usd": float(row[4]),
            }
        except Exception as e:
            logger.error(f"Channel snapshot failed: {e}")
            return {}

    def save_daily_snapshot(self) -> None:
        snapshot = self.fetch_channel_snapshot()
        if not snapshot:
            return

        best_style = self._find_best_style_today()

        with get_db() as db:
            today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            existing = db.query(ChannelAnalytics).filter(ChannelAnalytics.date == today).first()
            if existing:
                existing.total_views = snapshot["total_views"]
                existing.total_watch_time_hours = snapshot["total_watch_time_hours"]
                existing.subscriber_gain = snapshot["subscriber_gain"]
                existing.subscriber_loss = snapshot["subscriber_loss"]
                existing.net_subscribers = snapshot["subscriber_gain"] - snapshot["subscriber_loss"]
                existing.total_revenue_usd = snapshot["total_revenue_usd"]
                existing.best_performing_style = best_style
            else:
                db.add(ChannelAnalytics(
                    date=today,
                    total_views=snapshot["total_views"],
                    total_watch_time_hours=snapshot["total_watch_time_hours"],
                    subscriber_gain=snapshot["subscriber_gain"],
                    subscriber_loss=snapshot["subscriber_loss"],
                    net_subscribers=snapshot["subscriber_gain"] - snapshot["subscriber_loss"],
                    total_revenue_usd=snapshot["total_revenue_usd"],
                    best_performing_style=best_style,
                ))

    def _find_best_style_today(self) -> str | None:
        """Return the style with the best avg retention today."""
        from sqlalchemy import func
        with get_db() as db:
            result = (
                db.query(VideoJob.style, func.avg(YouTubeVideo.avg_view_duration_pct).label("avg_ret"))
                .join(YouTubeVideo, VideoJob.id == YouTubeVideo.job_id)
                .filter(YouTubeVideo.analytics_updated_at >= datetime.now(timezone.utc) - timedelta(hours=24))
                .group_by(VideoJob.style)
                .order_by(func.avg(YouTubeVideo.avg_view_duration_pct).desc())
                .first()
            )
            return result.style.value if result else None

    def generate_performance_report(self) -> dict:
        """Summary report for human review."""
        with get_db() as db:
            from sqlalchemy import func
            from database.models import VideoStatus

            total_published = db.query(VideoJob).filter(VideoJob.status == VideoStatus.PUBLISHED).count()
            total_failed = db.query(VideoJob).filter(VideoJob.status == VideoStatus.FAILED).count()

            top_videos = (
                db.query(YouTubeVideo)
                .filter(YouTubeVideo.youtube_id.isnot(None))
                .order_by(YouTubeVideo.views.desc())
                .limit(5)
                .all()
            )

            avg_ctr = db.query(func.avg(YouTubeVideo.ctr)).scalar() or 0
            avg_retention = db.query(func.avg(YouTubeVideo.avg_view_duration_pct)).scalar() or 0
            total_revenue = db.query(func.sum(YouTubeVideo.revenue_usd)).scalar() or 0

        return {
            "total_published": total_published,
            "total_failed": total_failed,
            "success_rate": total_published / max(1, total_published + total_failed),
            "avg_ctr": round(avg_ctr * 100, 2),
            "avg_retention_pct": round(avg_retention * 100, 1),
            "total_revenue_usd": round(total_revenue, 2),
            "top_videos": [
                {
                    "youtube_id": v.youtube_id,
                    "url": v.youtube_url,
                    "views": v.views,
                    "ctr": round(v.ctr * 100, 2),
                    "retention": round(v.avg_view_duration_pct * 100, 1),
                }
                for v in top_videos
            ],
        }


analytics_service = AnalyticsService()
