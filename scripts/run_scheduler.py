"""
Daily scheduler daemon — replaces Celery beat.

Runs a single Python process that:
  - fires the daily batch every day at 04:00 UTC
  - fires an analytics refresh every 6 hours
  - fires a channel snapshot every day at 23:30 UTC

Usage:
    python scripts/run_scheduler.py

To run as a background service on Windows: register this script with
the Task Scheduler, or launch it from a startup .bat / NSSM.

To trigger an immediate one-off batch (without waiting for the cron):
    python scripts/run_scheduler.py --now
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from config.settings import settings
from database.db import init_db


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(settings.LOGS_DIR / "scheduler.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger("run_scheduler")


def job_daily_batch() -> None:
    from scheduler.pipeline import scheduler as daily
    logger.info("=== Daily batch trigger ===")
    results = daily.run_daily_batch()
    logger.info(f"Daily batch finished: {len(results)} videos OK")


def job_update_analytics() -> None:
    from analytics.analyzer import analytics_service
    logger.info("=== Analytics refresh ===")
    n = analytics_service.update_all_video_analytics()
    logger.info(f"Analytics updated for {n} videos")


def job_daily_snapshot() -> None:
    from analytics.analyzer import analytics_service
    logger.info("=== Daily channel snapshot ===")
    analytics_service.save_daily_snapshot()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--now", action="store_true",
                        help="Run the daily batch immediately and exit (no daemon).")
    args = parser.parse_args()

    settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    init_db()

    if args.now:
        job_daily_batch()
        return

    sched = BlockingScheduler(timezone="UTC")

    sched.add_job(
        job_daily_batch,
        trigger=CronTrigger(hour=4, minute=0, timezone="UTC"),
        id="daily_batch",
        max_instances=1,
        coalesce=True,
    )
    sched.add_job(
        job_update_analytics,
        trigger=CronTrigger(hour="*/6", minute=15, timezone="UTC"),
        id="update_analytics",
        max_instances=1,
        coalesce=True,
    )
    sched.add_job(
        job_daily_snapshot,
        trigger=CronTrigger(hour=23, minute=30, timezone="UTC"),
        id="daily_snapshot",
        max_instances=1,
        coalesce=True,
    )

    logger.info("Scheduler started. Cron jobs:")
    for j in sched.get_jobs():
        logger.info(f"  - {j.id}: next run at {j.next_run_time}")

    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
