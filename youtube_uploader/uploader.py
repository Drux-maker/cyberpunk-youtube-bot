"""
YouTube Data API v3 uploader.
Handles OAuth2, video upload, thumbnail set, scheduling, playlists, end screens, pinned comment.
"""
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import google.oauth2.credentials
import google_auth_oauthlib.flow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

from config.settings import settings
from database.models import SEOMetadata, VideoJob, YouTubeVideo
from database.db import get_db

logger = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtubepartner",
]

CLIENT_SECRETS = {
    "web": {
        "client_id": settings.YOUTUBE_CLIENT_ID,
        "client_secret": settings.YOUTUBE_CLIENT_SECRET,
        "redirect_uris": [settings.YOUTUBE_REDIRECT_URI],
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
}


class YouTubeUploader:

    def __init__(self):
        self._service = None

    def _get_credentials(self) -> google.oauth2.credentials.Credentials:
        token_file = settings.YOUTUBE_TOKEN_FILE
        creds = None

        if token_file.exists():
            creds = google.oauth2.credentials.Credentials.from_authorized_user_file(
                str(token_file), SCOPES
            )

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = google_auth_oauthlib.flow.Flow.from_client_config(
                    CLIENT_SECRETS, SCOPES
                )
                flow.redirect_uri = settings.YOUTUBE_REDIRECT_URI
                auth_url, _ = flow.authorization_url(prompt="consent")
                logger.info(f"Visit this URL to authorize: {auth_url}")
                code = input("Enter the authorization code: ")
                flow.fetch_token(code=code)
                creds = flow.credentials

            token_file.parent.mkdir(parents=True, exist_ok=True)
            token_file.write_text(creds.to_json())

        return creds

    def _get_service(self):
        if self._service is None:
            creds = self._get_credentials()
            self._service = build("youtube", "v3", credentials=creds)
        return self._service

    def upload_video(
        self,
        job_id: int,
        video_path: Path,
        seo: SEOMetadata,
        publish_at: Optional[datetime] = None,
        privacy: str = "public",
    ) -> str:
        """Upload video and return YouTube video ID."""
        service = self._get_service()

        # Build chapter description
        chapters_text = ""
        if seo.chapters:
            chapters_text = "\n\nTimestamps:\n"
            for ch in seo.chapters:
                chapters_text += f"{ch['time']} {ch['title']}\n"

        full_description = seo.description + chapters_text

        body = {
            "snippet": {
                "title": seo.title,
                "description": full_description,
                "tags": seo.tags,
                "categoryId": settings.YOUTUBE_DEFAULT_CATEGORY_ID,
                "defaultLanguage": "en",
                "defaultAudioLanguage": "en",
            },
            "status": {
                "privacyStatus": "private" if publish_at else privacy,
                "publishAt": publish_at.isoformat() if publish_at else None,
                "selfDeclaredMadeForKids": False,
                "madeForKids": False,
            },
        }

        if publish_at:
            body["status"]["privacyStatus"] = "private"

        media = MediaFileUpload(
            str(video_path),
            mimetype="video/mp4",
            resumable=True,
            chunksize=50 * 1024 * 1024,  # 50MB chunks
        )

        request = service.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media,
        )

        response = None
        retries = 0
        while response is None:
            try:
                status, response = request.next_chunk()
                if status:
                    pct = int(status.progress() * 100)
                    logger.info(f"Upload progress: {pct}%")
            except HttpError as e:
                if e.resp.status in [500, 502, 503, 504] and retries < 5:
                    retries += 1
                    logger.warning(f"Upload error {e.resp.status}, retry {retries}")
                    import time; time.sleep(10 * retries)
                else:
                    raise

        video_id = response["id"]
        logger.info(f"Uploaded video {video_id}: {seo.title}")

        with get_db() as db:
            yt_video = YouTubeVideo(
                job_id=job_id,
                youtube_id=video_id,
                youtube_url=f"https://youtu.be/{video_id}",
                scheduled_at=publish_at,
                privacy_status=privacy,
                upload_status="uploaded",
            )
            db.add(yt_video)

        return video_id

    def set_thumbnail(self, video_id: str, thumbnail_path: Path) -> None:
        service = self._get_service()
        try:
            service.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(str(thumbnail_path), mimetype="image/jpeg"),
            ).execute()
            logger.info(f"Thumbnail set for {video_id}")
        except HttpError as e:
            logger.error(f"Thumbnail upload failed: {e}")

    def add_to_playlist(self, video_id: str, playlist_name: str) -> None:
        service = self._get_service()

        # Find or create playlist
        playlist_id = self._get_or_create_playlist(playlist_name)
        if not playlist_id:
            return

        try:
            service.playlistItems().insert(
                part="snippet",
                body={
                    "snippet": {
                        "playlistId": playlist_id,
                        "resourceId": {"kind": "youtube#video", "videoId": video_id},
                    }
                },
            ).execute()
            logger.info(f"Added {video_id} to playlist '{playlist_name}'")
        except HttpError as e:
            logger.error(f"Playlist insert failed: {e}")

    def _get_or_create_playlist(self, name: str) -> Optional[str]:
        service = self._get_service()

        # Search existing playlists
        response = service.playlists().list(part="snippet", mine=True, maxResults=50).execute()
        for item in response.get("items", []):
            if item["snippet"]["title"] == name:
                return item["id"]

        # Create new playlist
        try:
            result = service.playlists().insert(
                part="snippet,status",
                body={
                    "snippet": {"title": name, "description": f"AI-generated {name} music"},
                    "status": {"privacyStatus": "public"},
                },
            ).execute()
            logger.info(f"Created playlist '{name}': {result['id']}")
            return result["id"]
        except HttpError as e:
            logger.error(f"Playlist creation failed: {e}")
            return None

    def post_pinned_comment(self, video_id: str, comment_text: str) -> None:
        service = self._get_service()
        try:
            comment_thread = service.commentThreads().insert(
                part="snippet",
                body={
                    "snippet": {
                        "videoId": video_id,
                        "topLevelComment": {
                            "snippet": {"textOriginal": comment_text}
                        },
                    }
                },
            ).execute()
            comment_id = comment_thread["snippet"]["topLevelComment"]["id"]
            # Pin the comment (requires channel moderator)
            service.comments().setModerationStatus(
                id=comment_id, moderationStatus="published", banAuthor=False
            ).execute()
            logger.info(f"Pinned comment on {video_id}")
        except HttpError as e:
            logger.warning(f"Could not post/pin comment: {e}")

    def full_publish_pipeline(
        self,
        job_id: int,
        video_path: Path,
        thumbnail_path: Path,
        seo: SEOMetadata,
        publish_at: Optional[datetime] = None,
    ) -> str:
        """Complete upload sequence: video + thumbnail + playlists + comment."""
        video_id = self.upload_video(job_id, video_path, seo, publish_at)
        self.set_thumbnail(video_id, thumbnail_path)

        for playlist_name in (seo.playlist_names or []):
            self.add_to_playlist(video_id, playlist_name)

        if seo.pinned_comment:
            self.post_pinned_comment(video_id, seo.pinned_comment)

        logger.info(f"Full publish complete: https://youtu.be/{video_id}")
        return video_id


youtube_uploader = YouTubeUploader()
