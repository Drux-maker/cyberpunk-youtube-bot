"""
One-time script to authorize YouTube API access.
Run this ONCE on your server to generate the OAuth token.
Usage: python scripts/setup_youtube_auth.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from youtube_uploader.uploader import YouTubeUploader

if __name__ == "__main__":
    print("Starting YouTube OAuth2 authorization...")
    print("A URL will be printed — open it in your browser, authorize, then paste the code here.")
    uploader = YouTubeUploader()
    creds = uploader._get_credentials()
    print(f"\nAuthorization successful! Token saved to: {uploader._token_file_path()}")
    print("You can now run the bot without manual authorization.")
