"""
Downloads free cyberpunk-style fonts for thumbnails.
Run once during setup: python scripts/download_fonts.py
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import settings

FONTS = {
    "BebasNeue-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/bebasneue/BebasNeue-Regular.ttf",
    "Orbitron-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/orbitron/Orbitron%5Bwght%5D.ttf",
    "RobotoCondensed-Bold.ttf": "https://github.com/google/fonts/raw/main/apache/robotocondensed/RobotoCondensed-Bold.ttf",
}

if __name__ == "__main__":
    settings.FONTS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in FONTS.items():
        dest = settings.FONTS_DIR / filename
        if dest.exists():
            print(f"  Already exists: {filename}")
            continue
        print(f"  Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, str(dest))
            print(f"  OK: {dest}")
        except Exception as e:
            print(f"  FAILED: {e}")
    print("Done.")
