FROM python:3.12-slim

# Install system dependencies: FFmpeg, libsndfile, fonts
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    fonts-liberation \
    fontconfig \
    curl \
    && fc-cache -fv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create asset directories
RUN mkdir -p assets/music assets/visuals assets/videos assets/thumbnails assets/fonts logs config

# Non-root user for security
RUN useradd -m -u 1000 botuser && chown -R botuser:botuser /app
USER botuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
