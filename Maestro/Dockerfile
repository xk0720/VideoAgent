# Maestro — CPU-friendly container so "upload to server and run" is literal.
#
# Build:   docker build -t maestro:0.2.2 .
# Smoke:   docker run --rm maestro:0.2.2 python -m maestro.cli smoke
# Serve:   docker run --rm -p 8000:8000 maestro:0.2.2
#
# Health:  curl http://localhost:8000/health
#
# v0.3 GPU note: swap base to nvidia/cuda:12.4.0-runtime-ubuntu22.04 + add
# torch/omniweaving in a separate stage. The application contract above does
# NOT change — only the backend wiring inside models/video_gen_backends.py.

FROM python:3.11-slim

# ffmpeg/ffprobe is the only system dep — VideoConcat/FrameExtract/AssemblyTool
# all fall back gracefully if it's missing, but the real pipeline needs it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (Docker layer cache).
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Now the source.
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY tests/ tests/
COPY README.md REPORT_AND_INSTRUCTIONS.md ./

# Install the package so `python -m maestro.cli` and `uvicorn maestro.server:app`
# both work without PYTHONPATH gymnastics.
RUN pip install --no-cache-dir -e .

ENV MAESTRO_HOST=0.0.0.0 \
    MAESTRO_PORT=8000 \
    MAESTRO_OUTPUT_ROOT=/app/outputs \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Healthcheck hits the same endpoint UniVA standardized on, so existing
# orchestrators (k8s liveness, docker-compose healthcheck) just work.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; \
    sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=3).status==200 else 1)"

CMD ["uvicorn", "maestro.server:app", "--host", "0.0.0.0", "--port", "8000"]
