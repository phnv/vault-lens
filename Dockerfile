FROM python:3.11@sha256:55c475496e35ad8d898659c85f92b42d3b1c58f25f5d79023baf0840b0b4a462

# PYTHONDONTWRITEBYTECODE=1: Tells Python not to write .pyc bytecode cache files (__pycache__/) to disk when importing modules.
# PYTHONUNBUFFERED=1: Forces Python's stdout and stderr streams to be unbuffered (flushed immediately on every print() or log line).
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Non-root user; /data holds saved searches and sync state (named volume in compose).
RUN useradd --create-home --uid 1000 app && mkdir /data /models && chown app:app /data /models
ENV HF_HOME=/models

# CPU-only PyTorch first in its own layer so sentence-transformers does not pull the multi-GB
# CUDA wheels and torch stays cached when pyproject.toml changes.
RUN --mount=type=cache,target=/root/.cache/pip \
  pip install torch --index-url https://download.pytorch.org/whl/cpu

# Dependencies first for layer caching. Dev extras (pytest, ruff) are included so `make test` works
# inside the container; split into a prod target if you ever publish the image.
COPY pyproject.toml ./
# Stub README.md and src/vaultlens so setuptools can install third-party deps before copying docs/source.
RUN --mount=type=cache,target=/root/.cache/pip \
  mkdir -p src/vaultlens && touch README.md src/vaultlens/__init__.py \
  && pip install -e ".[dev]"
COPY README.md ./
COPY src ./src
RUN pip install --no-build-isolation --no-deps -e .
COPY config ./config
COPY tests ./tests

USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "src/vaultlens/ui/app.py", \
  "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
