FROM python:3.11

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Non-root user; /data holds saved searches and sync state (named volume in compose).
RUN useradd --create-home --uid 1000 app && mkdir /data /models && chown app:app /data /models
ENV HF_HOME=/models

# CPU-only PyTorch first in its own layer so sentence-transformers does not pull the multi-GB
# CUDA wheels and torch stays cached when pyproject.toml changes.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Dependencies first for layer caching. Dev extras (pytest, ruff) are included so `make test` works
# inside the container; split into a prod target if you ever publish the image.
COPY pyproject.toml README.md ./
# Stub src/vaultlens so setuptools can install third-party deps before copying source code.
RUN mkdir -p src/vaultlens && touch src/vaultlens/__init__.py \
    && pip install -e ".[dev]"
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
