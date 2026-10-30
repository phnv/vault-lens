FROM python:3.11

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Non-root user; /data holds saved searches and sync state (named volume in compose).
RUN useradd --create-home --uid 1000 app && mkdir /data && chown app:app /data

# Dependencies first for layer caching. Dev extras (pytest, ruff) are included so `make test` works
# inside the container; split into a prod target if you ever publish the image.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e ".[dev]"
COPY config ./config
COPY tests ./tests

USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "src/vaultlens/ui/app.py", \
  "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
