# InkSight inference service.
#
# Only needed to serve the model to *other* machines. If you are working on the
# Linux box that holds the model, skip this and use LocalBackend in-process.
#
# Linux-only by necessity: tensorflow-text ships no Windows wheels.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    INKSIGHT_MODEL=Derendering/InkSight-Small-p

WORKDIR /app

# Build context is the repo root (see compose.yaml) because the service
# imports ink_tokenizer.tf_runtime -- the engine is shared with LocalBackend
# rather than duplicated here.
COPY pyproject.toml README.md ./
COPY ink_tokenizer ./ink_tokenizer
# For GPU, change the tensorflow pin in pyproject.toml to tensorflow[and-cuda].
RUN pip install --no-cache-dir ".[service]"

COPY services/inksight/app.py ./

# Checkpoint (~692 MB) downloads on first run into the /models volume.
VOLUME ["/models"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
