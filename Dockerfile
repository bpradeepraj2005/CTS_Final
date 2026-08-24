# Portable container: binds $PORT when the host sets one (Render sets 10000),
# otherwise 7860 for local runs.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Run as a non-root user, and own the app directory so the append-only audit
# log (prior_auth/data/audit_log.jsonl) stays writable inside the container.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user:user requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user:user . ./

# No EXPOSE: Render detects the listening port, and a stale EXPOSE would point
# it at the wrong one. For local runs: docker run -p 7860:7860 <image>
#
# `sh -c` so ${PORT} expands; `exec` so uvicorn replaces the shell as PID 1 and
# receives SIGTERM directly on shutdown.
#
# Single worker: the corpus and BM25 index are built once per process, and the
# sync endpoints already run in Starlette's thread pool.
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
