# ==============================================================================
#  Zelt AI Production Container
#  Minimal, unprivileged, high-throughput reasoning microservice.
# ==============================================================================
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt README.md LICENSE ./
COPY zelt/ zelt/
COPY earlystop/ earlystop/

RUN pip install --no-cache-dir build && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels .

# --- Final Runtime Stage ---
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN groupadd -r zeltuser && useradd -r -g zeltuser zeltuser

COPY --from=builder /app/wheels /app/wheels
RUN pip install --no-cache-dir /app/wheels/* && rm -rf /app/wheels

USER zeltuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["zelt"]
CMD ["ui", "--host", "0.0.0.0", "--port", "8000"]
