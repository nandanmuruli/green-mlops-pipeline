# Containerized training/evaluation workload for the Green MLOps pipeline.
# Matches the expose's "CI/CD & Orchestration: containerized training
# workloads via Docker" requirement. Kubernetes orchestration on top of
# this image was scoped out (see README "Documented scope adaptation") --
# this container is designed to be runnable directly or as a CI/CD job step.

FROM python:3.10-slim

WORKDIR /app

# System deps needed by torch/codecarbon for CPU-only operation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Default: run the full DVC-tracked pipeline. Override the command to run
# a single stage, e.g.:
#   docker run <image> python -m src.pipeline.compress
CMD ["dvc", "repro"]
