# Install docker
# curl -fsSL https://get.docker.com | sh
# sudo usermod -aG docker $USER
# newgrp docker          # apply group change without logout
# docker --version       # verify

# Build image
# Usage 1: docker build -t tenmo:latest .
# Usage 2: docker build --build-arg EXAMPLES="mnist_unified" -t tenmo:mnist .
# Usage 3: docker build -t tenmo:mnist .
# Push
# docker login -u <username>
# docker tag tenmo:mnist ratulb/tenmo:latest
# docker push ratulb/tenmo:latest
# Run
# docker run -it ratulb/tenmo:latest
# ── Build arguments ────────────────────────────────────────────────
# MOJO_VERSION: leave empty to auto-detect from pixi.toml, or set
#               explicitly (e.g. --build-arg MOJO_VERSION=1.0.0b1).
# EXAMPLES:     space-separated list of example names to build.
ARG MOJO_VERSION=""
#ARG EXAMPLES="mnist mnist_unified word2vec_cbow xor"
ARG EXAMPLES="mnist"

# ── Builder stage ──────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# System dependencies
RUN apt-get update && apt-get install -y build-essential libopenblas-dev

# Redeclare ARGs to bring them into this stage's scope
ARG MOJO_VERSION
ARG EXAMPLES

WORKDIR /app

# Copy only pixi.toml first — extract Mojo version for pip install,
# then discard this layer (pixi.toml rarely changes, good Docker cache)
COPY pixi.toml .
RUN MOJO_VERSION="${MOJO_VERSION:-$(grep 'mojo =' pixi.toml | sed 's/.*"==\(.*\)"/\1/')}" && \
    echo "Using Mojo ${MOJO_VERSION}" && \
    pip install --upgrade pip && \
    pip install --pre "mojo==${MOJO_VERSION}" tiktoken mnist-datasets pure-cifar-10

# Full source copy and build
COPY . .

RUN mkdir -p /app/bin && for name in $EXAMPLES; do \
      mojo build -I . -Xlinker -lm "examples/${name}.mojo" -o "/app/bin/${name}"; \
    done

# Persist the resolved Mojo version for the runtime stage
RUN grep 'mojo =' pixi.toml | sed 's/.*"==\(.*\)"/\1/' > /tmp/mojo_version.txt

# ── Runtime stage ──────────────────────────────────────────────────
FROM python:3.12-slim

# Minimum system deps for compiled Mojo binaries
RUN apt-get update && apt-get install -y libopenblas-dev && rm -rf /var/lib/apt/lists/*

# Install same Mojo version as builder (needed for runtime .so loading)
COPY --from=builder /tmp/mojo_version.txt /tmp/mojo_version.txt
ARG MOJO_VERSION
RUN MOJO_VERSION="${MOJO_VERSION:-$(cat /tmp/mojo_version.txt)}" && \
    pip install --pre "mojo==${MOJO_VERSION}" tiktoken mnist-datasets pure-cifar-10

# Copy compiled binaries from builder
COPY --from=builder /app/bin /app/bin

# Entrypoint: list binaries or exec requested one
COPY --chmod=+x <<"EOF" /entrypoint.sh
#!/bin/sh
if [ $# -eq 0 ]; then
   echo "Available:"
   ls /app/bin/
   exit 1
fi
exec "$@"
EOF
ENTRYPOINT ["/entrypoint.sh"]
