# ORPflow HFT Paper Trading - Multi-stage Dockerfile
# OCaml + Rust (Jane Street Style - No Python in Hot Path)
# Single unified Rust binary handles market data + strategy + API
# WITH ONNX Runtime for ML inference in hot path

# ============================================================================
# Stage 1: Rust Builder with ONNX Runtime
# ============================================================================
FROM rust:1.85-slim-bookworm AS rust-builder

# Install build dependencies + ONNX Runtime
# CRITICAL: g++ is required for libstdc++ linking (ONNX Runtime dependency)
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    curl \
    ca-certificates \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Download and install ONNX Runtime for building
ENV ORT_VERSION=1.19.2
RUN curl -L https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz \
    -o /tmp/onnxruntime.tgz \
    && tar -xzf /tmp/onnxruntime.tgz -C /opt \
    && rm /tmp/onnxruntime.tgz

# Set ONNX Runtime paths for building
ENV ORT_LIB_LOCATION=/opt/onnxruntime-linux-x64-1.19.2
ENV LD_LIBRARY_PATH=/opt/onnxruntime-linux-x64-1.19.2/lib

WORKDIR /app/market-data

# Copy Cargo manifests to lock dependency versions in CI builds
COPY market-data/Cargo.toml ./
COPY market-data/Cargo.lock ./

# Copy actual source code
COPY market-data/src ./src

# Copy benchmarks (required by Cargo.toml)
COPY market-data/benches ./benches

# Copy tests for ONNX parity
COPY market-data/tests ./tests

# Build release binary WITH ML feature (ONNX support)
RUN cargo build --release --features ml --locked

# ============================================================================
# Stage 2: OCaml Builder
# ============================================================================
FROM ocaml/opam:debian-12-ocaml-5.1 AS ocaml-builder

# Install system dependencies as root first
USER root
RUN apt-get update && apt-get install -y \
    pkg-config \
    libgmp-dev \
    && rm -rf /var/lib/apt/lists/*

USER opam
WORKDIR /home/opam/app

# Install OCaml dependencies
RUN opam update && opam install -y \
    dune \
    core \
    core_unix \
    yojson \
    ppx_deriving \
    ppx_deriving_yojson \
    ppx_sexp_conv \
    lwt \
    lwt_ppx \
    cohttp-lwt-unix \
    alcotest

# Copy project files
COPY --chown=opam:opam core/ .

# Build
RUN eval $(opam env) && dune build --release

# ============================================================================
# Stage 3: Final Runtime Image (No Python - Jane Street Style)
# ============================================================================
FROM debian:bookworm-slim AS runtime

# Install runtime dependencies (minimal - no Python)
# CRITICAL: libstdc++6 is required for ONNX Runtime at runtime
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    curl \
    supervisor \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Copy ONNX Runtime libraries for inference
COPY --from=rust-builder /opt/onnxruntime-linux-x64-1.19.2/lib /opt/onnxruntime/lib

# Set LD_LIBRARY_PATH for ONNX Runtime
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib

WORKDIR /app

# Copy Rust binary (unified: market-data + strategy + API)
COPY --from=rust-builder /app/market-data/target/release/orp-flow-market-data /app/bin/market-data

# Copy OCaml binary (risk gateway)
COPY --from=ocaml-builder /home/opam/app/_build/default/bin/risk_gateway.exe /app/bin/risk_gateway

# Copy ONNX models for ML inference
COPY trained/onnx /app/models/onnx

# Copy supervisor configuration
COPY deploy/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy entrypoint script
COPY deploy/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create data directory for SQLite
RUN mkdir -p /data

# Set ONNX model path environment variable
ENV ONNX_MODEL_DIR=/app/models/onnx

# Expose ports:
# 8000 - Main API (health, status, trades, positions)
# 9090 - Metrics/Health checks
EXPOSE 8000 9090

# Health check - single Rust binary serves everything
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start all services via supervisor
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
