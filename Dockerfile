# ORPflow HFT Paper Trading - Multi-stage Dockerfile
# OCaml + Rust (Jane Street Style - No Python in Hot Path)
# Single unified Rust binary handles market data + strategy + API

# ============================================================================
# Stage 1: Rust Builder
# ============================================================================
FROM rust:1.83-slim-bookworm AS rust-builder

RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/market-data

# Copy Cargo.toml only (Cargo.lock generated during build)
COPY market-data/Cargo.toml ./

# Copy actual source code
COPY market-data/src ./src

# Copy benchmarks (required by Cargo.toml)
COPY market-data/benches ./benches

# Build release binary (generates Cargo.lock automatically)
RUN cargo build --release

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
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    curl \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Rust binary (unified: market-data + strategy + API)
COPY --from=rust-builder /app/market-data/target/release/orp-flow-market-data /app/bin/market-data

# Copy OCaml binary (risk gateway)
COPY --from=ocaml-builder /home/opam/app/_build/default/bin/risk_gateway.exe /app/bin/risk_gateway

# Copy supervisor configuration
COPY deploy/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy entrypoint script
COPY deploy/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create data directory for SQLite
RUN mkdir -p /data

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
