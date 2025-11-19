# syntax=docker/dockerfile:1.4

FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive

# Base build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    cmake \
    git \
    libmpfr-dev \
    libssl-dev \
    ccache \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app/polyfem

# Copy source code into the container
COPY . .

# If .git is present, update submodules; otherwise just skip
RUN if [ -d .git ]; then git submodule update --init --recursive; fi

# Configure PolyFEM with ccache enabled
# (BuildKit cache mount for ccache)
RUN --mount=type=cache,target=/root/.ccache \
    cmake -S . -B build \
    -DPOLYFEM_WITH_TESTS=OFF \
    -DPOLYFEM_WITH_CCACHE=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS_DEBUGNOSYMBOLS=""

# Build PolyFEM using ccache
RUN --mount=type=cache,target=/root/.ccache \
    cmake --build build -j"$(nproc)"

# Runtime working directory
WORKDIR /data

ENTRYPOINT ["/app/polyfem/build/PolyFEM_bin"]
