# syntax=docker/dockerfile:1.4

FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    g++ \
    cmake \
    python3 \
    python3-pip \
    zenity \
    libcgal-dev \
    libgl1-mesa-dev \
    libx11-dev \
    wget \
    gnupg \
    software-properties-common \
    libssl-dev \
    ccache \
    && rm -rf /var/lib/apt/lists/*

# Install CMake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | \
    gpg --dearmor - | \
    tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' && \
    apt-get update && \
    apt-get install -y cmake

# Set workdir
WORKDIR /app/polyfem

# Copy source code into the container
COPY . .

# Update submodules
RUN git submodule update --init --recursive

# Build PolyFEM
WORKDIR /app/polyfem/build

# Configure PolyFEM with ccache enabled
RUN --mount=type=cache,target=/root/.ccache \
    cmake .. \
    -DPOLYFEM_WITH_TESTS=OFF \
    -DPOLYFEM_WITH_CCACHE=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS_DEBUGNOSYMBOLS=""

# Build PolyFEM using ccache
RUN --mount=type=cache,target=/root/.ccache \
    make -j $(nproc)

WORKDIR /data
ENTRYPOINT ["/app/polyfem/build/PolyFEM_bin"]