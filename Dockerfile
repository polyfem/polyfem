# ---- Build Stage ----
FROM ubuntu:20.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive

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
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/Kitware/CMake/releases/download/v3.27.5/cmake-3.27.5-linux-x86_64.sh -O /tmp/cmake.sh && \
    chmod +x /tmp/cmake.sh && \
    /tmp/cmake.sh --skip-license --prefix=/usr/local && \
    rm /tmp/cmake.sh

WORKDIR /app
RUN git clone https://github.com/polyfem/polyfem --recursive

WORKDIR /app/polyfem/build
# Disable building tests and ensure data dir is not used
RUN cmake .. -DPOLYFEM_WITH_TESTS=OFF -DPOLYFEM_USE_EXISTING_DATA_DIR=OFF
RUN make -j 4

# ---- Release Stage ----
FROM ubuntu:20.04 AS release
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libx11-dev \
    zenity \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/polyfem

COPY --from=builder /app/polyfem/json-specs/ ./json-specs/
COPY --from=builder /app/polyfem/build/json-specs/ ./build/json-specs/

COPY --from=builder /app/polyfem/build/PolyFEM_bin ./build/PolyFEM_bin

WORKDIR /data

ENTRYPOINT ["/app/polyfem/build/PolyFEM_bin"]
