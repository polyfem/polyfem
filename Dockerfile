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

RUN pip3 install numpy sympy quadpy

WORKDIR /app
RUN git clone https://github.com/polyfem/polyfem --recursive

WORKDIR /app/polyfem/build
RUN cmake .. && make -j 4

# ---- Release Stage ----
FROM ubuntu:20.04 AS release
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libx11-dev \
    zenity \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/polyfem
COPY --from=builder /app/polyfem/build/PolyFEM_bin ./build/PolyFEM_bin

WORKDIR /data

ENTRYPOINT ["/app/polyfem/build/PolyFEM_bin"]
