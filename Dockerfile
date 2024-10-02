FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    g++ \
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

RUN wget https://github.com/Kitware/CMake/releases/download/v3.27.5/cmake-3.27.5-linux-x86_64.sh -O /tmp/cmake.sh && \
    chmod +x /tmp/cmake.sh && \
    /tmp/cmake.sh --skip-license --prefix=/usr/local && \
    rm /tmp/cmake.sh

WORKDIR /app

RUN git clone https://github.com/polyfem/polyfem --recursive

WORKDIR /app/polyfem/build
RUN cmake .. && make

WORKDIR /data

# Define the default entrypoint to run PolyFEM
ENTRYPOINT ["/app/polyfem/build/PolyFEM_bin"]


## To build the PolyFEM image, use:
# docker build -t polyfem .
## To run PolyFEM with:
# docker run --rm -v "$(pwd)":/data polyfem [PolyFEM arguments]
