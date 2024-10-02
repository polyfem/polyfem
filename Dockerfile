FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    g++ \
    python3 \
    python3-pip \
    zenity \
    libcgal-dev \
    libgl1-mesa-dev \
    libx11-dev

RUN pip3 install numpy sympy quadpy

WORKDIR /app

RUN git clone https://github.com/polyfem/polyfem --recursive
WORKDIR /app/polyfem/build
RUN cmake .. && make -j4

WORKDIR /data

ENTRYPOINT ["/app/polyfem/build/PolyFEM_bin"]

## To build the PolyFEM image, use:
# docker build -t polyfem .
## To run PolyFEM with:
# docker run --rm -v "$(pwd)":/data polyfem [PolyFEM arguments]
