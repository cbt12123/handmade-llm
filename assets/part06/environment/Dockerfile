# Public default; BASE_IMAGE can select an existing compatible devel image.
ARG BASE_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 g++ ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN command -v nvcc && command -v compute-sanitizer \
    && test -f /usr/local/cuda/include/cublas_v2.h \
    && python3 --version && g++ --version
WORKDIR /workspace
CMD ["bash"]
