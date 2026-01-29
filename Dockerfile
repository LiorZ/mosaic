# Mosaic: Multi-objective protein design using continuous relaxation
# Docker image with CUDA support for GPU-accelerated structure prediction

# Use Vast.ai PyTorch base image with CUDA 12.8.1
FROM vastai/pytorch:cuda-12.8.1-auto

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and add deadsnakes PPA for Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    # Python 3.12
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    # Build tools
    build-essential \
    cmake \
    ninja-build \
    # Git for dependency fetching
    git \
    git-lfs \
    # Networking tools
    curl \
    wget \
    ca-certificates \
    # HDF5 for data handling
    libhdf5-dev \
    # OpenBLAS for linear algebra
    libopenblas-dev \
    # Other dependencies
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install uv for fast Python package management
ENV UV_VERSION=0.5.14
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY README.md ./
COPY src/ ./src/

# Create virtual environment and install dependencies with CUDA support
RUN uv venv .venv --python=3.12

# Activate venv and install dependencies
RUN . .venv/bin/activate && uv sync --group jax-cuda

# Set environment variables for JAX/CUDA
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
ENV TF_GPU_ALLOCATOR=cuda_malloc_async
ENV JAX_PLATFORMS=cuda

# Ensure the virtual environment is activated by default
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

# Set Python path to include the source
ENV PYTHONPATH="/app/src"

# Create directories for model weights and cache
RUN mkdir -p /root/.boltz /root/.cache/huggingface /root/.cache/torch

# Copy example files (optional, for running examples)
COPY examples/ ./examples/

# Set up volumes for persistent model weights and data
VOLUME ["/root/.boltz", "/root/.cache/huggingface", "/root/.cache/torch", "/app/data"]

# Expose port for marimo notebook server
EXPOSE 2718

# Default command: start marimo server
CMD ["marimo", "edit", "--host", "0.0.0.0", "--port", "2718", "--headless"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:2718/ || exit 1

# Labels
LABEL maintainer="Mosaic Project"
LABEL description="Multi-objective protein design using continuous relaxation"
LABEL version="0.1.0"
