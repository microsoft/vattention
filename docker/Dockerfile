FROM nvcr.io/nvidia/pytorch:24.03-py3

SHELL ["/bin/bash", "-o", "pipefail", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LIBTORCH_PATH=/opt/libtorch \
    PYTORCH_SKIP_VERSION_CHECK=1 \
    TORCH_CUDA_ARCH_LIST=8.6 \
    CXXFLAGS=-D_GLIBCXX_USE_CXX11_ABI=1 \
    MAX_JOBS=4 \
    NVCC_THREADS=4 \
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    FLASHINFER_ENABLE_AOT=1 \
    HF_HOME=/root/.cache/huggingface \
    TORCH_HOME=/root/.cache/torch \
    PIP_CACHE_DIR=/root/.cache/pip

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install \
    ddsketch==3.0.1 \
    fastapi==0.129.0 \
    grpcio==1.62.1 \
    kaleido==1.2.0 \
    numpy==1.24.4 \
    openai==2.21.0 \
    pandas==1.5.3 \
    packaging \
    pillow==10.2.0 \
    plotly==6.5.2 \
    plotly-express==0.4.1 \
    psutil \
    pyarrow==14.0.1 \
    ray==2.53.0 \
    seaborn==0.13.2 \
    sentencepiece==0.2.1 \
    tiktoken==0.12.0 \
    transformers==4.44.2 \
    uvicorn==0.41.0 \
    wandb==0.25.0

RUN python -m pip install --no-build-isolation flash-attn==2.5.9.post1

RUN rm -rf /tmp/flashinfer && \
    git clone https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer && \
    cd /tmp/flashinfer && \
    git checkout c146e068bae01750c3afdbe8a14879183941cb06 && \
    sed -i 's|git@github.com:|https://github.com/|g' .gitmodules && \
    sed -i 's|ssh://git@github.com/|https://github.com/|g' .gitmodules && \
    git submodule sync --recursive && \
    git submodule update --init --recursive && \
    cd python && \
    python -m pip install --no-build-isolation .

CMD ["sleep", "infinity"]
