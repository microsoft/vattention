# Multi-User Docker Workflow

This workflow keeps the Arch host unchanged and gives each Linux user on the machine a separate long-lived container. Shared host prerequisites remain in `/opt`, while each user gets their own container name and cache directories.

## Host prerequisites

These only need to be installed once on the machine:

- Docker with NVIDIA runtime support
- `/opt/cuda-12.1`
- `/opt/libtorch`
- Access to the repository in each user's home directory

Recommended host checks:

```bash
docker info | grep -i runtime
ls /opt/cuda-12.1
ls /opt/libtorch
```

## First-time setup for each user

From the repository root:

```bash
scripts/docker/build-image.sh
scripts/docker/create-container.sh
scripts/docker/bootstrap-workspace.sh
```

What each step does:

- `build-image.sh` builds the shared base image from `docker/Dockerfile`
- `create-container.sh` creates a per-user container named `vattn-$USER`
- `bootstrap-workspace.sh` installs the repo's editable packages from the mounted workspace

The image bakes in the verified dependency set from the working `vattn_research` container:

- NVIDIA PyTorch `24.03`
- `flash-attn==2.5.9.post1`
- FlashInfer commit `c146e068bae01750c3afdbe8a14879183941cb06`
- `transformers==4.44.2`
- `ray==2.53.0`
- the rest of the Python serving stack used by the working container

## Normal daily workflow

Enter the container:

```bash
scripts/docker/enter-container.sh
```

Start the API server:

```bash
scripts/docker/start-server.sh \
  --model_name 01-ai/Yi-6B-200k \
  --model_tensor_parallel_degree 4 \
  --model_attention_backend fa_vattn \
  --model_load_format auto \
  --model_max_model_len 32768 \
  --gpu_memory_utilization 0.8 \
  --host 0.0.0.0 \
  --port 8000
```

## Rebuilding after code changes

Python-only changes are picked up immediately because the repo is bind-mounted into `/workspace`.

If you change compiled code, rerun:

```bash
scripts/docker/bootstrap-workspace.sh
```

That rebuilds:

- `sarathi-lean`
- `pod_attn`
- `vattention`

## Useful overrides

These scripts can be customized with environment variables:

```bash
VATTN_IMAGE_NAME=my-vattention:dev scripts/docker/build-image.sh
VATTN_CONTAINER_NAME=vattn-alice scripts/docker/create-container.sh
VATTN_TORCH_CUDA_ARCH_LIST=8.6 scripts/docker/create-container.sh
VATTN_WORKSPACE_HOST=$HOME/repos/vattention scripts/docker/create-container.sh
```

## Notes

- Containers are isolated per user; do not share one mutable container across multiple accounts.
- `/opt/cuda-12.1` and `/opt/libtorch` stay on the host and are mounted read-only into each container.
- `PYTHONPATH`, `LIBTORCH_PATH`, `PYTORCH_SKIP_VERSION_CHECK`, and ABI flags are set when the container is created so `docker exec` shells start with the expected environment.
