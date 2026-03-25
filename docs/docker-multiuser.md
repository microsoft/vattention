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

Expected output:

- `docker info | grep -i runtime`
  You should see the NVIDIA runtime listed, and often a default runtime as well. For example:

  ```text
   Runtimes: io.containerd.runc.v2 nvidia runc
   Default Runtime: runc
  ```

- `ls /opt/cuda-12.1`
  You should see the CUDA toolkit directories. For example:

  ```text
  cuda  ld.so.conf.d  lib  profile.d  share
  ```

- `ls /opt/libtorch`
  You should see the LibTorch install directories. For example:

  ```text
  bin  build-hash  build-version  include  lib  share
  ```

The exact order may vary slightly by host, but if `nvidia` is missing from the Docker runtimes or either `/opt` path does not exist, stop and fix the host setup before continuing.

## Clone the repository

Each user should clone their own copy of the repository into their home directory. A simple location is `~/repos`.

If you do not already have a `repos` folder, create it:

```bash
mkdir -p ~/repos
cd ~/repos
```

Clone the repository with HTTPS:

```bash
git clone https://github.com/Anodyine/vattention.git
cd vattention
```

To confirm that the clone worked, you should see files such as `README.md`, `docker/`, and `scripts/`:

```bash
ls
```

## First-time setup for each user

From the repository root:

```bash
scripts/docker/build-image.sh
scripts/docker/create-container.sh
scripts/docker/bootstrap-workspace.sh
```

The first image build can take a while and may print a lot of output. That is normal, especially on the first run.

What each step does:

- `build-image.sh` builds the shared base image from `docker/Dockerfile`
- `create-container.sh` creates a per-user container named `vattn-$USER`
- `bootstrap-workspace.sh` installs the repo's editable packages from the mounted workspace. This one will take a while.

After setup finishes, a simple success check is:

```bash
docker ps -a | grep vattn-$USER
```

You should see your container listed.

The image bakes in the verified dependency set from the working `vattn_research` container:

- NVIDIA PyTorch `24.03`
- `flash-attn==2.5.9.post1`
- FlashInfer commit `c146e068bae01750c3afdbe8a14879183941cb06`
- `transformers==4.44.2`
- `ray==2.53.0`
- the rest of the Python serving stack used by the working container

## Normal daily workflow

Use one of these workflows depending on what you want to do.

Open an interactive shell inside the container for debugging, inspection, or manual commands:

```bash
scripts/docker/enter-container.sh
```

OR start the API server from the host shell:

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

Or start the same Yi-6B server with the checked-in preset wrapper, also from the host shell:

```bash
scripts/docker/start-server-yi6b.sh
```

For the current known-good `DeepSeek-V2-Lite` bring-up on `2 x 24 GB` GPUs, use the checked-in preset wrapper from the host shell:

```bash
scripts/docker/start-server-deepseek-v2-lite.sh
```

That preset currently uses the tight startup settings that were verified to reach real serving for `DeepSeek-V2-Lite` in this repo and defaults to `CUDA_VISIBLE_DEVICES=0,1` inside the container:

- `--model_tensor_parallel_degree 2`
- `--gpu_memory_utilization 1.0`
- `--replica_scheduler_max_batch_size 1`

The wrapper also auto-selects a default `--model_max_model_len` based on the requested tensor-parallel degree unless you override it explicitly:

- `128` for the default `TP=2` bring-up
- `512` when you pass `--model_tensor_parallel_degree 4`

To target a different GPU pair, override the wrapper-local env var from the host shell:

```bash
DEEPSEEK_V2_LITE_CUDA_VISIBLE_DEVICES=2,3 scripts/docker/start-server-deepseek-v2-lite.sh
```

To override the max context directly, pass it on the command line:

```bash
DEEPSEEK_V2_LITE_CUDA_VISIBLE_DEVICES=0,1,2,3 \
scripts/docker/start-server-deepseek-v2-lite.sh \
  --model_tensor_parallel_degree 4 \
  --model_max_model_len 768
```

Do not run `start-server.sh` or `start-server-yi6b.sh` from inside the container shell opened by `enter-container.sh`. Those wrapper scripts are intended to be launched from the host and will `docker exec` into the container for you.

The same guidance applies to `start-server-deepseek-v2-lite.sh`.

By default, the server wrapper writes generated runtime files such as `config.yml` and `benchmark_config.yml` to a container-local directory under `/tmp/vattention/<container-name>` instead of modifying files in the repo checkout.

To override that location explicitly:

```bash
VATTN_SERVER_OUTPUT_DIR=/tmp/vattention/custom-run scripts/docker/start-server-yi6b.sh
```

Once the server is running, send a simple test prompt from another host shell:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "01-ai/Yi-6B-200k",
    "prompt": "The primary advantage of using virtual memory for LLM KV-cache management is",
    "max_tokens": 64,
    "temperature": 0.3
  }'
```

You should get back a JSON response with a `choices` array containing generated text.

If you started the server on a different port, replace `8000` in the URL to match.

## Rebuilding after code changes

Python-only changes are picked up immediately because the repo is bind-mounted into `/workspace`.

If you change compiled code or package install metadata, rerun only the component you changed:

```bash
scripts/docker/bootstrap-sarathi.sh
scripts/docker/bootstrap-pod-attn.sh
scripts/docker/bootstrap-vattention.sh
```

Use the matching script as a rule of thumb:

- Changed `sarathi-lean`: run `scripts/docker/bootstrap-sarathi.sh`
- Changed `pod_attn`: run `scripts/docker/bootstrap-pod-attn.sh`
- Changed `vattention` install-time code or packaging: run `scripts/docker/bootstrap-vattention.sh`
- Changed multiple components or want a clean reset: run `scripts/docker/bootstrap-workspace.sh`

The full bootstrap script is still available:

```bash
scripts/docker/bootstrap-workspace.sh
```

That rebuilds everything:

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

## Troubleshooting

- If you see `command not found`, make sure you are in the repository root (`cd ~/repos/vattention`) and that you typed the script path exactly as shown.
- If you see `permission denied`, you may not have permission to talk to Docker or execute the script in your current environment. Check with whoever manages the server setup.
