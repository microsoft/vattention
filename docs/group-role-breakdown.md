# Group Role Breakdown

This document summarizes the team-wide division of labor for the fragmentation study and report.

## Overall project goal

This project studies how `vAttention` reduces KV-cache memory fragmentation for long-context inference. The team is measuring and explaining fragmentation as a function of context length, with a focus on comparing dense attention variants and MLA-style compressed caching. The goal is to combine both theory and experiments to explain why `vAttention` is a strong fit for this problem.

In practice, the workflow is:

1. start the serving stack with `vAttention`
2. send requests at controlled context lengths
3. record per-request fragmentation metrics
4. analyze the results with theory, plots, and report discussion

## Example demo commands

These are simple commands you can run to get started.

Start the server:

```bash
scripts/docker/start-server-yi6b.sh
```

If you want the run artifacts to be easy to inspect from the host, write them directly into the bind-mounted workspace:

```bash
VATTN_SERVER_OUTPUT_DIR=/workspace/server-output/demo-run scripts/docker/start-server-yi6b.sh
```

In a second shell, send a simple completion request:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "01-ai/Yi-6B-200k",
    "prompt": "The main goal of this vAttention project is to",
    "max_tokens": 32,
    "temperature": 0.0
  }'
```

You can also show the served models endpoint:

```bash
curl http://localhost:8000/v1/models
```

## Live monitoring during the demo

While the server is running, open another shell and monitor GPU memory usage:

```bash
nvidia-smi -l 1
```

This refreshes every second and gives a simple live view of how GPU memory changes while requests are running.

## Viewing results from the host

If you use a workspace-backed output directory such as `/workspace/server-output/demo-run`, the generated files are immediately visible from the host at the matching repo path:

```bash
ls ~/repos/vattention/server-output/demo-run
```

That makes it easy to inspect files such as `config.yml`, `benchmark_config.yml`, and `sequence_metrics.csv` from your normal host shell without entering the container.


## Roles and responsibilities

### Kyle

- Theory and architecture side of the project.
- Create the theoretical fragmentation expressions versus context length for GQA and MLA.
- Contributed the existing theoretical expressions for MHA.
- Contributed the MLA implementation work already completed, including support for an MLA model in the codebase.

### Josh

- Own the request-level metrics pipeline in the serving stack.
- Add and validate fragmentation-related metrics so they are emitted alongside context-length information in `sequence_metrics.csv`.
- Ensure the data needed for downstream plotting and analysis is captured consistently for each completed request.

### Michel

- Own the sequential request-sweep driver used to run controlled context-length experiments.
- Generate deterministic request runs that pair cleanly with Josh's metrics pipeline.
- Serve as report lead by compiling everyone's draft sections into a coherent final 3-page report.

### Londy

- Write and present the project background on existing memory-fragmentation reduction strategies.
- Explain why the `vAttention` approach is the best fit for this problem setting.
- Own the plotting and results-visualization workflow.
- Turn Josh's collected metrics into publication-ready plots of context length versus fragmentation.

## Expected workflow across the team

1. Kyle provides the theoretical framework and MLA/MHA/GQA fragmentation analysis.
2. Michel runs the controlled context-length sweep experiments and coordinates report assembly.
3. Josh captures the request-level fragmentation metrics in the codebase.
4. Londy produces the figures and presents the background and motivation for the chosen approach.
