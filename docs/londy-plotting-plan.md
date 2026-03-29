# Plotting Plan: Context Length vs Fragmentation

This plan is for you to read the metrics produced by Josh's pipeline and generate publication-ready plots of context length vs fragmentation percentage.

## Goal

Produce a reliable plotting workflow that:

- loads Sarathi metrics output files
- validates required columns exist
- filters and cleans rows safely
- plots `context length` vs `fragmentation %`
- saves figures and a small summary table for reporting

Target x/y for the main figure:

- x-axis: `request_num_prefill_tokens`
- y-axis: `kv_fragmentation_percent`

## Theoretical curves to plot first

Before reading Josh's real metrics output, start by plotting the theoretical amortization curves described in the proposal.

The relevant expressions from the proposal are:

1. Average system-wide fragmentation waste:

`Wavg = B * k * L * Psize`

This comes from assuming half of each of the `2 * L` tail pages is wasted on each worker, which simplifies to `B * k * L * Psize`.

2. Percentage waste as a function of context length `C`:

`W_percent(C) = Wavg / (C * Stoken + Wavg)`

3. Memory footprint per token:

`Stoken = 2 * L * H * Dhead * Pbyte`

Where:

- `L` = number of transformer layers
- `B` = batch size
- `k` = tensor parallelism degree
- `Psize` = physical page size in bytes
- `C` = context length in tokens
- `Stoken` = bytes used per token across all layers
- `H` = number of attention heads
- `Dhead` = head dimension
- `Pbyte` = bytes per element of KV storage precision

For the proposal's worked example:

- `L = 94`
- `B = 1`
- `Psize = 2 * 1024 * 1024`
- `Stoken ~= 94 * 1024` bytes
- compare `k = 1` and `k = 2`

### First theoretical plotting task

Plot `context length` vs `fragmentation %` using the formula above for one or more values of `k`.

### Minimal example

```python
#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def wavg_bytes(batch_size: int, tp_degree: int, num_layers: int, page_size_bytes: int) -> float:
    return batch_size * tp_degree * num_layers * page_size_bytes


def waste_percent(
    context_lengths: np.ndarray,
    *,
    batch_size: int,
    tp_degree: int,
    num_layers: int,
    page_size_bytes: int,
    bytes_per_token: float,
) -> np.ndarray:
    fixed_waste = wavg_bytes(batch_size, tp_degree, num_layers, page_size_bytes)
    utilized = context_lengths * bytes_per_token
    return 100.0 * fixed_waste / (utilized + fixed_waste)


def main() -> None:
    out_path = Path("~/repos/vattention/tmp/theoretical_fragmentation_curve.png").expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    context_lengths = np.array([1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000])

    L = 94
    B = 1
    page_size_bytes = 2 * 1024 * 1024
    bytes_per_token = 94 * 1024

    fig, ax = plt.subplots(figsize=(9, 6), dpi=140)

    for k in [1, 2]:
        y = waste_percent(
            context_lengths,
            batch_size=B,
            tp_degree=k,
            num_layers=L,
            page_size_bytes=page_size_bytes,
            bytes_per_token=bytes_per_token,
        )
        ax.plot(context_lengths, y, marker="o", linewidth=2.0, label=f"TP degree k={k}")

    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Fragmentation (%)")
    ax.set_title("Theoretical Fragmentation Amortization Curve")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
```

### Example with a denser smooth curve

Use this version if you want a continuous-looking line instead of a small table of hand-picked context lengths.

```python
context_lengths = np.linspace(1000, 256000, 300)
```

If lower-context behavior is important, a log-scaled x-axis is often easier to read:

```python
ax.set_xscale("log")
```

### Example with proposal-style MB tables

If you want to reproduce the proposal tables more directly, it can help to compute utilized memory, fixed waste, and total allocated memory in MB before plotting.

```python
def bytes_to_mb(x: np.ndarray | float) -> np.ndarray | float:
    return x / (1024 * 1024)


context_lengths = np.array([1000, 32000, 64000, 128000, 256000])
fixed_waste = wavg_bytes(B, 2, L, page_size_bytes)
utilized = context_lengths * bytes_per_token
total = utilized + fixed_waste
percent = 100.0 * fixed_waste / total

for c, u, t, p in zip(context_lengths, bytes_to_mb(utilized), bytes_to_mb(total), percent):
    print(f"context={c:6d}  utilized_mb={u:8.1f}  total_mb={t:8.1f}  waste_pct={p:5.1f}")
```

### Suggested first outputs

Before touching real metrics, produce:

- one theoretical curve plot for `k = 1`
- one theoretical curve plot comparing `k = 1` vs `k = 2`
- one small CSV or printed table reproducing the proposal's example values

## Inputs and assumptions

Assume Josh's metrics system writes request-level metrics to:

- `/tmp/vattention/<container-name>/sequence_metrics.csv`

And that `sequence_metrics.csv` includes at least:

- `Request Id`
- `request_num_prefill_tokens`
- `kv_fragmentation_percent`

Optional useful columns (if available):

- `kv_blocks_mapped`
- `request_num_decode_tokens`
- `request_num_ignored`

## Does this need Docker?

No, not usually.

- Your work is offline analysis of CSV output that Josh's system already wrote.
- If `sequence_metrics.csv` is visible on the host filesystem, you can do everything from a normal local Python environment.
- Docker is only relevant if the metrics file exists only inside the container and has not been written to a host-visible path.

The normal path for this work should be:

- run the server in Docker
- let Josh's metrics system write `sequence_metrics.csv`
- read that CSV from the host and plot it locally

## Directory and environment setup

The easiest setup for you is a small local `uv` virtual environment in the repo.

### 1. Create a plotting workspace

```bash
mkdir -p ~/repos/vattention/scripts/plotting
```

### 2. Create a local virtual environment with `uv`

From the repo root:

```bash
cd ~/repos/vattention
uv venv .venv-londy
```

This creates a dedicated virtual environment at `.venv-londy`.

### 3. Activate the virtual environment

For `zsh` or `bash`:

```bash
source ~/repos/vattention/.venv-londy/bin/activate
```

After activation, `python` and `pip` should point into `.venv-londy`.

### 4. Install the plotting dependencies with `uv`

```bash
uv pip install pandas matplotlib numpy
```

These are enough for the first version of the plotting script.

### 5. Verify the environment before writing code

```bash
python -c "import pandas, matplotlib, numpy; print('ok')"
```

If this prints `ok`, the environment is ready.

### 6. Run the plotting script from the same environment

Example pattern:

```bash
source ~/repos/vattention/.venv-londy/bin/activate
python ~/repos/vattention/scripts/plotting/plot_context_vs_fragmentation.py --help
```

Why this setup is preferable:

- it avoids mixing plotting dependencies into the shared Docker runtime
- it keeps your work lightweight and independent from model-serving code
- it makes it easier to rerun analysis without touching the server setup

## Step 1: Load and validate metrics

Create `scripts/plotting/plot_context_vs_fragmentation.py`.

```python
#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {
    "Request Id",
    "request_num_prefill_tokens",
    "kv_fragmentation_percent",
}


def load_metrics(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "Missing required columns in sequence_metrics.csv: "
            + ", ".join(sorted(missing))
        )
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()

    df = load_metrics(args.input)
    print("Loaded rows:", len(df))
    print("Columns:", sorted(df.columns))


if __name__ == "__main__":
    main()
```

Why this step matters:

- Fails fast if Josh's columns are missing
- Gives an immediate schema sanity check before plotting

## Step 2: Clean and filter rows for plotting

Add a cleaning function to handle NaNs, ignored requests, and invalid values.

```python
import numpy as np


def clean_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    # Keep only relevant columns (plus optional ones)
    keep_cols = [
        c for c in [
            "Request Id",
            "request_num_prefill_tokens",
            "kv_fragmentation_percent",
            "request_num_ignored",
            "request_num_decode_tokens",
            "kv_blocks_mapped",
        ]
        if c in work.columns
    ]
    work = work[keep_cols]

    # Remove ignored requests if field exists
    if "request_num_ignored" in work.columns:
        work = work[work["request_num_ignored"] == 0]

    # Force numeric types and drop invalid rows
    work["request_num_prefill_tokens"] = pd.to_numeric(
        work["request_num_prefill_tokens"], errors="coerce"
    )
    work["kv_fragmentation_percent"] = pd.to_numeric(
        work["kv_fragmentation_percent"], errors="coerce"
    )
    work = work.dropna(subset=["request_num_prefill_tokens", "kv_fragmentation_percent"])

    # Keep physically meaningful ranges
    work = work[work["request_num_prefill_tokens"] > 0]
    work = work[(work["kv_fragmentation_percent"] >= 0) & (work["kv_fragmentation_percent"] <= 100)]

    return work
```

Why this step matters:

- Removes noisy records that can distort regression/trend lines
- Ensures axis values are interpretable

## Step 3: Create the main scatter plot

Add a plotting function.

```python
import matplotlib.pyplot as plt


def plot_scatter(df: pd.DataFrame, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), dpi=140)

    ax.scatter(
        df["request_num_prefill_tokens"],
        df["kv_fragmentation_percent"],
        alpha=0.75,
        s=28,
        edgecolors="none",
    )

    ax.set_title(title)
    ax.set_xlabel("Context Length (prefill tokens)")
    ax.set_ylabel("Fragmentation (%)")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    # Optional: log x-axis when context lengths span wide range
    # ax.set_xscale("log")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
```

Why this step matters:

- Scatter directly shows per-request variability
- Best first plot for debugging and trend discovery

## Step 4: Add binned trend line for readability

For reports, a binned mean trend is often easier to read than raw scatter alone.

```python

def add_binned_trend(df: pd.DataFrame, ax, bins: int = 20) -> None:
    binned = df.copy()
    binned["ctx_bin"] = pd.cut(
        binned["request_num_prefill_tokens"],
        bins=bins,
        duplicates="drop",
    )

    trend = (
        binned.groupby("ctx_bin", observed=True)
        .agg(
            ctx_mid=("request_num_prefill_tokens", "median"),
            frag_mean=("kv_fragmentation_percent", "mean"),
            frag_std=("kv_fragmentation_percent", "std"),
            n=("kv_fragmentation_percent", "size"),
        )
        .dropna(subset=["ctx_mid", "frag_mean"])
        .sort_values("ctx_mid")
    )

    ax.plot(trend["ctx_mid"], trend["frag_mean"], linewidth=2.0, label="Binned mean")

    # Optional uncertainty band
    if "frag_std" in trend.columns:
        lower = trend["frag_mean"] - trend["frag_std"].fillna(0)
        upper = trend["frag_mean"] + trend["frag_std"].fillna(0)
        ax.fill_between(trend["ctx_mid"], lower, upper, alpha=0.15, label="±1 std")

    ax.legend()
```

Use it in plotting:

```python
fig, ax = plt.subplots(figsize=(9, 6), dpi=140)
ax.scatter(df["request_num_prefill_tokens"], df["kv_fragmentation_percent"], alpha=0.35, s=20)
add_binned_trend(df, ax, bins=20)
ax.set_xlabel("Context Length (prefill tokens)")
ax.set_ylabel("Fragmentation (%)")
ax.set_title("Context Length vs Fragmentation")
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
fig.tight_layout()
```

## Step 5: Save summary table for downstream analysis

Add compact aggregate outputs:

```python

def write_summary(df: pd.DataFrame, out_csv: Path) -> None:
    summary = pd.DataFrame(
        {
            "n_requests": [len(df)],
            "min_context": [df["request_num_prefill_tokens"].min()],
            "max_context": [df["request_num_prefill_tokens"].max()],
            "mean_fragmentation": [df["kv_fragmentation_percent"].mean()],
            "median_fragmentation": [df["kv_fragmentation_percent"].median()],
            "p90_fragmentation": [df["kv_fragmentation_percent"].quantile(0.90)],
        }
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
```

Why this step matters:

- Produces quick numeric checkpoints for report text
- Helps compare models/runs without opening figures each time

## Step 6: Full runnable script template

```python
#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

REQUIRED_COLUMNS = {"Request Id", "request_num_prefill_tokens", "kv_fragmentation_percent"}


def load_metrics(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    return df


def clean_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "request_num_ignored" in work.columns:
        work = work[work["request_num_ignored"] == 0]
    work["request_num_prefill_tokens"] = pd.to_numeric(work["request_num_prefill_tokens"], errors="coerce")
    work["kv_fragmentation_percent"] = pd.to_numeric(work["kv_fragmentation_percent"], errors="coerce")
    work = work.dropna(subset=["request_num_prefill_tokens", "kv_fragmentation_percent"])
    work = work[work["request_num_prefill_tokens"] > 0]
    work = work[(work["kv_fragmentation_percent"] >= 0) & (work["kv_fragmentation_percent"] <= 100)]
    return work


def add_binned_trend(df: pd.DataFrame, ax, bins: int = 20) -> None:
    binned = df.copy()
    binned["ctx_bin"] = pd.cut(binned["request_num_prefill_tokens"], bins=bins, duplicates="drop")
    trend = (
        binned.groupby("ctx_bin", observed=True)
        .agg(
            ctx_mid=("request_num_prefill_tokens", "median"),
            frag_mean=("kv_fragmentation_percent", "mean"),
            frag_std=("kv_fragmentation_percent", "std"),
        )
        .dropna(subset=["ctx_mid", "frag_mean"])
        .sort_values("ctx_mid")
    )
    ax.plot(trend["ctx_mid"], trend["frag_mean"], linewidth=2.0, label="Binned mean")
    lower = trend["frag_mean"] - trend["frag_std"].fillna(0)
    upper = trend["frag_mean"] + trend["frag_std"].fillna(0)
    ax.fill_between(trend["ctx_mid"], lower, upper, alpha=0.15, label="±1 std")


def write_summary(df: pd.DataFrame, out_csv: Path) -> None:
    summary = pd.DataFrame(
        {
            "n_requests": [len(df)],
            "min_context": [df["request_num_prefill_tokens"].min()],
            "max_context": [df["request_num_prefill_tokens"].max()],
            "mean_fragmentation": [df["kv_fragmentation_percent"].mean()],
            "median_fragmentation": [df["kv_fragmentation_percent"].median()],
            "p90_fragmentation": [df["kv_fragmentation_percent"].quantile(0.90)],
        }
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot context length vs fragmentation")
    parser.add_argument("--input", type=Path, required=True, help="Path to sequence_metrics.csv")
    parser.add_argument("--out-plot", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--out-summary", type=Path, required=True, help="Output summary CSV path")
    parser.add_argument("--title", type=str, default="Context Length vs Fragmentation")
    parser.add_argument("--bins", type=int, default=20)
    args = parser.parse_args()

    raw = load_metrics(args.input)
    df = clean_for_plot(raw)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=140)
    ax.scatter(df["request_num_prefill_tokens"], df["kv_fragmentation_percent"], alpha=0.35, s=20, label="Requests")
    add_binned_trend(df, ax, bins=args.bins)
    ax.set_xlabel("Context Length (prefill tokens)")
    ax.set_ylabel("Fragmentation (%)")
    ax.set_title(args.title)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend()
    fig.tight_layout()

    args.out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_plot)
    plt.close(fig)

    write_summary(df, args.out_summary)

    print(f"Plotted {len(df)} requests")
    print(f"Plot: {args.out_plot}")
    print(f"Summary: {args.out_summary}")


if __name__ == "__main__":
    main()
```

## Step 7: Example commands

```bash
python ~/repos/vattention/scripts/plotting/plot_context_vs_fragmentation.py \
  --input /tmp/vattention/vattn-anodyine/sequence_metrics.csv \
  --out-plot /tmp/vattention/vattn-anodyine/plots/context_vs_fragmentation.png \
  --out-summary /tmp/vattention/vattn-anodyine/plots/context_vs_fragmentation_summary.csv \
  --title "Yi-6B: Context Length vs Fragmentation" \
  --bins 16
```

## Quality checklist before sharing results

- Required columns exist in CSV
- Number of plotted rows matches expectation
- No obvious out-of-range fragmentation values (<0 or >100)
- Plot title includes model name/run tag
- Summary CSV is saved with the figure

## First milestone

Deliver this first:

- one script that reads `sequence_metrics.csv`
- one scatter + trend plot (`context length` vs `fragmentation %`)
- one small summary CSV with aggregate stats

This is enough to unblock comparison across models/runs.
