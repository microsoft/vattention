# Londy Plotting Plan: Context Length vs Fragmentation

This plan is for Londy to read the metrics produced by Josh's pipeline and generate publication-ready plots of context length vs fragmentation percentage.

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

## Directory and environment setup

Create a small plotting workspace under `scripts/` or a dedicated analysis folder.

Example:

```bash
mkdir -p /home/anodyine/repos/vattention/scripts/plotting
```

Install dependencies in your active environment if needed:

```bash
pip install pandas matplotlib numpy
```

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
python /home/anodyine/repos/vattention/scripts/plotting/plot_context_vs_fragmentation.py \
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
