#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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


def clean_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    if "request_num_ignored" in work.columns:
        ignored = pd.to_numeric(work["request_num_ignored"], errors="coerce").fillna(0)
        work = work[ignored == 0]

    work["request_num_prefill_tokens"] = pd.to_numeric(
        work["request_num_prefill_tokens"], errors="coerce"
    )
    work["kv_fragmentation_percent"] = pd.to_numeric(
        work["kv_fragmentation_percent"], errors="coerce"
    )

    work = work.dropna(
        subset=["request_num_prefill_tokens", "kv_fragmentation_percent"]
    )
    work = work[work["request_num_prefill_tokens"] > 0]
    work = work[
        (work["kv_fragmentation_percent"] >= 0)
        & (work["kv_fragmentation_percent"] <= 100)
    ]
    return work.sort_values("request_num_prefill_tokens")


def add_binned_trend(df: pd.DataFrame, ax, bins: int) -> None:
    effective_bins = max(1, min(bins, len(df)))
    binned = df.copy()
    binned["ctx_bin"] = pd.cut(
        binned["request_num_prefill_tokens"],
        bins=effective_bins,
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

    if trend.empty:
        return

    ax.plot(
        trend["ctx_mid"],
        trend["frag_mean"],
        linewidth=2.0,
        color="#cf5c36",
        label="Binned mean",
    )

    lower = trend["frag_mean"] - trend["frag_std"].fillna(0)
    upper = trend["frag_mean"] + trend["frag_std"].fillna(0)
    ax.fill_between(
        trend["ctx_mid"],
        lower,
        upper,
        alpha=0.15,
        color="#cf5c36",
        label="\u00b11 std",
    )


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


def plot_raw_sawtooth(df: pd.DataFrame, out_plot: Path, title: str) -> None:
    has_blocks = "kv_blocks_mapped" in df.columns
    if has_blocks:
        df = df.copy()
        df["kv_blocks_mapped"] = pd.to_numeric(df["kv_blocks_mapped"], errors="coerce")

    if has_blocks:
        fig, (ax_top, ax_bottom) = plt.subplots(
            2,
            1,
            figsize=(10, 7),
            dpi=140,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
    else:
        fig, ax_top = plt.subplots(figsize=(10, 6), dpi=140)
        ax_bottom = None

    ax_top.plot(
        df["request_num_prefill_tokens"],
        df["kv_fragmentation_percent"],
        linewidth=1.8,
        color="#cf5c36",
        alpha=0.95,
        zorder=2,
        label="Fragmentation trajectory",
    )
    scatter = ax_top.scatter(
        df["request_num_prefill_tokens"],
        df["kv_fragmentation_percent"],
        c=df["kv_blocks_mapped"] if has_blocks else "#1f6feb",
        cmap="viridis" if has_blocks else None,
        s=42,
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
        label="Requests",
    )

    ax_top.set_title(title)
    ax_top.set_ylabel("Fragmentation (%)")
    ax_top.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax_top.legend(loc="upper right")

    if has_blocks:
        colorbar = fig.colorbar(scatter, ax=ax_top, pad=0.01)
        colorbar.set_label("KV blocks mapped")

        step_df = (
            df.dropna(subset=["kv_blocks_mapped"])
            .drop_duplicates(subset=["request_num_prefill_tokens", "kv_blocks_mapped"])
            .sort_values("request_num_prefill_tokens")
        )
        ax_bottom.step(
            step_df["request_num_prefill_tokens"],
            step_df["kv_blocks_mapped"],
            where="post",
            linewidth=2.0,
            color="#1f6feb",
        )
        ax_bottom.scatter(
            step_df["request_num_prefill_tokens"],
            step_df["kv_blocks_mapped"],
            color="#1f6feb",
            s=28,
            zorder=3,
        )
        ax_bottom.set_xlabel("Context Length (prefill tokens)")
        ax_bottom.set_ylabel("Blocks")
        ax_bottom.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    else:
        ax_top.set_xlabel("Context Length (prefill tokens)")

    fig.tight_layout()
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot)
    plt.close(fig)


def plot_context_vs_fragmentation(
    df: pd.DataFrame, out_plot: Path, title: str, bins: int
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), dpi=140)
    ax.scatter(
        df["request_num_prefill_tokens"],
        df["kv_fragmentation_percent"],
        alpha=0.75,
        s=36,
        color="#1f6feb",
        edgecolors="none",
        label="Requests",
    )
    add_binned_trend(df, ax, bins=bins)
    ax.set_title(title)
    ax.set_xlabel("Context Length (prefill tokens)")
    ax.set_ylabel("Fragmentation (%)")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend()
    fig.tight_layout()

    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot context length vs fragmentation")
    parser.add_argument("--input", type=Path, required=True, help="Path to sequence_metrics.csv")
    parser.add_argument("--out-plot", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--out-summary", type=Path, required=True, help="Output summary CSV path")
    parser.add_argument("--title", type=str, default="Context Length vs Fragmentation")
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument(
        "--plot-style",
        choices=("raw_sawtooth", "scatter_binned"),
        default="raw_sawtooth",
        help="Plot the raw sawtooth pattern directly, or use the earlier scatter+binned trend view.",
    )
    args = parser.parse_args()

    raw = load_metrics(args.input)
    df = clean_for_plot(raw)
    if df.empty:
        raise RuntimeError("No valid rows remained after cleaning sequence_metrics.csv")

    if args.plot_style == "raw_sawtooth":
        plot_raw_sawtooth(df, args.out_plot, args.title)
    else:
        plot_context_vs_fragmentation(df, args.out_plot, args.title, args.bins)
    write_summary(df, args.out_summary)

    print(f"Plotted {len(df)} requests")
    print(f"Plot: {args.out_plot}")
    print(f"Summary: {args.out_summary}")


if __name__ == "__main__":
    main()
