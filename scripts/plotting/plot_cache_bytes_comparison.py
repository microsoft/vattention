#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {
    "request_num_prefill_tokens",
    "kv_blocks_mapped",
    "kv_fragmentation_percent",
}


def load_top_level_yaml(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    parsed: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip().strip("'").strip('"')
    return parsed


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


def clean_metrics(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    if "request_num_ignored" in work.columns:
        ignored = pd.to_numeric(work["request_num_ignored"], errors="coerce").fillna(0)
        work = work[ignored == 0]

    for column in REQUIRED_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    work = work.dropna(subset=list(REQUIRED_COLUMNS))
    work = work[work["request_num_prefill_tokens"] > 0]
    work = work[work["kv_blocks_mapped"] > 0]
    work = work[
        (work["kv_fragmentation_percent"] >= 0)
        & (work["kv_fragmentation_percent"] <= 100)
    ]
    return work.sort_values("request_num_prefill_tokens")


def add_cache_byte_columns(df: pd.DataFrame, *, block_size_bytes: int) -> pd.DataFrame:
    work = df.copy()
    work["allocated_cache_bytes"] = work["kv_blocks_mapped"] * block_size_bytes
    work["allocated_cache_mib"] = work["allocated_cache_bytes"] / (1024 * 1024)
    work["waste_cache_bytes"] = (
        work["allocated_cache_bytes"] * work["kv_fragmentation_percent"] / 100.0
    )
    work["waste_cache_mib"] = work["waste_cache_bytes"] / (1024 * 1024)
    return work


def plot_comparison(
    *,
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_label: str,
    right_label: str,
    out_plot: Path,
    title: str,
) -> None:
    fig, (ax_alloc, ax_waste) = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        dpi=140,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )

    series_specs = [
        (left_df, left_label, "#cf5c36"),
        (right_df, right_label, "#1f6feb"),
    ]

    for df, label, color in series_specs:
        ax_alloc.step(
            df["request_num_prefill_tokens"],
            df["allocated_cache_mib"],
            where="post",
            linewidth=2.0,
            color=color,
            label=label,
        )
        ax_alloc.scatter(
            df["request_num_prefill_tokens"],
            df["allocated_cache_mib"],
            s=26,
            color=color,
            alpha=0.9,
        )

        ax_waste.plot(
            df["request_num_prefill_tokens"],
            df["waste_cache_mib"],
            linewidth=1.8,
            color=color,
            alpha=0.95,
            label=label,
        )
        ax_waste.scatter(
            df["request_num_prefill_tokens"],
            df["waste_cache_mib"],
            s=24,
            color=color,
            alpha=0.9,
        )

    ax_alloc.set_title(title)
    ax_alloc.set_ylabel("Allocated Cache (MiB)")
    ax_alloc.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax_alloc.legend(loc="upper left")

    ax_waste.set_xlabel("Context Length (prefill tokens)")
    ax_waste.set_ylabel("Estimated Waste (MiB)")
    ax_waste.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    fig.tight_layout()
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot)
    plt.close(fig)


def write_summary(
    *,
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_label: str,
    right_label: str,
    out_csv: Path,
) -> None:
    rows = []
    for df, label in [(left_df, left_label), (right_df, right_label)]:
        rows.append(
            {
                "label": label,
                "n_requests": len(df),
                "min_context": df["request_num_prefill_tokens"].min(),
                "max_context": df["request_num_prefill_tokens"].max(),
                "max_allocated_cache_mib": df["allocated_cache_mib"].max(),
                "max_waste_cache_mib": df["waste_cache_mib"].max(),
                "mean_waste_cache_mib": df["waste_cache_mib"].mean(),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare allocated cache bytes vs context length across two model runs."
    )
    parser.add_argument("--left-input", type=Path, required=True)
    parser.add_argument("--left-config", type=Path, required=True)
    parser.add_argument("--left-label", type=str, required=True)
    parser.add_argument("--right-input", type=Path, required=True)
    parser.add_argument("--right-config", type=Path, required=True)
    parser.add_argument("--right-label", type=str, required=True)
    parser.add_argument("--out-plot", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument(
        "--title",
        type=str,
        default="Allocated Cache Bytes vs Context Length",
    )
    args = parser.parse_args()

    left_cfg = load_top_level_yaml(args.left_config)
    right_cfg = load_top_level_yaml(args.right_config)
    left_block_size = int(left_cfg["model_block_size"])
    right_block_size = int(right_cfg["model_block_size"])

    left_df = add_cache_byte_columns(
        clean_metrics(load_metrics(args.left_input)),
        block_size_bytes=left_block_size,
    )
    right_df = add_cache_byte_columns(
        clean_metrics(load_metrics(args.right_input)),
        block_size_bytes=right_block_size,
    )

    plot_comparison(
        left_df=left_df,
        right_df=right_df,
        left_label=args.left_label,
        right_label=args.right_label,
        out_plot=args.out_plot,
        title=args.title,
    )
    write_summary(
        left_df=left_df,
        right_df=right_df,
        left_label=args.left_label,
        right_label=args.right_label,
        out_csv=args.out_summary,
    )

    print(f"Plot: {args.out_plot}")
    print(f"Summary: {args.out_summary}")


if __name__ == "__main__":
    main()
