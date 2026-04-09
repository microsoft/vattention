#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {
    "request_num_prefill_tokens",
    "kv_blocks_mapped",
    "kv_fragmentation_percent",
}


@dataclass
class SeriesSpec:
    input_path: Path
    config_path: Path
    label: str
    color: str


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


def parse_series_args(raw_series: list[str]) -> list[SeriesSpec]:
    if len(raw_series) < 2:
        raise ValueError("Expected at least two --series entries.")

    specs: list[SeriesSpec] = []
    for entry in raw_series:
        parts = entry.split("|")
        if len(parts) not in {3, 4}:
            raise ValueError(
                "--series entries must have the form "
                "'<metrics_csv>|<config_yml>|<label>[|<color>]'"
            )
        input_path = Path(parts[0]).expanduser()
        config_path = Path(parts[1]).expanduser()
        label = parts[2]
        color = parts[3] if len(parts) == 4 else ""
        specs.append(
            SeriesSpec(
                input_path=input_path,
                config_path=config_path,
                label=label,
                color=color,
            )
        )
    return specs


def assign_default_colors(specs: list[SeriesSpec]) -> list[SeriesSpec]:
    default_palette = ["#cf5c36", "#1f6feb", "#2da44e", "#8250df", "#bf8700", "#d1242f"]
    assigned: list[SeriesSpec] = []
    for idx, spec in enumerate(specs):
        color = spec.color or default_palette[idx % len(default_palette)]
        assigned.append(
            SeriesSpec(
                input_path=spec.input_path,
                config_path=spec.config_path,
                label=spec.label,
                color=color,
            )
        )
    return assigned


def load_series(specs: list[SeriesSpec]) -> list[tuple[SeriesSpec, pd.DataFrame]]:
    loaded: list[tuple[SeriesSpec, pd.DataFrame]] = []
    for spec in specs:
        cfg = load_top_level_yaml(spec.config_path)
        block_size = int(cfg["model_block_size"])
        df = add_cache_byte_columns(
            clean_metrics(load_metrics(spec.input_path)),
            block_size_bytes=block_size,
        )
        loaded.append((spec, df))
    return loaded


def plot_comparison(
    *,
    series_data: list[tuple[SeriesSpec, pd.DataFrame]],
    out_plot: Path,
    title: str,
) -> None:
    fig, ax_alloc = plt.subplots(
        1,
        1,
        figsize=(11, 5.8),
        dpi=140,
    )

    for spec, df in series_data:
        ax_alloc.step(
            df["request_num_prefill_tokens"],
            df["allocated_cache_mib"],
            where="post",
            linewidth=2.0,
            color=spec.color,
            label=spec.label,
        )
        ax_alloc.scatter(
            df["request_num_prefill_tokens"],
            df["allocated_cache_mib"],
            s=20,
            color=spec.color,
            alpha=0.9,
        )

    ax_alloc.set_title(title)
    ax_alloc.set_xlabel("Context Length (prefill tokens)")
    ax_alloc.set_ylabel("Allocated Cache (MiB)")
    ax_alloc.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax_alloc.legend(loc="upper left")

    fig.tight_layout()
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot)
    plt.close(fig)


def write_summary(
    *,
    series_data: list[tuple[SeriesSpec, pd.DataFrame]],
    out_csv: Path,
) -> None:
    rows = []
    for spec, df in series_data:
        rows.append(
            {
                "label": spec.label,
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
        description="Compare allocated cache bytes vs context length across multiple model runs."
    )
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        help=(
            "Series in the form "
            "'<metrics_csv>|<config_yml>|<label>[|<color>]'. "
            "Pass this flag multiple times."
        ),
    )
    parser.add_argument("--out-plot", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument(
        "--title",
        type=str,
        default="Allocated Cache Bytes vs Context Length",
    )
    args = parser.parse_args()

    specs = assign_default_colors(parse_series_args(args.series))
    series_data = load_series(specs)

    plot_comparison(
        series_data=series_data,
        out_plot=args.out_plot,
        title=args.title,
    )
    write_summary(
        series_data=series_data,
        out_csv=args.out_summary,
    )

    print(f"Plot: {args.out_plot}")
    print(f"Summary: {args.out_summary}")


if __name__ == "__main__":
    main()
