from typing import Optional
import argparse
import random
import time
import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plots = os.path.join(src, "plots")

colors = {
    '16': 'lightgreen',
    '32': 'cadetblue',
    '64': 'chocolate',
    '128': 'wheat',
}

hatches = {
    '16': '',
    '32': '\\\\',
    '64': '-',
    '128': '//',
}
opacity = 0.65

batch_sizes=[1, 2, 4, 8, 16]
x_ticks = ["1*16K", "2*16K", "4*16K", "8*16K", "16*16K"]
def plot_figure(df):
    fig, ax = plt.subplots(figsize=(15, 7))
    width = 0.15
    df["16_norm"] = df["16"] / df["16"]
    df["32_norm"] = df["32"] / df["16"]
    df["64_norm"] = df["64"] / df["16"]
    df["128_norm"] = df["128"] / df["16"]
    y_max = df[["16_norm", "32_norm", "64_norm", "128_norm"]].max().max()
    x_pos = np.arange(len(batch_sizes))
    ax.bar(x_pos - 1.5*width, df["16_norm"], width, label="16", color=colors["16"], hatch=hatches["16"], alpha=opacity, edgecolor='black')
    bars1 = ax.bar(x_pos - 0.5*width, df["32_norm"], width, label="32", color=colors["32"], hatch=hatches["32"], alpha=opacity, edgecolor='black')
    perf1 = df["32_norm"] #/ df["16"]
    for i, v in enumerate(perf1):
        ax.text(i - 0.5*width, df["32_norm"].iloc[i] + 0.01, f"{v:.2f}x", color='black', ha='center', va='bottom', fontsize=20, rotation=90)
    bars2 = ax.bar(x_pos + 0.5*width, df["64_norm"], width, label="64", color=colors["64"], hatch=hatches["64"], alpha=opacity, edgecolor='black')
    perf2 = df["64_norm"] #/ df["16"]
    for i, v in enumerate(perf2):
        ax.text(i + 0.5*width, df["64_norm"].iloc[i] + 0.01, f"{v:.2f}x", color='black', ha='center', va='bottom', fontsize=20, rotation=90)
    bars3 = ax.bar(x_pos + 1.5*width, df["128_norm"], width, label="128", color=colors["128"], hatch=hatches["128"], alpha=opacity, edgecolor='black')
    perf3 = df["128_norm"] #/ df["16"]
    for i, v in enumerate(perf3):
        ax.text(i + 1.5*width, df["128_norm"].iloc[i] + 0.01, f"{v:.2f}x", color='black', ha='center', va='bottom', fontsize=20, rotation=90)

    plt.xticks(x_pos, x_ticks, fontsize=28)
    plt.yticks(np.arange(0, y_max * 2, 0.5), fontsize=28)
    ax.grid(axis='y', linestyle='--')
    ax.set_xlabel('Batch Size * Context Length', fontweight='bold', fontsize=28)
    #plt.ylabel("Latency (ms)", fontweight='bold', fontsize=28)
    plt.ylabel("Normalized Runtime", fontweight='bold', fontsize=28)
    #plt.legend(loc='upper left', bbox_to_anchor=(0.5, 1.25), ncol=4, frameon=True, fontsize=28)
    plt.legend(loc='upper center', ncol=4, frameon=True, fontsize=28, title="Block Size", title_fontsize="28")
    plt.tight_layout()
    os.makedirs(plots, exist_ok=True)
    plt.savefig(os.path.join(plots, "figure_3.pdf"))

df = pd.read_csv(os.path.join(src, "logs/figure_3.csv"))
plot_figure(df)