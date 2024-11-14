import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

helpers = os.path.dirname(os.path.abspath(__file__))
src = os.path.dirname(helpers)
plots = os.path.join(src, "plots")

colors = {
    'fa': 'lightgreen',
    'fa_paged': 'cadetblue',
    'fi': 'chocolate',
    'fi_paged': 'wheat',
}

hatches = {
    'fa': '',
    'fa_paged': '\\\\',
    'fi': '-',
    'fi_paged': '//',
}
opacity = 0.65

df = pd.read_csv(os.path.join(src, "logs/figure_2.csv"))

context_lens = [1024, 2048, 4096, 8192, 16384, 32768]
context_lens_ticks = ["1K", "2K", "4K", "8K", "16K", "32K"]

def plot_figure(df):
    fig, ax = plt.subplots(figsize=(15, 7))
    width = 0.15
    x_pos = np.arange(len(context_lens_ticks))
    ax.bar(x_pos - 1.5*width, df["fa"], width, label="FA", color=colors["fa"], hatch=hatches["fa"], alpha=opacity, edgecolor='black')
    bars1 = ax.bar(x_pos - 0.5*width, df["fa_paged"], width, label="FA_Paged", color=colors["fa_paged"], hatch=hatches["fa_paged"], alpha=opacity, edgecolor='black')
    perf = df["fa_paged"] / df["fa"]
    for i, v in enumerate(perf):
        ax.text(i - 0.5*width, v + 0.02, f"{v:.2f}x", color='black', ha='center', va='bottom', fontsize=20, rotation=90)
    ax.bar(x_pos + 0.5*width, df["fi"], width, label="FI", color=colors["fi"], hatch=hatches["fi"], alpha=opacity, edgecolor='black')
    bars2 = ax.bar(x_pos + 1.5*width, df["fi_paged"], width, label="FI_Paged", color=colors["fi_paged"], hatch=hatches["fi_paged"], alpha=opacity, edgecolor='black')
    perf2 = df["fi_paged"] / df["fi"]
    positions = df["fi_paged"] / df["fa"]
    for i, v in enumerate(perf2):
        ax.text(i + 1.5*width, df["fi_paged"].iloc[i] + 0.02, f"{v:.2f}x", color='black', ha='center', va='bottom', fontsize=22, rotation=90)

    plt.xticks(x_pos, context_lens_ticks, fontsize=28)
    plt.yticks(np.arange(0, 1.71, 0.2), fontsize=28)
    ax.grid(axis='y', linestyle='--')
    ax.set_xlabel('Context Length', fontweight='bold', fontsize=28)
    plt.ylabel("Normalized Runtime", fontweight='bold', fontsize=28)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=4, frameon=True, fontsize=28)
    plt.tight_layout()
    os.makedirs(plots, exist_ok=True)
    plt.savefig(os.path.join(plots, "figure_2.pdf"))

df['fa'] = 1
df['fa_paged'] = df['fa_paged_latency'] / df['fa_latency']
df['fi'] = df['fi_latency'] / df['fa_latency']
df['fi_paged'] = df['fi_paged_latency'] / df['fa_latency']
plot_figure(df)