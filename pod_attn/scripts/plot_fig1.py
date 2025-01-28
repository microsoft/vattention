import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

#import argparse
#parser = argparse.ArgumentParser(description='Plotting script for FlashAttention')
#parser.add_argument('--sysname', type=str, default='pod', help='Input CSV file')
#args = parser.parse_args()

sysname = 'POD-Attention' #if args.sysname == 'pod' else 'SANGAM'

plt.rcParams.update({'font.size': 28})
plt.rcParams.update({'font.family': 'Sans Serif'})

configs = ['sm-util', 'dram-util']

legends = {
    'sm-util': 'Compute Utilization',
    'dram-util': 'Mem BW Utilization'
}

colors = {
    'sm-util': 'green',
    'dram-util': 'chocolate'
}

linestyles = {
    'sm-util': '-',
    'dram-util': '--'
}

markerstyles = {
    'sm-util': 'o',
    'dram-util': 's'
}

titles = {
    'prefill': 'Prefill',
    'decode': 'Decode',
    'fused': sysname
}

p_cl = [1024, 2048, 4096, 8192, 16384]
p_x_ticks = ['1K', '2K', '4K', '8K', '16K']

d_bs = [16, 32, 64, 128, 256]

# fused_cfgs = ['(1K, 32)', '(2K, 64)', '(4K, 128)', '(8K, 256)']
fused_cfgs = ['C0', 'C1', 'C2']

def plot_utilization(df, figname, version, title):
    plt.figure(figsize=(10, 7))
    x_ticks = np.arange(len(df['cl'])) if version == 'prefill' else np.arange(len(df['bs']))
    for i, cfg in enumerate(configs):
        plt.plot(x_ticks, df[cfg], label=legends[cfg], color=colors[cfg], linestyle = linestyles[cfg], linewidth=4, marker=markerstyles[cfg], markersize=10)

    x_label = 'Context Length' if version == 'prefill' else \
                'Batch Size' if version == 'decode' else \
                'Hybrid Batch Config'
    plt.xlabel(x_label, fontsize=28, fontweight='bold')
    plt.ylabel('Utilization (%)', fontsize=28, fontweight='bold')
    x_tick_labels = p_x_ticks if version == 'prefill' else \
                    d_bs if version == 'decode' else \
                    fused_cfgs
    plt.xticks(ticks=x_ticks, labels=x_tick_labels, fontsize=28)
    plt.yticks(np.arange(0, 110, 20))
    plt.grid()
    l_pos = 'lower right' if version == 'fused' else 'best'
    plt.legend(loc=l_pos, fontsize=28)
    plt.title(titles[version], fontsize=28, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figname)

runtime_colors = {
    'fa_p': 'tab:orange',
    'fa_d': 'tab:orange',
    'fi_p': 'tab:cyan',
    'fi_d': 'tab:cyan',
    'fi_fused': 'tab:purple',
    'fa_fused': 'tab:green'
}

runtime_hatches = {
    'fa_p': '',
    'fa_d': '/',
    'fi_p': '',
    'fi_d': '\\',
    'fi_fused': '-',
    'fa_fused': ''
}

runtime_alpha = {
    'fa_p': 0.6,
    'fa_d': 0.6,
    'fi_p': 0.6,
    'fi_d': 0.6,
    'fi_fused': 0.6,
    'fa_fused': 0.6
}


hybrid_configs = ['C0', 'C1', 'C2']

sys_label = "POD" if sysname == 'POD-Attention' else "SANGAM"
def plot_runtime(df):
    fig, ax = plt.subplots(figsize=(10, 7))
    width = 0.20
    x_pos = np.arange(len(hybrid_configs))
    i = 0
    for cfg in hybrid_configs:
        row = df[df['config'] == cfg]
        ax.bar(x_pos[i] - 1.5 * width, row["fa_p"], width, color=runtime_colors['fa_p'], hatch=runtime_hatches['fa_p'], alpha=runtime_alpha['fa_p'], edgecolor='black')
        ax.bar(x_pos[i] - 1.5 * width, row["fa_d"], width, bottom=row["fa_p"], color=runtime_colors['fa_d'], hatch=runtime_hatches['fa_d'], alpha=runtime_alpha['fa_d'], edgecolor='black')
        ax.bar(x_pos[i] - 0.5 * width, row["fi_p"], width, color=runtime_colors['fi_p'], hatch=runtime_hatches['fi_p'], alpha=runtime_alpha['fi_p'], edgecolor='black')
        ax.bar(x_pos[i] - 0.5 * width, row["fi_d"], width, bottom=row["fi_p"], color=runtime_colors['fi_d'], hatch=runtime_hatches['fi_d'], alpha=runtime_alpha['fi_d'], edgecolor='black')
        ax.bar(x_pos[i] + 0.5 * width, row["fi_batched"], width, color=runtime_colors['fi_fused'], hatch=runtime_hatches['fi_fused'], alpha=runtime_alpha['fi_fused'], edgecolor='black')
        ax.bar(x_pos[i] + 1.5 * width, row["pod"], width, color=runtime_colors['fa_fused'], hatch=runtime_hatches['fa_fused'], alpha=runtime_alpha['fa_fused'], edgecolor='black')
        i += 1

    # Add legend entries only once
    ax.bar(0, 0, color=runtime_colors['fa_p'], label="FA_Prefill", hatch=runtime_hatches['fa_p'], alpha=runtime_alpha['fa_p'], edgecolor='black')
    ax.bar(0, 0, color=runtime_colors['fa_d'], label="FA_Decode", hatch=runtime_hatches['fa_d'], alpha=runtime_alpha['fa_d'], edgecolor='black')
    ax.bar(0, 0, color=runtime_colors['fi_p'], label="FI_Prefill", hatch=runtime_hatches['fi_p'], alpha=runtime_alpha['fi_p'], edgecolor='black')
    ax.bar(0, 0, color=runtime_colors['fi_d'], label="FI_Decode", hatch=runtime_hatches['fi_d'], alpha=runtime_alpha['fi_d'], edgecolor='black')
    ax.bar(0, 0, color=runtime_colors['fi_fused'], label="FI_Batched", hatch=runtime_hatches['fi_fused'], alpha=runtime_alpha['fi_fused'], edgecolor='black')
    ax.bar(0, 0, color=runtime_colors['fa_fused'], label=sys_label, hatch=runtime_hatches['fa_fused'], alpha=runtime_alpha['fa_fused'], edgecolor='black')

    plt.xticks(x_pos, fused_cfgs, fontsize=28)
    plt.yticks(np.arange(0, 10.1, 2), fontsize=32)
    ax.grid(axis='y', linestyle='--')
    ax.set_xlabel('Hybrid Batch Config', fontweight='bold', fontsize=28)
    plt.ylabel("Attention Time (ms)", fontweight='bold', fontsize=28)
    plt.legend(loc='upper left', ncol=1, frameon=True, fontsize=22)
    # plt.title(f"", fontweight='bold', fontsize=36)
    plt.tight_layout()
    plt.savefig(f"runtime.pdf")

def plot_runtime_normalized(df, outfile):
    fig, ax = plt.subplots(figsize=(11, 9))
    width = 0.20
    x_pos = np.arange(len(hybrid_configs))
    i = 0
    for cfg in hybrid_configs:
        row = df[df['config'] == cfg]
        scale = row["fa_p"] + row["fa_d"]
        ax.bar(x_pos[i] - 1 * width, row["fa_p"] / scale, width, color=runtime_colors['fa_p'], hatch=runtime_hatches['fa_p'], alpha=runtime_alpha['fa_p'], edgecolor='black')
        ax.bar(x_pos[i] - 1 * width, row["fa_d"] / scale, width, bottom=row["fa_p"] / scale, color=runtime_colors['fa_d'], hatch=runtime_hatches['fa_d'], alpha=runtime_alpha['fa_d'], edgecolor='black')
        ax.bar(x_pos[i] - 0 * width, row["fi_p"] / scale, width, color=runtime_colors['fi_p'], hatch=runtime_hatches['fi_p'], alpha=runtime_alpha['fi_p'], edgecolor='black')
        ax.bar(x_pos[i] - 0 * width, row["fi_d"] / scale, width, bottom=row["fi_p"] / scale, color=runtime_colors['fi_d'], hatch=runtime_hatches['fi_d'], alpha=runtime_alpha['fi_d'], edgecolor='black')
        #ax.bar(x_pos[i] + 0.5 * width, row["fi_batched"] / scale, width, color=runtime_colors['fi_fused'], hatch=runtime_hatches['fi_fused'], alpha=runtime_alpha['fi_fused'], edgecolor='black')
        ax.bar(x_pos[i] + 1 * width, row["pod"] / scale, width, color=runtime_colors['fa_fused'], hatch=runtime_hatches['fa_fused'], alpha=runtime_alpha['fa_fused'], edgecolor='black')
        i += 1

    # Add legend entries only once
    ax.bar(0, 0, color=runtime_colors['fa_p'], label="FA_Prefill", hatch=runtime_hatches['fa_p'], alpha=runtime_alpha['fa_p'], edgecolor='black')
    ax.bar(0, 0, color=runtime_colors['fa_d'], label="FA_Decode", hatch=runtime_hatches['fa_d'], alpha=runtime_alpha['fa_d'], edgecolor='black')
    ax.bar(0, 0, color=runtime_colors['fi_p'], label="FI_Prefill", hatch=runtime_hatches['fi_p'], alpha=runtime_alpha['fi_p'], edgecolor='black')
    ax.bar(0, 0, color=runtime_colors['fi_d'], label="FI_Decode", hatch=runtime_hatches['fi_d'], alpha=runtime_alpha['fi_d'], edgecolor='black')
    #ax.bar(0, 0, color=runtime_colors['fi_fused'], label="FI_Batched", hatch=runtime_hatches['fi_fused'], alpha=runtime_alpha['fi_fused'], edgecolor='black')
    ax.bar(0, 0, color=runtime_colors['fa_fused'], label=sys_label, hatch=runtime_hatches['fa_fused'], alpha=runtime_alpha['fa_fused'], edgecolor='black')

    plt.xticks(x_pos, fused_cfgs, fontsize=28)
    plt.yticks(np.arange(0, 1.3, 0.2), fontsize=32)
    ax.grid(axis='y', linestyle='--')
    ax.set_xlabel('Hybrid Batch Config', fontweight='bold', fontsize=28)
    plt.ylabel("Normalized Runtime", fontweight='bold', fontsize=28)
    #plt.legend(loc='upper left', ncol=2, frameon=True, fontsize=22)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=True, fontsize=22)
    # plt.title(f"", fontweight='bold', fontsize=36)
    plt.tight_layout()
    plt.savefig(outfile)

import sys
indir = sys.argv[1]
outdir = sys.argv[2]

df_prefill = pd.read_csv(indir + 'prefill.csv', sep=';')
df_decode = pd.read_csv(indir + 'decode.csv', sep=';')
df_fused = pd.read_csv(indir + 'fused.csv', sep=';')
df_runtime = pd.read_csv(indir + 'runtime.csv', sep=';')

plot_utilization(df_prefill, outdir + 'fig1a.png', 'prefill', '')
plot_utilization(df_decode, outdir + 'fig1b.png', 'decode', '')
plot_utilization(df_fused, outdir + 'fig1c.png', 'fused', '')
#plot_runtime(df_runtime)
plot_runtime_normalized(df_runtime, outdir + 'fig1d.png')