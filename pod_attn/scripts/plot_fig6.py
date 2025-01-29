import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import argparse

#parser = argparse.ArgumentParser(description='Plotting script for FlashAttention')
#parser.add_argument('--sysname', type=str, default='pod', help='Input CSV file')
#args = parser.parse_args()

sysname = 'POD' #if args.sysname == 'pod' else 'SANGAM'

plt.rcParams.update({'font.size': 28})
plt.rcParams.update({'font.family': 'Sans Serif'})

legends = {
    'fa_serial': 'FA_Serial',
    'fa_streams': 'FA_Streams',
    'fi_serial': 'FI_Serial',
    'fi_fused': 'FI_Batched',
    'Hfuse': 'FA_HFuse',
    'fa_fused': sysname
}

colors = {
    'fa_serial': 'tab:orange',
    'fa_streams': 'tab:brown',
    'fi_serial': 'cadetblue',
    'fi_fused': 'tab:purple',
    'Hfuse': 'tab:red',
    'fa_fused': 'tab:green'
}

linestyles = {
    'fa_serial': '-',
    'fa_streams': '--',
    'fi_serial': '-.',
    'fi_fused': ':',
    'Hfuse': '-.',
    'fa_fused': ':'
}

markerstyles = {
    'fa_serial': 'o',
    'fa_streams': 's',
    'fi_serial': 'D',
    'fi_fused': 'x',
    'Hfuse': 'x',
    'fa_fused': 'D'
}

bs_no_quantization = 54
bs_quantization = 55

configs = ['fa_serial', 'fa_streams', 'Hfuse', 'fa_fused']
def plot_runtime(ax, df, y_label):
    x_ticks = np.arange(len(df['chunk_id']))
    handles = []
    labels = []
    for i, cfg in enumerate(configs):
        line, = ax.plot(x_ticks, df[cfg], label=legends[cfg], color=colors[cfg], linestyle=linestyles[cfg], linewidth=3, marker=markerstyles[cfg], markersize=6)
        handles.append(line)
        labels.append(legends[cfg])

    x_label = 'Chunk Id'
    ax.set_xlabel(x_label, fontsize=24, fontweight='bold')
    if y_label:
        ax.set_ylabel('Runtime (ms)', fontsize=28, fontweight='bold')
    x_tick_labels = df['chunk_id'].unique()
    x_tick_labels = [label if i % 4 == 0 else '' for i, label in enumerate(x_tick_labels)]
    x_tick_labels[-1] = df['chunk_id'].max()
    ax.set_xticks(ticks=x_ticks)
    ax.set_xticklabels(labels=x_tick_labels, fontsize=32)
    ax.set_yticks(np.arange(0, 3.1, 0.5))
    ax.grid(axis='y')
    return handles, labels

#df = pd.read_csv('chunking_fusedattn_overlap_quantization_a100_tp8.csv', sep=';')
import sys
infile = sys.argv[1]
outfile = sys.argv[2]


df = pd.read_csv(infile, sep=';')
df_no_quantization = df[df['bs'] == bs_no_quantization]
df_quantization = df[df['bs'] == bs_quantization]

fig, axs = plt.subplots(1, 2, figsize=(18, 8))
handles, labels = plot_runtime(axs[0], df_no_quantization, y_label=True)
axs[0].set_title(f'w/o quantization (d_bs={bs_no_quantization})', fontsize=28, fontweight='bold')
plot_runtime(axs[1], df_quantization, y_label=False)
axs[1].set_title(f'w/ quantization (d_bs={bs_quantization})', fontsize=28, fontweight='bold')

fig.legend(handles, labels, loc='upper center', ncol=6, fontsize=28, frameon=True)

plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig(outfile, bbox_inches='tight', pad_inches=0)
sys.exit()