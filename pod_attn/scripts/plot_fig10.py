import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

import sys
indir = sys.argv[1]
outdir = sys.argv[2]

plt.rcParams.update({'font.size': 28})
plt.rcParams.update({'font.family': 'Sans Serif'})

legends = {
    'sm-util': 'SM Utilization',
    'dram-util': 'DRAM Utilization'
}

colors = {
    '8': 'tab:green',
    '16': 'tab:orange',
    '32': 'tab:blue'
}

linestyles = {
    '8': '-',
    '16': '--',
    '32': '-.'
}

markerstyles = {
    '8': 'o',
    '16': 's',
    '32': '^'
}

titles = {
    'compute': 'Compute Utilization',
    'dram': 'DRAM Utilization'
}

batch_sizes = ['8', '16', '32']

def plot_utilization(df, figname, version):
    plt.figure(figsize=(10, 7))
    x_ticks = np.arange(len(df['tile_dim']))
    for i, bs in enumerate(batch_sizes):
        plt.plot(x_ticks, df[bs], label=bs, color=colors[bs], linestyle = linestyles[bs], linewidth=4, marker=markerstyles[bs], markersize=10)
        # plt.plot(x_ticks, df[bs], label=bs)

    plt.xlabel('Tile Dimension (Q, K/V)', fontsize=32, fontweight='bold')
    y_label = 'Compute Utilization (%)' if version == 'compute' else 'HBM BW Utilization (%)'
    plt.ylabel(y_label, fontsize=28, fontweight='bold')
    x_tick_labels = df['tile_dim']
    plt.xticks(ticks=x_ticks, labels=x_tick_labels, fontsize=30)
    plt.yticks(np.arange(0, 110, 20))
    plt.grid()
    plt.legend(loc='best', ncol=3, fontsize=28)
    plt.title('context length = 4K', fontsize=28, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figname)

df_compute = pd.read_csv(indir + 'compute.csv', sep=';')
df_dram = pd.read_csv(indir + 'dram.csv', sep=';')
plot_utilization(df_compute, outdir + 'a.png', 'compute')
plot_utilization(df_dram, outdir + 'b.png', 'dram')