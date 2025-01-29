import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

import sys
infile = sys.argv[1]
outfile = sys.argv[2]

plt.rcParams.update({'font.size': 28})
plt.rcParams.update({'font.family': 'Sans Serif'})

configs = ['equal-llama', 'proportional-llama', 'equal-yi', 'proportional-yi']

legends = {
    'equal-llama': 'Llama-3-8B (50:50)',
    'proportional-llama': 'Llama-3-8B (proportional)',
    'equal-yi': 'Yi-6B (50:50)',
    'proportional-yi': 'Yi-6B (proportional)'
}

colors = {
    'equal-llama': 'tab:orange',
    'proportional-llama': 'tab:blue',
    'equal-yi': 'tab:cyan',
    'proportional-yi': 'tab:purple'
}

linestyles = {
    'equal-llama': '-',
    'proportional-llama': '-',
    'equal-yi': '--',
    'proportional-yi': '--'
}

markerstyles = {
    'equal-llama': 'o',
    'proportional-llama': 's',
    'equal-yi': 'o',
    'proportional-yi': 's'
}

def plot_perf(df, figname):
    plt.figure(figsize=(16, 6.5))
    x_ticks = np.arange(len(df['bs']))
    for i, cfg in enumerate(configs):
        plt.plot(x_ticks, df[cfg], label=legends[cfg], color=colors[cfg], linestyle = linestyles[cfg], linewidth=3, marker=markerstyles[cfg], markersize=10)

    x_label = 'Batch Size'
    plt.xlabel(x_label, fontsize=32, fontweight='bold')
    plt.ylabel('Time (ms)', fontsize=32, fontweight='bold')
    x_tick_labels = df['bs'].tolist()
    plt.xticks(ticks=x_ticks, labels=x_tick_labels, fontsize=32)
    plt.yticks(np.arange(0, 7, 1), fontsize=32)
    plt.grid()
    plt.legend(loc='best', fontsize=30, ncol=2)
    #plt.title('Llama-3-8B (TP-2)', fontsize=22, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figname)

df = pd.read_csv(infile, sep='\t')
plot_perf(df, outfile)