import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import argparse
import sys
infile = sys.argv[1]
outfile = sys.argv[2]

#parser = argparse.ArgumentParser(description='Plotting script for FlashAttention')
#parser.add_argument('--sysname', type=str, default='pod', help='Input CSV file')
#args = parser.parse_args()
sysname = 'POD (SM-aware)' #if args.sysname == 'pod' else 'SANGAM (SM-aware)'

plt.rcParams.update({'font.size': 28})
plt.rcParams.update({'font.family': 'Sans Serif'})

#configs = ['serial', 'kernel(streams)', 'cta(sequential)', 'cta(interleaved)', 'warp', 'cta(sm-aware)']
configs = ['serial', 'intra-thread', 'cta(sequential)', 'optimal', 'kernel(streams)', 'cta(sm-aware)']
legends = {
    'serial': 'Serial',
    'kernel(streams)': 'Kernel (streams)',
    'cta(sequential)': 'CTA',
    'warp': 'Warp',
    'intra-thread': 'Intra-thread',
    'intra-thread-barrier': 'Intra-thread (Barrier)',
    'optimal': "Optimal",
    'cta(sm-aware)': "SM-aware CTA (Ours)",
}

colors = {
    'serial': 'tab:orange',
    'kernel(streams)': 'tab:brown',
    'cta(sequential)': 'tab:cyan',
    'cta(interleaved)': 'tab:purple',
    'warp': 'tab:blue',
    'intra-thread': 'tab:pink',
    'intra-thread-barrier': 'tab:brown',
    'optimal': "tab:blue",
    'cta(sm-aware)': 'tab:green',
}

linestyles = {
    'serial': '-',
    'kernel(streams)': '--',
    'cta(sequential)': '-.',
    'cta(interleaved)': ':',
    'warp': '--',
    'intra-thread': ':',
    'intra-thread-barrier': ':',
    'optimal': "--",
    'cta(sm-aware)': '-'
}

markerstyles = {
    'serial': 'o',
    'kernel(streams)': 's',
    'cta(sequential)': 'D',
    'cta(interleaved)': 'X',
    'warp': 'P',
    'intra-thread': 'o',
    'intra-thread-barrier': 's',
    'optimal': "P",
    'cta(sm-aware)': 'D'
}

def plot_perf(plot, df):
    x_ticks = np.arange(len(df['computeiters']))
    handles = []
    labels = []
    for i, cfg in enumerate(configs):
        lw=4
        if cfg == 'cta(sm-aware)':
            lw=6
        line, = plot.plot(x_ticks, df[cfg] / 1000, label=legends[cfg], color=colors[cfg], linestyle = linestyles[cfg], linewidth=lw, marker=markerstyles[cfg], markersize=10)
        handles.append(line)
        labels.append(legends[cfg])

    x_label = 'Compute Iterations'
    x_tick_labels = df['computeiters'].tolist()
    if(plot == plt):
        plot.xlabel(x_label, fontsize=32, fontweight='bold')
        plot.ylabel('Runtime (ms)', fontsize=32, fontweight='bold')
        plot.xticks(ticks=x_ticks, labels=x_tick_labels, fontsize=32)
        plot.yticks(np.arange(0, 160, 20), fontsize=32)
        plot.text(1, 140, 'Memory-heavy', fontsize=28, fontweight='bold', color='red')
        plot.text(5.5, 140, 'Compute-heavy', fontsize=28, fontweight='bold', color='green')
    else:
        plot.set_xlabel(x_label, fontsize=28, fontweight='bold')
        plot.set_ylabel('Runtime (ms)', fontsize=28, fontweight='bold')
        plot.set_xticks(ticks=x_ticks, labels=x_tick_labels, fontsize=24)
        plot.set_yticks(ticks=np.arange(0, 160, 20),labels=np.arange(0, 160, 20), fontsize=24)
        plot.text(0, 120, 'Memory-heavy', fontsize=24, fontweight='bold', color='red')
        plot.text(4.5, 120, 'Compute-heavy', fontsize=24, fontweight='bold', color='green')
    plot.axvspan(0, 4, color='lightcoral', alpha=0.15)
    plot.axvspan(4, max(x_ticks), color='lightgreen', alpha=0.15)
    plot.grid()
    return handles, labels
    #plt.legend(loc='best', fontsize=30, ncol=2)

df = pd.read_csv(infile, sep='\t')
# Compute optimal
Mb = df["Mb"].tolist()
Cb = df["Cb"].tolist()
opt = [max(i, j) for i,j in zip(Mb, Cb)]

df2 = df.filter(regex='computeiters|serial|kernel.*|-barrier$')
df2.columns = df2.columns.str.replace('-barrier$', '', regex=True)
df2.insert(0, 'optimal', opt)

plt.figure(figsize=(20, 8))
plot_perf(plt, df2)
plt.xlim(0, 9)
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), fontsize=28, ncol=5)
plt.legend(loc='lower right', fontsize=28, ncol=3, frameon=True)
plt.savefig(outfile, bbox_inches='tight', pad_inches=0)

'''
fig, axs = plt.subplots(1, 2, figsize=(20, 8))
handles, labels = plot_perf(axs[0], df)
axs[0].set_title(f'No barriers', fontsize=32, fontweight='bold')
plot_perf(axs[1], df2)
axs[1].set_title(f'With barriers', fontsize=32, fontweight='bold')
fig.legend(handles, labels, loc='upper center', fontsize=20, ncol=5, frameon=True, bbox_to_anchor=(0.5, 1.05))
plt.savefig('perf-case-study2.pdf', bbox_inches='tight', pad_inches=0)
'''