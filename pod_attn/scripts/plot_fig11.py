import re
import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse

#parser = argparse.ArgumentParser(description='Plotting script for FlashAttention')
#parser.add_argument('--sysname', type=str, default='pod', help='Input CSV file')
#args = parser.parse_args()
sysname = 'POD' #if args.sysname == 'pod' else 'SANGAM'

#model;cl;cs;kv_len;chunk_id;bs;fa_p;fa_d;fa_serial;fa_stream;fi_p;fi_d;fi_serial;fi_batched;HFuse;fa_fused;best_fused_op;speedup_fa_serial;
# Define the regex pattern for the header
header_pattern = re.compile(r'.*;.*;.*;.*;.*;.*;(.*);(.*);(.*);(.*);.*;.*;(.*);(.*);(.*);(.*);.*;.*;')

import sys
csv_file_path = sys.argv[1]
outfile = sys.argv[2]

peak = []
fa_serial = []
fa_stream = []
fi_serial = []
fi_batched = []
hfuse = []
fa_fused = []
with open(csv_file_path, newline='') as csvfile:
    skip = 0
    for line in csvfile:
        # Skip first line
        if skip == 0:
            skip = 1
            continue
        match = header_pattern.search(line)
        if match:
            # Filter
            fa_p = float(match.group(1))
            fa_d = float(match.group(2))
            fa_s = float(match.group(3))

            if(fa_p < 0.2 * fa_s or fa_d < 0.2 * fa_s):
                continue
            # Append to results
            peak.append(max(fa_p, fa_d) + (0.1 * min(fa_p, fa_d)))
            fa_serial.append(float(match.group(3)))
            fa_stream.append(float(match.group(4)))
            fi_serial.append(float(match.group(5)))
            fi_batched.append(float(match.group(6)))
            hfuse.append(float(match.group(7)))
            fa_fused.append(float(match.group(8)))

speedup = []
#speedup.append([i / j for i, j in zip(fa_serial, peak)])
speedup.append([i / j for i, j in zip(fa_serial, fa_stream)])
speedup.append([i / j for i, j in zip(fa_serial, fi_serial)])
speedup.append([i / j for i, j in zip(fa_serial, fi_batched)])
speedup.append([i / j for i, j in zip(fa_serial, hfuse)])
speedup.append([i / j for i, j in zip(fa_serial, fa_fused)])

schemes = [
    'fa_streams',
    'fi_serial',
    'fi_fused',
    'Hfuse',
    'fa_fused',
]

colors = {
    'fa_streams': 'tab:brown',
    'fi_serial': 'cadetblue',
    'fi_fused': 'tab:purple',
    'Hfuse': 'tab:red',
    'fa_fused': 'tab:green'
}

labels = {
    'fa_streams': 'FA_Streams',
    'fi_serial': 'FI_Serial',
    'fi_fused': 'FI_Batched',
    'Hfuse': 'FA_HFuse',
    'fa_fused': sysname
}


#titles = data[0]
#values = np.transpose(data[1:])

# Create the violin plot
plt.figure(figsize=(18, 7))
#fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
axs = ['a']
axs[0] = plt.subplot(111)
# plot violin plot
vp = axs[0].violinplot(speedup,
                  showmeans=False,
                  showmedians=True)
#axs[0].set_title('Violin plot')

# plot box plot
#axs[1].boxplot(speedup)
#axs[1].set_title('Box plot')

lines = ['cbars', 'cmaxes', 'cmedians', 'cmins']

# adding horizontal grid lines
for ax in axs:
    ax.yaxis.grid(True)
    ax.set_xticks([y + 1 for y in range(len(speedup))],
                  labels=[labels[i] for i in schemes],
                  fontsize=30)
    for line in lines:
        vp[line].set_color("black")
        vp[line].set_alpha(0.5)
        vp[line].set_linewidth(2.0)

    x_vals = np.array(ax.get_xlim())
    y_vals = 1 + 0 * x_vals
    plt.plot(x_vals, y_vals, '-', color='black', linewidth=2.5)

    vals = ax.get_yticks()
    ax.set_yticklabels(['{:.0%}'.format(x - 1.0) for x in vals],
                       fontsize=26)
    
    #ax.set_xlabel('')
    ax.set_ylabel('Normalized speedup', fontsize=30, fontweight='bold')
    for it, title in enumerate(schemes):
        vp['bodies'][it].set_facecolor(colors[title])
        vp['bodies'][it].set_alpha(0.75)
        #vp['bodies'][it].set_label(labels[title])
# Show the plot
#plt.show()
plt.savefig(outfile, bbox_inches='tight', pad_inches=0) 