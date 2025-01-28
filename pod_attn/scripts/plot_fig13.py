import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import re
import sys

context_lens = [1024, 2048, 4096, 8192, 16384]
d_batch_sizes = [8, 16, 32, 64, 128] #get_d_batch_sizes(model, cl)

#model;num_heads;num_kv_heads;head_size;bs;cl;fa_p;fa_d;fu_2cta;fu_4cta;
# Define the regex pattern for the header
header_pattern = re.compile(r'.*;.*;.*;.*;(.*);(.*);.*;.*;([0-9]+\.[0-9]+);([0-9]+\.[0-9]+);')

infile = sys.argv[1]
outfile = sys.argv[2]

speedup_2cta = []
speedup_4cta = []

with open(infile, newline='') as csvfile:
    for line in csvfile:
        match = header_pattern.search(line)
        if match:
            # Filter
            bs = int(match.group(1))
            cl = int(match.group(2))
            fu_2cta = float(match.group(3))
            fu_4cta = float(match.group(4))
            speedup_2cta.append(fu_2cta / min(fu_2cta, fu_4cta))
            speedup_4cta.append(fu_4cta / min(fu_2cta, fu_4cta))

speedup_2cta = np.array(speedup_2cta).reshape(len(context_lens), len(d_batch_sizes))
speedup_4cta = np.array(speedup_4cta).reshape(len(context_lens), len(d_batch_sizes))

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    #cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    #cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(range(data.shape[1]), labels=col_labels)
    ax.set_yticks(range(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w")
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

fig, (ax1, ax2) = plt.subplots(1, 2)

from  matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list('rg',["g", "y", "r"], N=256) 

im = heatmap(speedup_2cta, context_lens, d_batch_sizes, ax=ax1,
                   cmap=cmap, cbarlabel="harvest [t/year]")
texts = annotate_heatmap(im, valfmt="{x:.2f}")
ax1.set_title("2 CTAs per SM")
ax1.set_xlabel("Batch size")
ax1.xaxis.set_label_position('top') 
ax1.set_ylabel("Context length")


im = heatmap(speedup_4cta, context_lens, d_batch_sizes, ax=ax2,
                   cmap=cmap, cbarlabel="harvest [t/year]")
texts = annotate_heatmap(im, valfmt="{x:.2f}")
ax2.set_title("4 CTAs per SM")
ax2.set_xlabel("Batch size")
ax2.xaxis.set_label_position('top') 
ax2.set_ylabel("Context length")

fig.tight_layout()

plt.savefig(outfile, bbox_inches='tight', pad_inches=0) 