import matplotlib.pyplot as plt
import csv
import glob
import sys
import re
from pathlib import PurePath
from collections import defaultdict
from statistics import mean, median

plt.rcParams['figure.dpi'] = 300

def bar_plot(ax, data, labels=None, colors=None, total_width=0.8, single_width=1, legend=True, y_max=None):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    labels:

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    xticks = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
        
        xticks = []

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])
            xticks.append(x)

            if y_max and y > y_max:
                rect = bar[0]
                plt.text(rect.get_x() + rect.get_width() / 2.0, y_max, f'{y:.0f}x', ha='center', va='bottom', fontsize=3)

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())

    if labels:
        ax.set_xticks(xticks)
        ax.set_xticklabels(labels, fontsize=7)

    if y_max:
        ax.set_ylim([0, y_max])


def get_kind(k):
    if row['kind'] == "DENSE" or row['kind'] == "SPARSE":
        return row['kind'].title()
    else:
        return row['kind']

def process_data(data):
    out = {"Dense": [], "Sparse": [], "RLE": [], "RLEP": [], "LZ77": []}
    for k in data.keys():
        for i, val in enumerate(data[k]):
            out[k].append(data["Dense"][i]/val)
    return out


if __name__ == "__main__":
    # all_path = '/data/scratch/danielbd/artifact/out/spmv/spmv_all.csv' if len(sys.argv) < 2 else sys.argv[1]
    all_path = '/data/scratch/danielbd/artifact/out/spmv_rlep/spmv_all.csv' if len(sys.argv) < 2 else sys.argv[1]
    out_loc = './' if len(sys.argv) < 3 else sys.argv[2]
    data_mean = {"Dense": [], "Sparse": [], "RLE": [], "RLEP": [], "LZ77": []}
    data_size = {"Dense": [], "Sparse": [], "RLE": [], "RLEP": [], "LZ77": []}
    labels = []

    with open(all_path, 'r') as fin:
        csvin = csv.DictReader(fin)
        for row in csvin:
            # source,index,kind,total_vals,total_bytes,mean,stddev,median
            kind = get_kind(row['kind'])
            source = row['source']
            data_mean[kind].append(float(row['mean']))
            data_size[kind].append(int(row['total_bytes']))
            if kind == "Dense":
                labels.append(source)
            else:
                assert source == labels[-1]

    # print(data_mean)
    # print(data_size)

    data_thpt = {"Dense": [], "Sparse": [], "RLE": [], "RLEP": [], "LZ77": []}
    for k in data_mean.keys():
        for i in range(len(data_mean[k])):
            data_thpt[k].append(data_size[k][i] / data_mean[k][i])

    rle_sparse_overhead = []
    for i in range(len(data_thpt["Sparse"])):
        rle_sparse_overhead.append(data_thpt["Sparse"][i] / data_thpt["RLE"][i])

    print(rle_sparse_overhead)
    print(mean(rle_sparse_overhead))
    print(median(rle_sparse_overhead))


    data_mean = process_data(data_mean)
    data_size = process_data(data_size)


    # print(data_mean)
    # print(data_size)

    
            # if row['kind'] == "DENSE":
            #     pass
            # elif row['kind'] == "SPARSE":
            #     pass
            # elif row['kind'] == "RLE":
            #     pass
            # elif row['kind'] == "LZ77":
            #     pass
            # else: 
            #     print("ignoring unknown kind: {}".format(row['kind']))


    fig, ax = plt.subplots()
    bar_plot(ax, data_mean, labels=labels, total_width=.8, single_width=.9, y_max=4)
    plt.savefig("{}spmv_mean.png".format(out_loc))
    plt.close()

    fig, ax = plt.subplots()
    bar_plot(ax, data_size, labels=labels, total_width=.8, single_width=.9, y_max=10)
    plt.savefig("{}spmv_size.png".format(out_loc))
    plt.close()

    fig, ax = plt.subplots()
    bar_plot(ax, data_thpt, labels=labels, total_width=.8, single_width=.9)
    plt.savefig("{}spmv_thpt.png".format(out_loc))
    plt.close()
