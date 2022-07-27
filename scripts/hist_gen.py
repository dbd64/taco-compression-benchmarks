# import matplotlib
# matplotlib.use('GTK4Agg') 

import matplotlib.pyplot as plt
import csv
import glob
import sys
import re
from pathlib import PurePath
from collections import defaultdict
from statistics import mean, median

plt.rcParams['figure.dpi'] = 300

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("ERROR")
        exit(1)
    hist_csv = sys.argv[1]
    out_loc = sys.argv[2]

    runs = []
    counts = []

    with open(hist_csv, 'r') as fin:
        csvin = csv.DictReader(fin)
        for row in csvin:
            runs.append(int(row['run length']))
            counts.append(int(row['count']))

    print("len : {}".format(len(runs)))
    if len(runs) != len(counts):
        print("len counts: {}".format(len(counts)))


    fig, ax = plt.subplots()
    ax.set_yscale("log")
    # ax.set_xscale("log")
    bound = 2**15
    bound = 785
    if len(runs) > bound:
        ax.bar(runs[:bound],counts[:bound])
    else:
        ax.bar(runs,counts)
    plt.savefig("{}".format(out_loc))
    plt.close()
