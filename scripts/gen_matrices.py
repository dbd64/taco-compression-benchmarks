import sys
import numpy as np
from numpy.random import default_rng
import argparse

def write_mtx_to_csv(mtx, outfile):
    np.savetxt(outfile, np.asarray(mtx), delimiter=',', fmt='%i')


def matrix(args):
    rng = default_rng(args.seed)
    ii64 = np.iinfo(np.int16)


    nrows = args.width
    ncols = args.height 
    mtx = np.empty((nrows, ncols))

    run_cnt_rem = args.run_count
    lit_cnt_rem = args.lit_count

    choice_probs = to_dist(np.asarray([run_cnt_rem, lit_cnt_rem]))

    for col in range(nrows):
        print("*** COL: {}".format(col))
        row = 0
        while row < ncols:
            # print(" * ROW: {}".format(row))
            # print(choice_probs)
            if rng.choice([True, False], p=choice_probs):
                run_len = rng.choice(args.run_hist[:,0], p=args.run_dist)
                run_len = min(np.int64(run_len), ncols-row)
                run_val = rng.integers(low=ii64.min, high=ii64.max)
                # ii = np.r_[row:(row+run_len)]
                mtx[col, row:(row+run_len)] = run_val
                row += run_len
                # for i in range(run_len):
                #     mtx[row,col] = run_val
                #     row+=1
                # run_cnt_rem -= run_len
            else:
                lit_len = rng.choice(args.lit_hist[:,0], p=args.lit_dist)
                lit_len = min(np.int64(lit_len), ncols-row)

                mtx[col,row:(row+lit_len)] = rng.integers(low=ii64.min, high=ii64.max,size=lit_len)
                row += lit_len


                # for i in range(lit_len):
                #     val = rng.integers(low=ii64.min, high=ii64.max)
                #     mtx[row,col] = val
                #     row+=1

                # lit_cnt_rem -= lit_len
    return np.transpose(mtx)

    # elements = [1.1, 2.2, 3.3]
    # probabilities = [0.2, 0.5, 0.3]
    # np.random.choice(elements, 10, p=probabilities)


    # pass

# Load histogram csv files
def load_hist(filename):
    arr = np.genfromtxt(filename, delimiter=',')
    arr = arr[1:,:] # Remove header
    return arr

# Normalize to ensure this is a distribution
def to_dist(hist):
    return hist / np.sum(hist) 

def main(args):
    # Load histograms and compute derived values
    args.run_hist = load_hist(args.run_hist)
    args.lit_hist = load_hist(args.lit_hist)

    args.run_dist = to_dist(args.run_hist[:,1])
    args.lit_dist = to_dist(args.lit_hist[:,1])

    args.run_count = np.sum(args.run_hist[:,1])
    args.lit_count = np.sum(args.lit_hist[:,1])

    mtx = matrix(args)
    write_mtx_to_csv(mtx, args.out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run_hist", help="filename containing run counts table", default="")
    parser.add_argument("-l", "--lit_hist", help="filename containing literal length count table", default="")
    parser.add_argument("-o", "--out_file", help="output filename", default="/data/scratch/danielbd/spmv_data/rand.csv")
    parser.add_argument("-w", "--width", type=int, help="Width of the generated matrix", default=54)
    parser.add_argument("-n", "--height", type=int, help="Height of the generated matrix", default=581012)
    parser.add_argument("-s", "--seed", type=int, help="Random Seed", default=0)
    args = parser.parse_args()

    main(args)