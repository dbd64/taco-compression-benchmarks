import csv
import glob
import sys
import re

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def merge_csv(pattern, outf):
    print(pattern)
    with open(outf, 'w') as out:
        csvout = csv.writer(out)
        files = glob.glob(pattern)
        files = natural_sort(files)
        header = True
        for f in files:
            if f == outf:
                continue
            with open(f, 'r') as fin:
                if header:
                    header = False
                else:
                    fin.__next__()
                for line in csv.reader(fin):
                    csvout.writerow(line)



def main(argv):
    folder = argv[0] if argv[0][-1] == '/' else argv[0] + '/'
    out = argv[1]
    prefix = argv[2] if len(argv) > 2 else ""
    postfix = argv[3] if len(argv) > 3 else ""

    merge_csv(folder + prefix + r"*" + postfix + '.csv', out)

if __name__ == "__main__":
   main(sys.argv[1:])
