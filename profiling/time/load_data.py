import sys
sys.path.insert(0, '../util/')

import prof_util as pu
import numpy

data = []

#import pdb; pdb.set_trace()

with open("bld/time_log.txt") as f:
    for line in f:
        data.append([float(e) for e in line.split("|")[1:-1]])

pu.graph_data(data, "Run time (ms)")
