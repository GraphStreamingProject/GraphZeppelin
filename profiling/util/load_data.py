import sys
import prof_util as pu
import numpy

data = []

#import pdb; pdb.set_trace()

with open(sys.argv[1]) as f:
    for line in f:
        data.append([float(e) for e in line.split("|")[1:-1]])

pu.graph_data(data, sys.argv[2])
