import sys 
sys.path.insert(0, '../util/')

import subprocess
import prof_util as pu
import numpy as np
import matplotlib.pyplot as plt

domain = np.arange(100, 200, 10)
num_trials = 10

req_data = np.zeros((len(domain), 3))
extra_data = np.zeros((len(domain), 3))

for k in range(len(domain)):
    print("Profiling on " + str(domain[k]) + " vertex graphs")
    for i in range(1, num_trials+1):
        subprocess.run(["./bld/stream_gen", str(domain[k]),
            "stream.txt"]) 
        for algo in range(3):
            subprocess.run(["valgrind", "--tool=massif", "--depth=1",
                "--massif-out-file=memlog.txt", "./bld/runner", 
                str(algo), "stream.txt"])

            req_heap = 0;
            extra_heap = 0; 
            with open("memlog.txt", "r") as f:
                line = prev1 = prev2 = prev3 = f.readline()
                while line != "heap_tree=peak\n":
                    prev3 = prev2
                    prev2 = prev1
                    prev1 = line
                    line = f.readline()
                req_heap = int(prev3[prev3.find("=")+1:-1])
                extra_heap = int(prev2[prev2.find("=")+1:-1])

            req_data[k][algo] = ((i - 1) * req_data[k][algo] 
                    + req_heap) / i 
            extra_data[k][algo] = ((i - 1) * extra_data[k][algo] 
                    + extra_heap) / i

total_data = req_data + extra_data

req_data = np.concatenate((domain.reshape((len(domain), 1)), 
    req_data), axis=1)
total_data = np.concatenate((domain.reshape((len(domain), 1)), 
    total_data), axis=1)

pu.format_data(req_data, "req_mem.txt")
pu.format_data(total_data, "total_mem.txt")
pu.graph_data(req_data, "Peak memory consumption (bytes)")
pu.graph_data(total_data, "Peak memory consumption (bytes)")
