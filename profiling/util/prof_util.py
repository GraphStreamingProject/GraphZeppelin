import numpy as np
import matplotlib.pyplot as plt

def graph_data (data, y_label):
    data = np.array(data)
    plt.plot(data[:,0], data[:,1], "o-", label="Sketch-based CC")
    plt.plot(data[:,0], data[:,2], "d-", label="Sketch-based bipartite test")
    plt.plot(data[:,0], data[:,3], "s-", label="BGL bipartite test")
    plt.legend()
    plt.xlabel("Number of Vertices")
    plt.ylabel(y_label)
    plt.grid(linestyle="--", axis="y")
    plt.ylim(0, data.max())
    plt.show()
         
def format_data (data, filename):
    with open(filename, "w") as f:
        for row in data: 
            line = "|"
            for elem in row:
                line += str(elem) + "|"
            line += "\n"

            f.write(line)    
