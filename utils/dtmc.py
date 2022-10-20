from pydtmc import MarkovChain, plot_graph
import numpy as np
from matplotlib import pyplot as plt


class DTMCHandler:
    def __init__(self, dtmc_array, node_labels=None):
        self.dtmc_array = dtmc_array
        self.node_labels = [''.join(filter(str.isalnum, l))
                            for l in node_labels]
        self.mc = MarkovChain(
            self.dtmc_array, self.node_labels)

    def save_dtmc_array(self, path):
        np.save(path, self.dtmc_array)

    def save_dtmc_graph(self, path):
        plt.ioff()
        fig, ax = plot_graph(self.mc)
        ax.axis('off')
        fig.savefig(path)
