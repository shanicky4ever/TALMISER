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

    def save_dtmc(self, path):
        save_array = np.vstack((np.array(self.node_labels), self.dtmc_array))
        np.save(path, save_array)

    def save_dtmc_graph(self, path):
        plt.ioff()
        fig, ax = plot_graph(self.mc)
        ax.axis('off')
        fig.savefig(path)

    def load_dtmc(self, path):
        load_array = np.load(path)
        self.node_labels = load_array[0]
        self.dtmc_array = load_array[1:]
        self.mc = MarkovChain(
            self.dtmc_array, self.node_labels)

    def get_dtmc_array(self):
        return self.dtmc_arrays

    def get_dtmc_node_labels(self):
        return self.node_labels

    def get_dtmc_controller(self):
        return self.mc
