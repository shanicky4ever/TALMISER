from pydtmc import MarkovChain, plot_graph
from matplotlib import pyplot as plt
from utils.helper_function import load_obj, save_obj
import numpy as np


class DTMCHandler:
    def __init__(self, dtmc_array=None, node_labels=None):
        if dtmc_array is not None:
            self.dtmc_array = dtmc_array
            self.node_labels = [''.join(filter(str.isalnum, l))
                                for l in node_labels]
            self.mc = MarkovChain(
                self.dtmc_array, self.node_labels)

    def save_dtmc(self, path):
        save_dict = {
            'node_labels': self.node_labels,
            'dtmc_array': self.dtmc_array.tolist()
        }
        save_obj(save_dict, path)

    def save_dtmc_graph(self, path):
        plt.ioff()
        fig, ax = plot_graph(self.mc)
        ax.axis('off')
        fig.savefig(path)

    def load_dtmc(self, path):
        load_dict = load_obj(path)
        self.node_labels = load_dict['node_labels']
        self.dtmc_array = np.array(load_dict['dtmc_array'])
        self.mc = MarkovChain(
            self.dtmc_array, self.node_labels)

    def get_dtmc_array(self):
        return self.dtmc_arrays

    def get_dtmc_node_labels(self):
        return self.node_labels

    def get_dtmc_controller(self):
        return self.mc
