from pydtmc import MarkovChain, plot_graph
from utils.helper_function import load_obj, save_obj
import numpy as np


class DTMCHandler:
    def __init__(self, dtmc_array=None, node_labels=None):
        if dtmc_array is not None:
            self.dtmc_array = dtmc_array
            self.node_labels = [get_valid_labels(node_label)
                                for node_label in node_labels]
            self.mc = MarkovChain(
                self.dtmc_array, self.node_labels)

    def save_dtmc(self, path):
        save_dict = {
            'node_labels': self.node_labels,
            'dtmc_array': self.dtmc_array.tolist()
        }
        save_obj(save_dict, path)

    def save_dtmc_graph(self, path):
        fig, _ = plot_graph(self.mc)

        # A Workaround to complete figure
        fig_size = fig.get_size_inches()
        fig.set_size_inches(fig_size[0], fig_size[1]*1.5)

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


def get_valid_labels(node_label):
    label = ''.join(filter(str.isalnum, node_label))
    if label[0].isdigit():
        label = 'in' + label
    return label
