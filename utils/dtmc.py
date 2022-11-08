from pydtmc import MarkovChain, plot_graph
from utils.helper_function import load_obj, save_obj
import numpy as np


class DTMCHandler:

    def __init__(self):
        pass

    def build_dtmc(self, stats, attribute_name, para_names, label_names):
        self.para_names = [get_valid_name(name) for name in para_names]
        self.label_names = [get_valid_name(name) for name in label_names]
        self.node_names = [get_valid_name(attribute_name)] + \
            self.para_names + self.label_names

        dtmc_nodes_num = 1 + stats.shape[0] + stats.shape[1]
        self.dtmc_array = np.zeros((dtmc_nodes_num, dtmc_nodes_num))
        for i in range(stats.shape[0]):
            self.dtmc_array[0][i+1] = sum(stats[i])/sum(sum(stats))
            for j in range(stats.shape[1]):
                self.dtmc_array[1+i][1+stats.shape[0] +
                                     j] = stats[i][j]/sum(stats[i])
        for j in range(stats.shape[1]):
            self.dtmc_array[1+stats.shape[0]+j][1+stats.shape[0]+j] = 1.0
        self.mc = MarkovChain(self.dtmc_array, self.node_names)

    def save_dtmc(self, path):
        save_dict = {
            'node_names': self.node_names,
            'dtmc_array': self.dtmc_array.tolist(),
            'para_names': self.para_names,
            'label_names': self.label_names
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
        self.nodel_names = load_dict['node_names']
        self.para_names = load_dict['para_names']
        self.label_names = load_dict['label_names']
        self.dtmc_array = np.array(load_dict['dtmc_array'])
        self.mc = MarkovChain(self.dtmc_array, self.nodel_names)

    def get_dtmc_array(self):
        return self.dtmc_arrays

    def get_dtmc_nodel_names(self):
        return self.nodel_names

    def get_dtmc_controller(self):
        return self.mc


def get_valid_name(nodel_name):
    label = ''.join(filter(str.isalnum, nodel_name))
    if label[0].isdigit():
        label = 'in' + label
    return label
