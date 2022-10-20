import numpy as np
import pandas as pd
from utils.dtmc import DTMCHandler


class BaseSolver:

    def __init__(self, dataHandler, configs, pretrained_model_path=None):
        self.dataHandler = dataHandler
        self.configs = configs
        self.pretrained_model_path = pretrained_model_path

    def train(self):
        raise NotImplementedError

    def generate_DTMC(self, attribute_name):
        raise NotImplementedError

    def _stats_to_DTMCHandler(self, stats, attribute_name):
        dtmc_nodes_num = 1 + stats.shape[0] + stats.shape[1]
        DTMC = np.zeros((dtmc_nodes_num, dtmc_nodes_num))
        for i in range(stats.shape[0]):
            DTMC[0][i+1] = sum(stats[i])/sum(sum(stats))
            for j in range(stats.shape[1]):
                DTMC[1+i][1+stats.shape[0]+j] = stats[i][j]/sum(stats[i])
        for j in range(stats.shape[1]):
            DTMC[1+stats.shape[0]+j][1+stats.shape[0]+j] = 1.0

        node_labels = [attribute_name] + \
            [self.dataHandler.get_decode_data(
                attribute_name, i) for i in range(stats.shape[0])] +\
            [self.dataHandler.get_decode_data(
                self.dataHandler.get_label_name(), i) for i in range(stats.shape[1])]
        return DTMCHandler(DTMC, node_labels)
