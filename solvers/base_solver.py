import numpy as np
from utils.dtmc import DTMCHandler
import os
import pandas as pd


class BaseSolver:

    def __init__(self, dataHandler=None, configs=None, pretrained_model_path=None):
        self.dataHandler = dataHandler
        self.configs = configs
        self.pretrained_model_path = pretrained_model_path

    def train(self):
        raise NotImplementedError

    def _get_pred(self):
        raise NotImplementedError

    def generate_DTMCHandler(self, attribute_name, is_plot):
        # TODO: Add a function to fit the formula
        results = self._get_pred()
        value_distr = self.dataHandler.get_data()[attribute_name]
        stats = np.zeros((len(value_distr.unique()),
                          len(pd.unique(results))))
        for v, r in zip(value_distr, results):
            stats[v][r] += 1
        return self._stats_to_DTMCHandler(stats, attribute_name, is_plot=is_plot)

    def _stats_to_DTMCHandler(self, stats, attribute_name, is_plot=False):
        print(stats)
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
        handler = DTMCHandler(DTMC, node_labels)
        self.combine_name = _generate_combine_name(
            folder=self.dataHandler.get_dtmc_folder(),
            dataset=self.dataHandler.get_dataset_name(), model_name=self.configs['model'], attribute_name=attribute_name)
        handler.save_dtmc(f"{self.combine_name}.yaml")
        if is_plot:
            handler.save_dtmc_graph(f"{self.combine_name}.png")
        return handler

    def read_dtmc(self, dataset, model_name, attribute_name, configs=None):
        if not configs and not self.configs:
            raise ValueError("Configs must be provided")
        if configs:
            dtmc_folder = configs['base_config']['dtmc_folder']
            self.combine_name = _generate_combine_name(
                dtmc_folder, dataset, model_name, attribute_name)
        handler = DTMCHandler()
        handler.load_dtmc(f"{self.combine_name}.yaml")


def _generate_combine_name(folder, dataset, model_name, attribute_name):
    return os.path.join(folder, f"{dataset}_{model_name}_{attribute_name}")
