import numpy as np
from utils.dtmc import DTMCHandler
import os
import math
import logging


class BaseSolver:

    def __init__(self, dataHandler=None, configs=None, pretrained_model_path=None):
        self.dataHandler = dataHandler
        self.configs = configs
        self.pretrained_model_path = pretrained_model_path

    def train(self):
        raise NotImplementedError

    def _get_mutation_pred(self, attribute_name, attribute_value):
        raise NotImplementedError

    def _get_fairness_pred(self, attribute_name, attribute_value):
        raise NotImplementedError

    def generate_DTMC_by_actual_class(self, attribute_name, is_plot, eps=0.01, delta=0.05):
        label_unique = self.dataHandler.get_label_unique()
        value_distr = self.dataHandler.get_data()[attribute_name].unique()
        stats = np.zeros((len(value_distr), len(label_unique)))
        logging.info(f"{len(value_distr)} unique values in {attribute_name}")
        while len(_mutation_list := _calc_mutation_list(stats, eps, delta)) > 0:
            mutate_value = _mutation_list[0]
            results = self._get_mutation_pred(attribute_name, mutate_value)
            for r in results:
                stats[mutate_value][r] += 1
        return self._stats_to_DTMCHandler(stats, attribute_name, is_plot=is_plot)

    def generate_DTMC_by_fair_attribute(self, attribute_name, is_plot, eps=0.01, delta=0.05):
        stats = np.zeros((1, 2))
        while len(_mutation_list := _calc_mutation_list(stats, eps, delta)) > 0:
            mutate_value = _mutation_list[0]
            results = self._get_fairness_pred(attribute_name, mutate_value)
            for r in results:
                stats[mutate_value][r] += 1
        return self._stats_to_DTMCHandler(stats, attribute_name, is_plot=is_plot, fairness_DTMC=True)

    def _stats_to_DTMCHandler(self, stats, attribute_name, is_plot=False, fairness_DTMC=False):
        if not fairness_DTMC:
            para_names = [self.dataHandler.get_decode_data(
                attribute_name, i) for i in range(stats.shape[0])]
            label_names = [self.dataHandler.get_decode_data(
                self.dataHandler.get_label_name(), i) for i in range(stats.shape[1])]
            attr_name = attribute_name
        else:
            para_names = (attribute_name,)
            label_names = ["diff", "same"]
            attr_name = self.dataHandler.get_dataset_name()

        appendix = "fairness" if fairness_DTMC else None

        self.combine_name = _generate_combine_name(
            folder=self.dataHandler.get_dtmc_folder(),
            dataset=self.dataHandler.get_dataset_name(),
            model_name=self.configs['model'],
            attribute_name=attribute_name,
            appendix=appendix)

        handler = DTMCHandler()
        handler.build_dtmc(stats, attribute_name=attr_name,
                           para_names=para_names, label_names=label_names)
        handler.save_dtmc(f"{self.combine_name}.yaml")
        if is_plot:
            handler.save_dtmc_graph(f"{self.combine_name}.png")
        return handler

    def read_dtmc(self, dataset, model_name, attribute_name, configs=None):
        if not configs and not self.configs:
            raise ValueError("Configs must be provided")
        if configs:
            dtmc_folder = configs['base_config']['dtmc_folder']
            model = configs[f'{model_name}_config']['model']
            self.combine_name = _generate_combine_name(
                dtmc_folder, dataset, model, attribute_name)
        self.dtmc_handler = DTMCHandler()
        dtmc_file = f"{self.combine_name}.yaml"
        if not os.path.exists(dtmc_file):
            os.system(
                f"python DTMC_generator.py -d {dataset} -m {model_name} -a {attribute_name} --plot")
        self.dtmc_handler.load_dtmc(dtmc_file)

    def get_fair_pairs(self, fair_diff=0.05):
        assert self.dtmc_handler is not None
        return self.dtmc_handler.get_fair_pairs(fair_diff)

    def _all_num_equal(self, nums):
        ori = nums[0]
        for n in nums:
            if n != ori:
                return False
        return True


def _generate_combine_name(folder, dataset, model_name, attribute_name, appendix=None):
    name_list = [dataset, model_name, attribute_name]
    if appendix:
        name_list.append(appendix)
    return os.path.join(folder, model_name, '_'.join(name_list))


def _calc_mutation_list(stats, eps, delta):
    m_list = []
    for i, st in enumerate(stats):
        if sum(st) == 0:
            m_list.append(i)
            continue
        threshold = 2/(eps**2) * math.log(2/delta) * \
            (1/4 - (1/2-min(st)/sum(st)-2/3*eps)**2)
        if sum(st) < threshold:
            m_list.append(i)
    return m_list
