from solvers.base_solver import BaseSolver
from models.svm import modelHandler


class SvmSolver(BaseSolver):
    def __init__(self, dataHandler=None, configs=None, pretrained=False):
        super().__init__(dataHandler, configs, pretrained)
        if dataHandler:
            pretrained_model_path = None if not pretrained else self.configs['weight_file']
            self.modelHandler = modelHandler(configs, pretrained_model_path)

    def train(self):
        X_train, X_val, y_train, y_val = self.dataHandler.split_train_val()
        self.modelHandler.train(X_train, X_val, y_train, y_val)

    def _get_mutation_pred(self, attribute_name, attribute_value):
        mutated_data = self.dataHandler.mutator.mutate(
            attribute_name, attribute_value)
        results = self.modelHandler.simple_forward(mutated_data)
        return results

    def _get_fairness_pred(self, attribute_name, attribute_value):
        outputs = []
        for attr in self.dataHandler.get_attr_unique(attribute_name):
            mutated_data = self.dataHandler.mutator.mutate(
                attribute_name, attr, assign_fair_value=True)
            outputs.append(self.modelHandler.simple_forward(mutated_data))
        results = [1 if self._all_num_equal([ou[i] for ou in outputs]) else 0
                   for i in range(len(outputs[0]))]
        return results
