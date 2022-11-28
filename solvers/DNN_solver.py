from solvers.base_solver import BaseSolver
from models.DNN import modelHandler


class DNNSolver(BaseSolver):
    def __init__(self, dataHandler=None, configs=None, pretrained=False):
        super().__init__(dataHandler, configs, pretrained)
        if dataHandler:
            input_dim = dataHandler.get_input_dim()
            pretrained_model_path = None if not pretrained else self.configs['weight_file']
            self.modelHandler = modelHandler(
                input_dim=input_dim, dnn_config=configs,
                pretrained_model_path=pretrained_model_path)

    def train(self):
        train_loader, val_loader = self.dataHandler.generate_dataloader_for_DNN(
            batch_size=self.configs['batch_size'])
        self.modelHandler.train(train_loader, val_loader)

    def _get_mutation_pred(self, attribute_name, attribute_value):
        mutated_data = self.dataHandler.mutator.mutate(
            attribute_name, attribute_value)
        mutated_loader = self.dataHandler.generate_dataloader_for_DNN(
            data=mutated_data, batch_size=self.configs['batch_size'], split=False)
        results = self.modelHandler.simple_forward(mutated_loader)
        return results

    def _get_fairness_pred(self, attribute_name, attribute_value):
        outputs = []
        for attr in self.dataHandler.get_attr_unique(attribute_name):
            mutated_data = self.dataHandler.mutator.mutate(
                attribute_name, attr, assign_fair_value=True)
            mutated_loader = self.dataHandler.generate_dataloader_for_DNN(
                data=mutated_data, batch_size=self.configs['batch_size'], split=False)
            outputs.append(self.modelHandler.simple_forward(mutated_loader))
        results = [1 if self._all_num_equal([ou[i] for ou in outputs]) else 0
                   for i in range(len(outputs[0]))]
        return results
