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

    # def generate_DTMCHandler(self, attribute_name, is_plot=False):
    #     # TODO: Add a function to fit the formula
    #     all_loader = self.dataHandler.generate_dataloader_for_DNN(
    #         batch_size=self.configs['batch_size'], split=False)
    #     results = self.modelHandler.simple_forward(all_loader)
    #     value_distr = self.dataHandler.get_data()[attribute_name]
    #     stats = np.zeros((len(value_distr.unique()),
    #                       len(pd.unique(results))))
    #     for v, r in zip(value_distr, results):
    #         stats[v][r] += 1
    #     return self._stats_to_DTMCHandler(stats, attribute_name, is_plot=is_plot)
