from solvers.base_solver import BaseSolver
from models.DNN import modelHandler


class DNNSolver(BaseSolver):
    def __init__(self, dataHandler, configs):
        super().__init__(dataHandler, configs)
        input_dim = dataHandler.get_input_dim()
        self.modelHandler = modelHandler(
            input_dim=input_dim, dnn_config=configs)

    def train(self):
        train_loader, val_loader = self.dataHandler.generate_dataloader_for_DNN()
        self.modelHandler.train(train_loader, val_loader)
