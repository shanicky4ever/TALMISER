class BaseSolver:

    def __init__(self, dataHandler, configs):
        self.dataHandler = dataHandler
        self.configs = configs

    def train(self):
        raise NotImplementedError
