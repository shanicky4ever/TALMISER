from torch import nn
import torch
import tqdm


class MLP(nn.Module):
    '''
        Multilayer Perceptron for regression.
    '''

    def __init__(self, input_dim):
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum
    return acc


class modelHandler:
    def __init__(self, input_dim, dnn_config):
        self.input_dim = input_dim
        self.dnn_config = dnn_config
        self.model = MLP(self.input_dim)

    def train(self, train_loader, val_loader):
        self.model.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.dnn_config['lr'])
        loss_function = nn.BCEWithLogitsLoss()

        for _ in tqdm.tqdm(range(self.dnn_config['epochs']), ncols=160):
            for _, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()
                y_hat = self.model(x.float())
                loss = loss_function(y_hat, y.float().view(-1, 1))
                loss.backward()
                optimizer.step()
        self.evaluate(val_loader=val_loader)

    def evaluate(self, val_loader):
        self.model.eval()
        acc = 0
        for x, y in val_loader:
            y_pred = self.model(x.float())
            acc += binary_acc(y_pred, y.float().view(-1, 1))
        print(f'val acc is {acc/val_loader.dataset.__len__()}')

    def get_model(self):
        return self.model

    def save_model(self):
        torch.save(self.model.state_dict(), self.dnn_config["weight_file"])
