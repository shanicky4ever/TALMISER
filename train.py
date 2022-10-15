import argparse
from utils.helper_function import load_obj, makedir
from utils.data import tabularDataHandler
from models import DNN

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='census',
                        type=str, choices=['census', ])
    parser.add_argument('--model', default='DNN', type=str, choices=['DNN', ])
    args = parser.parse_args()

    configs = load_obj(f"configs/{args.dataset}.yaml")
    makedir("trained_model", del_before=False)

    dataHandler = tabularDataHandler(configs['base_config'])

    if args.model == 'DNN':
        train_loader, val_loader = dataHandler.generate_dataloader_for_DNN()
        model_handler = DNN.modelHandler(
            dataHandler.get_input_dim(), configs['DNN_config'])
        model_handler.train(train_loader, val_loader)
