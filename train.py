import argparse
from utils.helper_function import load_obj, makedir
from utils.data import tabularDataHandler
from solvers.solver_provider import get_solver

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='census',
                        type=str, choices=['census', ])
    parser.add_argument('--model', default='DNN', type=str, choices=['DNN', ])
    args = parser.parse_args()

    configs = load_obj(f"configs/{args.dataset}.yaml")
    makedir("trained_model", del_before=False)

    dataHandler = tabularDataHandler(configs['base_config'], encode=True)

    solver = get_solver(args.model)(
        dataHandler, configs[f'{args.model}_config'])
    solver.train()
