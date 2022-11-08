import argparse
from utils.helper_function import load_obj, makedir
from utils.data import tabularDataHandler
from solvers.solver_provider import get_solver

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='census',
                        type=str, choices=['census', ])
    parser.add_argument('--model', default='DNN', type=str, choices=['DNN', ])
    parser.add_argument('--attribute', default='sex', type=str)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--epsilon', default=0.01, type=float)
    parser.add_argument('--delta', default=0.05, type=float)
    args = parser.parse_args()

    configs = load_obj(f"configs/{args.dataset}.yaml")
    makedir("dtmc_results", del_before=False)
    makedir(configs['base_config']['dtmc_folder'], del_before=False)

    dataHandler = tabularDataHandler(configs['base_config'], encode=False)

    solver = get_solver(args.model)(
        dataHandler=dataHandler, configs=configs[f'{args.model}_config'],
        pretrained=True)
    dtmc_handler = solver.generate_DTMCHandler(
        args.attribute, is_plot=args.plot, eps=args.epsilon, delta=args.delta)
