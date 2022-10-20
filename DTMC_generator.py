import argparse
from utils.helper_function import load_obj, makedir
from utils.data import tabularDataHandler
from solvers.solver_provider import get_solver
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='census',
                        type=str, choices=['census', ])
    parser.add_argument('--model', default='DNN', type=str, choices=['DNN', ])
    parser.add_argument('--attribute', default='sex', type=str)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    configs = load_obj(f"configs/{args.dataset}.yaml")
    makedir("dtmc_results", del_before=False)
    makedir(configs['base_config']['dtmc_folder'], del_before=False)

    dataHandler = tabularDataHandler(configs['base_config'], encode=False)

    solver = get_solver(args.model)(
        dataHandler=dataHandler, configs=configs[f'{args.model}_config'],
        pretrained=True)
    dtmc_handler = solver.generate_DTMCHandler(args.attribute)
    combine_name = os.path.join(
        configs['base_config']['dtmc_folder'], f"{args.dataset}_{args.model}_{args.attribute}")
    dtmc_handler.save_dtmc(f"{combine_name}.npy")
    if args.plot:
        dtmc_handler.save_dtmc_graph(f"{combine_name}.png")
