import argparse
from utils.helper_function import load_obj, makedir
from utils.data import tabularDataHandler
from solvers import get_solver
import os
import logging
logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='census',)
    parser.add_argument('-m', '--model', default='DNN')
    parser.add_argument('-a', '--attribute', default='sex', type=str)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--epsilon', default=0.01, type=float)
    parser.add_argument('--delta', default=0.05, type=float)
    parser.add_argument('--o', default='fair', choices=['fair', 'ori'])
    args = parser.parse_args()

    configs = load_obj(f"configs/{args.dataset}.yaml")
    makedir("dtmc_results", del_before=False)
    makedir(configs['base_config']['dtmc_folder'], del_before=False)
    makedir(os.path.join(configs['base_config']
            ['dtmc_folder'], configs[f'{args.model}_config']['model']), del_before=False)

    dataHandler = tabularDataHandler(configs['base_config'], encode=False)

    solver = get_solver(args.model)(
        dataHandler=dataHandler, configs=configs[f'{args.model}_config'],
        pretrained=True)
    if args.o == 'fair':
        dtmc_handler = solver.generate_DTMC_by_fair_attribute(
            args.attribute, is_plot=args.plot, eps=args.epsilon, delta=args.delta)
    else:
        dtmc_handler = solver.generate_DTMC_by_actual_class(
            args.attribute, is_plot=args.plot, eps=args.epsilon, delta=args.delta)
