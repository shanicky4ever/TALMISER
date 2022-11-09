from solvers import get_solver
from utils.helper_function import load_obj
import argparse
import logging
logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='census',)
    parser.add_argument('-m', '--model', default='DNN')
    parser.add_argument('-a', '--attribute', default='sex', type=str)
    parser.add_argument('-fd', '--fair_diff', default=0.05, type=float)
    args = parser.parse_args()

    configs = load_obj(f"configs/{args.dataset}.yaml")
    solver = get_solver(args.model)()
    solver.read_dtmc(args.dataset, args.model, args.attribute, configs)

    if len(fair_pairs := solver.get_fair_pairs(args.fair_diff)) > 0:
        logging.info(f"Found {len(fair_pairs)} fair pairs as follows:")
        for index, fair_pair in enumerate(fair_pairs):
            print(f"{index}. {fair_pair}")
    else:
        logging.warning("No fair pairs found.")
