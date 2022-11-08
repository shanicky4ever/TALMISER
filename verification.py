
from solvers import get_solver
from utils.helper_function import load_obj
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='census',
                        type=str, choices=['census', ])
    parser.add_argument('--model', default='DNN', type=str, choices=['DNN', ])
    parser.add_argument('-a', '--attribute', default='sex', type=str)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('-fd', '--fair_diff', default=0.05, type=float)
    args = parser.parse_args()

    configs = load_obj(f"configs/{args.dataset}.yaml")
    solver = get_solver(args.model)()
    solver.read_dtmc(args.dataset, args.model, args.attribute, configs)

    for fair_pair in (fair_pairs := solver.get_fair_pairs(args.fair_diff)):
        print(fair_pair)
    if len(fair_pairs) == 0:
        print("No fair pairs found.")
