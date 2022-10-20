
from solvers.solver_provider import get_solver
from utils.helper_function import load_obj
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='census',
                        type=str, choices=['census', ])
    parser.add_argument('--model', default='DNN', type=str, choices=['DNN', ])
    parser.add_argument('--attribute', default='sex', type=str)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    configs = load_obj(f"configs/{args.dataset}.yaml")
    solver = get_solver(args.model)()
    solver.read_dtmc(args.dataset, args.model, args.attribute, configs)
