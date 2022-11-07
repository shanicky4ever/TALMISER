from solvers import DNNSolver
from solvers.mutator import TabularMutator


def get_solver(model_class):
    solver_dict = {
        'DNN': DNNSolver,
    }
    return solver_dict[model_class]


def get_mutator(data_type):
    mutator_dict = {
        'tabular': TabularMutator,
    }
    return mutator_dict(data_type)
