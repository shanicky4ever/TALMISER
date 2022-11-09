from .DNN_solver import DNNSolver
from .svm_solver import SvmSolver
from .random_forest_solver import RandomForestSolver


def get_solver(model_class):
    solver_dict = {
        'DNN': DNNSolver,
        'svm': SvmSolver,
        'rf': RandomForestSolver
    }
    assert model_class in solver_dict
    return solver_dict[model_class]
