from .DNN_solver import DNNSolver
from .svm_solver import SvmSolver


def get_solver(model_class):
    solver_dict = {
        'DNN': DNNSolver,
        'svm': SvmSolver
    }
    assert model_class in solver_dict
    return solver_dict[model_class]
