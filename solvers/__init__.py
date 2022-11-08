from .DNN_solver import DNNSolver


def get_solver(model_class):
    solver_dict = {
        'DNN': DNNSolver,
    }
    return solver_dict[model_class]
