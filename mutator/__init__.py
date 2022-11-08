from .tabular_mutator import TabularMutator


def get_mutator(data_type):
    mutator_dict = {
        'tabular': TabularMutator,
    }
    return mutator_dict[data_type]
