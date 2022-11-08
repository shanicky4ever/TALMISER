

class BaseMutator:
    def __init__(self, data, bound=1/5) -> None:
        self.data = data
        assert 0 <= bound <= 1, "bound should be in (0,1)"
        self.bound = bound
        self.mutated_data = None
        self.last_mutate = -1

    def mutate(self, attribute_name, attribute_value):
        if attribute_value != self.last_mutate:
            self.reset_mutated_data()
        self.last_mutate = attribute_value
        return self._mutate(attribute_name, attribute_value)

    def _mutate(self, attribute_name, attribute_value):
        raise NotImplementedError

    def reset_mutated_data(self):
        self.mutated_data = None
