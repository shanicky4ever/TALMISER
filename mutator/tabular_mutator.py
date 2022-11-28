from mutator.base_mutator import BaseMutator
import random


class TabularMutator(BaseMutator):

    def _mutate(self, attribute_name, attribute_value):
        if self.mutated_data is None:
            self.mutated_data = self.data[self.data[attribute_name]
                                          == attribute_value]
        else:
            attr_max = max(self.data[attribute_name])
            attr_min = min(self.data[attribute_name])
            for index, row in self.mutated_data.iterrows():
                while (change_attr := random.choice(self.mutated_data.columns)) == attribute_name:
                    continue
                change_ratio = random.uniform(-self.bound, self.bound)
                if (row[attribute_name] == attr_max and change_ratio > 0) \
                        or (row[attribute_name] == attr_min and change_ratio < 0):
                    change_ratio = -change_ratio

                sym = 1 if change_ratio > 0 else -1
                change_value = 0
                if attr_max-attr_min >= 1/self.bound:
                    change_value = change_ratio*(attr_max-attr_min)
                    if row[attribute_name].dtype == 'int64':
                        change_value = int((abs(change_value)+0.5)*sym)
                else:
                    change_value = sym if change_ratio > self.bound/2 else 0
                self.mutated_data.at[index, change_attr] += change_value

        return self.mutated_data
